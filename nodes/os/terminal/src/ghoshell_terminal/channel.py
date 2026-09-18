"""The model-facing channel: commands become cards, cards become processes.

The pose this channel takes is that the human is a busy collaborator, not an
approval gate. Issuing a command returns a receipt immediately — the model never
blocks on a human's response time. The command becomes a card on the terminal
surface, and whatever happens to it next (approved, denied, talked about,
expired) reaches the model as a signal, not as a return value.

Every card settles exactly once; the waiter tasks spawned here are the single
place where ``awaiting`` turns into ``running`` or ``rejected``. The web surface
only hands down verdicts — it never writes card state itself.

Signals go out through an injected ``signaler`` rather than through
``CommandUtil``, for the same reason the surface is injected: the node wires both
to ``Matrix`` and tests wire both to a list.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from typing import Any, Protocol

from ghoshell_moss.contracts.subprocesses import (
    CaptureSpec,
    ManagedProcess,
    Subprocesses,
)
from ghoshell_moss.core.blueprint.channel_builder import CommandUtil, new_channel
from ghoshell_moss.core.blueprint.mindflow import Priority
from ghoshell_moss.core.concepts.channel import Channel
from ghoshell_moss.message import Message
from ghoshell_moss.signals import NotifySignalMeta

from .card import Card, CardState, CardType
from .poller import stream_output
from .store import CardStore, Mode
from .surface import StopHandles

__all__ = ["build_terminal_channel"]

_LEVELS: dict[str, Priority] = {
    "background": Priority.BACKGROUND,
    "info": Priority.INFO,
    "warning": Priority.WARNING,
}

_OUTPUT_THRESHOLD = 8000
"""Above this many characters the full output is a file, and the model reads the
path instead of the text."""

_BUFFER_LINES = 500
"""In-memory tail window the subprocess layer keeps. Also the ceiling on how much
output can go unseen between two polls."""

_TAG_SAFE = re.compile(r"[^A-Za-z0-9_]")


class _Surface(Protocol):
    async def broadcast(self, frame: dict[str, Any]) -> None: ...


class _NoSurface:
    async def broadcast(self, frame: dict[str, Any]) -> None:
        return None


def _notice_name(thread: str) -> str:
    return f"thread_{_TAG_SAFE.sub('_', thread)}"


def build_terminal_channel(
    store: CardStore,
    processes: Subprocesses,
    *,
    surface: _Surface | None = None,
    signaler: Callable[[Any], None] | None = None,
    stops: StopHandles | None = None,
    name: str = "terminal",
    description: str | None = None,
    enabled: Callable[[], bool] | None = None,
) -> Channel:
    """Compose the terminal channel over a store and a subprocess owner.

    :param store: cards, threads, rules and mode — the single source of truth,
        shared with the web surface.
    :param processes: the subprocess owner. The node passes ``matrix.processes``
        so subprocess lifetimes belong to the node, not to this channel.
    :param surface: the web broadcaster. None = headless (tests, or a node run
        without a browser).
    :param signaler: how a message reaches the ghost. Plain callable so the node
        can pass ``matrix.send_signal_to_ghost`` and tests can pass a list.
    :param enabled: live availability gate. Return False and every command drops
        out of the model's interface (that is how ``mode=disabled`` reads).
    """
    surface = surface or _NoSurface()
    enabled = enabled or (lambda: True)

    async def _signal(card: Card) -> None:
        """Tell the ghost a card settled. Never lost, always guaranteed a turn."""
        if signaler is None:
            return
        if card.output_chars > _OUTPUT_THRESHOLD and card.output_file:
            result = f"output: {card.output_chars} chars, full text at {card.output_file}"
        elif card.output_tail:
            tail = "".join(card.output_tail).rstrip()
            result = f"output:\n{tail}"
        else:
            result = "no output"
        text = (
            f"[{name} #{card.id}] {card.type.value} '{card.title}' "
            f"state={card.state.value} exit={card.exit_code} "
            f"({card.elapsed()}s)\n{result}\n"
            f"read({card.id}) for detail"
        )
        level = card.level if card.level in _LEVELS else "info"
        signal = NotifySignalMeta(next=True).to_signal(
            Message.new(tag=name, name=card.thread or name).with_content(text),
            description=f"terminal card #{card.id} {card.state.value}",
            priority=_LEVELS[level],
        )
        signaler(signal)

    async def _emit(frame: dict[str, Any]) -> None:
        await surface.broadcast(frame)

    async def _head(card: Card) -> None:
        await _emit({"type": "card.head", "card": card.view()})

    async def _tail(card: Card) -> None:
        await _emit({"type": "card.tail", "card": card.view()})

    async def _full(card: Card) -> None:
        await _emit({"type": "card.full", "card": card.view()})

    _live: dict[int, ManagedProcess] = {}

    # -- running a settled command ------------------------------------------

    async def _run(card: Card) -> None:
        try:
            managed = await processes.shell(
                card.content,
                name=f"{name}:{card.thread}" if card.thread else name,
                description=card.description,
                cwd=card.cwd or None,
                capture=CaptureSpec(buffer_lines=_BUFFER_LINES),
            )
        except Exception as e:
            store.set_state(card.id, CardState.ERROR)
            store.append_output(card.id, [f"[failed to spawn] {e}\n"])
            await _full(card)
            await _signal(card)
            return
        _live[card.id] = managed
        card.process_index = managed.meta.index
        await _full(card)

        async def on_lines(lines: list[str]) -> None:
            store.append_output(card.id, lines)
            await _emit({"type": "card.output", "id": card.id, "lines": lines})

        try:
            await stream_output(managed, on_lines)
        finally:
            _live.pop(card.id, None)
        code = managed.process.returncode
        store.set_state(
            card.id, CardState.DONE if code == 0 else CardState.ERROR, exit_code=code
        )
        await _full(card)
        await _signal(card)

    async def _await_verdict(card: Card) -> None:
        """Park until the human (or the model's own cancel) settles the card."""
        verdict = await store.waiter(card.id)
        if store.get(card.id) is None or store.get(card.id).state is not CardState.AWAITING:
            return
        if verdict == "accept":
            store.set_state(card.id, CardState.RUNNING)
            await _full(card)
            await _run(card)
        else:
            store.set_state(card.id, CardState.REJECTED)
            await _full(card)

    async def _await_rule(card: Card) -> None:
        verdict = await store.waiter(card.id)
        if store.get(card.id) is None or store.get(card.id).state is not CardState.AWAITING:
            return
        if verdict == "accept":
            store.activate_rule(card.id)
            store.set_state(card.id, CardState.DONE)
        else:
            store.set_state(card.id, CardState.REJECTED)
        await _full(card)

    # -- channel ------------------------------------------------------------

    chan = new_channel(name=name, description=description or (
        "run shell commands as cards. Every command is a proposal the human "
        "sees, may question, and accepts or denies; the model is never blocked "
        "on that decision."
    ))

    @chan.build.instruction
    def instruction() -> str:
        root = store.root
        return (
            "Shell commands become cards on a human-facing terminal. exec() returns "
            "a receipt at once — it does not wait for the human. A card is settled "
            "by the human (accept / deny / ask) and again when the process ends; both "
            "arrive as signals, so read(id) is how you learn what happened. Commands "
            "run in a thread: open(thread, cwd, description) first, then exec into it. "
            f"Every cwd must live inside {root}. "
            "rule() proposes a regex; once the human accepts it, matching commands run "
            "without asking while the terminal is in auto mode. The terminal has three "
            "modes — approval (ask every time), auto (rules decide), disabled (your "
            "commands are not available). The mode and pending/running counts appear "
            "in this channel's notice."
        )

    # -- threads ------------------------------------------------------------

    @chan.build.command(name="open", always_observe=False, available=enabled)
    async def open_thread(thread: str, cwd: str = "", description: str = "") -> str:
        """Open a named thread: where commands run and what they are for.

        ``thread`` is your handle for every later exec(). ``cwd`` must resolve
        inside the terminal root; empty = the root itself.
        """
        try:
            t = store.open_thread(thread, cwd, description)
        except ValueError as e:
            CommandUtil.raise_observe(str(e))
        return f"thread {t.name!r} opened @ {t.cwd}"

    @chan.build.command(name="threads", always_observe=True, available=enabled)
    async def threads() -> str:
        """List the threads you have open."""
        lines = [
            f"- {t.name} @ {t.cwd}" + (f"  {t.description}" if t.description else "")
            for t in store.threads()
        ]
        return "\n".join(lines) if lines else "(no threads open — open() one first)"

    # -- commands -----------------------------------------------------------

    @chan.build.command(name="exec", blocking=False, always_observe=False, available=enabled)
    async def exec_cmd(
        thread: str, chunks__: str, desc: str = "", level: str = "info"
    ) -> str:
        """Run a shell line in ``thread``. Returns a receipt immediately.

        The line streams onto a card the human watches appear. ``desc`` is the
        card's subtitle — say what this command is for. ``level`` picks how loudly
        the completion signal reaches you: background / info / warning.
        """
        t = store.get_thread(thread)
        if t is None:
            CommandUtil.raise_observe(f"no thread {thread!r} — open() one first")
        card = store.new_card(
            CardType.COMMAND,
            title=thread,
            description=desc,
            thread=thread,
            cwd=t.cwd,
            level=level if level in _LEVELS else "info",
        )
        await _head(card)
        try:
            async for chunk in chunks__:
                store.append_content(card.id, chunk)
                await _emit({"type": "card.delta", "id": card.id, "text": chunk})
        except asyncio.CancelledError:
            # The model's streaming was cut short (interpreter stopped). Note it
            # and let the framework turn this into a STOPPED the model can read.
            store.set_state(card.id, CardState.CANCELLED)
            asyncio.get_running_loop().create_task(_tail(card))
            CommandUtil.reraise_stopped(
                f"[{name} #{card.id}] command cancelled while being written"
            )
            return ""

        if not card.content.strip():
            store.set_state(card.id, CardState.CANCELLED)
            await _tail(card)
            CommandUtil.raise_observe(
                f"[{name} #{card.id}] empty command — put the shell line in the tag body"
            )

        auto = store.mode == Mode.AUTO and store.match_rule(card.content) is not None
        if auto:
            store.set_state(card.id, CardState.RUNNING)
            await _tail(card)
            CommandUtil.create_task(_run(card))
            return f"[{name} #{card.id}] {thread} running (auto-approved)"
        store.set_state(card.id, CardState.AWAITING)
        await _tail(card)
        CommandUtil.create_task(_await_verdict(card))
        return (
            f"[{name} #{card.id}] {thread} awaiting approval — the human decides on "
            f"the terminal surface; you will be signalled"
        )

    @chan.build.command(name="rule", always_observe=False, available=enabled)
    async def rule(name_: str, pattern: str, description: str = "") -> str:
        """Propose an auto-approval regex. The human decides whether it goes live.

        Once accepted, any command whose text matches it runs without asking while
        the terminal is in auto mode. Matched with re.search.
        """
        try:
            re.compile(pattern)
        except re.error as e:
            CommandUtil.raise_observe(f"not a usable regex: {e}")
        card = store.new_card(
            CardType.RULE, title=name_, description=description, thread="", cwd=""
        )
        store.append_content(card.id, pattern)
        store.set_state(card.id, CardState.AWAITING)
        await _head(card)
        await _tail(card)
        CommandUtil.create_task(_await_rule(card))
        return f"[{name} #{card.id}] rule proposal '{name_}' awaiting approval"

    # -- reading and stopping -----------------------------------------------

    def _report(card: Card) -> str:
        lines = [f"[{name} #{card.id}] {card.type.value} state={card.state.value}"]
        if card.title:
            lines.append(f"title: {card.title}")
        if card.description:
            lines.append(f"desc: {card.description}")
        if card.thread:
            lines.append(f"thread: {card.thread} cwd: {card.cwd}")
        if card.process_index is not None:
            lines.append(f"process: #{card.process_index}")
        if card.exit_code is not None:
            lines.append(f"exit: {card.exit_code}")
        lines.append(f"elapsed: {card.elapsed()}s")
        if card.state is CardState.AWAITING:
            lines.append("(awaiting the human's verdict on the terminal surface)")
        for d in card.dialogue:
            lines.append(f"{d.author}: {d.text}")
        if card.output_chars > _OUTPUT_THRESHOLD and card.output_file:
            lines.append(f"output ({card.output_chars} chars): {card.output_file}")
        elif card.output_tail:
            lines.append("output:\n" + "".join(card.output_tail).rstrip())
        else:
            lines.append("output: (none yet)")
        return "\n".join(lines)

    @chan.build.command(name="read", always_observe=True, available=enabled)
    async def read_card(card_id: int) -> str:
        """Read one card: state, process, exit code, and its output.

        Long output is kept in a file — then this returns the path, not the text.
        """
        card = store.get(card_id)
        if card is None:
            CommandUtil.raise_observe(f"no card #{card_id} — cards() to list them")
        return _report(card)

    @chan.build.command(name="cards", always_observe=True, available=enabled)
    async def cards() -> str:
        """List every card on the terminal, newest last."""
        out = []
        for c in store.cards():
            preview = c.content.strip().splitlines()[0][:60] if c.content.strip() else ""
            out.append(f"#{c.id} {c.type.value} {c.state.value} [{c.thread}] {preview}")
        return "\n".join(out) if out else "(no cards yet)"

    async def _stop_card(card_id: int) -> str:
        card = store.get(card_id)
        if card is None:
            return f"[{name} #{card_id}] no such card"
        managed = _live.get(card_id)
        if managed is None:
            return f"[{name} #{card_id}] nothing running (state={card.state.value})"
        await managed.stop()
        return f"[{name} #{card_id}] stopped, exit={managed.process.returncode}"

    async def _stop_all() -> str:
        targets = list(_live.items())
        for _, managed in targets:
            await managed.stop()
        return f"[{name}] stopped {len(targets)} process(es)"

    if stops is not None:
        stops.stop = _stop_card
        stops.stop_all = _stop_all

    @chan.build.command(name="stop", always_observe=False, available=enabled)
    async def stop(card_id: int) -> str:
        """Stop the process behind a running card (SIGINT, then SIGKILL)."""
        if store.get(card_id) is None:
            CommandUtil.raise_observe(f"no card #{card_id} — cards() to list them")
        return await _stop_card(card_id)

    @chan.build.command(name="stop_all", always_observe=False, available=enabled)
    async def stop_all() -> str:
        """Stop every running card."""
        return await _stop_all()

    @chan.build.command(name="cancel", always_observe=False, available=enabled)
    async def cancel(card_id: int) -> str:
        """Withdraw a card that is still awaiting the human's verdict."""
        card = store.get(card_id)
        if card is None:
            CommandUtil.raise_observe(f"no card #{card_id} — cards() to list them")
        if store.settle(card_id, "cancel"):
            store.set_state(card_id, CardState.CANCELLED)
            await _full(card)
            return f"[{name} #{card_id}] withdrawn"
        return f"[{name} #{card_id}] not awaiting anything (state={card.state.value})"

    # -- context ------------------------------------------------------------

    @chan.build.named_notices
    def notices() -> dict[str, str]:
        """Warm state: the mode, and each thread's cwd with its card counts.

        Counts only — never ticking values, which would rewrite the fragment every
        frame for no information.
        """
        awaiting = store.awaiting()
        running = store.running()
        out = {
            "terminal": (
                f"mode: {store.mode} | awaiting: {len(awaiting)} | "
                f"running: {len(running)}"
            )
        }
        for t in store.threads():
            n_await = sum(1 for c in awaiting if c.thread == t.name)
            n_run = sum(1 for c in running if c.thread == t.name)
            desc = f" | {t.description}" if t.description else ""
            out[_notice_name(t.name)] = (
                f"cwd: {t.cwd} | awaiting: {n_await} | running: {n_run}{desc}"
            )
        return out

    @chan.build.context_messages
    def cards_context() -> list[str]:
        hot = [c for c in store.cards() if not c.settled]
        if not hot:
            return []
        lines = [f"[{name}] cards in flight (read(id) for detail):"]
        for c in hot:
            preview = c.content.strip().splitlines()[0][:60] if c.content.strip() else ""
            lines.append(
                f"  #{c.id} {c.state.value} [{c.thread}] {c.elapsed()}s {preview!r}"
            )
        return ["\n".join(lines)]

    return chan
