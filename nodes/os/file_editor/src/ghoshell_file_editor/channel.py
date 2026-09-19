"""The model-facing channel: every action becomes a card on the human surface.

The pose: the human and the model share one perception of the same text. A
``read`` therefore does two things at once — it hands the text back to the model
immediately, and it leaves a card behind so the human sees exactly what the model
saw. An edit lands in memory the moment it is written; nothing asks permission.
The only action that waits for a human is ``export``, because landing on disk is
the only real side effect.

``export`` returns a receipt at once — the model never blocks on a person's
response time. Its verdict arrives as a signal, and ``history(thread)`` says
where the thread ended up.

Signals go out through an injected ``signaler`` rather than through
``CommandUtil``, for the same reason the surface is injected: the node wires both
to ``Matrix`` and tests wire both to a list.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from ghoshell_moss.core.blueprint.channel_builder import CommandUtil, new_channel
from ghoshell_moss.core.blueprint.mindflow import Priority
from ghoshell_moss.core.concepts.channel import Channel
from ghoshell_moss.message import Message
from ghoshell_moss.signals import AsideSignalMeta, NotifySignalMeta

from .projection import action_view, thread_view
from .store import DocStore
from .structure import Action, Thread, line_count, slice_region

__all__ = ["build_file_editor_channel"]

_READ_EXCERPT = 4000
"""How much of a read stays on its card. The model gets the whole text; the card
only needs to be enough for a human to recognize what was read."""

_READ_LIMIT = 200_000
"""Ceiling on the text ``read`` returns, so one command cannot flood a context."""

_LEVELS: dict[str, Priority] = {
    "background": Priority.BACKGROUND,
    "info": Priority.INFO,
    "warning": Priority.WARNING,
}

_SAFE = re.compile(r"[^A-Za-z0-9_]")


def _notice_name(thread_id: str) -> str:
    return f"thread_{_SAFE.sub('_', thread_id)}"


class _Surface(Protocol):
    async def broadcast(self, frame: dict[str, Any]) -> None: ...


class _NoSurface:
    async def broadcast(self, frame: dict[str, Any]) -> None:
        return None


def parse_ops(raw: str) -> list[tuple[str, str]]:
    """Parse the ``str_replace`` body: a str_replace-protocol payload.

    Accepts one op or a list, each ``{"old_str": ..., "new_str": ...}``. The body
    arrives as text so the model can wrap it in CDATA and keep multi-line strings
    readable.
    """
    text = raw.strip()
    if not text:
        raise ValueError("the body is empty — put the JSON ops in the tag body")
    try:
        payload = json.loads(text)
    except ValueError as e:
        raise ValueError(f"the body is not valid JSON: {e}")
    if isinstance(payload, dict) and "edits" in payload:
        payload = payload["edits"]
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list) or not payload:
        raise ValueError("expected one op or a non-empty list of ops")
    ops: list[tuple[str, str]] = []
    for i, op in enumerate(payload):
        if not isinstance(op, dict):
            raise ValueError(f"op {i + 1} is not an object")
        old = op.get("old_str")
        new = op.get("new_str")
        if not isinstance(old, str) or not isinstance(new, str):
            raise ValueError(
                f"op {i + 1} needs string old_str and new_str (got "
                f"{sorted(op)!r})"
            )
        ops.append((old, new))
    return ops


def build_file_editor_channel(
    store: DocStore,
    *,
    surface: _Surface | None = None,
    signaler: Callable[[Any], None] | None = None,
    enabled: Callable[[], bool] | None = None,
    name: str = "file_editor",
    description: str | None = None,
    surface_url: str | Callable[[], str] | None = None,
) -> Channel:
    """Compose the file-editor channel over a store.

    :param store: threads, actions and drafts — the single source of truth,
        shared with the web surface.
    :param surface: the web broadcaster. None = headless (tests, or a node run
        without a browser).
    :param signaler: how a message reaches the ghost. Plain callable so the node
        can pass ``matrix.send_signal_to_ghost`` and tests can pass a list.
    :param enabled: live availability gate. Return False and every command drops
        out of the model's interface.
    :param surface_url: where the human surface lives, surfaced as a warm ``url``
        notice fragment so the model discovers it without a fixed port. May be a
        callable (resolved lazily, after the surface binds its ephemeral port).
    """
    surface = surface or _NoSurface()
    enabled = enabled or (lambda: True)

    def _url() -> str:
        if surface_url is None:
            return ""
        return surface_url() if callable(surface_url) else surface_url

    async def _emit(frame: dict[str, Any]) -> None:
        await surface.broadcast(frame)

    async def _card(thread: Thread, action: Action) -> None:
        await _emit(
            {"type": "action", "thread": thread.id, **action_view(action)}
        )

    async def _threads() -> None:
        await _emit(
            {"type": "threads", "threads": [thread_view(t) for t in store.threads()]}
        )

    def _notify(text: str, level: str = "info", *, next_: bool = True) -> None:
        """A must-not-lose message. ``next`` guarantees the ghost a turn."""
        if signaler is None:
            return
        signal = NotifySignalMeta(next=next_).to_signal(
            Message.new(tag=name).with_content(f"[{name}] {text}"),
            description=text[:120],
            priority=_LEVELS.get(level, Priority.INFO),
        )
        signaler(signal)

    def _aside(text: str) -> None:
        """A fact the ghost notices without being interrupted."""
        if signaler is None:
            return
        signal = AsideSignalMeta().to_signal(
            Message.new(tag=name).with_content(f"[{name}] {text}"),
            description=text[:120],
        )
        signaler(signal)

    def _get_thread(thread_id: str) -> Thread:
        thread = store.get(thread_id)
        if thread is None:
            CommandUtil.raise_observe(
                f"no thread {thread_id!r} — threads() lists them"
            )
        return thread

    chan = new_channel(
        name=name,
        description=description
        or (
            "a shared working copy of some text: every action becomes a card the "
            "human watches, and export is the only step that touches the disk"
        ),
    )

    @chan.build.instruction
    def instruction() -> str:
        return (
            "You and the human share one working copy per thread. open() starts "
            "one — from a file, from a draft, or blank (say what it is for with "
            "label). read() returns the text to you at once and leaves a card so "
            "the human sees the same thing; every edit (write / append / "
            "str_replace / rewind) lands in memory immediately and becomes a card "
            "too. Nothing asks permission except export(): the text reaches disk "
            "only once the human accepts it, and that ends the thread — open a new "
            "one to keep working on the file. history(thread) is the index of "
            "actions on a thread."
        )

    # -- lifecycle ----------------------------------------------------------

    @chan.build.command(name="open", always_observe=False, available=enabled)
    async def open_thread(
        thread: str, path: str = "", label: str = "", draft: str = ""
    ) -> str:
        """Start an editable object. ``thread`` is your handle for every later
        command.

        ``path`` loads a readable text file as the baseline; ``draft`` continues
        from a working copy left behind by a crash (see the drafts notice); with
        neither, the thread starts blank. An editable object does not have to come
        from a document — ``label`` is what the human sees it called.
        """
        try:
            opened = store.open(thread, label, path=path or None, draft=draft or None)
        except (KeyError, ValueError) as e:
            CommandUtil.raise_observe(str(e))
        await _threads()
        loc = f" @ {opened.path}" if opened.path else ""
        return (
            f"[{name}] thread {opened.id!r} open{loc} — v0, "
            f"{line_count(opened.content)} lines. read({opened.id!r}) to see it."
        )

    @chan.build.command(name="close", always_observe=False, available=enabled)
    async def close_thread(thread: str) -> str:
        """Abandon a thread: the working copy is dropped and nothing is written.

        Use it when the work is going nowhere — it frees the thread's slot.
        """
        target = store.get(thread)
        if target is None:
            CommandUtil.raise_observe(f"no thread {thread!r}")
        store.close(thread)
        await _threads()
        return f"[{name}] thread {thread!r} closed (nothing written)"

    # -- reading ------------------------------------------------------------

    @chan.build.command(name="read", always_observe=True, available=enabled)
    async def read(thread: str, region: str = "") -> str:
        """Return a thread's text to you, and leave a card showing the same text.

        ``region`` is a line range like ``"10-40"`` (1-based, inclusive); empty
        reads the whole working copy. Reading changes nothing — but the card is
        how the human shares your view of the file instead of guessing at it.
        """
        target = _get_thread(thread)
        if target.state != "live":
            CommandUtil.raise_observe(
                f"thread {thread!r} is {target.state} — read the file itself, or "
                f"open a new thread"
            )
        try:
            text = slice_region(target.content, region)
        except ValueError as e:
            CommandUtil.raise_observe(str(e))
        excerpt = text[:_READ_EXCERPT]
        if len(text) > _READ_EXCERPT:
            excerpt += "\n… (truncated on the card)"
        action = store.record(
            target.id, "read", f"read {region or 'all'}", text=excerpt, payload=region
        )
        await _card(target, action)
        if len(text) > _READ_LIMIT:
            return (
                f"{text[:_READ_LIMIT]}\n… (truncated at {_READ_LIMIT} chars — read a "
                f"region for the rest)"
            )
        return text

    # -- mutations ----------------------------------------------------------

    @chan.build.command(name="write", always_observe=False, available=enabled)
    async def write(thread: str, chunks__: str, label: str = "") -> str:
        """Replace the whole working copy with what you write.

        The body streams in and lands the moment the tag closes. ``label`` is the
        card's title — say what the new text is.
        """
        return await _stream(thread, "write", chunks__, label or "rewrite")

    @chan.build.command(name="append", always_observe=False, available=enabled)
    async def append(thread: str, chunks__: str, label: str = "") -> str:
        """Add what you write to the end of the working copy.

        The way to build a long document a section at a time: the card shows the
        segment you added, not the whole file.
        """
        return await _stream(thread, "append", chunks__, label or "append")

    async def _stream(thread: str, kind: str, chunks: Any, label: str) -> str:
        target = store.get(thread)
        if target is None:
            CommandUtil.raise_observe(f"no thread {thread!r} — threads() lists them")
        try:
            action = store.begin(thread, kind, label)
        except (KeyError, ValueError) as e:
            CommandUtil.raise_observe(str(e))
        await _card(target, action)
        try:
            async for chunk in chunks:
                store.feed(thread, action.n, chunk)
        except asyncio.CancelledError:
            store.cancel(thread, action.n)
            await _card(target, action)
            CommandUtil.reraise_stopped(
                f"[{name}] {kind} on {thread!r} cancelled while being written"
            )
            return ""
        try:
            landed = store.tail(thread, action.n)
        except ValueError as e:
            CommandUtil.raise_observe(str(e))
        await _card(target, landed)
        await _threads()
        return (
            f"[{name}] {thread!r} v{target.version} — {label} "
            f"({line_count(target.content)} lines)"
        )

    @chan.build.command(name="str_replace", always_observe=False, available=enabled)
    async def str_replace(thread: str, text__: str, label: str = "") -> str:
        """Edit by swapping exact snippets — one card per op.

        The body is JSON in the tag body (CDATA it, so quotes and newlines are
        safe): one op, or a list of them, each ``{"old_str": ..., "new_str": ...}``.
        Every ``old_str`` must appear exactly once in the text at the moment its op
        runs, so include enough surrounding context to be unique. Ops apply in
        order, so a later op sees the earlier ones' result.

        <![CDATA[ [{"old_str": "## Usage\\n", "new_str": "## Usage\\n\\nRun `moss start`.\\n"}] ]]>
        """
        target = store.get(thread)
        if target is None:
            CommandUtil.raise_observe(f"no thread {thread!r} — threads() lists them")
        try:
            ops = parse_ops(text__)
            actions = store.replace(thread, ops, label or f"str_replace ×{len(ops)}")
        except (KeyError, ValueError) as e:
            CommandUtil.raise_observe(str(e))
        for action in actions:
            await _card(target, action)
        await _threads()
        return (
            f"[{name}] {thread!r} v{target.version} — {len(actions)} card(s), "
            f"{line_count(target.content)} lines"
        )

    @chan.build.command(name="rewind", always_observe=False, available=enabled)
    async def rewind(thread: str, n: int) -> str:
        """Put the working copy back where action ``n`` left it (``0`` = baseline).

        Append-only: this adds an action, it does not erase the ones after it. Use
        history(thread) to pick ``n``.
        """
        target = store.get(thread)
        if target is None:
            CommandUtil.raise_observe(f"no thread {thread!r} — threads() lists them")
        try:
            action = store.rewind(thread, n)
        except (KeyError, ValueError) as e:
            CommandUtil.raise_observe(str(e))
        await _card(target, action)
        await _threads()
        return f"[{name}] {thread!r} v{target.version} — back to v{n}"

    # -- the one side effect ------------------------------------------------

    @chan.build.command(name="export", always_observe=False, available=enabled)
    async def export(thread: str, path: str = "") -> str:
        """Propose writing the working copy to ``path`` (default: the file it was
        opened from).

        Nothing reaches the disk until the human accepts — unless the thread is
        auto-trusted and this export goes to its established target, in which case
        it writes at once. Accepting ends the thread; open a new one to keep
        editing that file. Returns a receipt at once.
        """
        target = _get_thread(thread)
        if target.state != "live":
            CommandUtil.raise_observe(f"thread {thread!r} is {target.state}")
        dest = path or target.path
        if not dest:
            CommandUtil.raise_observe(
                f"thread {thread!r} has no path — pass one to export"
            )
        try:
            dest = store.resolve_target(dest)
        except ValueError as e:
            CommandUtil.raise_observe(str(e))
        # Auto only when there is an established target and this export goes to
        # it — writing somewhere new is establishing a new target, and that still
        # asks. A pathless thread can never be auto (trust without an object).
        auto = target.auto and bool(target.path) and dest == target.path
        action = store.record(
            thread,
            "export",
            f"export to {dest}",
            text=target.content,
            payload=dest,
            state=("applied" if auto else "awaiting"),
        )
        await _card(target, action)
        if auto:
            CommandUtil.create_task(_do_export(target, action, dest))
            return (
                f"[{name}] {thread!r} auto-exporting to {dest} — the thread is "
                f"trusted; you will be signalled"
            )
        CommandUtil.create_task(_await_verdict(target, action, dest))
        return (
            f"[{name}] {thread!r} export to {dest} awaiting approval — the human "
            f"decides on the file editor surface; you will be signalled"
        )

    async def _await_verdict(target: Thread, action: Action, dest: str) -> None:
        verdict = await store.waiter(target.id, action.n)
        if verdict != "accept":
            store.finish_export(target.id, action.n, "rejected")
            await _card(target, action)
            _aside(f"export of {target.id!r} to {dest} was denied")
            await _maybe_batch_done()
            return
        await _do_export(target, action, dest)
        await _maybe_batch_done()

    async def _do_export(target: Thread, action: Action, dest: str) -> None:
        text = target.content
        try:
            path = Path(dest)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        except OSError as e:
            store.finish_export(target.id, action.n, "failed")
            store.say(target.id, action.n, "g", f"write failed: {e}")
            await _card(target, action)
            _notify(f"export of {target.id!r} to {dest} failed: {e}", level="warning")
            return
        landed = store.finish_export(target.id, action.n, "written")
        store.mark_exported(target.id, dest)
        await _card(target, landed)
        await _threads()
        _notify(
            f"thread {target.id!r} exported to {dest} "
            f"({line_count(text)} lines) — the thread is closed"
        )

    async def _maybe_batch_done() -> None:
        """One knock when the last awaiting export settles — not one per verdict."""
        if store.pending_exports() or signaler is None:
            return
        _notify("every pending export is now decided — history(thread) to review")

    # -- queries ------------------------------------------------------------

    @chan.build.command(name="threads", always_observe=True, available=enabled)
    async def threads() -> str:
        """List the threads you have, with their current version."""
        out = []
        for t in store.threads():
            loc = t.exported_to or t.path or "(blank)"
            out.append(
                f"- {t.id} [{t.state}] v{t.version} | {t.label} | "
                f"{line_count(t.content)} lines | {loc}"
            )
        return "\n".join(out) if out else "(no threads — open() one)"

    @chan.build.command(name="history", always_observe=True, available=enabled)
    async def history(thread: str) -> str:
        """The index of a thread: every action with its verdict, and the head.

        An index, not a transcript — ask for a version with read() after a rewind
        if you need the text back.
        """
        target = store.get(thread)
        if target is None:
            CommandUtil.raise_observe(f"no thread {thread!r} — threads() lists them")
        marks = {
            "applied": "·",
            "awaiting": "?",
            "written": "→",
            "rejected": "✗",
            "failed": "!",
            "cancelled": "⨯",
            "streaming": "…",
        }
        loc = target.exported_to or target.path or "(blank)"
        lines = [
            f"[{name}] {target.id} [{target.state}] v{target.version} | "
            f"{target.label} | {loc}"
        ]
        for a in target.actions:
            who = "<u>" if a.author == "u" else ""
            lines.append(
                f"  {a.n} {marks.get(a.state, '?')} {a.kind} {who} {a.label}".rstrip()
            )
        if not target.actions:
            lines.append("  (nothing yet)")
        return "\n".join(lines)

    # -- warm state ---------------------------------------------------------

    @chan.build.named_notices
    def notices() -> dict[str, str]:
        """The current version of every thread, re-emitted whenever it moves.

        This is the fragment that keeps the model oriented after a context
        compaction: it can lose the conversation but not the fact that it is
        standing on v7 of 'readme'. Counts only — no ticking values.
        """
        pending = store.pending_exports()
        out = {
            "file_editor": (
                f"live: {len(store.live())} | exported: "
                f"{sum(1 for t in store.threads() if t.state == 'exported')} | "
                f"export pending: {len(pending)}"
            )
        }
        url = _url()
        if url:
            out["url"] = url
        for t in store.threads():
            loc = t.exported_to or t.path or "(blank)"
            waiting = " | export pending" if any(
                tid == t.id for tid, _ in pending
            ) else ""
            flag = " [auto]" if t.auto else ""
            out[_notice_name(t.id)] = (
                f"v{t.version} | {t.label} | {line_count(t.content)} lines | "
                f"{loc}{waiting}{flag}"
            )
        drafts = store.recoverable()
        if drafts:
            out["drafts"] = "unclaimed working copies — open(draft=...): " + ", ".join(
                drafts
            )
        return out

    return chan
