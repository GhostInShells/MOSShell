"""file_editor channel — the model-facing control surface (axis 2).

Every command takes an explicit ``thread`` — there is no hidden "current
thread" state. Actions and verdicts mutate the :class:`ThreadStore`; each
mutation also broadcasts the matching WS frame (``action.head`` /
``action.delta`` / ``action.full`` / ``burst``) so the human surface streams it
live.

The channel itself emits no signal to the model: the model's own commands
return observes, and human-side events signal via the surface. Queries
(``threads`` / ``thread`` / ``history``) project store state into model-readable
text — the same store the human surface reads.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from ghoshell_moss.core.blueprint.channel_builder import CommandUtil, new_channel
from ghoshell_moss.core.concepts.channel import Channel

from .projection import action_view
from .store import ThreadStore

__all__ = ["build_file_editor_channel", "stream_write"]


class _Surface(Protocol):
    async def broadcast(self, frame: dict[str, Any]) -> None: ...


class _NoSurface:
    async def broadcast(self, frame: dict[str, Any]) -> None:
        return None


def _action_full(thread_id: str, action) -> dict[str, Any]:
    return {"type": "action.full", **action_view(thread_id, action)}


def _thread_text(thread) -> str:
    head = thread.head
    loc = f" @ {thread.path}" if thread.path else ""
    lines = [f"[{thread.id}] {thread.label}{loc}  ({len(thread.actions)} actions)"]
    marks = {"pending": "·", "confirmed": "✓", "rejected": "✗"}
    for a in thread.actions.values():
        who = f"<{a.author}>" if a.author == "u" else ""
        lines.append(f"  {a.seq.n} {marks[a.verdict]} {a.kind} {who} {a.description}")
    lines.append(f"head: {head.seq.n if head is not None else 'base'}")
    return "\n".join(lines)


def _history_text(thread) -> str:
    lines = []
    for a in thread.versions:
        diff = a.effect.diff or ""
        lines.append(f"--- {a.seq.n} {a.kind} ({a.description})\n{diff.rstrip()}")
    return "\n".join(lines) if lines else "(no confirmed changes yet)"


async def stream_write(
    store: ThreadStore,
    broadcast: Callable[[dict[str, Any]], Any],
    thread: str,
    chunks: Any,
    description: str,
) -> str:
    """Stream a full-text write: open → deltas → tail, broadcasting each stage.

    ``chunks`` is an async iterable of str (the model's ``chunks__`` delta arg).
    Extracted from the command so the streaming path is testable directly.
    """
    seq = store.open_action(thread, "g", "write", description or "write")
    await broadcast({
        "type": "action.head", "thread": thread, "seq": seq.n,
        "kind": "write", "author": "g",
        "description": description or "write", "state": "streaming",
    })
    async for chunk in chunks:
        store.append_delta(thread, seq.n, chunk)
        await broadcast({
            "type": "action.delta", "thread": thread, "seq": seq.n, "text": chunk,
        })
    action = store.tail_action(thread, seq.n)
    await broadcast(_action_full(thread, action))
    return f"ok [{thread}:{seq.n}] {len(action.payload)} chars"


def build_file_editor_channel(
    store: ThreadStore,
    *,
    surface: _Surface | None = None,
    enabled: Callable[[], bool] | None = None,
) -> Channel:
    """Build the file_editor channel over a store.

    :param store: the thread store — the single source of truth.
    :param surface: an optional WS broadcaster (axis 3). None = silent.
    :param enabled: a live availability gate shared by every command. When it
        returns False, all command signatures drop out of the model's interface
        (the "temporarily disabled" switch).
    """
    surface = surface or _NoSurface()
    enabled = enabled or (lambda: True)

    async def _broadcast(frame: dict[str, Any]) -> None:
        await surface.broadcast(frame)

    chan = new_channel(
        name="file_editor",
        description=(
            "a dialogue thread over a readable text file — open a thread, stream "
            "write proposals, confirm/reject/rewind, reply with diffs."
        ),
    )

    @chan.build.instruction
    def instruction() -> str:
        return (
            "Dialogue over a text file, not an editing tool. open() a thread over "
            "a file (or a blank one), then write() full-text proposals that the "
            "human watches stream in real time. The human may confirm, reject, or "
            "reply with a diff; read thread() to see where the line stands. "
            "Export is the only real side effect: it lands on disk only once "
            "confirmed."
        )

    # -- open -------------------------------------------------------

    @chan.build.command(name="open", always_observe=True, available=enabled)
    async def open_thread(
        thread: str, path: str = "", label: str = "", motivation: str = "",
    ) -> str:
        """Open a dialogue thread. ``thread`` is your chosen handle; use it in
        every later command. If ``path`` points at a readable text file its
        content becomes the baseline (v0); otherwise the line starts blank."""
        base = ""
        if path:
            p = Path(path)
            if not p.exists():
                CommandUtil.raise_observe(f"no such file: {path!r}")
            try:
                base = p.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                CommandUtil.raise_observe(
                    f"{path!r} is not readable text — only text files are supported"
                )
        try:
            store.open_thread(
                thread, label or thread, path=path or None,
                motivation=motivation, base_content=base,
            )
        except (KeyError, ValueError) as e:
            CommandUtil.raise_observe(str(e))
        return f"thread {thread!r} opened ({len(base)} chars baseline)"

    # -- write (streaming) ------------------------------------------

    @chan.build.command(name="write", always_observe=False, available=enabled)
    async def write(thread: str, chunks__: str, description: str = "") -> str:
        """Stream the full new content of a thread. The body streams token by
        token: the human watches it appear, then sees the diff. This is the one
        mutation kind — the payload is the complete resulting text."""
        return await stream_write(store, _broadcast, thread, chunks__, description)

    # -- atomic actions ---------------------------------------------

    @chan.build.command(name="rewind", always_observe=False, available=enabled)
    async def rewind(thread: str, target: str) -> str:
        """Append a rewind action back to an earlier action (or 'base')."""
        seq = store.append_action(thread, "g", "rewind", f"rewind to {target}", target)
        action = store.get_action(thread, seq.n)
        await _broadcast(_action_full(thread, action))
        return f"ok [{thread}:{seq.n}] rewind to {target}"

    @chan.build.command(name="reference", always_observe=False, available=enabled)
    async def reference(thread: str, region: str) -> str:
        """Display a region of the current text for shared view (no effect)."""
        seq = store.append_action(thread, "g", "reference", f"view {region}", region)
        action = store.get_action(thread, seq.n)
        await _broadcast(_action_full(thread, action))
        return f"ok [{thread}:{seq.n}] reference {region}"

    @chan.build.command(name="export", always_observe=False, available=enabled)
    async def export(thread: str, path: str) -> str:
        """Propose writing the current text to ``path``. Nothing lands on disk
        until this action is confirmed — its dialogue is the approval."""
        seq = store.append_action(thread, "g", "export", f"export to {path}", path)
        action = store.get_action(thread, seq.n)
        await _broadcast(_action_full(thread, action))
        return f"ok [{thread}:{seq.n}] export {path} (pending confirmation)"

    # -- verdicts + dialogue (author g) -----------------------------

    @chan.build.command(name="confirm", always_observe=False, available=enabled)
    async def confirm(thread: str, n: int) -> str:
        """Confirm up to action ``n`` — every pending action at or before it."""
        flipped = store.confirm(thread, n, by="g")
        for a in flipped:
            await _broadcast(_action_full(thread, a))
        return f"ok confirmed through [{thread}:{n}]"

    @chan.build.command(name="reject", always_observe=False, available=enabled)
    async def reject(thread: str, n: int) -> str:
        """Reject action ``n`` and every later pending action on the thread."""
        cascaded = store.reject(thread, n, by="g")
        for seq in cascaded:
            action = store.get_action(thread, seq.n)
            await _broadcast(_action_full(thread, action))
        return f"ok rejected from [{thread}:{n}]"

    @chan.build.command(name="reply", always_observe=False, available=enabled)
    async def reply(thread: str, n: int, text: str, anchor: str = "intent") -> str:
        """Attach a dialogue entry to action ``n``. Decides nothing."""
        store.reply(thread, n, "g", anchor, text=text)
        action = store.get_action(thread, n)
        await _broadcast(_action_full(thread, action))
        return f"ok replied on [{thread}:{n}]"

    # -- queries ----------------------------------------------------

    @chan.build.command(name="threads", always_observe=True, available=enabled)
    async def threads() -> str:
        """List open dialogue threads."""
        lines = []
        for t in store.threads():
            if t.state == "open":
                loc = f" @ {t.path}" if t.path else ""
                lines.append(f"- {t.id} ({t.label}){loc}  {len(t.actions)} actions")
        return "\n".join(lines) if lines else "(no threads open)"

    @chan.build.command(name="thread", always_observe=True, available=enabled)
    async def thread(thread: str) -> str:
        """Read one thread's line: every action with verdict + the head."""
        t = store.get_thread(thread)
        if t is None:
            CommandUtil.raise_observe(f"no thread {thread!r}. threads() to list.")
        return _thread_text(t)

    @chan.build.command(name="history", always_observe=True, available=enabled)
    async def history(thread: str) -> str:
        """The confirmed state changes so far (the version list as diffs)."""
        t = store.get_thread(thread)
        if t is None:
            CommandUtil.raise_observe(f"no thread {thread!r}. threads() to list.")
        return _history_text(t)

    return chan
