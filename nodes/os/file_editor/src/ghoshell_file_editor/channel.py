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
import time
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
from .structure import (
    REGION_LINES,
    Action,
    Thread,
    change_stat,
    line_count,
    region_span,
    slice_region,
)

__all__ = ["build_file_editor_channel"]

_READ_EXCERPT = 4000
"""How much of a read stays on its card. The model gets the whole text; the card
only needs to be enough for a human to recognize what was read."""

_READ_LIMIT = 200_000


def _size_txt(n: int) -> str:
    if n >= 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.0f} KB"
    return f"{n} B"


def _meta_of(thread: Thread) -> str:
    """What we will say about a file too big to hand over — facts, then the way in.

    A memory is not a context: holding a 10MB working copy costs bytes, reading it
    costs the model's whole frame and does not come back. So the first answer for
    a big file is its shape — size, lines, line endings — and the two moves that
    actually work on it: search for a line, read a region, edit a region.
    """
    text = thread.content
    endings = "CRLF" if "\r\n" in text else "LF"
    return (
        f"[file_editor] {thread.id!r} — {thread.lines} lines · "
        f"{_size_txt(thread.bytes)} · utf-8, {endings} · not sent whole\n"
        f"too big to hand over in one frame. Working moves:\n"
        f"  find(pattern) → line numbers\n"
        f"  read(region=\"a-b\") → at most {REGION_LINES} lines per call\n"
        f"  edit(region=\"a-b\") → replace those lines without quoting them back"
    )
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

    def layout_state(self) -> tuple[str, str]: ...

    async def set_layout(self, mode: str) -> str: ...

    async def set_cursor(
        self, thread: str, start: int, end: int | None = None, *, version: int = 0,
        by: str = "model",
    ) -> dict[str, Any]: ...

    def cursors(self) -> dict[str, dict[str, Any]]: ...

    def cursor_of(self, thread: str) -> dict[str, Any] | None: ...

    def view_at_state(self) -> tuple[str, int]: ...

    def selection_state(self) -> dict[str, Any] | None: ...

    def format_state(self) -> tuple[str, str]: ...

    async def set_format(self, mode: str, by: str = "model") -> str: ...


class _NoSurface:
    """Headless stand-in: no page to tell, but the knob still has a value, so a
    test or a browser-less node reads back an answer instead of an error."""

    def __init__(self) -> None:
        self._layout, self._layout_by = "page", "default"
        self._cursor: dict[str, Any] = {"thread": "", "start": 0, "end": 0}

    async def broadcast(self, frame: dict[str, Any]) -> None:
        return None

    def layout_state(self) -> tuple[str, str]:
        return self._layout, self._layout_by

    async def set_cursor(
        self, thread: str, start: int, end: int | None = None, *, version: int = 0,
        by: str = "model",
    ) -> dict[str, Any]:
        lo = max(1, int(start))
        return {"start": lo, "end": max(lo, int(end or lo)), "version": version, "by": by}

    def cursors(self) -> dict[str, dict[str, Any]]:
        return {}

    def cursor_of(self, thread: str) -> dict[str, Any] | None:
        return None

    def view_at_state(self) -> tuple[str, int]:
        return ("", 0)

    def selection_state(self) -> dict[str, Any] | None:
        return None

    def format_state(self) -> tuple[str, str]:
        return ("auto", "default")

    async def set_format(self, mode: str, by: str = "model") -> str:
        return (mode or "").strip().lower()

    async def set_layout(self, mode: str) -> str:
        chosen = (mode or "").strip().lower()
        if chosen not in ("split", "page"):
            raise ValueError(f"layout must be 'split' or 'page', got {mode!r}")
        self._layout, self._layout_by = chosen, "model"
        return chosen


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
        await _emit({
            "type": "threads",
            "threads": [thread_view(t) for t in store.threads()],
            # Working copies nobody owns — a crash's leftovers. They are the
            # model's to adopt, but the human should not have to hear about them
            # second-hand: the chip in their header is the same list.
            "drafts": store.recoverable(),
        })

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

    def _guard_lock(thread_id: str) -> None:
        """Writes wait while the human holds the pen.

        The refusal names the lines and the time: a write that fails with "busy"
        leaves me choosing between retrying blindly and giving up, and neither is
        a decision I can make from "busy".
        """
        refusal = surface.lock_refusal(thread_id)
        if refusal:
            CommandUtil.raise_observe(refusal)

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
            "only once the human accepts it, and the thread stays yours after "
            "that — keep editing and export again. history(thread) is the index of "
            "actions on a thread. Every card carries two things: the mechanics "
            "(which region, how many lines) which the node fills in itself, and "
            "the label — your motive, which only you can give. Write it: a card "
            "that says only 'read all' leaves the human unable to tell whether it "
            "is worth opening."
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
        """End a thread: nothing is written to disk, and its slot is freed.

        An unsaved working copy is kept as a recoverable draft — it turns up in
        the notice and reopens with ``open(draft=...)``; a copy that already
        matches the disk is dropped. Closing is "I am done with this thread", not
        "throw the text away".
        """
        target = store.get(thread)
        if target is None:
            CommandUtil.raise_observe(f"no thread {thread!r}")
        store.close(thread)
        await _threads()
        return f"[{name}] thread {thread!r} closed (nothing written)"

    # -- reading ------------------------------------------------------------

    @chan.build.command(name="read", always_observe=True, available=enabled)
    async def read(thread: str, region: str = "", label: str = "") -> str:
        """Return a thread's text to you, and leave a card showing the same text.

        ``region`` is a line range like ``"10-40"`` (1-based, inclusive); empty
        reads the whole working copy. Reading changes nothing — but the card is
        how the human shares your view of the file instead of guessing at it.

        ``label`` is the card's title: say *why* you are reading. The card always
        carries the mechanics beside it (which region, how many lines), so the
        motive is the part only you can supply — a card that says "read all" tells
        the human nothing about whether it is worth opening.

        Reading also *moves the cursor onto those lines* — the page scrolls there
        and lights them, so the human follows your eyes instead of guessing which
        part of the file you are talking about.
        """
        target = _get_thread(thread)
        if target.state != "live":
            CommandUtil.raise_observe(
                f"thread {thread!r} is {target.state} — read the file itself, or "
                f"open a new thread"
            )
        if target.big and not region:
            meta = _meta_of(target)
            action = store.record(
                target.id,
                "read",
                label or "read meta",
                text=meta,
                payload="",
                fact=f"meta only · {target.lines} lines · {_size_txt(target.bytes)}",
            )
            await _card(target, action)
            return meta
        try:
            text = slice_region(target.content, region)
            start, end = region_span(target.content, region)
        except ValueError as e:
            CommandUtil.raise_observe(str(e))
        if target.big and end - start + 1 > REGION_LINES:
            shown = REGION_LINES
            text = "".join(text.splitlines(keepends=True)[:shown])
            text += (
                f"\n… (region capped at {REGION_LINES} lines; lines "
                f"{start + shown}-{end} not sent — narrow the region)"
            )
        excerpt = text[:_READ_EXCERPT]
        if len(text) > _READ_EXCERPT:
            excerpt += "\n… (truncated on the card)"
        action = store.record(
            target.id,
            "read",
            label or f"read {region or 'all'}",
            text=excerpt,
            payload=region,
            fact=f"{region or 'all'} · {line_count(text)} lines",
        )
        await _card(target, action)
        await surface.set_cursor(
            target.id, start, end, version=target.version, by="model"
        )
        if len(text) > _READ_LIMIT:
            return (
                f"{text[:_READ_LIMIT]}\n… (truncated at {_READ_LIMIT} chars — read a "
                f"region for the rest)"
            )
        return text

    @chan.build.command(name="format", available=enabled)
    async def format(mode: str = "") -> str:
        """Which renderer draws the document.

        ``auto`` — the default — lets the file's extension decide. Any renderer in
        the node's ``renderers/`` directory can be named (``raw``, ``markdown``,
        ``code``, ``plain``, or whatever was added since); ``tmp`` is the draft
        pushed with ``preview``.

        ``raw`` is the addressable face — every pointer, selection and line range
        means something there, and it is what everything falls back to. A rendered
        face is for reading: its blocks still carry their line ranges, so pointing
        and selecting work, but per block rather than per character.

        Empty = read the mode back.
        """
        if not mode:
            current, by = surface.format_state()
            return f"[{name}] format: {current} (chosen by {by})"
        try:
            chosen = await surface.set_format(mode, by="model")
        except ValueError as e:
            CommandUtil.raise_observe(str(e))
        return f"[{name}] format: {chosen} (chosen by model)"

    @chan.build.command(name="preview")
    async def preview(thread: str, text__: str, label: str = "") -> str:
        """Push a draft renderer for one thread and look at the document through it.

        The tag body is the body of ``render(text)``: it receives the working copy's
        text and returns HTML. It lands in the thread's own string of drafts — the
        page labels it 临时, and the next one does not throw it away: ``show`` comes
        back to it. Nothing is written to disk; a draft that turns out worth keeping
        becomes a file in ``renderers/`` and joins the library for everyone.

        A draft is a *material* for this thread, not a separate document: it reads
        the same working copy, so it always reflects the latest edit.

        Keep it short. A draft that needs more than a few dozen lines is telling you
        it wants to be a file.
        """
        item = await surface.push_preview(thread, text__, label)
        return (
            f"[{name}] draft #{item['seq']} '{item['label']}' pushed on {thread!r} "
            f"— the page shows it under 临时"
        )

    @chan.build.command(name="renderings")
    async def renderings(thread: str = "") -> str:
        """List the drafts pushed on a thread, and which face the page is showing.

        The cold read for my own perception: ``preview`` is not fire-and-forget,
        and without this I would be guessing which of the faces I drew is on the
        screen now. With no thread it lists every thread that has drafts.
        """
        threads = [thread] if thread else surface.draft_threads()
        if not threads:
            return f"[{name}] renderings: none pushed yet"
        lines = [f"[{name}] renderings:"]
        for t in threads:
            items = surface.drafts(t)
            face = surface.current_face(t)
            now = ""
            if face:
                now = f"  · showing {face.get('renderer')}"
                if face.get("seq"):
                    now += f" #{face['seq']}"
                now += f" ({face.get('by')})"
            lines.append(f"  {t}:{now}")
            if not items:
                lines.append("    (no drafts)")
            for d in items:
                lines.append(f"    #{d['seq']:<3} {d['label']}")
        return "\n".join(lines)

    @chan.build.command(name="show")
    async def show(thread: str, seq: int) -> str:
        """Draw one of the drafts already pushed — go back to an earlier view.

        Switching is mine to do: the drafts belong to the thread's context, and
        comparing the face I had two minutes ago with the one I have now is how
        a view gets judged. ``seq`` comes from ``renderings``.
        """
        try:
            item = await surface.show_draft(thread, int(seq))
        except ValueError as e:
            CommandUtil.raise_observe(str(e))
        return f"[{name}] showing draft #{item['seq']} '{item['label']}' on {thread!r}"

    @chan.build.command(name="renderers")
    async def renderers() -> str:
        """List the renderer library — the cold read.

        The page already knows the library and shows the picker; this is for me.
        The state face carries what is *relevant* (the current face, the broken
        one), not the catalogue — so "what faces does this node have" has to be
        asked rather than remembered, or I would confuse "I have not seen it" with
        "it does not exist".
        """
        items = surface.list_renderers()
        if not items:
            return f"[{name}] renderers: (none found — falling back to raw only)"
        lines = [f"[{name}] renderers ({len(items)}):"]
        for r in items:
            exts = " ".join(r.get("ext") or []) or "(fallback)"
            lines.append(
                f"  {r['name']:<10} {exts:<40} {r.get('description') or ''}".rstrip()
            )
        return "\n".join(lines)

    @chan.build.command(name="reload")
    async def reload_file(thread: str, label: str = "") -> str:
        """Read the file again into the working copy — one action, no merge.

        For the case this node does not lock against: the file changed on disk
        while we were editing it. The previous text is not thrown away, it stays
        in the history where a diff can be asked for; what this does is say "the
        file is the truth now".

        Calling it when nothing changed is not an error, it is just visible: the
        receipt names the drift it was called to fix.
        """
        target = _get_thread(thread)
        drift = store.disk_drift(target) or "no drift"
        try:
            action = store.reload_from_disk(target.id, label or "reloaded from disk")
        except (KeyError, ValueError) as e:
            CommandUtil.raise_observe(str(e))
        action.fact = change_stat(action.effect)
        await _card(target, action)
        await _threads()
        return (
            f"[{name}] {thread!r} reloaded to v{target.version} ({drift}) — {action.fact}"
        )

    @chan.build.command(name="diff")
    async def diff(thread: str, label: str = "") -> str:
        """Compare the working copy with the file on disk — as an action.

        The third step of the persistence story: detect the drift, read it back,
        and when merging actually matters, look at the two side by side. The diff
        is recorded, not just shown, because "someone compared these" is itself a
        fact about the edit that both of us should be able to find later.

        Direction is file → working copy: the question is what *we* are about to
        write, not what the file did to us.
        """
        target = _get_thread(thread)
        try:
            action, text = store.diff_against_disk(target.id, label or "diff against the file")
        except (KeyError, ValueError) as e:
            CommandUtil.raise_observe(str(e))
        await _card(target, action)
        await _threads()
        if not text:
            return f"[{name}] {thread!r} is identical to the file on disk — nothing to merge"
        return f"[{name}] {thread!r} — {action.fact}\n{text}"

    @chan.build.command(name="point", available=enabled)
    async def point(thread: str, region: str = "", label: str = "") -> str:
        """Put the reading point on these lines — without reading them out.

        The gesture for talking about a document: the page scrolls to the region
        and lights it, so whoever is watching sees the fragment under discussion
        instead of a line number in a sentence. ``region`` is a range like
        ``"330-346"``; empty points at the whole thread. It returns a one-line
        receipt naming the lines, so a mis-aimed point shows up as numbers rather
        than as someone staring at the wrong paragraph.

        With no thread it only *reads back* where the reading point is — who moved
        it included, because "he scrolled somewhere" and "I pointed somewhere" are
        two different sentences about the same number.
        """
        if not thread.strip():
            marks = []
            for one in store.threads():
                mark = surface.cursor_of(one.id)
                if not mark:
                    continue
                # A marker set on an older version is history, not an address: say
                # so instead of quoting line numbers that have since moved.
                if int(mark.get("version", 0)) == one.version:
                    marks.append(
                        f"point: {one.id!r} lines {mark['start']}-{mark['end']}"
                    )
                else:
                    marks.append(
                        f"point: {one.id!r} lines {mark['start']}-{mark['end']} "
                        f"(stale — text moved to v{one.version})"
                    )
            seen, line = surface.view_at_state()
            pos = f"you: line {line} of {seen!r}" if seen else "you: not moved"
            sel = surface.selection_state()
            picked = ""
            if sel:
                picked = (
                    f" · selected: {sel['thread']!r} lines {sel['start']}-{sel['end']}"
                    f" «{sel.get('text', '')[:80]}»".rstrip()
                )
            body = " · ".join(marks) if marks else "point: unset"
            return f"[{name}] {body} · {pos}{picked}"
        target = _get_thread(thread)
        try:
            start, end = region_span(target.content, region)
        except ValueError as e:
            CommandUtil.raise_observe(str(e))
        await surface.set_cursor(
            target.id, start, end, version=target.version, by="model"
        )
        tail = f" — {label}" if label else ""
        return f"[{name}] cursor on {thread!r} lines {start}-{end}{tail}"

    @chan.build.command(name="find", always_observe=True, available=enabled)
    async def find(
        thread: str,
        pattern: str,
        region: str = "",
        limit: int = 40,
        ignore_case: bool = False,
        label: str = "",
    ) -> str:
        """Search the working copy and get line numbers — not the text.

        The register to read a long file by: search for what you know, then read or
        edit the region the match names. Line numbers are the coordinate every
        other move accepts, which is why this returns them instead of excerpts.

        ``pattern`` is a regular expression; ``region`` narrows the search the same
        way ``read`` does; ``limit`` caps how many matches are listed (``total``
        still counts them all). ``label`` is the card's title.
        """
        target = _get_thread(thread)
        try:
            found = store.find(
                target.id, pattern, region=region, limit=limit, ignore_case=ignore_case
            )
        except (KeyError, ValueError) as e:
            CommandUtil.raise_observe(str(e))
        rows = [
            f"{m['line']}:{m['col']}  {m['text']}" for m in found["matches"]
        ]
        body = "\n".join(rows) if rows else "(no matches)"
        action = store.record(
            target.id,
            "find",
            label or f"find {pattern[:40]}",
            text=body,
            payload=pattern,
            fact=f"{found['total']} match(es), {found['shown']} shown · {found['lines']} lines",
        )
        await _card(target, action)
        matches = found["matches"]
        if matches:
            # A search is a way of looking somewhere: put the cursor on the first
            # hit so the page is already where the answer is.
            await surface.set_cursor(
                target.id,
                int(matches[0]["line"]),
                int(matches[0]["line"]),
                version=target.version,
                by="model",
            )
        return (
            f"[{name}] {thread!r} — {found['total']} match(es) in "
            f"{found['lines']} lines\n{body}"
        )

    @chan.build.command(name="edit", always_observe=False, available=enabled)
    async def edit(thread: str, region: str, text__: str = "", label: str = "") -> str:
        """Replace the lines a region names — the edit that follows a ``find``.

        ``str_replace`` asks you to quote what you are replacing, which is fine for
        a paragraph and hostile for a long file: one wrong character and the edit
        does not land. A region replaces by address instead. ``text__`` is the new
        body for those lines; an empty body deletes them. ``label`` is the card's
        title — say why the lines are going.
        """
        _guard_lock(thread)
        target = _get_thread(thread)
        try:
            action = store.edit_region(
                target.id, region, text__, label or f"edit {region}"
            )
        except (KeyError, ValueError) as e:
            CommandUtil.raise_observe(str(e))
        action.fact = change_stat(action.effect)
        await _card(target, action)
        await _threads()
        return (
            f"[{name}] {thread!r} v{target.version} — {action.fact}, "
            f"{line_count(target.content)} lines"
        )

    # -- mutations ----------------------------------------------------------

    @chan.build.command(name="write", always_observe=False, available=enabled)
    async def write(thread: str, chunks__: str, label: str = "") -> str:
        """Replace the whole working copy with what you write.

        The body streams in and lands the moment the tag closes. ``label`` is the
        card's title — say what the new text is.
        """
        _guard_lock(thread)
        return await _stream(thread, "write", chunks__, label or "rewrite")

    @chan.build.command(name="append", always_observe=False, available=enabled)
    async def append(thread: str, chunks__: str, label: str = "") -> str:
        """Add what you write to the end of the working copy.

        The way to build a long document a section at a time: the card shows the
        segment you added, not the whole file.
        """
        _guard_lock(thread)
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
        # The card says what moved as well as why: the diff is only known once the
        # edit has landed.
        landed.fact = change_stat(landed.effect)
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
        _guard_lock(thread)
        target = store.get(thread)
        if target is None:
            CommandUtil.raise_observe(f"no thread {thread!r} — threads() lists them")
        try:
            ops = parse_ops(text__)
            actions = store.replace(thread, ops, label or f"str_replace ×{len(ops)}")
        except (KeyError, ValueError) as e:
            CommandUtil.raise_observe(str(e))
        for action in actions:
            action.fact = change_stat(action.effect)
            await _card(target, action)
        await _threads()
        return (
            f"[{name}] {thread!r} v{target.version} — {len(actions)} card(s), "
            f"{line_count(target.content)} lines"
        )

    @chan.build.command(name="rewind", always_observe=False, available=enabled)
    async def rewind(thread: str, n: int, label: str = "") -> str:
        """Put the working copy back where action ``n`` left it (``0`` = baseline).

        Append-only: this adds an action, it does not erase the ones after it. Use
        history(thread) to pick ``n``. ``label`` is the card's title — say why you
        are going back.
        """
        target = store.get(thread)
        if target is None:
            CommandUtil.raise_observe(f"no thread {thread!r} — threads() lists them")
        try:
            action = store.rewind(thread, n, label=label)
        except (KeyError, ValueError) as e:
            CommandUtil.raise_observe(str(e))
        action.fact = change_stat(action.effect)
        await _card(target, action)
        await _threads()
        return f"[{name}] {thread!r} v{target.version} — back to v{n}"

    # -- the one side effect ------------------------------------------------

    @chan.build.command(name="export", always_observe=False, available=enabled)
    async def export(thread: str, path: str = "", label: str = "") -> str:
        """Propose writing the working copy to ``path`` (default: the file it was
        opened from).

        Nothing reaches the disk until the human accepts — unless the thread is
        auto-trusted and this export goes to its established target, in which case
        it writes at once. The thread stays live afterwards: an export records
        which version is on disk, not the end of the object, so keep editing and
        export again. Returns a receipt at once.
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
        # Two ways a write goes through without asking, and both are the human's
        # grant, never mine: the thread is auto-trusted and this is its established
        # target, or the destination sits in a directory the human trusted. Writing
        # somewhere new is establishing a new target, and that still asks.
        everything = store.global_trust()
        directory_trust = store.is_trusted(dest)
        auto = directory_trust or (
            target.auto and bool(target.path) and dest == target.path
        )
        action = store.record(
            thread,
            "export",
            label or f"export to {dest}",
            text=target.content,
            payload=dest,
            state=("applied" if auto else "awaiting"),
            fact=f"→ {dest} · {line_count(target.content)} lines",
        )
        await _card(target, action)
        if auto:
            CommandUtil.create_task(_do_export(target, action, dest))
            why = (
                "trust is on for this whole surface"
                if everything else "the directory is trusted"
                if directory_trust else "the thread is trusted"
            )
            return (
                f"[{name}] {thread!r} auto-exporting to {dest} — {why}; "
                f"you will be signalled"
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
        store.note_export(target.id, dest, target.version)
        await _card(target, landed)
        await _threads()
        # A finished export is not news that deserves the floor: it is buffered
        # and rides the next moment. Only a refusal or a failure claims a turn.
        _notify(
            f"thread {target.id!r} wrote v{target.version} to {dest} "
            f"({line_count(text)} lines) — still open, keep editing or export again",
            level="background",
            next_=False,
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
            written = f"wrote v{t.exported_at}" if t.exported_to else "nothing written"
            out.append(
                f"- {t.id} [{t.state}] v{t.version} | {t.label} | "
                f"{line_count(t.content)} lines | {loc} | {written}"
                + (" | unsaved changes" if t.unsaved else "")
            )
        return "\n".join(out) if out else "(no threads — open() one)"

    def _age(seconds: float) -> str:
        if seconds < 3600:
            return f"{int(seconds // 60)}m"
        if seconds < 86400:
            return f"{int(seconds // 3600)}h"
        return f"{int(seconds // 86400)}d"

    @chan.build.command(name="orphans", always_observe=True, available=enabled)
    async def orphans() -> str:
        """List the working copies nobody owns — what a crash left behind.

        One line each: name, size, age. Pass one to ``open(draft=...)`` to carry on
        from it, or to ``discard`` to throw it away. Pull it when you are about to
        need it; the notice carries only a count, because a stale name repeated
        every turn is noise and noise is what buries the one that matters.
        """
        names = store.recoverable()
        if not names:
            return "(no unclaimed working copies)"
        now = time.time()
        lines = []
        for draft in names:
            try:
                st = (store.drafts_dir / draft).stat()
            except OSError:
                lines.append(f"- {draft} (gone)")
                continue
            lines.append(f"- {draft} | {st.st_size} bytes | {_age(now - st.st_mtime)} old")
        return "\n".join(lines)

    @chan.build.command(name="discard", available=enabled)
    async def discard(draft: str) -> str:
        """Throw an unclaimed working copy away.

        Only a draft no thread owns — a live thread's copy is that thread's text, so
        close the thread first if that is what you mean. ``orphans()`` lists what
        there is to discard. The human's header chip refreshes with it.
        """
        try:
            gone = store.discard(str(draft))
        except ValueError as e:
            CommandUtil.raise_observe(str(e))
        await _threads()
        return (
            f"[{name}] draft {draft} discarded"
            if gone else f"[{name}] draft {draft} was already gone"
        )

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

    # -- presentation -------------------------------------------------------

    @chan.build.command(name="view", always_observe=True, available=enabled)
    async def view(mode: str = "") -> str:
        """Switch the page between the three readings of a thread — or read it back.

        ``document`` is the working copy, read through the current renderer;
        ``edit`` is the addressable face — line numbers, selection and the inline
        edit box — and entering it is what makes the human's selection mean
        something; ``history`` is what has happened to the thread; ``meta`` is its
        own facts. Which one is on the table is a view, not a layout: the human
        clicks it, and so may you. Empty = read back the current mode and who
        chose it.
        """
        if not mode:
            current, by = surface.view_state()
            return f"[{name}] view: {current} (chosen by {by})"
        try:
            chosen = await surface.set_view(mode)
        except ValueError as e:
            CommandUtil.raise_observe(str(e))
        return f"[{name}] view: {chosen} (chosen by model)"

    @chan.build.command(name="layout", always_observe=True, available=enabled)
    async def layout(mode: str = "") -> str:
        """How an open card is shown — my half of the presentation contract.

        ``split`` keeps the card list beside it; ``page`` gives it the whole
        window. Which card is open is the human's click; this decides only how
        big it opens, and the page says who chose it. Empty = read the mode back.
        """
        if not mode:
            current, by = surface.layout_state()
            return f"[{name}] layout: {current} (chosen by {by})"
        chosen = await surface.set_layout(mode)
        return f"[{name}] layout: {chosen} (chosen by model)"

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
                f"live: {len(store.live())} | closed: "
                f"{sum(1 for t in store.threads() if t.state != 'live')} | "
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
            # Where the disk stands relative to the working copy — a thread that
            # has moved past what was written says so, because "I already
            # exported" is exactly the belief that goes stale.
            written = f" | wrote v{t.exported_at}" if t.exported_to else ""
            lock = surface.lock_of(t.id)
            held = (
                f" | being edited {lock['start']}–{lock['end']} since {lock['since']}"
                if lock else ""
            )
            # A file that changed underneath us is a fact about this thread, not a
            # verdict on it — it belongs in the state, where it costs one clause
            # and prevents a whole paragraph of wrong reasoning.
            drift = store.disk_drift(t)
            moved = f" | {drift} — reload(thread) to take it" if drift else ""
            out[_notice_name(t.id)] = (
                f"v{t.version}{written} | {t.label} | {line_count(t.content)} lines | "
                f"{loc}{waiting}{flag}{held}{moved}"
            )
        if store.global_trust():
            out["trust"] = "on — every export here writes at once"
        elif store.trusted():
            out["trust"] = "exports write without asking under: " + ", ".join(
                store.trusted()
            )
        drafts = store.recoverable()
        if drafts:
            # 名单不进 notice: 一份没人认领的旧工作副本每轮重复出现只是噪音，而噪音会
            # 把真正的异常淹掉。改成 orphans() 按需拉、discard() 当场删 —— 状态留给我读,
            # 不该持续推给我看。
            out["drafts"] = f"{len(drafts)} unclaimed working copies — orphans() lists them"
        return out

    return chan
