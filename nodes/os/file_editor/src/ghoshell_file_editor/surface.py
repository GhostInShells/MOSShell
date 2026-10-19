"""The human-facing web surface: cards in, verdicts and questions out.

One port serves the page and the WebSocket. Downlink carries the card stream
(``action`` / ``threads`` / ``detail``) plus a full ``snapshot`` on connect.
Uplink carries the human's three moves — accept, deny, ask — plus detail
requests (content is fetched when a card is clicked, never streamed with it) and
the enable toggle.

Verdicts are handed down, never written here: ``store.settle`` wakes the
channel's waiter, and the channel is the only place a thread moves. What this
module does own is the signal to the ghost — a human's question is a fact the
model must learn, and it travels as a must-not-lose notify that guarantees a
turn; a verdict rides an aside, because it is a fact the ghost notices when it
is free rather than an interruption.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from websockets.asyncio.server import ServerConnection, serve

from ghoshell_moss.message import Message
from ghoshell_moss.signals import AsideSignalMeta, NotifySignalMeta

from .structure import line_count

from .projection import action_view, detail_view, snapshot, thread_view
from .store import DocStore

__all__ = ["FileEditorSurface"]

_DEBOUNCE_SECONDS = 0.4
"""A verdict is one decision. Repeated clicks inside this window are the same
decision arriving twice, not a second one."""

_LAYOUTS = ("split", "page")
# How many drafts one thread keeps. A draft is a working view; when the ring
# wraps, the survivor should have been written to renderers/ as a file.
_DRAFT_RING = 8
# `auto` plus whatever the renderer library happens to contain — the list cannot
# be a constant here, because the library is a directory anyone can add a file to.
# `raw` and `tmp` are always legal: raw is the last resort, tmp is the draft layer.
_FORMATS = ("auto", "raw", "tmp")
_VIEWS = ("document", "edit", "history", "meta")
"""How an open card is shown: a side panel beside the card list, or the whole
window. This is the model's half of the presentation contract — the human owns
the click that opens a card, the model owns how big it opens."""


class FileEditorSurface:
    def __init__(
        self,
        store: DocStore,
        *,
        send_signal: Callable[[Any], None],
        self_identity: str,
        on_toggle: Callable[[bool], None] | None = None,
        host: str = "127.0.0.1",
        port: int = 0,
        html_path: Path,
    ) -> None:
        self._store = store
        self._send_signal = send_signal
        self._identity = self_identity
        self._on_toggle = on_toggle
        self._host = host
        self._html_path = html_path
        self._clients: set[ServerConnection] = set()
        self._server: Any = None
        self._decided: set[tuple[str, int]] = set()
        self._last: dict[tuple[str, int, str], float] = {}
        self._drafts: dict[str, list[dict]] = {}
        self._faces: dict[str, dict] = {}
        self._locks: dict[str, dict] = {}
        # Inline by default: the card opens under itself, inside the list. "page"
        # is the deliberate exception for reading one card at a time.
        self._layout = "split"
        self._layout_by = "default"
        # Which of the three readings a page is on. The human's click moves it,
        # and so may the model: showing the other side *the thing you are talking
        # about* is half of being understood.
        self._view = "document"
        self._view_by = "default"
        # Where the reader is in the text. One point, two authors: the model
        # moves it to show what it is talking about, the human moves it by
        # scrolling. It is state, not a signal — a signal has to win attention to
        # arrive, a state only has to be read.
        # One marker per thread: a pointer is an address *inside* a document, so
        # keeping one global marker meant switching threads lost the place and
        # coming back did not restore it. Each entry remembers the text version it
        # was set on — a line number that outlives its text is a lie.
        self._cursors: dict[str, dict[str, Any]] = {}
        # ...and where the human is, which is a different question: it moves when
        # they scroll and it never drags the page anywhere.
        self._view_at: tuple[str, int] = ("", 0)
        # What the human has selected, when they select something. A selection is
        # a sentence they cannot say in the chat ("this part") — reading it as
        # state is the only way it reaches me.
        self._selection: dict[str, Any] | None = None
        # How the document is drawn: raw (line numbers, addressable) or rendered
        # (what the file means). `auto` reads it off the suffix, because the file
        # already says which of the two is worth showing.
        self._format = "auto"
        self._format_by = "default"
        self.port = port

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self.port}"

    # -- downlink -----------------------------------------------------------

    async def broadcast(self, frame: dict[str, Any]) -> None:
        if not self._clients:
            return
        payload = json.dumps(frame, ensure_ascii=False)
        for client in list(self._clients):
            try:
                await client.send(payload)
            except Exception:
                self._clients.discard(client)

    def snapshot(self) -> dict[str, Any]:
        """The full state a connecting page needs — store plus surface-owned knobs."""
        frame = snapshot(self._store)
        frame["layout"] = self._layout
        frame["layout_by"] = self._layout_by
        frame["view"] = self._view
        frame["view_by"] = self._view_by
        frame["cursors"] = self._cursors
        frame["view_at"] = {"thread": self._view_at[0], "line": self._view_at[1]}
        frame["selection"] = self._selection
        frame["format"] = self._format
        frame["format_by"] = self._format_by
        # The draft faces ride along under their own key — `drafts` already means
        # "unclaimed working copies" in this protocol, and two meanings for one
        # word is how a snapshot starts lying. A page that reloaded (or a second
        # window) must see the same materials I pushed, or my perception would
        # depend on which window happens to be open.
        frame["render_drafts"] = {t: self.drafts(t) for t in self.draft_threads()}
        return frame

    # -- layout (the model's side of the presentation contract) -------------

    def layout_state(self) -> tuple[str, str]:
        return self._layout, self._layout_by

    async def set_layout(self, mode: str) -> str:
        """Set how an open card is shown, and remember who chose it.

        Every page is told, so the marker can name its author instead of hiding
        the decision — a card opened in the wrong size must not look like an
        accident.
        """
        chosen = (mode or "").strip().lower()
        if chosen not in _LAYOUTS:
            raise ValueError(f"layout must be one of {_LAYOUTS}, got {mode!r}")
        self._layout = chosen
        self._layout_by = "model"
        await self.broadcast({"type": "layout", "layout": chosen, "by": self._layout_by})
        return chosen

    # -- view (which reading is on the table) --------------------------------

    def view_state(self) -> tuple[str, str]:
        return self._view, self._view_by

    async def set_view(self, mode: str, by: str = "model") -> str:
        """Switch the page between document / history / meta, and say who did it."""
        chosen = (mode or "").strip().lower()
        if chosen not in _VIEWS:
            raise ValueError(f"view must be one of {_VIEWS}, got {mode!r}")
        self._view = chosen
        self._view_by = by
        await self.broadcast({"type": "view", "view": chosen, "by": by})
        return chosen

    # -- the reading point (mine) and the reading position (theirs) ----------

    def cursors(self) -> dict[str, dict[str, Any]]:
        """Every thread's marker: {thread: {start, end, version, by}}."""
        return self._cursors

    def cursor_of(self, thread: str) -> dict[str, Any] | None:
        return self._cursors.get(thread)

    def view_at_state(self) -> tuple[str, int]:
        """(thread, line) the human is parked on. Empty thread before they scroll."""
        return self._view_at

    def selection_state(self) -> dict[str, Any] | None:
        """What the human has picked: {thread, start, end, text}, or None."""
        return self._selection

    def format_state(self) -> tuple[str, str]:
        return self._format, self._format_by

    async def set_format(self, mode: str, by: str = "model") -> str:
        chosen = (mode or "").strip().lower()
        # The library is a directory, so the legal set is "the constants plus
        # every file in it" — a name is valid because it exists, not because a
        # constant somewhere remembered to list it.
        known = set(_FORMATS) | {r["name"] for r in self.list_renderers()}
        if chosen not in known:
            raise ValueError(
                f"format must be one of {tuple(sorted(known))}, got {mode!r}"
            )
        self._format = chosen
        self._format_by = by
        await self.broadcast({"type": "format", "format": chosen, "by": by})
        return chosen

    # -- the draft layer ----------------------------------------------------
    #
    # A draft is not a firework: it is one of a string of *materials* that belong
    # to a thread's context. I push several while working on one document and
    # switch between them, so the server keeps them (a short ring per thread) and
    # the page can be told to show any of them again. Only the reference rides on
    # history — the script itself lives here.

    def draft_threads(self) -> list[str]:
        """Threads that have drafts — so listing them needs no private access."""
        return sorted(self._drafts)

    def drafts(self, thread: str) -> list[dict]:
        """The drafts pushed for one thread, oldest first."""
        return [dict(d) for d in self._drafts.get(thread, [])]

    def current_face(self, thread: str) -> dict:
        """What the page says it is drawing right now: {renderer, seq, by}."""
        return dict(self._faces.get(thread) or {})

    async def push_preview(self, thread: str, script: str, label: str = "") -> dict:
        """Keep a draft and hand it to the page.

        The ring is small on purpose: a draft is a working view, and dozens of
        them is a sign that the useful one should have been written to disk.
        """
        ring = self._drafts.setdefault(thread, [])
        seq = (ring[-1]["seq"] + 1) if ring else 1
        item = {"seq": seq, "label": label or f"草稿 {seq}", "script": script}
        ring.append(item)
        del ring[:-_DRAFT_RING]
        await self.broadcast({"type": "preview", "thread": thread, **item})
        return item

    async def show_draft(self, thread: str, seq: int) -> dict:
        """Ask the page to draw one of the drafts already pushed."""
        for d in self._drafts.get(thread, []):
            if d["seq"] == seq:
                await self.broadcast({"type": "preview_show", "thread": thread, "seq": seq})
                return dict(d)
        raise ValueError(f"no draft #{seq} on thread {thread!r}")

    def set_face(self, thread: str, renderer: str, seq: int | None, by: str) -> None:
        self._faces[thread] = {"renderer": renderer, "seq": seq, "by": by}

    # -- the editing lock ---------------------------------------------------
    #
    # The human edits; I wait. Not because I cannot merge — because merging is
    # where the cost is, and the first version of shared editing does not need it.
    # The lock is a *state*, visible in the thread's notice: a lock I cannot see
    # is indistinguishable from a broken tool.

    def lock_of(self, thread: str) -> dict:
        return dict(self._locks.get(thread) or {})

    def locked(self, thread: str) -> bool:
        return thread in self._locks

    def lock_refusal(self, thread: str) -> str:
        """What a write gets told when the human holds the pen.

        The message is the state, in one sentence: which lines, since when. A
        generic "thread busy" would leave me to guess whether to retry, wait, or
        give up.
        """
        lock = self._locks.get(thread)
        if not lock:
            return ""
        span = f"lines {lock['start']}–{lock['end']}" if lock.get("end") else "a region"
        return (
            f"he is editing {span} in {thread!r} since {lock['since']} — wait for the "
            f"commit, then read the thread again"
        )

    async def begin_edit(self, thread: str, start: int, end: int) -> dict:
        lock = {"start": int(start), "end": int(end), "since": time.strftime("%H:%M:%S")}
        self._locks[thread] = lock
        await self.broadcast({"type": "lock", "thread": thread, "lock": lock})
        return lock

    async def cancel_edit(self, thread: str) -> None:
        if self._locks.pop(thread, None) is not None:
            await self.broadcast({"type": "lock", "thread": thread, "lock": None})

    async def commit_edit(
        self, thread: str, region: str, text: str, label: str = "", client_id: str = ""
    ) -> Any:
        """Land the human's edit as an action — the same shape as mine.

        One action, author ``u``, in the same history, drawn with the same diff.
        That is the whole point: "who wrote this line" must have one answer.
        """
        action = self._store.edit_region(
            thread, region, text, label or f"edited {region}", author="u"
        )
        await self.cancel_edit(thread)
        # The action itself rides back with the receipt: the page already has the
        # text it just wrote, and making it pull the whole thread again to learn
        # one card is why the echo felt slow.
        await self.broadcast(
            {
                "type": "landed",
                "thread": thread,
                "n": action.n,
                "client_id": client_id,
                "by": "human",
                "action": action_view(action),
            }
        )
        return action

    async def set_cursor(
            self,
            thread: str,
            start: int,
            end: int | None = None,
            *,
            version: int = 0,
            by: str = "model",
    ) -> dict[str, Any]:
        """Point at lines. This is a *marker*, so it stays where I put it.

        It used to be one value with two authors, which made the human's scroll
        erase my pointer and made my pointer fight their scroll. Two questions,
        two answers: a point says "look at these lines" and only I set it; the
        position says "where the reader is" and only the page sets it.
        """
        lo = max(1, int(start))
        hi = max(lo, int(end if end is not None else start))
        marker = {"start": lo, "end": hi, "version": int(version), "by": by}
        self._cursors[thread] = marker
        await self.broadcast(
            {"type": "cursor", "thread": thread, "by": by, **marker}
        )
        return marker

    async def set_selection(
            self, thread: str, start: int, end: int, text: str = ""
    ) -> None:
        """Record a selection. Empty range clears it — a collapsed cursor is not a choice."""
        if not thread or start <= 0 or end < start:
            self._selection = None
        else:
            self._selection = {
                "thread": thread,
                "start": int(start),
                "end": int(end),
                "text": text[:400],
            }
        await self.broadcast({"type": "selection", "selection": self._selection})

    async def set_view_at(self, thread: str, line: int) -> None:
        """Record where the human has scrolled to — no jump, no redraw of my marker.

        Line 0 means "nothing to report": a page that has not been touched yet says
        so instead of leaving an old number behind. A position nobody chose is not
        a perception, it is a ghost of one, and the model would read it as the
        human pointing somewhere.
        """
        if not thread:
            return
        line = max(0, int(line))
        if self._view_at == (thread, line):
            return
        self._view_at = (thread, line)
        await self.broadcast({"type": "view_at", "thread": thread, "line": line})

    # -- uplink -------------------------------------------------------------

    def _debounced(self, thread: str, n: int, action: str) -> bool:
        now = time.monotonic()
        key = (thread, n, action)
        if now - self._last.get(key, 0.0) < _DEBOUNCE_SECONDS:
            return True
        self._last[key] = now
        return False

    async def _verdict(self, thread: str, n: int, verdict: str) -> None:
        key = (thread, n)
        if key in self._decided or self._debounced(thread, n, verdict):
            return
        if not self._store.settle(thread, n, verdict):
            return
        self._decided.add(key)
        where = "accepted — writing it now" if verdict == "accept" else "denied"
        self._aside(f"export of {thread!r} #{n} was {where}")

    async def _ask(self, thread: str, n: int, text: str) -> None:
        if self._debounced(thread, n, "ask"):
            return
        try:
            action = self._store.say(thread, n, "u", text or "(wants to talk)")
        except KeyError:
            await self.broadcast({"type": "error", "text": f"no card {thread}#{n}"})
            return
        target = self._store.get(thread)
        if target is not None:
            await self.broadcast(
                {"type": "action", "thread": thread, **action_view(action)}
            )
        self._notify(f"the human asks about {thread!r} #{n}: {text}")

    async def _detail(self, thread: str, n: int) -> None:
        target = self._store.get(thread)
        if target is None:
            await self.broadcast({"type": "error", "text": f"no thread {thread!r}"})
            return
        action = target.get(n)
        if action is None:
            await self.broadcast(
                {"type": "error", "text": f"no card {thread}#{n}"}
            )
            return
        await self.broadcast({"type": "detail", **detail_view(target, action)})

    async def _text(self, thread: str) -> None:
        """Hand the page the working copy itself — what the document view reads.

        Pull, not push: a page asks when it is showing a thread, so a session with
        ten threads never ships ten documents nobody is looking at. The version
        rides along so the page can tell "I already have this one" from "this
        moved" without diffing text.
        """
        target = self._store.get(thread)
        if target is None:
            await self.broadcast({"type": "error", "text": f"no thread {thread!r}"})
            return
        text = target.content
        if len(text) > 200_000:
            text = text[:200_000] + "\n… (capped on the page at 200k chars)"
        await self.broadcast(
            {
                "type": "text",
                "thread": thread,
                "version": target.version,
                "lines": line_count(target.content),
                "text": text,
            }
        )

    async def _set_thread_auto(self, thread: str, auto: bool) -> None:
        try:
            self._store.set_thread_auto(thread, auto)
        except (KeyError, ValueError) as e:
            await self.broadcast({"type": "error", "text": str(e)})
            return
        await self.broadcast({
            "type": "threads",
            "threads": [thread_view(t) for t in self._store.threads()],
        })

    async def _set_dir_trust(self, path: str, trusted: bool) -> None:
        """The human flips trust for a directory — the model never does."""
        try:
            self._store.set_dir_trust(path, trusted)
        except ValueError as e:
            await self.broadcast({"type": "error", "text": str(e)})
            return
        # State only: the notice carries `trust` / `[auto]`, and a signal here
        # would interrupt the model mid-sentence to tell it something it reads
        # every turn anyway.
        await self.broadcast(
            {"type": "trusted", "trusted": self._store.trusted()}
        )

    async def _set_global_trust(self, on: bool) -> None:
        """Hand the approvals over, or take them back — the human's switch."""
        store_on = self._store.set_global_trust(on)
        await self.broadcast({
            "type": "trusted",
            "trusted": self._store.trusted(),
            "global_trust": store_on,
        })

    async def _set_enabled(self, enabled: bool) -> None:
        if self._on_toggle is not None:
            self._on_toggle(enabled)

    def _notify(self, text: str) -> None:
        if self._send_signal is None:
            return
        signal = NotifySignalMeta(next=True).to_signal(
            Message.new(tag="file_editor", name=self._identity).with_content(
                f"[file_editor] {text}"
            ),
            description=text[:120],
        )
        self._send_signal(signal)

    def _aside(self, text: str) -> None:
        if self._send_signal is None:
            return
        signal = AsideSignalMeta().to_signal(
            Message.new(tag="file_editor", name=self._identity).with_content(
                f"[file_editor] {text}"
            ),
            description=text[:120],
        )
        self._send_signal(signal)

    # -- ws server ----------------------------------------------------------

    def _process_request(self, connection: ServerConnection, request: Any):
        path = request.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            try:
                html = self._html_path.read_text(encoding="utf-8")
            except OSError:
                return connection.respond(500, "index.html not found")
            response = connection.respond(200, html)
            del response.headers["Content-Type"]
            response.headers["Content-Type"] = "text/html; charset=utf-8"
            # A reload must show the page as it is on disk. Without this the browser
            # may keep the old one, and a change looks like it never happened.
            response.headers["Cache-Control"] = "no-store"
            return response
        if path == "/ws":
            return None
        if path == "/renderers" or path.startswith("/renderers/"):
            return self._renderer_response(connection, path)
        if path == "/__ver":
            # The page's fingerprint, so an open window can tell it runs old code.
            # Pull-based and cheap: the client asks, the server hashes the file,
            # nobody has to remember to refresh. Same mechanism as the screen.
            response = connection.respond(200, json.dumps({"ver": self._page_ver()}))
            del response.headers["Content-Type"]
            response.headers["Content-Type"] = "application/json"
            response.headers["Cache-Control"] = "no-store, max-age=0"
            return response
        return connection.respond(404, "not found")

    def _page_ver(self) -> str:
        try:
            return hashlib.sha1(self._html_path.read_bytes()).hexdigest()[:10]
        except OSError:
            return ""

    # -- renderers ----------------------------------------------------------

    @property
    def _renderers_dir(self) -> Path:
        """Where the renderer library lives: beside the page, not inside it.

        A renderer is a file. Adding one is writing a file — no command, no page
        edit — which is the whole point of the contract in DESIGN.md: the library
        has to be extendable by whoever is looking at the document, including the
        model, without a build step.
        """
        return self._html_path.parent / "renderers"

    @staticmethod
    def _renderer_meta(source: str, fallback: str) -> dict:
        """Read a renderer's self-description out of its header comment.

        The file explains itself (`// @name`, `// @ext`, `// @description`) so
        listing the library never means executing model-written code. Parsing a
        comment is a poor substitute for running the module — and that is exactly
        why it is the right side of the line: discovery must be free of side
        effects.
        """
        meta = {"name": fallback, "ext": [], "description": ""}
        for line in source.splitlines()[:12]:
            m = re.match(r"^\s*//\s*@(\w+)\s?(.*)$", line)
            if not m:
                continue
            key, value = m.group(1), m.group(2).strip()
            if key == "name" and value:
                meta["name"] = value
            elif key == "ext":
                meta["ext"] = [e for e in value.split() if e]
            elif key == "description":
                meta["description"] = value
        return meta

    def list_renderers(self) -> list[dict]:
        out: list[dict] = []
        try:
            paths = sorted(self._renderers_dir.glob("*.js"))
        except OSError:
            return out
        for p in paths:
            try:
                meta = self._renderer_meta(p.read_text(encoding="utf-8"), p.stem)
            except OSError:
                continue
            meta["file"] = p.name
            out.append(meta)
        return out

    def _renderer_response(self, connection: Any, path: str) -> Any:
        if path == "/renderers":
            response = connection.respond(
                200, json.dumps({"renderers": self.list_renderers()})
            )
            del response.headers["Content-Type"]
            response.headers["Content-Type"] = "application/json"
            response.headers["Cache-Control"] = "no-store, max-age=0"
            return response
        # `/renderers/<name>.js` — one file, as a module. `Path.name` throws away
        # any `..` before it reaches the disk, so a traversal attempt can only
        # ever name a file inside the library.
        name = Path(path[len("/renderers/"):]).name
        if not name.endswith(".js"):
            return connection.respond(404, "not found")
        target = self._renderers_dir / name
        try:
            body = target.read_text(encoding="utf-8")
        except OSError:
            return connection.respond(404, "not found")
        response = connection.respond(200, body)
        del response.headers["Content-Type"]
        response.headers["Content-Type"] = "text/javascript; charset=utf-8"
        # A renderer that changed on disk must not be served from cache: the whole
        # contract is "edit the file, see the face change".
        response.headers["Cache-Control"] = "no-store"
        return response

    async def _handler(self, connection: ServerConnection) -> None:
        self._clients.add(connection)
        try:
            await connection.send(
                json.dumps(self.snapshot(), ensure_ascii=False)
            )
            async for raw in connection:
                try:
                    frame = json.loads(raw)
                except (TypeError, ValueError):
                    continue
                kind = frame.get("type")
                thread = str(frame.get("thread", ""))
                n = int(frame.get("n", 0))
                if kind == "accept":
                    await self._verdict(thread, n, "accept")
                elif kind == "deny":
                    await self._verdict(thread, n, "deny")
                elif kind == "ask":
                    await self._ask(thread, n, str(frame.get("text", "")))
                elif kind == "detail":
                    await self._detail(thread, n)
                elif kind == "text":
                    await self._text(thread)
                elif kind == "view":
                    try:
                        await self.set_view(str(frame.get("view", "")), by="human")
                    except ValueError:
                        pass
                elif kind == "selection":
                    await self.set_selection(
                        thread,
                        int(frame.get("start", 0) or 0),
                        int(frame.get("end", 0) or 0),
                        str(frame.get("text", "")),
                    )
                elif kind == "cursor":
                    # The human scrolled: record where they are. It is read back as
                    # state, so "where is he" needs no signal to win my attention.
                    await self.set_view_at(thread, int(frame.get("start", 0) or 0))
                elif kind == "auto":
                    await self._set_thread_auto(thread, bool(frame.get("auto")))
                elif kind == "global_trust":
                    await self._set_global_trust(bool(frame.get("on")))
                elif kind == "trust_dir":
                    await self._set_dir_trust(
                        str(frame.get("path", "")), bool(frame.get("trusted"))
                    )
                elif kind == "toggle":
                    await self._set_enabled(bool(frame.get("enabled", True)))
                elif kind == "actions":
                    # The page pulls a thread's history instead of waiting to be
                    # pushed: a window that connected late, or a page that was
                    # reloaded, has no other way to learn what already happened.
                    snap = snapshot(self._store)
                    mine = [a for a in snap.get("actions", []) if a.get("thread") == thread]
                    await connection.send(json.dumps(
                        {"type": "actions", "thread": thread, "actions": mine}
                    ))
                elif kind == "edit_begin":
                    await self.begin_edit(
                        thread,
                        int(frame.get("start", 0) or 0),
                        int(frame.get("end", 0) or 0),
                    )
                elif kind == "edit_cancel":
                    await self.cancel_edit(thread)
                elif kind == "edit_commit":
                    try:
                        await self.commit_edit(
                            thread,
                            str(frame.get("region", "")),
                            str(frame.get("text", "")),
                            str(frame.get("label", "")),
                            str(frame.get("client_id", "")),
                        )
                    except (KeyError, ValueError):
                        # A rejected edit is not silence: the page keeps its
                        # pending action and is told it did not land.
                        await self.broadcast(
                            {"type": "rejected", "thread": thread,
                             "client_id": str(frame.get("client_id", ""))}
                        )
                elif kind == "face":
                    # What the page is actually drawing. Without this the state
                    # face only knows what *I* asked for, and the human's own
                    # switch would leave me describing a face nobody is looking at.
                    self.set_face(
                        thread,
                        str(frame.get("renderer", "")),
                        int(frame["seq"]) if frame.get("seq") else None,
                        str(frame.get("by", "human")),
                    )
        finally:
            self._clients.discard(connection)

    async def start(self) -> None:
        self._server = await serve(
            self._handler,
            self._host,
            self.port,
            process_request=self._process_request,
        )
        sockets = getattr(self._server, "sockets", None)
        if sockets:
            self.port = sockets[0].getsockname()[1]

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
