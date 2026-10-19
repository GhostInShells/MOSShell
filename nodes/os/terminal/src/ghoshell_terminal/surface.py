"""The human-facing web surface: cards in, verdicts out.

One port serves the page and the WebSocket. Downlink carries the card stream
(``card.head`` / ``card.delta`` / ``card.tail`` / ``card.output`` / ``card.full``)
plus a full snapshot on connect, the global ``mode`` and the layout mode. Uplink carries the
human's three actions (``accept`` / ``deny`` / ``ask``), the two stops, and the
mode switch.

Verdicts are handed down, never written here: ``store.settle`` wakes the channel's
waiter, and the channel is the only place card state moves — and the only place a
verdict is reported to the model. This module keeps the human's side whole: it
broadcasts the card stream and it records what the human said (the ``ask`` box's
dialogue), which is where the channel reads a denial's reason from. Its own signal
to the ghost is the ``ask`` question — a question, not a decision.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from websockets.asyncio.server import ServerConnection, serve

from ghoshell_moss.ground import DEFAULT_L0_FILENAME
from ghoshell_moss.message import Message
from ghoshell_moss.signals import NotifySignalMeta

from .card import CardState
from .store import CardStore

__all__ = ["TerminalSurface", "StopHandles"]

_DEBOUNCE_SECONDS = 0.4
"""A verdict is one decision. Repeated clicks inside this window are the same
decision arriving twice, not a second one."""

_LAYOUTS = ("split", "page")
"""How an open card is shown — the model's half of the presentation contract.

``split`` keeps the list beside the open card; ``page`` gives the card the whole
window. Which card is open is the human's click; this decides only how big it
opens, and ``_layout_by`` records who chose it so the page can say so. The default
is ``page``: an open card is the subject, and the list is not worth sharing room
with it. ``split`` is what the model reaches for when the window is roomy.
"""

_ANALYZE_INSTRUCTION = (
    "You are a second-opinion shell reviewer. The prompt is an XML document: a "
    "<cwd> and <command> describe the command under review, and one or more "
    "<turn> blocks carry the human's <question> and your earlier <answer>. "
    "Answer the newest (last, unanswered) question directly and concretely. "
    "Assess the command as written — do not cheerlead or assume intent. "
    "Reply in the same language the human used in their question."
)
"""The zero-context analyzer's fixed instruction. The prompt is a single growing
XML document (cwd + command + dialogue turns) — no assistant/role messages."""


def _analyze_prompt(cwd: str, command: str, history: list, question: str) -> str:
    """The zero-context prompt as one XML document that grows between rounds.

    Each turn appends a ``<turn>`` with its ``<question>`` and, once answered,
    an ``<answer>``. The final turn carries only the newest question — the model
    answers it. There are no assistant messages: the instruction is fixed and
    this document is the only thing that changes as the dialogue advances.
    """
    out = [
        "<analysis>",
        f"  <cwd>{escape(cwd)}</cwd>",
        f"  <command>{escape(command)}</command>",
    ]
    for turn in history or []:
        if not isinstance(turn, dict):
            continue
        out.append("  <turn>")
        out.append(f"    <question>{escape(str(turn.get('q', '')))}</question>")
        answer = str(turn.get("a", "") or "")
        if answer:
            out.append(f"    <answer>{escape(answer)}</answer>")
        out.append("  </turn>")
    out.append("  <turn>")
    out.append(f"    <question>{escape(question)}</question>")
    out.append("  </turn>")
    out.append("</analysis>")
    return "\n".join(out)


def _ground_root_for(cwd: Path) -> Path | None:
    """The nearest GROUND.md at or above ``cwd``, or None (mirrors the channel's)."""
    cwd = cwd.resolve()
    if (cwd / DEFAULT_L0_FILENAME).is_file():
        return cwd
    current = cwd.parent
    while True:
        if (current / DEFAULT_L0_FILENAME).is_file():
            return current
        if current == current.parent:
            return None
        current = current.parent


class StopHandles:
    """Filled in by the channel once its commands exist.

    The web surface owns stop / stop-all buttons, but only the channel knows which
    card holds which process. Rather than let the surface reach into the channel,
    main.py makes this holder, hands it to both, and the channel puts its stop
    commands in it.
    """

    def __init__(self) -> None:
        self.stop: Callable[[int], Awaitable[str]] | None = None
        self.stop_all: Callable[[], Awaitable[str]] | None = None


class TerminalSurface:
    def __init__(
        self,
        store: CardStore,
        *,
        send_signal: Callable[[Any], None],
        self_identity: str,
        host: str,
        port: int,
        html_path: Path,
        stops: StopHandles | None = None,
        llm_funcs: Callable[[], Any] | None = None,
        groundset: Any | None = None,
    ) -> None:
        self._store = store
        self._send_signal = send_signal
        self._identity = self_identity
        self._host = host
        self._html_path = html_path
        self._stops = stops or StopHandles()
        self._llm_funcs = llm_funcs
        self._groundset = groundset
        self._clients: set[ServerConnection] = set()
        self._server: Any = None
        self._settled: set[int] = set()
        self._last_action: dict[tuple[int, str], float] = {}
        self._layout = "page"
        self._layout_by = "default"
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
        return {
            "type": "snapshot",
            "mode": self._store.mode,
            "global_trust": self._store.global_trust(),
            "layout": self._layout,
            "layout_by": self._layout_by,
            "root": str(self._store.root),
            "threads": [t.model_dump(mode="json") for t in self._store.threads()],
            "cards": [c.view() for c in self._store.cards()],
        }

    # -- layout (the model's side of the presentation contract) -------------

    def layout_state(self) -> tuple[str, str]:
        return self._layout, self._layout_by

    async def set_layout(self, mode: str) -> str:
        """Set how an open card is shown, and remember who chose it.

        The human owns the click that opens a card; the model owns this. Every page
        is told, so the marker can name its author instead of hiding the decision.
        """
        chosen = (mode or "").strip().lower()
        if chosen not in _LAYOUTS:
            raise ValueError(f"layout must be one of {_LAYOUTS}, got {mode!r}")
        self._layout = chosen
        self._layout_by = "model"
        await self.broadcast({"type": "layout", "layout": chosen, "by": self._layout_by})
        return chosen

    # -- uplink -------------------------------------------------------------

    def _debounced(self, card_id: int, action: str) -> bool:
        now = time.monotonic()
        key = (card_id, action)
        last = self._last_action.get(key, 0.0)
        if now - last < _DEBOUNCE_SECONDS:
            return True
        self._last_action[key] = now
        return False

    async def _accept(self, card_id: int, text: str = "") -> None:
        if card_id in self._settled or self._debounced(card_id, "accept"):
            return
        if not self._store.settle(card_id, "accept"):
            return
        self._settled.add(card_id)
        card = self._store.get(card_id)
        if text:
            self._store.add_dialogue(card_id, "human", text)
            await self.broadcast({"type": "card.full", "card": card.view()})

    async def _deny(self, card_id: int, text: str = "") -> None:
        if card_id in self._settled or self._debounced(card_id, "deny"):
            return
        if not self._store.settle(card_id, "deny"):
            return
        self._settled.add(card_id)
        card = self._store.get(card_id)
        if text:
            self._store.add_dialogue(card_id, "human", text)
            await self.broadcast({"type": "card.full", "card": card.view()})

    async def _ask(self, card_id: int, text: str) -> None:
        if self._debounced(card_id, "ask"):
            return
        card = self._store.get(card_id)
        if card is None or card.state is not CardState.AWAITING:
            return
        self._store.add_dialogue(card_id, "human", text or "(wants to talk)")
        await self.broadcast({"type": "card.full", "card": card.view()})
        self._notify(
            f"[terminal #{card_id}] human asks about '{card.title}': {text}",
            next_=True,
        )

    async def _analyze(self, card_id: int, text: str, history: list) -> None:
        """Zero-context second opinion — a side-channel model reads the command.

        Deliberately divorced from the issuing model: the prompt is the command
        text, its cwd, and the human's dialogue — no title, no description, no
        ghost intent. The reply goes back to the surface, never into the ghost's
        own message stream.
        """
        card = self._store.get(card_id)
        if card is None:
            return
        llm_funcs = self._llm_funcs() if self._llm_funcs is not None else None
        if llm_funcs is None:
            await self.broadcast({
                "type": "analyze",
                "id": card_id,
                "unavailable": True,
                "text": "analyze is unavailable — no LLM func engine is configured",
            })
            return
        prompt = _analyze_prompt(card.cwd, card.content, history, text)
        try:
            result = await llm_funcs.call_messages(
                instruction=_ANALYZE_INSTRUCTION,
                prompt=[Message.new().with_content(prompt)],
            )
        except Exception as e:
            await self.broadcast({"type": "error", "text": f"analyze failed: {e}"})
            return
        await self.broadcast({
            "type": "analyze",
            "id": card_id,
            "text": (result.content or "").strip(),
        })

    async def _accept_all(self) -> None:
        for card in self._store.awaiting():
            await self._accept(card.id)

    async def _deny_all(self) -> None:
        for card in self._store.awaiting():
            await self._deny(card.id)

    async def _stop(self, card_id: int) -> None:
        if self._stops.stop is not None:
            await self._stops.stop(card_id)

    async def _stop_all(self) -> None:
        if self._stops.stop_all is not None:
            await self._stops.stop_all()

    async def _set_global_trust(self, on: bool) -> None:
        """Hand every approval over, or take it back — the human's switch.

        State only: the notice carries it, and a signal here would only interrupt.
        """
        store_on = self._store.set_global_trust(on)
        await self.broadcast({"type": "trust", "global_trust": store_on})

    async def _set_mode(self, mode: str) -> None:
        try:
            resolved = self._store.set_mode(mode)
        except ValueError as e:
            await self.broadcast({"type": "error", "text": str(e)})
            return
        await self.broadcast({"type": "mode", "mode": resolved})

    async def _set_thread_auto(self, name: str, auto: bool) -> None:
        try:
            self._store.set_thread_auto(name, auto)
        except KeyError:
            await self.broadcast({"type": "error", "text": f"no thread {name!r}"})
            return
        await self.sync_threads()
        # Trusting a thread also settles what is already waiting in it — the
        # human's "auto" means "run everything here", including cards that were
        # issued before the switch was flipped.
        if auto:
            for card in list(self._store.awaiting()):
                if card.thread == name:
                    await self._accept(card.id)

    async def _render_ground(self, thread: str) -> None:
        """Render a thread's cognitive field for the human to view (shared).

        The model reads the same field with ``ground(thread)``; this is the
        human's on-demand view, requested from the surface rather than cached
        in the DOM.
        """
        if self._groundset is None:
            await self.broadcast({"type": "error", "text": "ground is not available"})
            return
        t = self._store.get_thread(thread)
        if t is None:
            await self.broadcast({"type": "error", "text": f"no thread {thread!r}"})
            return
        root = _ground_root_for(Path(t.cwd))
        if root is None:
            await self.broadcast({
                "type": "ground", "thread": thread,
                "text": f"no ground (no GROUND.md) from {t.cwd} up to the filesystem root",
            })
            return
        try:
            opened = await self._groundset.open(root)
            view = await opened.render(cwd=Path(t.cwd))
        except Exception as e:
            await self.broadcast({"type": "error", "text": f"ground failed: {e}"})
            return
        await self.broadcast({"type": "ground", "thread": thread, "text": str(view)})

    async def sync_threads(self) -> None:
        """Push the thread list to every open page.

        A page only learns the threads from the snapshot it connected with, so
        anything that *adds* a thread — the channel's ``open()``, the human's
        auto switch — has to say so. Without this the detail panel silently
        drops a card's thread row (name, cwd, auto, ground), because the page
        looks the thread up in a map that was never updated.
        """
        await self.broadcast({
            "type": "threads",
            "threads": [t.model_dump(mode="json") for t in self._store.threads()],
        })

    def _notify(self, text: str, *, next_: bool = True) -> None:
        """A must-not-lose message. ``next`` guarantees the ghost a turn."""
        if self._send_signal is None:
            return
        signal = NotifySignalMeta(next=next_).to_signal(
            Message.new(tag="terminal", name=self._identity).with_content(text),
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
        if path == "/__ver":
            # The page's fingerprint, so an open window can tell it runs old code.
            # Pull-based and cheap: the client asks, the server hashes the file,
            # nobody has to remember to hit refresh. Same mechanism as the screen.
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

    # Frames whose work can take as long as a model call or a ground render.
    # They must not run inside the read loop: that loop is this page's only
    # uplink, so one slow frame parked in an ``await`` swallows every click
    # behind it — while broadcasts, being a different task, keep flowing. The
    # page then looks alive and is deaf, which is the worst of both.
    # (Found live, D19: an ``analyze`` blocked the loop and the human's accepts
    # never arrived; the downlink kept working, so "the link is fine" was true
    # of half of it.)
    _SLOW_FRAMES = frozenset({"ask", "analyze", "ground"})

    async def _run_frame(self, kind: str, card_id: int, work: Awaitable[None]) -> None:
        """Run one frame's work. A failure is reported, never swallowed."""
        try:
            await work
        except Exception as e:  # noqa: BLE001 — one bad frame must not kill the link
            print(f"[terminal] frame {kind}#{card_id} failed: {e!r}", file=sys.stderr)
            try:
                await self.broadcast({"type": "error", "text": f"{kind} failed: {e}"})
            except Exception:  # noqa: BLE001 — the report must not become the failure
                pass

    def _frame_work(self, kind: str, card_id: int, frame: dict[str, Any]) -> Awaitable[None] | None:
        """Map a frame to the work it asks for; ``None`` = not a frame we know."""
        text = str(frame.get("text", "") or "")
        if kind == "accept":
            return self._accept(card_id, text)
        if kind == "deny":
            return self._deny(card_id, text)
        if kind == "ask":
            return self._ask(card_id, str(frame.get("text", "")))
        if kind == "analyze":
            return self._analyze(card_id, text, frame.get("history") or [])
        if kind == "accept_all":
            return self._accept_all()
        if kind == "deny_all":
            return self._deny_all()
        if kind == "stop":
            return self._stop(card_id)
        if kind == "stop_all":
            return self._stop_all()
        if kind == "global_trust":
            return self._set_global_trust(bool(frame.get("on")))
        if kind == "mode":
            return self._set_mode(str(frame.get("mode", "")))
        if kind == "thread_auto":
            return self._set_thread_auto(str(frame.get("name", "")), bool(frame.get("auto")))
        if kind == "ground":
            return self._render_ground(str(frame.get("thread", "")))
        return None

    async def _handler(self, connection: ServerConnection) -> None:
        self._clients.add(connection)
        try:
            await connection.send(json.dumps(self.snapshot(), ensure_ascii=False))
            async for raw in connection:
                try:
                    frame = json.loads(raw)
                    if not isinstance(frame, dict):
                        continue
                    kind = str(frame.get("type", ""))
                    card_id = int(frame.get("id", 0) or 0)
                except (TypeError, ValueError) as e:
                    # A frame we cannot read is a fact about the page, not a
                    # reason to drop the link — say it and keep reading.
                    print(f"[terminal] bad frame {raw[:80]!r}: {e!r}", file=sys.stderr)
                    continue
                if kind == "ping":
                    # One cheap round trip, answered here so it can never queue
                    # behind real work: it is how the page tells a live link
                    # from a deaf one.
                    await connection.send(json.dumps({"type": "pong"}))
                    continue
                work = self._frame_work(kind, card_id, frame)
                if work is None:
                    print(f"[terminal] unknown frame {kind!r}", file=sys.stderr)
                    continue
                if kind in self._SLOW_FRAMES:
                    asyncio.create_task(self._run_frame(kind, card_id, work))
                else:
                    await self._run_frame(kind, card_id, work)
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
