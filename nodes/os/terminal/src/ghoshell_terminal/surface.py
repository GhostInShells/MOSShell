"""The human-facing web surface: cards in, verdicts out.

One port serves the page and the WebSocket. Downlink carries the card stream
(``card.head`` / ``card.delta`` / ``card.tail`` / ``card.output`` / ``card.full``)
plus a full snapshot on connect and the global ``mode``. Uplink carries the
human's three actions (``accept`` / ``deny`` / ``ask``), the two stops, and the
mode switch.

Verdicts are handed down, never written here: ``store.settle`` wakes the channel's
waiter, and the channel is the only place card state moves. What this module does
own is the signal to the ghost — the human's action is a fact the model must
learn, and it travels as a must-not-lose notify that guarantees a turn.
"""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from websockets.asyncio.server import ServerConnection, serve

from ghoshell_moss.ground import DEFAULT_L0_FILENAME
from ghoshell_moss.message import Message
from ghoshell_moss.signals import AsideSignalMeta, NotifySignalMeta

from .card import CardState
from .store import CardStore

__all__ = ["TerminalSurface", "StopHandles"]

_DEBOUNCE_SECONDS = 0.4
"""A verdict is one decision. Repeated clicks inside this window are the same
decision arriving twice, not a second one."""

_ANALYZE_INSTRUCTION = (
    "Explain what this shell command does, what it touches, and its risks. "
    "Answer the human's question directly and concretely. Assess the command "
    "as written — do not cheerlead or assume intent."
)
"""The zero-context analyzer's fixed instruction. It sees only the command text,
the cwd, and the human's question — nothing the issuing model intended."""


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
            "root": str(self._store.root),
            "threads": [t.model_dump(mode="json") for t in self._store.threads()],
            "cards": [c.view() for c in self._store.cards()],
        }

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
        self._aside(
            f"[terminal #{card_id}] accepted '{card.title}' — it is running now"
        )

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
        self._aside(
            f"[terminal #{card_id}] denied '{card.title}' — do not re-issue this one"
        )

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
        text, its cwd, and the human's question — no title, no description, no
        ghost intent. The reply goes back to the surface, never into the ghost's
        own message stream.
        """
        llm_funcs = self._llm_funcs() if self._llm_funcs is not None else None
        if llm_funcs is None:
            await self.broadcast({"type": "error", "text": "analyze: LLMFuncs not registered"})
            return
        card = self._store.get(card_id)
        if card is None:
            return
        lines = [f"cwd: {card.cwd}", f"command: {card.content}"]
        for turn in history or []:
            if isinstance(turn, dict):
                lines.append(f"Q: {turn.get('q', '')}")
                lines.append(f"A: {turn.get('a', '')}")
        lines.append(f"Q: {text}")
        try:
            result = await llm_funcs.call(
                instruction=_ANALYZE_INSTRUCTION,
                prompt="\n".join(lines),
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
        await self._broadcast_threads()
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

    async def _broadcast_threads(self) -> None:
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

    def _aside(self, text: str) -> None:
        """A decision the ghost notices without being interrupted. Verdicts are
        facts, not questions — they buffer until the ghost is free."""
        if self._send_signal is None:
            return
        signal = AsideSignalMeta().to_signal(
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
            return response
        if path == "/ws":
            return None
        return connection.respond(404, "not found")

    async def _handler(self, connection: ServerConnection) -> None:
        self._clients.add(connection)
        try:
            await connection.send(json.dumps(self.snapshot(), ensure_ascii=False))
            async for raw in connection:
                try:
                    frame = json.loads(raw)
                except (TypeError, ValueError):
                    continue
                kind = frame.get("type")
                card_id = int(frame.get("id", 0))
                if kind == "accept":
                    await self._accept(card_id, str(frame.get("text", "") or ""))
                elif kind == "deny":
                    await self._deny(card_id, str(frame.get("text", "") or ""))
                elif kind == "ask":
                    await self._ask(card_id, str(frame.get("text", "")))
                elif kind == "analyze":
                    await self._analyze(
                        card_id,
                        str(frame.get("text", "")),
                        frame.get("history") or [],
                    )
                elif kind == "accept_all":
                    await self._accept_all()
                elif kind == "deny_all":
                    await self._deny_all()
                elif kind == "stop":
                    await self._stop(card_id)
                elif kind == "stop_all":
                    await self._stop_all()
                elif kind == "mode":
                    await self._set_mode(str(frame.get("mode", "")))
                elif kind == "thread_auto":
                    await self._set_thread_auto(
                        str(frame.get("name", "")), bool(frame.get("auto"))
                    )
                elif kind == "ground":
                    await self._render_ground(str(frame.get("thread", "")))
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
