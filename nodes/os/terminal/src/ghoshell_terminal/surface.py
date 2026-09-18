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

from ghoshell_moss.message import Message
from ghoshell_moss.signals import NotifySignalMeta

from .card import CardState
from .store import CardStore

__all__ = ["TerminalSurface", "StopHandles"]

_DEBOUNCE_SECONDS = 0.4
"""A verdict is one decision. Repeated clicks inside this window are the same
decision arriving twice, not a second one."""


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
    ) -> None:
        self._store = store
        self._send_signal = send_signal
        self._identity = self_identity
        self._host = host
        self._html_path = html_path
        self._stops = stops or StopHandles()
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

    async def _accept(self, card_id: int) -> None:
        if card_id in self._settled or self._debounced(card_id, "accept"):
            return
        if not self._store.settle(card_id, "accept"):
            return
        self._settled.add(card_id)
        card = self._store.get(card_id)
        self._signal(
            f"[terminal #{card_id}] accepted '{card.title}' — it is running now",
            card.thread,
            next_=False,
        )

    async def _deny(self, card_id: int) -> None:
        if card_id in self._settled or self._debounced(card_id, "deny"):
            return
        if not self._store.settle(card_id, "deny"):
            return
        self._settled.add(card_id)
        card = self._store.get(card_id)
        self._signal(
            f"[terminal #{card_id}] denied '{card.title}' — do not re-issue this one",
            card.thread,
        )

    async def _ask(self, card_id: int, text: str) -> None:
        if self._debounced(card_id, "ask"):
            return
        card = self._store.get(card_id)
        if card is None or card.state is not CardState.AWAITING:
            return
        self._store.add_dialogue(card_id, "human", text or "(wants to talk)")
        await self.broadcast({"type": "card.full", "card": card.view()})
        self._signal(
            f"[terminal #{card_id}] human asks about '{card.title}': {text}",
            card.thread,
        )

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

    def _signal(self, text: str, thread: str = "", *, next_: bool = True) -> None:
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
                    await self._accept(card_id)
                elif kind == "deny":
                    await self._deny(card_id)
                elif kind == "ask":
                    await self._ask(card_id, str(frame.get("text", "")))
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
