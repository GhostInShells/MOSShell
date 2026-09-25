"""The human-facing web surface — the tree on the left, the detail on the right.

One port serves the page and the WebSocket. On connect the client receives a
full ``state`` frame; every mutation rebroadcasts it. Uplink carries the human's
moves — select a node, confirm an action card, open a path — as ``user.action``
frames that the channel relays onward.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from websockets.asyncio.server import ServerConnection, serve

__all__ = ["DecisionTreeSurface"]


class DecisionTreeSurface:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        html_path: Path,
        get_state: Callable[[], dict[str, Any]],
        on_action: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        self._host = host
        self._html_path = html_path
        self.get_state = get_state
        self.on_action = on_action
        self._clients: set[ServerConnection] = set()
        self._server: Any = None
        self.port = port

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self.port}"

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
            await connection.send(self._encode({"type": "state", **self.get_state()}))
            async for raw in connection:
                try:
                    frame = json.loads(raw)
                except (TypeError, ValueError):
                    continue
                if frame.get("type") == "user.action":
                    await self.on_action(frame)
        finally:
            self._clients.discard(connection)

    @staticmethod
    def _encode(frame: dict[str, Any]) -> str:
        return json.dumps(frame, ensure_ascii=False)

    async def broadcast(self, frame: dict[str, Any]) -> None:
        if not self._clients:
            return
        payload = self._encode(frame)
        await asyncio.gather(*(self._send(c, payload) for c in list(self._clients)))

    async def _send(self, connection: ServerConnection, payload: str) -> None:
        try:
            await connection.send(payload)
        except Exception:
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
            self._server = None
