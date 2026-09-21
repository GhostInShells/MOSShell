"""Web surface — one port serves the page and the WebSocket.

Downlink: ``snapshot`` on connect, then ``frame.head`` / ``frame.full`` as
the channel writes frames, plus ``enabled`` when the master switch flips.

Uplink: one thing only — ``toggle`` on the master switch. There is no
per-frame verdict; observation is the whole point of the surface.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from websockets.asyncio.server import ServerConnection, serve

from .store import FrameStore

__all__ = ["CoBrowserSurface"]


class CoBrowserSurface:
    def __init__(
        self,
        store: FrameStore,
        *,
        host: str,
        port: int,
        html_path: Path,
    ) -> None:
        self._store = store
        self._host = host
        self._html_path = html_path
        self._clients: set[ServerConnection] = set()
        self._server: Any = None
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
            "enabled": self._store.enabled,
            "frames": [f.view() for f in self._store.frames()],
        }

    # -- uplink -------------------------------------------------------------

    async def _toggle(self, enabled: bool) -> None:
        self._store.set_enabled(enabled)
        await self.broadcast({"type": "enabled", "enabled": self._store.enabled})

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
                if frame.get("type") == "toggle":
                    await self._toggle(bool(frame.get("enabled")))
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
