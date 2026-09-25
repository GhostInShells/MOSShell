"""The human-facing web surface: window frames down, steering up.

One port serves the page and the WebSocket. Downlink carries the window stream
(``snapshot`` on connect, then ``state`` / ``open`` / ``close`` / ``activate`` /
``fullscreen`` / ``veil``), the 5Hz audio samples, and a ``notice`` line for the
human's activity log.

Uplink carries the human's moves: switch to a group or the desktop, toggle
fullscreen on an item, and dismiss (send a window back to the desktop).

A human move is handed to the store; the store stays the only authority, and this
module re-broadcasts the change so every client stays in step. The move also
reaches the ghost as an aside — a fact it notices when free, not an interruption.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from websockets.asyncio.server import ServerConnection, serve

from ghoshell_moss.message import Message
from ghoshell_moss.signals import AsideSignalMeta

from . import projection as P
from .audio import AudioSource
from .model import ScreenModel

__all__ = ["ScreenSurface"]

_AUDIO_HZ = 5.0


class ScreenSurface:
    def __init__(
        self,
        model: ScreenModel,
        audio: AudioSource,
        *,
        send_signal: Callable[[Any], None],
        self_identity: str,
        host: str = "127.0.0.1",
        port: int = 0,
        html_path: Path,
    ) -> None:
        self._model = model
        self._audio = audio
        self._send_signal = send_signal
        self._identity = self_identity
        self._host = host
        self._html_path = html_path
        self._clients: set[ServerConnection] = set()
        self._server: Any = None
        self._audio_task: asyncio.Task | None = None
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

    # -- uplink -------------------------------------------------------------

    async def _switch_view(self, group: str) -> None:
        try:
            self._model.activate(group)
        except (KeyError, ValueError) as e:
            await self.broadcast({"type": "error", "text": str(e)})
            return
        await self.broadcast(P.activate_frame(self._model, by_model=False))
        where = f"#{group}" if group else "the desktop"
        self._aside(f"human switched to {where}")

    async def _toggle_fullscreen(self, item_id: str) -> None:
        current = self._model.fullscreen()
        new = None if (not item_id or current == item_id) else item_id
        try:
            self._model.set_fullscreen(new)
        except (KeyError, ValueError) as e:
            await self.broadcast({"type": "error", "text": str(e)})
            return
        await self.broadcast(P.fullscreen_frame(new))
        what = f"#{new}" if new else "off"
        self._aside(f"human toggled fullscreen {what}")

    async def _dismiss(self, item_id: str) -> None:
        if self._model.dismiss(item_id) is None:
            return
        await self.broadcast(P.state_frame(self._model))
        self._aside(f"human sent #{item_id} to the desktop")

    async def _tap(self, item_id: str) -> None:
        """A human tapped a desktop proxy — interest, not a move. Reach the ghost
        as an aside so it can decide whether to arrange; nothing on screen changes."""
        self._aside(f"human tapped #{item_id} on the desktop")

    def _aside(self, text: str) -> None:
        if self._send_signal is None:
            return
        signal = AsideSignalMeta().to_signal(
            Message.new(tag="webview_screen", name=self._identity).with_content(
                f"[webview_screen] {text}"
            ),
            description=text[:120],
        )
        self._send_signal(signal)

    # -- audio pump ---------------------------------------------------------

    async def _audio_loop(self) -> None:
        while True:
            await asyncio.sleep(1 / _AUDIO_HZ)
            if not self._clients:
                continue
            await self.broadcast(P.audio_frame(self._audio.sample()))

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
            await connection.send(
                json.dumps(P.snapshot(self._model), ensure_ascii=False)
            )
            async for raw in connection:
                try:
                    frame = json.loads(raw)
                except (TypeError, ValueError):
                    continue
                kind = frame.get("type")
                if kind == "switch_view":
                    await self._switch_view(str(frame.get("group", "")))
                elif kind == "fullscreen":
                    await self._toggle_fullscreen(str(frame.get("id", "")))
                elif kind == "dismiss":
                    await self._dismiss(str(frame.get("id", "")))
                elif kind == "tap":
                    await self._tap(str(frame.get("id", "")))
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
        self._audio_task = asyncio.create_task(self._audio_loop())

    async def stop(self) -> None:
        if self._audio_task is not None:
            self._audio_task.cancel()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
