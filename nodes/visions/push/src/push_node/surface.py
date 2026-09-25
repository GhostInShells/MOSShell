"""The human-facing web surface: pending requests in, verdicts out, live previews.

One aiohttp app serves three things on one port:
- ``GET /`` — the control page (index.html)
- ``GET /stream/<id>`` — a live MJPEG preview of one session (the page shows it
  as an ``<img>``, so previews self-refresh with no frame push over the socket)
- ``GET /ws`` — the control websocket (snapshot + session deltas down, actions up)

Verdicts are handed down, never written here: ``store.settle`` wakes the
channel's waiter, and the channel is the only place a session moves from
pending to live. This module owns the *signal to the ghost* — a human action is
a fact the model must learn, travelling as a must-not-lose notify.

aiohttp is used rather than the terminal's ``websockets`` because this node also
streams MJPEG, which aiohttp does natively — and it is already the visions
family venv's HTTP dep (the old camera viewer used it for exactly this).
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Optional

from aiohttp import WSMsgType, web

from ghoshell_moss.message import Message
from ghoshell_moss.signals import AsideSignalMeta, NotifySignalMeta

from .store import AcceptAll, PushStore

__all__ = ["PushSurface", "PushHandles"]

_BOUNDARY = "frame"


class PushHandles:
    """Filled by the channel once its commands exist.

    The surface owns the buttons; the channel owns the producers and the
    sessions. main wires this holder to both, and the channel puts its stop /
    open commands in it — the surface never reaches into the channel.
    """

    def __init__(self) -> None:
        self.stop: Callable[[int], Awaitable[None]] | None = None
        self.stop_all: Callable[[], Awaitable[None]] | None = None
        self.open_human: Callable[[str, str], Awaitable[int]] | None = None


class PushSurface:
    def __init__(
        self,
        store: PushStore,
        accept_all: AcceptAll,
        *,
        send_signal: Callable[[Any], None],
        self_identity: str,
        host: str,
        port: int,
        html_path: Path,
        handles: PushHandles | None = None,
        preview_fps: float = 5.0,
    ) -> None:
        self._store = store
        self._accept_all = accept_all
        self._frame: Callable[[int], Optional[bytes]] = lambda sid: None
        self._send_signal = send_signal
        self._identity = self_identity
        self._host = host
        self._html_path = html_path
        self._handles = handles or PushHandles()
        self._preview_fps = preview_fps
        self._clients: set[web.WebSocketResponse] = set()
        self._runner: Optional[web.AppRunner] = None
        self.port = port

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self.port}"

    def bind_frame(self, frame: Callable[[int], Optional[bytes]]) -> None:
        """Wire the producer's frame accessor in — the channel owns it, so this
        is set after the channel builds the producer manager."""
        self._frame = frame

    # -- downlink -----------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        return {
            "type": "snapshot",
            "accept_all": self._accept_all.enabled,
            "sessions": [s.view() for s in self._store.sessions()],
        }

    async def broadcast(self, frame: dict[str, Any]) -> None:
        if not self._clients:
            return
        payload = json.dumps(frame, ensure_ascii=False)
        for client in list(self._clients):
            try:
                await client.send_str(payload)
            except Exception:
                self._clients.discard(client)

    # -- uplink -------------------------------------------------------------

    async def _accept(self, session_id: int) -> None:
        if not self._store.settle(session_id, "accept"):
            return
        session = self._store.get(session_id)
        self._aside(f"[push #{session_id}] accepted '{session.label or session.source}' — streaming now")

    async def _deny(self, session_id: int) -> None:
        if not self._store.settle(session_id, "deny"):
            return
        session = self._store.get(session_id)
        self._aside(f"[push #{session_id}] denied '{session.label or session.source}'")

    async def _stop(self, session_id: int) -> None:
        if self._handles.stop is not None:
            await self._handles.stop(session_id)

    async def _stop_all(self) -> None:
        if self._handles.stop_all is not None:
            await self._handles.stop_all()

    async def _open(self, source: str, label: str) -> None:
        if self._handles.open_human is not None:
            session_id = await self._handles.open_human(source, label)
            self._notify(f"[push #{session_id}] human opened a {source} stream", next_=True)

    async def _set_accept_all(self, enabled: bool) -> None:
        self._accept_all.enabled = bool(enabled)
        await self.broadcast({"type": "accept_all", "enabled": self._accept_all.enabled})

    # -- signals ------------------------------------------------------------

    def _notify(self, text: str, *, next_: bool = True) -> None:
        if self._send_signal is None:
            return
        signal = NotifySignalMeta(next=next_).to_signal(
            Message.new(tag="push", name=self._identity).with_content(text),
            description=text[:120],
        )
        self._send_signal(signal)

    def _aside(self, text: str) -> None:
        if self._send_signal is None:
            return
        signal = AsideSignalMeta().to_signal(
            Message.new(tag="push", name=self._identity).with_content(text),
            description=text[:120],
        )
        self._send_signal(signal)

    # -- http routes --------------------------------------------------------

    async def _index(self, request: web.Request) -> web.Response:
        try:
            html = self._html_path.read_text(encoding="utf-8")
        except OSError:
            return web.Response(status=500, text="index.html not found")
        return web.Response(text=html, content_type="text/html")

    async def _stream(self, request: web.Request) -> web.StreamResponse:
        session_id = int(request.match_info["id"])
        resp = web.StreamResponse(
            headers={"Content-Type": f"multipart/x-mixed-replace; boundary={_BOUNDARY}"}
        )
        await resp.prepare(request)
        try:
            while True:
                jpeg = self._frame(session_id)
                if jpeg:
                    chunk = (
                        f"--{_BOUNDARY}\r\n"
                        "Content-Type: image/jpeg\r\n"
                        f"Content-Length: {len(jpeg)}\r\n\r\n"
                    ).encode("ascii") + jpeg + b"\r\n"
                    await resp.write(chunk)
                await asyncio.sleep(1.0 / self._preview_fps)
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        return resp

    async def _ws(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self._clients.add(ws)
        try:
            await ws.send_str(json.dumps(self.snapshot(), ensure_ascii=False))
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                try:
                    frame = json.loads(msg.data)
                except (TypeError, ValueError):
                    continue
                kind = frame.get("type")
                session_id = int(frame.get("id", 0))
                if kind == "accept":
                    await self._accept(session_id)
                elif kind == "deny":
                    await self._deny(session_id)
                elif kind == "stop":
                    await self._stop(session_id)
                elif kind == "stop_all":
                    await self._stop_all()
                elif kind == "open":
                    await self._open(str(frame.get("source", "")), str(frame.get("label", "")))
                elif kind == "accept_all":
                    await self._set_accept_all(bool(frame.get("enabled")))
        finally:
            self._clients.discard(ws)
        return ws

    # -- lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        app = web.Application()
        app.router.add_get("/", self._index)
        app.router.add_get("/stream/{id}", self._stream)
        app.router.add_get("/ws", self._ws)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._host, self.port)
        await site.start()
        # Read the bound port back (0 = ephemeral).
        if self.port == 0:
            for sock in site._server.sockets:
                self.port = sock.getsockname()[1]
                break

    async def stop(self) -> None:
        for ws in list(self._clients):
            await ws.close()
        self._clients.clear()
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
