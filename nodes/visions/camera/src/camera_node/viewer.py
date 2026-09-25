"""Minimal MJPEG viewer — lets a human see the ghost's camera field of view.

Runs an aiohttp server in the matrix event loop. A browser opening
``http://127.0.0.1:<port>/stream`` consumes a ``multipart/x-mixed-replace``
stream of JPEG frames pulled from the controller's rolling cache. No frontend
build: an ``<img src="/stream">`` or a direct page nav works.

This is the view-only baseline of the family contract's "minimal GUI". If we
later want the viewer to host interactive events (e.g. authorization clicks),
that's a heavier build (reflex/streamlit) and a separate decision.
"""
from __future__ import annotations

import asyncio
from typing import Callable, Optional

from aiohttp import web

_BOUNDARY = "frame"


class MjpegViewer:
    def __init__(
        self,
        latest_jpeg: Callable[[], Optional[bytes]],
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        fps: float = 10.0,
    ):
        self._latest_jpeg = latest_jpeg
        self._host = host
        self._port = port
        self._fps = fps
        self._runner: Optional[web.AppRunner] = None

    @property
    def port(self) -> int:
        return self._port

    async def start(self) -> None:
        app = web.Application()
        app.router.add_get("/", self._index)
        app.router.add_get("/stream", self._stream)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._host, self._port)
        await site.start()

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    async def _index(self, request: web.Request) -> web.Response:
        html = (
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<title>camera</title></head><body style='margin:0;background:#000'>"
            f"<img src='/stream' style='width:100vw;height:100vh;object-fit:contain'/></body></html>"
        )
        return web.Response(text=html, content_type="text/html")

    async def _stream(self, request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse(
            headers={"Content-Type": f"multipart/x-mixed-replace; boundary={_BOUNDARY}"},
        )
        await resp.prepare(request)
        try:
            while True:
                jpg = self._latest_jpeg()
                if jpg:
                    chunk = (
                        f"--{_BOUNDARY}\r\n"
                        "Content-Type: image/jpeg\r\n"
                        f"Content-Length: {len(jpg)}\r\n\r\n"
                    ).encode("ascii") + jpg + b"\r\n"
                    await resp.write(chunk)
                await asyncio.sleep(1.0 / self._fps)
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        return resp
