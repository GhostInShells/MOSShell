"""同源桥 — 一个 aiohttp server 同时提供页面 / 模型资产 / WebSocket.

**为什么必须同源**: 页面若来自 file:// 或别的端口, 它连本 node 的 WS 就是跨域, 握手被
浏览器拦下. 把三样东西放在同一个 origin 下, 页面用相对路径取模型, WS 用相对路径连,
零 CORS 配置.

    http://127.0.0.1:<port>/
    ├── /                 页面 (web/index.html)
    ├── /static/…         页面资源 (web/*)
    ├── /model/…          当前形象套件的模型资产 (avatars/<name>/model/*)
    ├── /vendor/…         Cubism Core + Framework bundle (本地自建, 不入库)
    └── /ws               事件流

WS 协议刻意保持简单: **页面只收不发**. 唯一上行是 `ready` 握手 (页面报告它加载完模型、
它认得哪些参数), 驱动不向它回读状态 —— command 即真相.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from aiohttp import WSMsgType, web

from .avatar import Avatar

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8767


class AvatarBridge:
    """把 Avatar 的事件流暴露到一个同源页面."""

    def __init__(
        self,
        avatar: Avatar,
        *,
        web_dir: Path,
        vendor_dir: Path,
        model_dir: Path,
        backdrop_dir: Path | None = None,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        logger: logging.Logger,
    ) -> None:
        self.avatar = avatar
        self.web_dir = Path(web_dir)
        self.vendor_dir = Path(vendor_dir)
        self.model_dir = Path(model_dir)
        self.backdrop_dir = Path(backdrop_dir) if backdrop_dir else None
        self.host = host
        self.port = port
        self.logger = logger
        self._runner: web.AppRunner | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    # ------------------------------------------------------------------ 路由

    def _app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/", self._index)
        app.router.add_get("/ws", self._ws)
        app.router.add_get("/healthz", self._healthz)
        # 页面资源 / 模型资产 / vendor 各自挂静态目录. 三者都要求目录真实存在.
        app.router.add_static("/static/", self.web_dir, show_index=False)
        app.router.add_static("/model/", self.model_dir, show_index=False)
        if self.vendor_dir.is_dir():
            app.router.add_static("/vendor/", self.vendor_dir, show_index=False)
        # 背板是页面自己的图层 (KD6), 与模型包解耦, 走 node 级的 backdrop/ 目录.
        if self.backdrop_dir is not None and self.backdrop_dir.is_dir():
            app.router.add_static("/backdrop/", self.backdrop_dir, show_index=False)
        return app

    async def _index(self, request: web.Request) -> web.StreamResponse:
        return web.FileResponse(self.web_dir / "index.html")

    async def _healthz(self, request: web.Request) -> web.StreamResponse:
        return web.json_response(
            {"avatar": self.avatar.name, "clients": self.avatar.client_count}
        )

    async def _ws(self, request: web.Request) -> web.StreamResponse:
        ws = web.WebSocketResponse(heartbeat=30)
        await ws.prepare(request)
        self.avatar.attach(ws)
        self.logger.info("avatar %s: page connected (%d)", self.avatar.name, self.avatar.client_count)
        # 全量快照: 晚连上的页面也能追上已经发生的命令.
        await ws.send_json(self.avatar.hello_frame())
        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                self._on_client_message(msg.json())
        finally:
            self.avatar.detach(ws)
            self.logger.info("avatar %s: page disconnected", self.avatar.name)
        return ws

    def _on_client_message(self, payload: dict) -> None:
        """页面 → 驱动的唯一通道. 只处理握手; 其余按调试信息记录.

        驱动不回读页面状态, 所以这里**故意**不做任何状态同步 —— 加了就破坏
        "command 即真相", 会让唇形/肢体出现双驱动抖动.
        """
        kind = payload.get("t")
        if kind == "ready":
            self.logger.info(
                "avatar %s: page ready (model=%s, params=%s)",
                self.avatar.name,
                payload.get("model"),
                len(payload.get("params") or ()),
            )
        elif kind == "interact":
            # 人类交互 (点击/拖拽) → 记录进感知面; 点击还触发 Tap 反馈动作.
            self._on_interact(payload)
        else:
            self.logger.debug("avatar %s: ignoring client message %r", self.avatar.name, kind)

    def _on_interact(self, payload: dict) -> None:
        act = payload.get("act")
        if act == "tap":
            self.avatar.on_tap()
        elif act == "drag":
            dx = payload.get("dx", 0.0)
            dy = payload.get("dy", 0.0)
            self.avatar.record_interaction(f"拖拽 {dx:+.1f},{dy:+.1f}")
        elif act == "press":
            self.avatar.record_interaction("按下")

    # ---------------------------------------------------------------- 生命周期

    async def start(self) -> None:
        self.avatar.start()
        runner = web.AppRunner(self._app())
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        self._runner = runner
        self.logger.info("avatar %s: page at %s", self.avatar.name, self.url)

    async def stop(self) -> None:
        await self.avatar.stop()
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    async def serve_forever(self) -> None:
        await asyncio.Event().wait()
