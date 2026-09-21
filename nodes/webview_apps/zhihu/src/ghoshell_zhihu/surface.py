"""人类侧的 web 面：授权输入 + action 卡流 + 裁决。

一个端口服务页面和 WebSocket。下行带 snapshot（连上即给）+ action 帧（channel 物化后
推来）+ auth_state；上行带人类的三个动作：``auth``（粘 Access Secret）、``approve`` /
``reject``（单条裁决）、``auto``（整个 type 放行/收回）。

secret 只经 ``cli.auth_set`` 走 stdin 进系统密钥链，surface 不存、不回显、不进广播。
裁决只调 ``store.settle`` 唤醒 channel 的 waiter —— CLI 执行仍归 channel，这里不碰。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from websockets.asyncio.server import ServerConnection, serve

from .cli import ZhihuCli
from .store import ZhihuStore

__all__ = ["ZhihuSurface"]


def _action_view(rec) -> dict[str, Any]:
    return {
        "id": rec.id,
        "type": rec.type,
        "args": rec.args,
        "transform": rec.transform,
        "render": rec.render,
        "state": rec.state,
        "identity": rec.identity,
        "error": rec.error,
        "result": rec.result,
        "at": rec.at,
    }


class ZhihuSurface:
    def __init__(
        self,
        store: ZhihuStore,
        cli: ZhihuCli,
        *,
        host: str,
        port: int,
        html_path: Path,
    ) -> None:
        self._store = store
        self._cli = cli
        self._host = host
        self._html_path = html_path
        self._clients: set[ServerConnection] = set()
        self._server: Any = None
        self.port = port

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self.port}"

    # -- downlink ---------------------------------------------------------

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
            "auth_configured": self._store.auth_configured,
            "identity": self._store.identity,
            "disabled": self._store.disabled,
            "auto": sorted(self._store._auto),
            "actions": [_action_view(r) for r in self._store.actions()],
        }

    # -- uplink -----------------------------------------------------------

    async def _auth(self, secret: str) -> None:
        if not secret.strip():
            await self.broadcast({"type": "error", "text": "empty secret"})
            return
        result = await self._cli.auth_set(secret.strip())
        if result.get("ok") is False:
            await self.broadcast({
                "type": "auth_state",
                "configured": False,
                "error": result.get("error", {}).get("message", "auth failed"),
            })
            return
        status = await self._cli.run("auth", "status")
        masked = status.get("masked", "") if status.get("ok") else ""
        self._store.auth_configured = True
        self._store.identity = masked or "authorized"
        await self.broadcast({
            "type": "auth_state",
            "configured": True,
            "identity": self._store.identity,
        })

    async def _approve(self, action_id: int) -> None:
        if self._store.settle(action_id, "approve"):
            await self._broadcast_action(action_id)

    async def _reject(self, action_id: int) -> None:
        if self._store.settle(action_id, "reject"):
            await self._broadcast_action(action_id)

    async def _set_disabled(self, on: bool) -> None:
        self._store.set_disabled(on)
        await self.broadcast({"type": "mode", "disabled": on})

    async def _set_auto(self, type_: str, on: bool) -> None:
        self._store.set_auto(type_, on)
        # "整个 type 放行" 也结清已经挂在等待里的同 type action。
        if on:
            for rec in list(self._store.awaiting()):
                if rec.type == type_:
                    self._store.settle(rec.id, "approve")
        await self._broadcast_auto()
        for rec in self._store.actions():
            if rec.type == type_ and rec.state in ("approved", "rejected"):
                await self._broadcast_action(rec.id)

    async def _broadcast_action(self, action_id: int) -> None:
        rec = self._store.get(action_id)
        if rec is not None:
            await self.broadcast({"type": "action", "action": _action_view(rec)})

    async def _broadcast_auto(self) -> None:
        await self.broadcast({"type": "auto", "auto": sorted(self._store._auto)})

    # -- ws server --------------------------------------------------------

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
                if kind == "auth":
                    await self._auth(str(frame.get("secret", "")))
                elif kind == "disable":
                    await self._set_disabled(bool(frame.get("on")))
                elif kind == "approve":
                    await self._approve(int(frame.get("id", 0)))
                elif kind == "reject":
                    await self._reject(int(frame.get("id", 0)))
                elif kind == "auto":
                    await self._set_auto(
                        str(frame.get("name", "")), bool(frame.get("on"))
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
