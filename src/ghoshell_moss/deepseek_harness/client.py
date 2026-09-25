"""
DshClient — dsh web 面 (0.1.5-rc.2) 的 unary Remote 动词 facade.

对齐 dsh 0.1.5 从 apiproxy 迁到 Typert Remote 之后的线协议 (弃旧点号 + 平铺 payload):
- URL: ``POST /api/{namespace}/{method}`` (斜杠)
- 信封: ``{type:'client-request', rpcId, method:'{ns}/{method}', payload:{args:{...}}}``
- 响应: ``{type:'server-response', rpcId, result:{ok:true,value} | {ok:false,error}}``

args 键 = Remote 方法的参数名: 多数方法叫 ``request``, ``session/list`` 叫 ``_request``,
无参方法 (``session/modelCatalog``) 为 ``{}``. 每个动词一个命名方法, method 字符串只出现一次.

依赖最小: 只组合 base_url + logger + timeout, 自建 http client. 不持有 launcher,
不感知 ws / subprocess / 进程生命周期.

只挂会话动词 (session.*); workspace / host / agent-preset / settings / credentials / llm /
skill / goal 的管理面动词在 0.1.5 迁移后已无消费者, 不在此暴露 (workspace 列表走
launcher 的 ``workspace/follow`` 流, 见 launcher.py).
"""

from __future__ import annotations

from typing import Any, TypeVar

import httpx
from pydantic import BaseModel

from ghoshell_moss.deepseek_harness.types import sessions
from ghoshell_moss.deepseek_harness.types.rpc import RpcError
from ghoshell_moss.contracts.logger import LoggerItf

__all__ = [
    "DshClient",
    "DshRpcException",
]

_ValueT = TypeVar("_ValueT", bound=BaseModel)


class DshRpcException(Exception):
    """dsh rpc 业务失败分支: 携带 RpcError(code/message/details)."""

    def __init__(self, method: str, error: RpcError) -> None:
        self.method = method
        self.error = error
        super().__init__(f"dsh rpc {method} failed [{error['code']}]: {error['message']}")


class DshClient:
    """强类型 unary Remote 动词 facade: 每个方法 = 一个 dsh Remote 动词, 不背业务逻辑."""

    def __init__(self, base_url: str, logger: LoggerItf, *, timeout: float = 10.0) -> None:
        self._base_url = base_url
        self._logger = logger
        self._http_client = httpx.AsyncClient(timeout=timeout)
        self._rpc_counter = 0

    async def call(
        self,
        method: str,
        params: BaseModel | None,
        value_cls: type[_ValueT],
        *,
        args_key: str | None = "request",
    ) -> _ValueT:
        """引擎: params 序列化 → args 包络 → 信封 POST → ok 则 value_cls.model_validate, 否则 raise.

        ``method`` 是 Remote 端点 (namespace/method, 斜杠). ``args_key`` 是参数名;
        None 表示无参动词 (args={}).
        """
        payload = params.model_dump(exclude_none=True, by_alias=True) if params is not None else {}
        args: dict[str, Any] = {} if args_key is None else {args_key: payload}
        self._rpc_counter += 1
        envelope = {
            "type": "client-request",
            "rpcId": f"rpc-{self._rpc_counter}",
            "method": method,
            "payload": {"args": args},
        }
        self._logger.debug("dsh rpc %s (rpcId=%s)", method, envelope["rpcId"])
        resp = await self._http_client.post(
            f"{self._base_url}/api/{method}",
            json=envelope,
        )
        resp.raise_for_status()
        result = resp.json()["result"]
        if not result["ok"]:
            error = result.get("error")
            if error is None:
                error = {"code": "internal", "message": "no error", "details": {}}
            raise DshRpcException(method, error)
        return value_cls.model_validate(result.get("value"))

    async def rpc(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        """裸 RPC: POST client-request 信封, 返回 result dict (不校验 value).

        供无 value 的动词用 (如 ``$events/result``). ``payload`` 原样作信封 payload.
        """
        self._rpc_counter += 1
        envelope = {
            "type": "client-request",
            "rpcId": f"rpc-{self._rpc_counter}",
            "method": method,
            "payload": payload,
        }
        self._logger.debug("dsh rpc %s (rpcId=%s)", method, envelope["rpcId"])
        resp = await self._http_client.post(
            f"{self._base_url}/api/{method}",
            json=envelope,
        )
        resp.raise_for_status()
        return resp.json()["result"]

    async def plugin_call(self, path: str, payload: dict | None = None) -> dict:
        """plugin webServer 面 (非 Remote): 裸 POST ``{base}{path}``, 返回响应 JSON dict.

        与 call() 分工: call() 走 Remote 的 client-request 信封 (POST /api/{ns}/{method});
        plugin 注册的 HTTP 路由走这里 — raw JSON body, 无信封, 响应为 plugin 自定义 JSON.
        """
        resp = await self._http_client.post(f"{self._base_url}{path}", json=payload or {})
        resp.raise_for_status()
        return resp.json()

    def set_cookies(self, cookies: dict[str, str]) -> None:
        """注入 dsh web 鉴权 cookie 到后续 /api 调用 (requestRejection 要求)."""
        self._http_client.cookies.update(cookies)

    async def close(self) -> None:
        await self._http_client.aclose()

    # ---- session 动词 ---- #

    async def session_create(self, params: sessions.SessionCreateParams) -> sessions.SessionCreateValue:
        return await self.call("session/create", params, sessions.SessionCreateValue)
