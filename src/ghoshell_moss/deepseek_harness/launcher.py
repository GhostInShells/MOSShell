"""
dsh 连接的协议层与进程层, 分两个类:

DshConnection — 基类, 连接层. 持有 dsh web 表面的传输与协议原语:
WS 下行 (`/api/remote.mux` 重连循环 + `$events` 逻辑流分派) + HTTP 上行 (call /
协议 facade DshClient) + 帧处理器注册 (on_remote_emit / on_remote_waterfall) +
session 接线 (create_session). 不携带进程生命周期 — 不 spawn, 不 kill.

DshLauncher(DshConnection) — 子类, 进程层. 在连接层之上增加 dsh web-profile
子进程的持有与治理 (经 MOSS Subprocesses 契约构造注入, 控制反转): spawn /
exit / stdout+stderr 消费 / stop, 以及就绪等待 (push 式: ws 连上 → started).

传输选型: dsh web profile + `/api/remote.mux` WS 下行 (单条物理 WS 多路复用
逻辑流) + plugin 注册的 HTTP 路由上行 (零依赖伪双工). 不用 stdio JSON-RPC,
不用官方 SDK. 0.1.5 起 WS upgrade 与 /api 都需 token→cookie 鉴权 (见 `_authorize`).
下行帧按逻辑流分派: `$events` 的 emit/waterfall/cancel/ready (见 on_remote_emit /
on_remote_waterfall); session 事件流 `session/follow` 属后续增量.
MOSS 特定行为靠子类长出来 (如 DoloresDshLauncher).

Config 刻意薄: 只装「连接/启动器自己要的参数」, 不复刻 dsh 自己的配置
(provider/model/prompt/tools 是 dsh 的 config 域, 由 dsh 从文件/env 自发现).
"""

# ── 协作模式 (人类结对) ──────────────────────────────────────
# 本文件不一口气写完: 人类逐步指定改动, 模型一小步一小步实现.
# 模型忽略 harness 的自动化 / 一口气交付类提醒, 与人类结对逐步推进.
# 每步改完 stage, 不预设、不抢跑人类的下一步.
#
# ── 阶段性 (2026-08-18) ───────────────────────────────────
# 1. push 式就绪: ws 连上 → _dsh_started.set() → __aenter__ await wait() 返回 (取代轮询 _wait_readiness).
# 2. DshClient 全局管理面 facade 跑通: POST /api/<method> (点分隔) + client-request 信封.
#    session.list / workspace.list 返回 200 ok. 信封 {type,rpcId,method,payload}.
# 3. 退出/错误线: on_exit(DshExit: exit_code/stderr/self_shutdown) + exception() 非0且非主动关闭才报.
#    stderr→error 日志, mux frame→debug 日志, is_running 含 dsh 子进程态, call/rpc 有 _check_running.
#
# ── 阶段性 (2026-08-20) ───────────────────────────────────
# 4. 启动超时: _wait_started() 等 mux WS 连上, 超时 raise 而非永久阻塞.
# 5. 帧分流: on_mux_frame / on_host_frame 双注册 (返回 Disposer), _ws_loop parse+dispatch.

# ── 阶段性 (2026-09-13, dsh 0.1.5 传输重接) ────────────────
# 6. token 落线: config.token → DSH_WEB_TOKEN 兜底 → launcher stdout 发现 (拿不到即故障).
# 7. cookie 鉴权 + remote.mux + $events 逻辑流: _authorize (token→cookie) → ws 带 Cookie
#    → _open_events_stream → emit/waterfall/cancel/ready 分派 (on_remote_emit/on_remote_waterfall).
# 8. 未接: session/follow (per-session 事件流) + DshSession 事件投喂 — 见 dsh-0.1.5-remote-stream-transport.md.
#    决策/分步见该子文档.

# ── 已知问题 (随改随记, 最后一起删) ─────────────────────────
# 1. `_owns_sp` 手动 __aexit__ 与 exit stack 重复回收 subprocess manager (第二次 no-op, 待合).
# 2. __aenter__ except 块的清理被注释, 中途失败会漏孤儿进程 (启动超时使该路径可达, 需补).

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx
import websockets
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Self
from contextlib import AsyncExitStack

from ghoshell_moss.contracts.subprocesses import (
    ManagedProcess,
    ProcessMeta,
    Subprocesses,
)
from ghoshell_moss.core.subprocesses import SubprocessesImpl
from ghoshell_moss.core.helpers.asyncio_utils import ThreadSafeEvent
from ghoshell_moss.contracts.logger import LoggerItf, get_moss_logger
from .types.nouns import WorkspaceView
from .client import DshClient
from .session import DshSession

__all__ = [
    "DshConnectionConfig",
    "DshConnection",
    "DshLauncherConfig",
    "DshLauncher",
    "DshExit",
]

# dsh web 鉴权 token 的环境变量兜底来源: 让无子进程的 DshConnection 也能独立起
# (launcher 拥有进程时从 stdout 发现, 见 DshLauncher._maybe_capture_token).
DSH_WEB_TOKEN_ENV = "DSH_WEB_TOKEN"

# $events 下行帧处理器: emit 单向通知 (event_name, args 位置参数).
RemoteEmitHandler = Callable[[str, list[Any]], Awaitable[None] | None]
# waterfall 处理器: 收 (event_name, request), 返回 outcome dict
#   {"kind":"next"} | {"kind":"result","value":...} | {"kind":"rejected","error":{...}};
#   返回 None 等价 {"kind":"next"}.
RemoteWaterfallHandler = Callable[
    [str, dict[str, Any]],
    Awaitable[dict[str, Any] | None] | dict[str, Any] | None,
]
# 解绑函数: on_remote_emit / on_remote_waterfall 返回, 调用即注销对应 handler.
Disposer = Callable[[], None]


@dataclass(frozen=True, slots=True)
class DshExit:
    """dsh 子进程退出信息 (冻结, 便于按需扩展)."""

    exit_code: int | None
    stderr: str
    self_shutdown: bool = False


class DshConnectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str = Field(default='127.0.0.1')
    port: int = Field(default=3083, description="web 端口; base_url/mux_url 由此派生.")
    connect_timeout: float = Field(default=10.0, description="连 WS / 单个 HTTP 请求的超时 (秒).")
    token: str | None = Field(
        default=None,
        description="dsh web 鉴权 token; None → 从 DSH_WEB_TOKEN 环境变量读一次.",
    )

    @property
    def mux_url(self) -> str:
        return f"ws://{self.host}:{self.port}/api/remote.mux"

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


class DshLauncherConfig(DshConnectionConfig):
    """拉起 dsh web profile 的参数面.

    Deliberately thin: 只装这个启动器 spawn 进程、连上 web 表面需要的参数.
    dsh 自己的配置 (provider/model/prompt/tools) 在 dsh 的文件/env 里由 dsh
    自发现, 这里不复刻. 可扩展靠"嵌套子配置 + 子类化", forbid 让拼写错当场失败.
    """

    binary: str = Field(default="dsh", description="dsh 可执行; 默认从 PATH 找.")
    home: Path | None = Field(default=None, description="DSH_HOME; None → 在 cwd 启动, 让 dsh 自发现 profile/config.")
    profile: str = Field(default="web", description="进程层 profile 选择, 不是 dsh 配置.")
    args: list[str] = Field(default_factory=list, description="启动器 flag 之后的 verbatim 参数.")
    readiness_path: str = Field(default="/plugin-api/ping", description="就绪探针: 轮询到它返回即视为 dsh+plugin 起来.")
    readiness_timeout: float = Field(default=30.0, description="等待就绪的时限 (秒).")
    shutdown_timeout: float = Field(default=5.0, description="拆除进程的时限 (秒).")


class DshConnection:
    """连接层: 持 dsh web 表面的传输与协议原语, 不带进程生命周期.

    职责 (协议原语形状, 不背业务逻辑):
    - outbound call: `call()` POST JSON 到 `{base}{path}` (MOSS→dsh).
    - 鉴权: `_authorize()` token→cookie, 注入 http client 与 DshClient; WS upgrade 带 Cookie.
    - inbound notify: `on_remote_emit` / `on_remote_waterfall` 双注册 (返回 Disposer),
      `_ws_loop` 下行重连 + `_dispatch_raw_frame` 按 `$events` 帧 type 分派.
    - session 接线: `create_session()` 把 DshSession 的 accept_host_event 挂到 emit 流,
      退出时 on_exit 解绑. (session/follow 事件流属后续增量.)
    - host 级 workspace: `workspaces()` / `workspace_for_path()` 走 workspace.list RPC
      (0.1.5 不再推 workspace 变更帧).

    生命周期只覆盖连接自身 (WS 循环 + HTTP client); 子进程的 spawn/治理/拆除
    属 DshLauncher. 连接层的 `_wait_started` / `_on_start_failed` 是空实现,
    由子类覆盖 — 基类单独可用时不依赖子进程, 也就没有"就绪等待/失败清理"。
    """

    def __init__(
            self,
            config: DshConnectionConfig,
            logger: LoggerItf | None = None,
    ) -> None:
        self._config = config
        self._token: str | None = config.token or os.environ.get(DSH_WEB_TOKEN_ENV) or None
        # prepare http client
        self._http_client = httpx.AsyncClient(timeout=self.config.connect_timeout)
        self._emit_handlers: list[RemoteEmitHandler] = []
        self._waterfall_handlers: list[RemoteWaterfallHandler] = []
        self._workspaces: dict[str, WorkspaceView] = {}
        # $events 逻辑流状态: clientId (waterfall 回话凭据) + 鉴权 cookie.
        self._remote_client_id: str | None = None
        self._cookies: dict[str, str] = {}
        self._cookie_header: str | None = None
        self._events_stream_id = "moss-events"
        self._logger: LoggerItf = logger or get_moss_logger()
        self.client = DshClient(self.config.base_url, self._logger, timeout=self.config.connect_timeout)
        self._aexit_stack = AsyncExitStack()
        self._dsh_started = ThreadSafeEvent()
        # 标记 dsh 是否已经运行.

        self._started = False
        self._stopped = False
        self._log_prefix: str = f"[DSHConnection] "

    @property
    def config(self) -> DshConnectionConfig:
        return self._config

    def token(self) -> str | None:
        """当前 dsh web 鉴权 token (config → DSH_WEB_TOKEN 兜底).

        子类 (DshLauncher) 可覆盖为运行时从 dsh stdout 发现的值。
        """
        return self._token

    # ---- 运行状态 ---- #

    def is_running(self) -> bool:
        return self._started and not self._stopped

    def _check_running(self) -> None:
        if not self.is_running():
            raise RuntimeError("DshLauncher not running (dsh subprocess not alive)")

    # ---- 生命周期 ---- #

    @contextlib.asynccontextmanager
    async def _ws_loop_ctx(self):
        task = asyncio.create_task(self._ws_loop())
        try:
            yield
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def _ws_loop(self) -> None:
        """mux WS 下行重连循环: 连上后 parse+dispatch 帧, 断开则重连."""
        while self.is_running():
            try:
                await self._authorize()
                headers = {"Cookie": self._cookie_header} if self._cookie_header else None
                async with websockets.connect(self.config.mux_url, additional_headers=headers) as ws:
                    self._dsh_started.set()
                    self._logger.info("%smux connected", self._log_prefix)
                    await self._open_events_stream(ws)
                    async for raw in ws:
                        await self._dispatch_raw_frame(raw)
            except asyncio.CancelledError:
                raise
            except ConnectionRefusedError as exc:
                self._logger.warning("mux TCP refused (dsh not up): %s", exc)
            except websockets.exceptions.InvalidHandshake as exc:
                self._logger.warning("mux handshake failed (mux not ready): %s", exc)
            except websockets.exceptions.ConnectionClosed as exc:
                self._logger.warning("mux connection closed: %s", exc)
            await asyncio.sleep(1.0)

    async def _authorize(self) -> None:
        """token → cookie 交换, 注入 http client 与 DshClient (后续 /api 调用需鉴权).

        幂等: 已持有 cookie 即返回. token 尚未就绪 (launcher 正在等 stdout) 时跳过,
        由 WS 重连循环下一轮重试。
        """
        if self._cookies:
            return
        token = self.token()
        if token is None:
            return
        try:
            resp = await self._http_client.get(
                f"{self.config.base_url}/?token={token}", follow_redirects=False,
            )
        except Exception as exc:
            self._logger.warning("%sweb auth exchange failed: %s", self._log_prefix, exc)
            return
        self._cookies = dict(self._http_client.cookies.items())
        if not self._cookies:
            self._logger.warning(
                "%sweb auth exchange produced no cookie (status %s)", self._log_prefix, resp.status_code
            )
            return
        self._cookie_header = "; ".join(f"{k}={v}" for k, v in self._cookies.items())
        self.client.set_cookies(self._cookies)
        self._logger.info("%sweb auth cookie acquired", self._log_prefix)

    async def _open_events_stream(self, ws: Any) -> None:
        """在已连上的 mux 上开 `$events` 逻辑流 (应用级转发事件)."""
        await ws.send(json.dumps({
            "type": "open",
            "streamId": self._events_stream_id,
            "endpoint": "$events",
            "payload": {"args": {}},
        }))

    async def _dispatch_raw_frame(self, raw: str) -> None:
        """解析 mux 下行帧 (`item`/`error`/`end`), 分派 `item.value` 到 $events handler.

        单帧解析失败只记日志、不断流 — 任何畸形帧都不该静默杀死整条 mux 链路。
        """
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return
        if not isinstance(msg, dict):
            return
        t = msg.get("type")
        if t == "item":
            await self._dispatch_remote_event(msg.get("value"))
        elif t == "error":
            self._logger.warning("%smux stream error: %s", self._log_prefix, msg.get("error"))
        elif t == "end":
            self._logger.debug("%smux stream ended: %s", self._log_prefix, msg.get("streamId"))
        # 其余帧类型静默忽略.

    async def _dispatch_remote_event(self, value: Any) -> None:
        """按 $events 下行帧 type 分派: ready(存 clientId) / emit / waterfall(回话) / cancel."""
        if not isinstance(value, dict):
            return
        t = value.get("type")
        if t == "ready":
            self._remote_client_id = value.get("clientId")
            self._logger.info("%s$events ready (clientId bound)", self._log_prefix)
            return
        if t == "emit":
            event = value.get("event", "")
            args = list(value.get("args") or [])
            for handler in list(self._emit_handlers):
                try:
                    result = handler(event, args)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception:
                    self._logger.exception("remote emit handler failed: %s", event)
            return
        if t == "waterfall":
            await self._dispatch_waterfall(value)
            return
        if t == "cancel":
            # 取消 pending waterfall — 当前不跟踪 pending 号, 仅日志.
            self._logger.debug("%s$events cancel: %s", self._log_prefix, value.get("eventId"))
            return

    async def _dispatch_waterfall(self, value: dict[str, Any]) -> None:
        """把一个 waterfall 交给注册 handler, 把 outcome 经 $events/result 回话."""
        event = value.get("event", "")
        event_id = value.get("eventId", "")
        request = dict(value.get("request") or {})
        outcome: dict[str, Any] = {"kind": "next"}
        for handler in list(self._waterfall_handlers):
            try:
                result = handler(event, request)
                if asyncio.iscoroutine(result):
                    result = await result
                if result is not None:
                    outcome = result
            except Exception:
                self._logger.exception("remote waterfall handler failed: %s", event)
        await self._send_event_result(event_id, outcome)

    async def _send_event_result(self, event_id: str, outcome: dict[str, Any]) -> None:
        """经 `$events/result` RPC 回一个 waterfall 结果 (client-request 信封)."""
        client_id = self._remote_client_id
        if client_id is None:
            self._logger.warning("%scannot answer waterfall %s: no clientId yet", self._log_prefix, event_id)
            return
        payload = {"args": {"clientId": client_id, "eventId": event_id, "outcome": outcome}}
        try:
            await self.client.rpc("$events/result", payload)
        except Exception:
            self._logger.exception("$events/result failed: %s", event_id)

    async def _enter_async_context(self, stack: AsyncExitStack) -> None:
        await stack.enter_async_context(self._ws_loop_ctx())

    async def _wait_started(self) -> None:
        pass

    async def _on_start_failed(self) -> None:
        pass

    async def __aenter__(self) -> Self:
        if self._started:
            return self
        try:
            self._started = True
            # 启动 aexit stack.
            await self._aexit_stack.__aenter__()
            await self._enter_async_context(self._aexit_stack)
            await self._wait_started()
        except BaseException:
            # 启动失败: 手动关掉已 spawn 的 subprocess (句柄在 _wait_started 之前已拿到).
            await self._on_start_failed()
            raise
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._stopped:
            return
        self._stopped = True
        # 退出所有的栈.
        await self._aexit_stack.__aexit__(exc_type, exc_val, exc_tb)

        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None
        await self.client.close()

    # ---- 协议原语 ---- #

    async def call(
            self,
            path: str,
            payload: dict[str, Any] | None = None,
            *,
            timeout: float | None = None,
    ) -> dict[str, Any]:
        """outbound 请求 (MOSS→dsh): POST JSON 到 `{base}{path}`, 返回解析后的响应 dict."""
        self._check_running()
        if self._http_client is None:
            raise RuntimeError("DshLauncher not started")
        resp = await self._http_client.post(
            f"{self.config.base_url}{path}",
            json=payload or {},
            timeout=timeout if timeout is not None else self.config.connect_timeout,
        )
        if resp.is_error:
            try:
                detail = resp.json().get("error", resp.text)
            except Exception:
                detail = resp.text
            raise RuntimeError(f"dsh RPC {path} failed ({resp.status_code}): {detail}")
        return resp.json()

    def on_remote_emit(self, handler: RemoteEmitHandler) -> Disposer:
        """注册 $events emit 下行处理器 (单向应用事件), 返回解绑函数."""
        self._emit_handlers.append(handler)

        def _remove() -> None:
            self._emit_handlers.remove(handler)

        return _remove

    def on_remote_waterfall(self, handler: RemoteWaterfallHandler) -> Disposer:
        """注册 $events waterfall 下行处理器 (需回话事件), 返回解绑函数."""
        self._waterfall_handlers.append(handler)

        def _remove() -> None:
            self._waterfall_handlers.remove(handler)

        return _remove

    async def workspaces(self, *, force: bool = False) -> list[WorkspaceView]:
        """host 级 workspace 列表 — 0.1.5 不再推 workspace 变更帧, 走 workspace.list RPC 拉取 (带缓存)."""
        if not force and self._workspaces:
            return list(self._workspaces.values())
        value = await self.client.workspace_list()
        self._workspaces = {w.workspaceId: w for w in value.items}
        return list(self._workspaces.values())

    async def workspace_for_path(self, path: str, *, force: bool = False) -> WorkspaceView | None:
        """按 canonical path 解析 workspace (与 session.cwd 匹配)."""
        for workspace in await self.workspaces(force=force):
            if workspace.path == path:
                return workspace
        return None

    def create_session(self, session_id: str, logger: LoggerItf | None = None) -> DshSession:
        """创建并接线一个 session facade: 注册 accept_host_event 到 $events emit 流, 退出时解绑.

        $events 流收 host 级运行态 (api-session/status·added·…). session 内部按 sessionId
        过滤, 只消费自己的帧. 不持久持有 session — 只经 handler 列表关联, session 关闭时
        on_exit 解绑断链. (session 事件流 session/follow 属后续增量.)
        """
        session = DshSession(session_id=session_id, client=self.client, logger=logger)
        session.on_exit(self.on_remote_emit(session.accept_host_event))
        return session


class DshLauncher(DshConnection):
    """进程层: 在连接层之上持有并治理一段 dsh web-profile 子进程.

    回答一个问题: "怎么把 dsh 进程拉起来, 并连上它的 web 表面".
    连接层没有的子进程生命周期都在此: spawn / exit / stdout+stderr 消费 /
    stop, 以及 push 式就绪等待 (ws 连上 → _dsh_started → __aenter__ 返回).

    subprocesses 走构造注入 (控制反转). 传入 ghost/owner 的 Subprocesses,
    dsh 骑 owner 的治理链; 不传则自建一个 SubprocessesImpl, 自包含可用.

    生命周期串接: 本类压栈子进程层 (subprocess_manager → dsh_process →
    consume), 再 `super()` 压入连接层的 WS 循环 — 退出时 LIFO 先拆 WS,
    再拆 consume, 再停子进程。
    """

    def __init__(
            self,
            config: DshLauncherConfig,
            subprocesses: Subprocesses | None = None,
            logger: LoggerItf | None = None,
    ) -> None:
        super().__init__(config=config, logger=logger)
        self._external_sp = subprocesses is not None
        self._subprocess_manager: Subprocesses = subprocesses or SubprocessesImpl()
        self._owns_sp = not self._external_sp
        self._dsh_process: ManagedProcess | None = None
        # 标记 dsh 子进程运行态: spawn 后 True, on_exit 回调翻 False.
        self._dsh_subprocess_is_running = False
        self._consume_dsh_process_out_task: asyncio.Task | None = None
        self._consume_dsh_process_err_task: asyncio.Task | None = None
        self._on_exit_callbacks: list[Callable[[DshExit], None]] = []
        self._exit: DshExit | None = None
        self._self_shutdown = False
        self._stderr_lines: list[str] = []
        self._discovered_token: str | None = None
        self._token_ready = ThreadSafeEvent()
        self._log_prefix: str = f"[DSHLauncher] "

    @property
    def config(self) -> DshLauncherConfig:
        return self._config

    def token(self) -> str | None:
        """运行时从 dsh stdout 发现的 token 优先, 否则回落到 config/env 兜底."""
        return self._discovered_token or super().token()

    def is_running(self) -> bool:
        return super().is_running() and self._dsh_subprocess_is_running

    async def _wait_started(self) -> None:
        """等待 dsh web token + mux WS 连上 (push 式就绪), 超时则失败而非永久阻塞."""
        await self._wait_token()
        try:
            await self._dsh_started.wait_for(self.config.readiness_timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"dsh 未在 {self.config.readiness_timeout}s 内就绪 (mux WS 未连接)"
            ) from None

    async def _wait_token(self) -> None:
        """launcher 拥有进程, 鉴权 token 必须可得 — 拿不到即故障.

        若 config.token 或 DSH_WEB_TOKEN 已提供则立即返回, 不依赖 stdout 发现。
        """
        if self.token() is not None:
            return
        try:
            await self._token_ready.wait_for(self.config.readiness_timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"dsh 未在 {self.config.readiness_timeout}s 内输出 web token (无法鉴权)"
            ) from None

    async def _on_start_failed(self) -> None:
        await self._stop_proc()

    async def _enter_async_context(self, stack: AsyncExitStack) -> None:
        if not self._subprocess_manager.is_running():
            await stack.enter_async_context(self._subprocess_manager)
        # spawn dsh
        await self._aexit_stack.enter_async_context(self._dsh_process_ctx())
        # 压栈子进程 rpc 协议消费逻辑.
        await self._aexit_stack.enter_async_context(self._consume_dsh_process_ctx())
        # 压栈 mux WS 下行重连循环 — 由基类负责, 在子进程层之后压入, 退出时先于进程拆除.
        await super()._enter_async_context(stack)

    @contextlib.asynccontextmanager
    async def _consume_dsh_process_ctx(self):
        try:
            # 创建子任务, 消费 dsh 的 stdout (json rpc 协议) + stderr (错误日志).
            self._consume_dsh_process_out_task = asyncio.create_task(self._consume_dsh_process_stdout())
            self._consume_dsh_process_err_task = asyncio.create_task(self._consume_dsh_process_stderr())
            yield
        finally:
            # 关闭消费循环.
            self._consume_dsh_process_out_task.cancel()
            self._consume_dsh_process_err_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._consume_dsh_process_out_task
                await self._consume_dsh_process_err_task
            await self._stop_proc()

    @contextlib.asynccontextmanager
    async def _dsh_process_ctx(self):
        self._dsh_process = await self._spawn_dsh()
        try:
            yield
        finally:
            await self._stop_proc()

    def on_exit(self, callback: Callable[[DshExit], None]) -> None:
        """注册子进程退出回调, 退出时按注册顺序层层调用."""
        self._on_exit_callbacks.append(callback)

    def exception(self) -> Exception | None:
        """子进程非 0 退出后返回结合 stderr 的异常; 运行中 / 正常退出 / 主动关闭返回 None."""
        if self._dsh_subprocess_is_running:
            return None
        exit_info = self._exit
        if exit_info is None or exit_info.self_shutdown or exit_info.exit_code in (0, None):
            return None
        return RuntimeError(f"dsh exited with code {exit_info.exit_code}: {exit_info.stderr}")

    # ---- 内部 ---- #

    async def _spawn_dsh(self) -> ManagedProcess:
        """spawn dsh subprocess"""
        args = [
            self.config.binary,
            "--profile", self.config.profile,
            "--port", str(self.config.port),
            # ego 起的 dsh 不自动开浏览器 — desktop mode 下 ego 把界面开进自己的 screen, 不借系统默认浏览器.
            "--no-open",
            *self.config.args,
        ]
        extra_env: dict[str, str] = {}
        if self.config.home is not None:
            extra_env["DSH_HOME"] = str(self.config.home)
        managed_process = await self._subprocess_manager.execute(
            *args,
            name="dsh",
            description=f"dsh {self.config.profile} profile on :{self.config.port}",
            cwd=self.config.home,
            extra_env=extra_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            on_exit=self._on_dsh_exit,
        )
        self._dsh_subprocess_is_running = True
        return managed_process

    def _on_dsh_exit(self, meta: ProcessMeta) -> None:
        self._dsh_subprocess_is_running = False
        exit_info = DshExit(
            exit_code=meta.exit_code,
            stderr="\n".join(self._stderr_lines[-20:]),
            self_shutdown=self._self_shutdown,
        )
        self._exit = exit_info
        for callback in self._on_exit_callbacks:
            callback(exit_info)

    async def _consume_dsh_process_stdout(self) -> None:
        proc = self._dsh_process
        if proc is None or proc.process.stdout is None:
            return
        stream = proc.process.stdout
        try:
            while self._dsh_subprocess_is_running:
                line = await stream.readline()
                if not line:
                    break
                text = line.decode(errors="replace").rstrip()
                self._maybe_capture_token(text)
                self._logger.debug("%sstdout: %s", self._log_prefix, self._redact_token(text))
        finally:
            self._logger.debug("%sstdout consume closed", self._log_prefix)

    # dsh web 打印的 token 形如 `dsh web: http://…/?token=<base64url>`.
    _TOKEN_RE = re.compile(r"token=([A-Za-z0-9_-]+)")

    def _maybe_capture_token(self, text: str) -> None:
        if self._discovered_token is not None:
            return
        m = self._TOKEN_RE.search(text)
        if m is None:
            return
        self._discovered_token = m.group(1)
        self._token_ready.set()
        self._logger.info("%sdsh web token discovered (value not logged)", self._log_prefix)

    def _redact_token(self, text: str) -> str:
        return self._TOKEN_RE.sub("token=***", text)

    async def _consume_dsh_process_stderr(self) -> None:
        proc = self._dsh_process
        if proc is None or proc.process.stderr is None:
            return
        stream = proc.process.stderr
        while self._dsh_subprocess_is_running:
            line = await stream.readline()
            if not line:
                break
            text = line.decode(errors="replace").rstrip()
            self._stderr_lines.append(text)
            if len(self._stderr_lines) > 400:
                del self._stderr_lines[:200]
            self._logger.error("%sstderr: %s", self._log_prefix, text)

    async def _stop_proc(self) -> None:
        proc = self._dsh_process
        if proc is None:
            return
        self._self_shutdown = True
        await proc.stop(timeout=self.config.shutdown_timeout)
        self._dsh_process = None

    def _dump_stderr(self) -> None:
        """启动失败时, 把 dsh 的 stderr tail 打出来帮助诊断."""
        proc = self._dsh_process
        if proc is None or proc.output is None:
            return
        try:
            tail = proc.output.stderr()
        except Exception:
            return
        if tail:
            self._logger.error("--- dsh stderr tail ---\n%s", tail[-2000:])
