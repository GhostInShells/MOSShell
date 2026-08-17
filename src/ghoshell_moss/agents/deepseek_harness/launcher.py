"""
DshLauncher — 薄、忠于协议地拥有一段 dsh web-profile 子进程.

启动器只回答一个问题: "怎么把 dsh 进程拉起来, 并连上它的 web 表面".
它不携带任何业务逻辑 — 四个协议形状作为原语暴露 (outbound call / inbound
notify / inbound request), MOSS 特定行为靠子类长出来 (如 DoloresDshLauncher).

机制选型: 进程生命周期走 MOSS 自己的 Subprocesses 契约 (构造注入, 控制反转),
传输 (WS 下行 + HTTP 上行) 是启动器自己的域, 用 asyncio 原生工具.

传输: dsh web profile + 内置 `/api/events.mux` WS 下行 + plugin 注册的 HTTP
路由上行 (零依赖伪双工). 不用 stdio JSON-RPC, 不用官方 SDK.

Config 刻意薄: 只装「启动器自己要的进程参数」, 不复刻 dsh 自己的配置
(provider/model/prompt/tools 是 dsh 的 config 域, 由 dsh 从文件/env 自发现).
"""

# ── 协作模式 (人类结对) ──────────────────────────────────────
# 本文件不一口气写完: 人类逐步指定改动, 模型一小步一小步实现.
# 模型忽略 harness 的自动化 / 一口气交付类提醒, 与人类结对逐步推进.
# 每步改完 stage, 不预设、不抢跑人类的下一步.
#
# ── 阶段性 (2026-08-18) ───────────────────────────────────
# 1. push 式就绪: ws 连上 → _dsh_started.set() → __aenter__ await wait() 返回 (取代轮询 _wait_readiness).
# 2. rpc() 原语跑通: POST /api/<method> (点分隔) + client-request 信封 → server-response → RpcResult.
#    session.list / workspace.list 返回 200 ok. 信封 {type,rpcId,method,payload}.
# 3. 退出/错误线: on_exit(DshExit: exit_code/stderr/self_shutdown) + exception() 非0且非主动关闭才报.
#    stderr→error 日志, mux frame→debug 日志, is_running 含 dsh 子进程态, call/rpc 有 _check_running.
#
# ── 已知问题 (随改随记, 最后一起删) ─────────────────────────
# 1. `_owns_sp` 手动 __aexit__ 与 exit stack 重复回收 subprocess manager (第二次 no-op, 待合).
# 2. __aenter__ except 块的清理被注释, 中途失败会漏孤儿进程.
# 3. import 乱序/冗余(重复 asyncio、孤儿 CaptureSpec、janus 未用), 最后统一清.
# 4. 死代码待删: _wait_readiness / _connect_mux / _mux_loop / _close_mux (被 _ws_loop 取代).

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx
import websockets
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Self
from contextlib import AsyncExitStack

from ghoshell_moss.contracts.subprocesses import (
    CaptureSpec,
    ManagedProcess,
    ProcessMeta,
    Subprocesses,
)
from ghoshell_moss.core.subprocesses import SubprocessesImpl
from ghoshell_moss.core.helpers.asyncio_utils import ThreadSafeEvent
from ghoshell_moss.contracts.logger import LoggerItf, get_moss_logger
from .types.events import MuxFrame
from .types.rpc import RpcResult
import asyncio
import janus

__all__ = [
    "DshLauncherConfig",
    "DshLauncher",
    "EventHandler",
    "DshExit",
]

# inbound 通知处理: 收到一个 mux 下行帧. 返回 None 或 awaitable (异步消费方).
EventHandler = Callable[[MuxFrame], Awaitable[None] | None]


@dataclass(frozen=True, slots=True)
class DshExit:
    """dsh 子进程退出信息 (冻结, 便于按需扩展)."""

    exit_code: int | None
    stderr: str
    self_shutdown: bool = False


class DshLauncherConfig(BaseModel):
    """拉起 dsh web profile 的参数面.

    Deliberately thin: 只装这个启动器 spawn 进程、连上 web 表面需要的参数.
    dsh 自己的配置 (provider/model/prompt/tools) 在 dsh 的文件/env 里由 dsh
    自发现, 这里不复刻. 可扩展靠"嵌套子配置 + 子类化", forbid 让拼写错当场失败.
    """

    model_config = ConfigDict(extra="forbid")

    binary: str = Field(default="dsh", description="dsh 可执行; 默认从 PATH 找.")
    home: Path | None = Field(default=None, description="DSH_HOME; None → 在 cwd 启动, 让 dsh 自发现 profile/config.")
    profile: str = Field(default="web", description="进程层 profile 选择, 不是 dsh 配置.")
    port: int = Field(default=3083, description="web 端口; base_url/mux_url 由此派生.")
    args: list[str] = Field(default_factory=list, description="启动器 flag 之后的 verbatim 参数.")
    readiness_path: str = Field(default="/plugin-api/ping", description="就绪探针: 轮询到它返回即视为 dsh+plugin 起来.")
    connect_timeout: float = Field(default=10.0, description="连 WS / 单个 HTTP 请求的超时 (秒).")
    readiness_timeout: float = Field(default=30.0, description="等待就绪的时限 (秒).")
    shutdown_timeout: float = Field(default=5.0, description="拆除进程的时限 (秒).")

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def mux_url(self) -> str:
        return f"ws://127.0.0.1:{self.port}/api/events.mux"


class DshLauncher:
    """薄启动器: lifecycle + 协议原语, 不背业务.

    subprocesses 走构造注入 (控制反转). 传入 ghost/owner 的 Subprocesses,
    dsh 骑 owner 的治理链; 不传则自建一个 SubprocessesImpl, 自包含可用.
    """

    def __init__(
            self,
            config: DshLauncherConfig,
            subprocesses: Subprocesses | None = None,
            logger: LoggerItf | None = None,
    ) -> None:
        self.config = config
        self._external_sp = subprocesses is not None
        self._subprocess_manager: Subprocesses = subprocesses or SubprocessesImpl()
        self._owns_sp = not self._external_sp
        self._dsh_process: ManagedProcess | None = None
        # prepare http client
        self._http_client = httpx.AsyncClient(timeout=self.config.connect_timeout)
        self._ws: Any | None = None
        self._mux_task: asyncio.Task[Any] | None = None
        self._handlers: list[EventHandler] = []
        self._logger: LoggerItf = logger or get_moss_logger()
        self._aexit_stack = AsyncExitStack()
        # 标记 dsh 是否已经运行.
        self._dsh_started = ThreadSafeEvent()
        self._started = False
        self._stopped = False
        # 标记 dsh 子进程运行态: spawn 后 True, on_exit 回调翻 False.
        self._dsh_subprocess_is_running = False
        self._consume_dsh_process_out_task: asyncio.Task | None = None
        self._consume_dsh_process_err_task: asyncio.Task | None = None
        self._on_exit_callbacks: list[Callable[[DshExit], None]] = []
        self._exit: DshExit | None = None
        self._self_shutdown = False
        self._rpc_counter = 0
        self._stderr_lines: list[str] = []
        self._log_prefix: str = f"[DSHLauncher] "

    # ---- 运行状态 ---- #
    def is_running(self) -> bool:
        return self._started and not self._stopped and self._dsh_subprocess_is_running

    def _check_running(self) -> None:
        if not self.is_running():
            raise RuntimeError("DshLauncher not running (dsh subprocess not alive)")

    # ---- 生命周期 ---- #

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

    @contextlib.asynccontextmanager
    async def _dsh_process_ctx(self):
        self._dsh_process = await self._spawn_dsh()
        try:
            yield
        finally:
            await self._stop_proc()

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
        """mux WS 下行重连循环: 连上后先 print 原始帧, 断开则重连."""
        while self._dsh_subprocess_is_running:
            try:
                async with websockets.connect(self.config.mux_url) as ws:
                    self._dsh_started.set()
                    print(f"{self._log_prefix}mux connected")
                    async for raw in ws:
                        self._logger.debug("mux event: %s", raw)
            except asyncio.CancelledError:
                raise
            except ConnectionRefusedError as exc:
                self._logger.warning("mux TCP refused (dsh not up): %s", exc)
            except websockets.exceptions.InvalidHandshake as exc:
                self._logger.warning("mux handshake failed (mux not ready): %s", exc)
            except websockets.exceptions.ConnectionClosed as exc:
                self._logger.warning("mux connection closed: %s", exc)
            await asyncio.sleep(1.0)

    async def __aenter__(self) -> Self:
        if self._started:
            return self
        try:
            self._started = True
            # 启动 aexit stack.
            await self._aexit_stack.__aenter__()
            if not self._subprocess_manager.is_running():
                await self._aexit_stack.enter_async_context(self._subprocess_manager)
            # spawn dsh
            await self._aexit_stack.enter_async_context(self._dsh_process_ctx())
            # 压栈子进程 rpc 协议消费逻辑.
            await self._aexit_stack.enter_async_context(self._consume_dsh_process_ctx())
            # 压栈 mux WS 下行重连循环.
            await self._aexit_stack.enter_async_context(self._ws_loop_ctx())
            # 阻塞到 ws 连上 (mux connected) 才返回, 作为 push 式就绪信号.
            await self._dsh_started.wait()

            # todo: 后续慢慢改.
            # await self._wait_readiness()
            # await self._connect_mux()
        except BaseException:
            # await self._close_mux()
            # self._dump_stderr()
            # await self._stop_proc()
            # if self._owns_sp:
            #     await self._subprocess_manager.__aexit__(None, None, None)
            # raise
            raise
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._stopped:
            return
        self._stopped = True
        # 退出所有的栈.
        await self._aexit_stack.__aexit__(exc_type, exc_val, exc_tb)

        await self._close_mux()
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None
        await self._stop_proc()
        if self._owns_sp:
            await self._subprocess_manager.__aexit__(None, None, None)

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
        resp.raise_for_status()
        return resp.json()

    async def rpc(
            self,
            method: str,
            payload: dict[str, Any] | None = None,
    ) -> RpcResult:
        """apiproxy 动词调用: POST /api/<method>, 包 client-request 信封, 解 server-response."""
        self._check_running()
        if self._http_client is None:
            raise RuntimeError("DshLauncher not started")
        self._rpc_counter += 1
        envelope = {
            "type": "client-request",
            "rpcId": f"rpc-{self._rpc_counter}",
            "method": method,
            "payload": payload or {},
        }
        resp = await self._http_client.post(
            f"{self.config.base_url}/api/{method}",
            json=envelope,
            timeout=self.config.connect_timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["result"]

    def on_event(self, handler: EventHandler) -> None:
        """inbound 单向通知: 注册一个消费 mux 下行帧的 callback (fire-and-forget)."""
        self._handlers.append(handler)

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
                print(f"{self._log_prefix}{line.decode(errors='replace').rstrip()}")
        finally:
            print(f"{self._log_prefix}stdout consume closed")

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

    async def _wait_readiness(self) -> None:
        """轮询就绪探针 (替代 demo 的固定 sleep), 直到 up 或超时."""
        deadline = asyncio.get_running_loop().time() + self.config.readiness_timeout
        while True:
            try:
                await self.call(self.config.readiness_path)
                return
            except Exception:
                if asyncio.get_running_loop().time() >= deadline:
                    raise TimeoutError(
                        f"dsh 未在 {self.config.readiness_timeout}s 内就绪 (probe {self.config.readiness_path})"
                    ) from None
                await asyncio.sleep(0.2)

    async def _connect_mux(self) -> None:
        self._ws = await websockets.connect(self.config.mux_url)
        self._mux_task = asyncio.create_task(self._mux_loop())

    async def _mux_loop(self) -> None:
        """读 mux 下行流, 按 server-request 信封解出 MuxFrame, 派发给所有 handler."""
        ws = self._ws
        try:
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(msg, dict) or msg.get("type") != "server-request":
                    continue
                frame = MuxFrame(type=msg.get("method", ""), **(msg.get("payload") or {}))
                for handler in list(self._handlers):
                    try:
                        result = handler(frame)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception as exc:  # noqa: BLE001 — 消费方出错不拖垮 mux reader
                        print(f"[dsh-launcher] handler error: {exc!r}")
        finally:
            if self._ws is not None:
                try:
                    await self._ws.close()
                except Exception:
                    # todo: 静默失败.
                    pass
                self._ws = None

    async def _close_mux(self) -> None:
        task = self._mux_task
        if task is not None:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
            self._mux_task = None
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

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
            print(f"--- dsh stderr tail ---\n{tail[-2000:]}")
