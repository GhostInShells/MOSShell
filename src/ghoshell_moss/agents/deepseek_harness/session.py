"""
DshSession — 持有 dsh session 句柄的会话级 facade.

与 DshClient(全局管理面) 分工: DshClient 回答「dsh 全局有哪些东西」, DshSession 回答
「某一个 session 自己能做什么」。两者同走 apiproxy http 面, 但 DshSession 绑定单个
sessionId, 把身份与原始 rpc 入参对象屏蔽掉。

屏蔽约定 (facade 的立身之本):
- 入参屏蔽: 方法收 plain args, 不收 Session*Params; sessionId 由 facade 自动填充,
  调用方永远不用传。
- 返回值不屏蔽: 直接返回 types/ 里已建的 pydantic value 模型 (SessionPromptValue 等),
  不做二次封装。

依赖方向 (不互绑, 防治理循环):
- DshSession 只持有 DshClient(叶子) 与 sessionId, 不持有 launcher。
- mux 帧流依赖被反转: session 暴露 accept_frame(MuxFrame | HostFrame) 入口, 由 owner
  单向注册喂帧 — host 流收运行态, mux 流收 session event。
- 退出解绑: session 关闭时 fire on_exit 回调, owner 借回调解绑 accept_frame, 引用链断。

生命周期:
- async with: __aenter__ 起消费 task, __aexit__ 拆除; 可提前 close() 幂等关闭。
- is_running() -> bool: 本对象的生命周期激活态 (_started and not _closed), 是消费门控,
  不是 dsh 运行态。
- close() 幂等: 停消费 task → fire on_exit 回调 (异常隔离)。

状态:
- 消费门控用 is_running() — 没启动就不监听, 防 queue 爆炸。
- dsh 运行态 (host/session-status{running}) 被 session 消费、存进 _dsh_running, 不通过
  is_running() 暴露; 生命周期变更检查下一轮用。

事件消费:
- 线性消费: accept_frame 只入队不处理(无背压, append + Event.set), 消费 task 逐帧处理。
- 运行态 (host/session-status{running}) 本文件消费; session event 的具体监听表面
  (on_turn_start/on_tool_call/...) 与生命周期变更检查, 下一轮再定, 本文件不展开。
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from typing import Callable, Literal

from typing_extensions import Self

from ghoshell_moss.agents.deepseek_harness.client import DshClient
from ghoshell_moss.agents.deepseek_harness.types import sessions
from ghoshell_moss.agents.deepseek_harness.types.events import HostFrame, MuxFrame
from ghoshell_moss.contracts.logger import LoggerItf, get_moss_logger

__all__ = ["DshSession"]

# 消费 task 连续处理 _YIELD_EVERY 帧后主动 sleep(0.0) 让出 loop, 防长队饿死其它任务.
_YIELD_EVERY = 64

# session 关闭回调: owner 用它解绑 accept_frame / 清理引用. 无参同步.
ExitCallback = Callable[[], None]


class DshSession:
    """会话级 facade: 绑定一个 sessionId, 屏蔽 rpc 入参对象, 封装驱动动词."""

    def __init__(
        self,
        *,
        session_id: str,
        client: DshClient,
        logger: LoggerItf | None = None,
    ) -> None:
        self._session_id = session_id
        self._client = client
        self._logger = logger or get_moss_logger()
        # dsh 运行态 (host/session-status{running} 由 session 自己消费、持有).
        self._dsh_running = False
        # 线性消费: deque 存帧, Event 唤醒消费 task (空等待阻塞, 不忙旋).
        self._queue: deque = deque()
        self._wakeup = asyncio.Event()
        self._consume_task: asyncio.Task | None = None
        self._on_exit_callbacks: list[ExitCallback] = []
        self._started = False
        self._closed = False

    # ---- 生命周期 ---- #

    async def __aenter__(self) -> Self:
        if self._started:
            return self
        self._started = True
        self._consume_task = asyncio.create_task(self._consume_loop())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def close(self) -> None:
        """幂等提前关闭: 停消费 task → fire on_exit 回调."""
        if self._closed:
            return
        self._closed = True
        task = self._consume_task
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            self._consume_task = None
        for callback in self._on_exit_callbacks:
            try:
                callback()
            except Exception:
                self._logger.exception("dsh session %s on_exit callback failed", self._session_id)

    def on_exit(self, callback: ExitCallback) -> None:
        """注册 session 关闭时的回调 (owner 用它解绑 accept_frame / 清理引用)."""
        self._on_exit_callbacks.append(callback)

    def is_running(self) -> bool:
        """本对象生命周期激活态 (_started and not _closed), 消费门控, 非 dsh 运行态."""
        return self._started and not self._closed

    # ---- 帧入口 (owner 喂帧, 反转依赖) ---- #

    def accept_frame(self, frame: MuxFrame | HostFrame) -> None:
        """owner 单向喂帧入口: 没启动不监听(防 queue 爆炸), 按 sessionId 分流, 只入队."""
        if not self.is_running():
            return
        if frame.sessionId != self._session_id:
            return
        self._queue.append(frame)
        self._wakeup.set()

    # ---- 驱动动词 (入参屏蔽, 返回值不屏蔽) ---- #

    async def prompt(
        self,
        *,
        content: list[sessions.PromptContentPart],
        mode: str | Literal["queue", "steer"] = "queue",
        client_timezone: str | None = None,
    ) -> sessions.SessionPromptValue:
        params = sessions.SessionPromptParams(
            sessionId=self._session_id,
            mode=mode,
            content=content,
            clientTimeZone=client_timezone,
        )
        return await self._client.call("session.prompt", params, sessions.SessionPromptValue)

    async def cancel(self) -> sessions.SessionCancelValue:
        params = sessions.SessionCancelParams(sessionId=self._session_id)
        return await self._client.call("session.cancel", params, sessions.SessionCancelValue)

    async def update_queue(
        self,
        *,
        item_id: str,
        action: sessions.QueueAction,
    ) -> sessions.SessionUpdateQueueValue:
        params = sessions.SessionUpdateQueueParams(
            sessionId=self._session_id,
            itemId=item_id,
            action=action,
        )
        return await self._client.call("session.updateQueue", params, sessions.SessionUpdateQueueValue)

    async def select_model(
        self,
        *,
        provider: str,
        model: str,
        reasoning_effort: str | None = None,
    ) -> sessions.SessionSelectModelValue:
        params = sessions.SessionSelectModelParams(
            sessionId=self._session_id,
            provider=provider,
            model=model,
            reasoningEffort=reasoning_effort,
        )
        return await self._client.call("session.selectModel", params, sessions.SessionSelectModelValue)

    async def models(self) -> sessions.SessionModels:
        params = sessions.SessionModelsParams(sessionId=self._session_id)
        return await self._client.call("session.models", params, sessions.SessionModels)

    async def history(
        self,
        *,
        before_seq: int | None = None,
        max_messages: int | None = None,
    ) -> sessions.SessionHistoryValue:
        params = sessions.SessionHistoryParams(
            sessionId=self._session_id,
            beforeSeq=before_seq,
            maxMessages=max_messages,
        )
        return await self._client.call("session.history", params, sessions.SessionHistoryValue)

    async def fork(self, *, at_seq: int | None = None) -> sessions.SessionForkValue:
        params = sessions.SessionForkParams(sessionId=self._session_id, atSeq=at_seq)
        return await self._client.call("session.fork", params, sessions.SessionForkValue)

    async def rename(self, *, title: str) -> sessions.SessionRenameValue:
        params = sessions.SessionRenameParams(sessionId=self._session_id, title=title)
        return await self._client.call("session.rename", params, sessions.SessionRenameValue)

    async def attachment(self, *, attachment_id: str) -> sessions.SessionAttachmentValue:
        params = sessions.SessionAttachmentParams(sessionId=self._session_id, attachmentId=attachment_id)
        return await self._client.call("session.attachment", params, sessions.SessionAttachmentValue)

    # ---- 消费 ---- #

    async def _consume_loop(self) -> None:
        """线性消费队列: 逐帧处理, 空则阻塞等 Event, 连续 _YIELD_EVERY 帧后 sleep(0.0) 让出."""
        n = 0
        while not self._closed:
            if not self._queue:
                self._wakeup.clear()
                if not self._queue:
                    await self._wakeup.wait()
                continue
            frame = self._queue.popleft()
            try:
                self._handle_frame(frame)
            except Exception:
                self._logger.exception("dsh session %s frame handling failed", self._session_id)
            n += 1
            if n % _YIELD_EVERY == 0:
                await asyncio.sleep(0.0)

    def _handle_frame(self, frame: MuxFrame | HostFrame) -> None:
        """按帧 type 分派. 本文件只处理运行态; session event 的 on_xxx 下一轮接."""
        if frame.type == "host/session-status":
            self._dsh_running = frame.running
