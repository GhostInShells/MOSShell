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
- dsh 运行态 (host/session-status{running}) 被 session 消费, 经 running 属性读取,
  经 when_running/when_idle 等待翻转 (状态镜像事件, 非生命周期态)。

事件消费:
- 线性消费: accept_frame 只入队不处理(无背压, append + Event.set), 消费 task 逐帧处理。
- 运行态 (host/session-status{running}) 与 token 记账 (assistant/message usage) 本文件消费;
  其余 session event 监听表面 (on_turn_start/on_tool_call/...) 下一轮再定, 本文件不展开。
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from typing import Awaitable, Callable, Literal

from typing_extensions import Self

from ghoshell_moss.deepseek_harness.client import DshClient
from ghoshell_moss.deepseek_harness.types import sessions
from ghoshell_moss.deepseek_harness.types.events import HostFrame, MuxFrame
from ghoshell_moss.deepseek_harness.types.session_events import (
    AssistantMessageEvent,
    RequestHeader,
    TokenUsage,
)
from ghoshell_moss.contracts.logger import LoggerItf, get_moss_logger

__all__ = ["DshSession"]

# 消费 task 连续处理 _YIELD_EVERY 帧后主动 sleep(0.0) 让出 loop, 防长队饿死其它任务.
_YIELD_EVERY = 64

# session 关闭回调: owner 用它解绑 accept_frame / 清理引用. 无参同步.
ExitCallback = Callable[[], None]

# token usage 更新回调: 收会话累计 TokenUsage, 可同步或异步 (返回 None 或 coroutine).
UsageCallback = Callable[[TokenUsage], Awaitable[None] | None]


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
        # 运行态镜像事件: running ⇄ idle 翻转时同步 set/clear, 对外经 when_running/when_idle 等待.
        # 初始假设新建 session 的 agent 处于 idle (host/session-status 首帧到来前不拉基线).
        self._running_event = asyncio.Event()
        self._idle_event = asyncio.Event()
        self._idle_event.set()
        # 会话累计 token 用量 (assistant/message usage 累加), 更新时通知 usage 回调.
        self._token_usage = TokenUsage()
        self._usage_callbacks: list[UsageCallback] = []
        # 观测状态镜像: instruction 由 request/header 帧同步; model/routable 由 session.models 拉取.
        self._instruction: str | None = None
        self._model_selection: sessions.ModelSelection | None = None
        self._routable: bool | None = None
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
                await self._handle_frame(frame)
            except Exception:
                self._logger.exception("dsh session %s frame handling failed", self._session_id)
            n += 1
            if n % _YIELD_EVERY == 0:
                await asyncio.sleep(0.0)

    async def _handle_frame(self, frame: MuxFrame | HostFrame) -> None:
        """按帧 type 分派. 本文件处理运行态 + token 记账 + instruction 快照; 其余 on_xxx 下一轮接."""
        if frame.type == "host/session-status":
            self._set_running(frame.running)
        elif frame.type == "session/event":
            event = frame.event
            if event is None:
                return
            if event.meta.type == "assistant/message":
                usage = AssistantMessageEvent.from_session_event(event)
                if usage is not None and usage.usage is not None:
                    await self._accumulate_usage(usage.usage)
            elif event.meta.type == "request/header":
                header = RequestHeader.from_session_event(event)
                if header is not None:
                    self._instruction = header.header.system

    def _set_running(self, running: bool) -> None:
        """翻转运行态镜像事件: running ⇄ idle 互斥 set/clear."""
        self._dsh_running = running
        if running:
            self._running_event.set()
            self._idle_event.clear()
        else:
            self._idle_event.set()
            self._running_event.clear()

    async def _accumulate_usage(self, usage: TokenUsage) -> None:
        """把一步的 usage 累进会话累计量, 然后逐个通知 usage update 回调 (异常隔离)."""
        total = self._token_usage
        total.inputTokens += usage.inputTokens or 0
        total.outputTokens += usage.outputTokens or 0
        if usage.cacheReadTokens is not None:
            total.cacheReadTokens = (total.cacheReadTokens or 0) + usage.cacheReadTokens
        if usage.cacheWriteTokens is not None:
            total.cacheWriteTokens = (total.cacheWriteTokens or 0) + usage.cacheWriteTokens
        if usage.reasoningTokens is not None:
            total.reasoningTokens = (total.reasoningTokens or 0) + usage.reasoningTokens
        for callback in list(self._usage_callbacks):
            try:
                result = callback(total)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                self._logger.exception(
                    "dsh session %s usage update callback failed", self._session_id
                )

    # ---- 对外状态面 (ego 消费) ---- #

    @property
    def running(self) -> bool:
        """dsh agent 当前运行态 — host/session-status 推帧实时更新, 非本对象生命周期态."""
        return self._dsh_running

    @property
    def token_usage(self) -> TokenUsage:
        """会话累计 token 用量 — assistant/message usage 累加, 每次更新通知 usage 回调."""
        return self._token_usage

    async def instruction(self, *, force: bool = False) -> str | None:
        """当前生效的 system prompt — request/header.header.system 的最后一次快照.

        监听同步 (mux request/header 帧) 命中缓存直返; force 或尚未收到时从 history
        折最新 request/header (冷锚基线). 会话尚未发出首个请求时返回 None.
        """
        if not force and self._instruction is not None:
            return self._instruction
        history = await self.history()
        for entry in reversed(history.events):
            header = RequestHeader.from_session_event(entry.event)
            if header is not None:
                self._instruction = header.header.system
                return self._instruction
        return self._instruction

    async def model_selection(self, *, force: bool = False) -> sessions.ModelSelection:
        """当前选中的模型 (provider/model/reasoningEffort) — session.models.current.

        pull-primary: 无完整推源 (request/context 缺 reasoningEffort / routable),
        故缓存命中直返, force 或空则拉 session.models.
        """
        if not force and self._model_selection is not None:
            return self._model_selection
        models = await self.models()
        self._model_selection = models.current
        self._routable = models.routable
        return self._model_selection

    async def routable(self, *, force: bool = False) -> bool:
        """当前模型路由是否可服务 — session.models.routable. 与 model_selection 同源拉取."""
        if not force and self._routable is not None:
            return self._routable
        models = await self.models()
        self._model_selection = models.current
        self._routable = models.routable
        return self._routable

    async def when_running(self) -> None:
        """等到 agent 处于 running. 已在 running 则立即返回 (状态镜像, 非边沿触发)."""
        await self._running_event.wait()

    async def when_idle(self) -> None:
        """等到 agent 处于 idle. 已在 idle 则立即返回 (状态镜像, 非边沿触发)."""
        await self._idle_event.wait()

    def on_usage_update(self, callback: UsageCallback) -> Callable[[], None]:
        """注册 token 用量更新回调 (收累计 TokenUsage), 返回解绑函数.

        回调可同步或异步; 异常被隔离记录, 不影响后续回调. 与 launcher 的
        on_mux_frame/on_host_frame 同一注册-解绑模式.
        """
        self._usage_callbacks.append(callback)

        def _remove() -> None:
            self._usage_callbacks.remove(callback)

        return _remove
