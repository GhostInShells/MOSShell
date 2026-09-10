"""
DshSession — 持有 dsh session 句柄的会话级 facade.

与 DshClient(全局管理面) 分工: DshClient 回答「dsh 全局有哪些东西」, DshSession 回答
「某一个 session 自己能做什么」。两者同走 apiproxy http 面, 但 DshSession 绑定单个
sessionId, 把身份与原始 rpc 入参对象屏蔽掉。

屏蔽约定 (facade 的立身之本):
- 入参屏蔽: 方法收 plain args, 不收 Session*Params; sessionId 由 facade 自动填充,
  调用方永远不用传。
- 返回值不屏蔽: 原始动词方法直接返回 types/ 里已建的 pydantic value 模型 (SessionPromptValue
  等), 不做二次封装。唯一例外是 run() — 它组合多帧产出 DshRunResult (对齐官方 SDK RunResult)。

两个驱动层次 (对齐官方 dsh SDK 的 client.session_prompt / Session.run 分层):
- 原始动词: prompt() / cancel() 是 dsh 原生动词的薄透传 (fire-and-return, 返回 accepted),
  不阻塞在 turn 上。适合"发出去就不管"的异步驱动。
- 单轮对话: run() 是组合出的阻塞单轮 (官方 SDK Session.run 的语义) — 发 prompt,
  等本轮 turn/end, 返回 final_response + finish_reason + events。这是"可以 loop 它的关键"。
  cancel() 是它对称的标准中断: 中断在跑的 turn, 使 pending run() 以 turn/end 结算。

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
- session/event 帧按事件名分派到 on_session_event* 注册的分派闭包 (阻塞消费, 逐 handler await)。
- 本文件经 on_session_event_model 注册 token 记账 (assistant/message usage); instruction /
  surface 走 plugin 路由 pull (见 instruction() / surface_messages())。
- close() 清空分派闭包集合, 释放回调引用 (消费方 aexit 时主动 close 帮助回收)。
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Self

from ghoshell_moss.contracts.logger import LoggerItf, get_moss_logger
from ghoshell_moss.deepseek_harness.client import DshClient
from ghoshell_moss.deepseek_harness.types import sessions
from ghoshell_moss.deepseek_harness.types.events import HostFrame, MuxFrame
from ghoshell_moss.deepseek_harness.types.session_events import (
    AssistantMessageEvent,
    Message,
    SessionEvent,
    SessionEventModel,
    TokenUsage,
    TurnEnd,
    TurnStart,
)

__all__ = ["DshRunResult", "DshSession"]

# 消费 task 连续处理 _YIELD_EVERY 帧后主动 sleep(0.0) 让出 loop, 防长队饿死其它任务.
_YIELD_EVERY = 64

# plugin 观测面路由 (与 moss-dolores-ghost-plugin.ts 的 DOLORES_SESSION_* 常量对齐, 跨语言契约).
_DOLORES_SESSION_INSTRUCTION = "/moss-api/ghost/dolores/session/instruction"
_DOLORES_SESSION_SURFACE = "/moss-api/ghost/dolores/session/surface"

# on_session_event 泛型参数: E 绑定具体模型类, 回调收该类强类型实例.
E = TypeVar("E", bound=SessionEventModel)

# session 关闭回调: owner 用它解绑 accept_frame / 清理引用. 无参同步.
ExitCallback = Callable[[], None]

# 分派闭包: on_session_event* 注册时生成, 收原始信封 SessionEvent, 内部完成强类型重建或原样
# 透传, 放进按事件名分组的集合. 消费方收的类型由注册方法决定 (raw 信封 vs 强类型模型).
EventDispatcher = Callable[[SessionEvent], Awaitable[None]]

# 通配事件名: on_session_event(WILDCARD_EVENT, cb) 注册 catch-all — 每个 session/event 帧都派发,
# 不挑事件名 (全量观测面, 如 DoloresRun 的 _events 累积 + _last_seq 推进). 仅 raw 注册有意义;
# on_session_event_model 绑定具体 model_cls.event_type(), 与通配无关.
WILDCARD_EVENT = "*"


class DshRunResult(BaseModel):
    """一轮 run() 的结果 — 对齐官方 SDK 的 RunResult (单轮: prompt → turn 结束).

    字段镜像官方 dsh Python SDK 的 ``RunResult`` (见 dsh 源码 python/sdk/.../api.py), 去掉
    传输专属的 ``notifications`` / ``session_root``, 保留对 loop 有用的四元组。
    """

    model_config = ConfigDict(extra="allow")

    session_id: str = Field(default="")
    final_response: str = Field(default="", description="本轮最后一条 assistant/message 的 text 块拼接.")
    finish_reason: str = Field(default="", description="本轮 turn/end 的 data.reason.kind; 无 turn/end 时为空串.")
    events: list[SessionEvent] = Field(default_factory=list, description="本轮 turn 区间内收集的原始事件.")


def _normalize_content(content: list[sessions.PromptContentPart] | str) -> list[sessions.PromptContentPart]:
    """str → ``[{type:text, text}]``, 与官方 SDK 的 normalize_input 同形; list 原样透传."""
    if isinstance(content, str):
        return [sessions.PromptContentPart(type="text", text=content)]
    return list(content)


def _final_response(events: list[SessionEvent]) -> str:
    """取区间内最后一条 assistant/message 的 text 块拼接 (镜像官方 SDK final_response)."""
    for event in reversed(events):
        message = AssistantMessageEvent.from_session_event(event)
        if message is None:
            continue
        return "".join(block.text or "" for block in message.message.content if block.type == "text")
    return ""


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
        # 会话累计 token 用量 (assistant/message usage 累加, 纯记账 — 消费方经 on_session_event* 订阅).
        self._token_usage = TokenUsage()
        # 会话级事件治理: 事件名 → 分派闭包集合 (on_session_event* 注册时由 model_cls+callback 包成).
        self._event_handlers: dict[str, set[EventDispatcher]] = {}
        # 内部治理回调 — token 记账: 与外部消费同一机制 (dogfooding on_session_event_model).
        self.on_session_event_model(AssistantMessageEvent, self._on_assistant_message)
        # 观测状态镜像: model/routable 由 session.models 拉取; cwd/agent_preset 是会话常量,
        # 由 host/session-added 帧或 session.list 拉取. instruction 走 plugin 路由 pull (见 instruction()).
        self._model_selection: sessions.ModelSelection | None = None
        self._routable: bool | None = None
        self._cwd: str | None = None
        self._agent_preset: str | None = None
        # 线性消费: deque 存帧, Event 唤醒消费 task (空等待阻塞, 不忙旋).
        self._queue: deque = deque()
        self._wakeup = asyncio.Event()
        self._consume_task: asyncio.Task | None = None
        self._on_exit_callbacks: list[ExitCallback] = []
        self._started = False
        self._closed = False
        # 单轮驱动门控: 同一 session 同时只允许一个 run() 在飞 (对齐 ACP "one in-flight request per session").
        self._run_active = False

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
        """幂等提前关闭: 停消费 task → 清事件分派闭包 → fire on_exit 回调.

        清空 _event_handlers 释放回调闭包引用 (消费方 aexit 时主动 close 帮助内存回收).
        """
        if self._closed:
            return
        self._closed = True
        task = self._consume_task
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            self._consume_task = None
        self._event_handlers.clear()
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
        """标准中断: 停一个活跃 turn, 保留队尾工作 (dsh session.cancel, keepInbox).

        使在跑的 :meth:`run` 以本轮 turn/end 结算 (reason 落到 interrupted/aborted)。
        与 run() 对称 — fire-and-return 的中断动词, 不等 turn 停下来。
        """
        params = sessions.SessionCancelParams(sessionId=self._session_id)
        return await self._client.call("session.cancel", params, sessions.SessionCancelValue)

    async def run(
        self,
        content: list[sessions.PromptContentPart] | str,
        *,
        mode: str | Literal["queue", "steer"] = "queue",
        client_timezone: str | None = None,
        timeout: float | None = None,
    ) -> DshRunResult:
        """单轮对话 (阻塞) — 对齐官方 dsh SDK 的 ``Session.run``, 传输改走 web profile。

        语义: 发 prompt → 等本轮 turn 结束 → 返回结果。这是 loop 一个 session 的基本原语:

            for message in messages:
                result = await session.run(message)

        与 :meth:`prompt` 的分工: prompt 是 fire-and-return 的原生动词 (返回 accepted);
        run 是组合出的阻塞单轮 (返回 final_response + finish_reason + events)。

        实现口径 (对齐 SDK 的可见契约, 两处适配 MOSS 传输):
        - 先挂 catch-all 收集器再发 prompt, 不丢本轮 ``turn/start``。
        - SDK 用 ``agent/inbox/spliced`` 回执门控起点; apiproxy ``session.prompt`` 不回
          messageId, 故改以**本轮第一个 turn/start** 为起点门控 (更强: 直接锚定 turn 号)。
        - SDK 停在 whole-agent idle; 这里停在**本轮 turn/end** — 才能保证"只跑一轮",
          不被队尾排队的工作拖住。
        - ``final_response`` = 区间内最后一条 assistant/message 的 text; ``finish_reason`` =
          该 turn/end 的 reason.kind。

        ``cancel()`` 对称: 中断在跑的 turn, run 随即以 turn/end 结算 (reason 反映中断)。

        :param content: prompt 内容 — str 自动包成单个 text 块, 或 ``PromptContentPart`` 列表。
        :param mode: dsh 的 queue (独占下一 turn) / steer (下个 step 边界)。
        :param timeout: 本轮等待上限 (秒); None = 一直等。超时抛 ``asyncio.TimeoutError``。
        :raises RuntimeError: 已有 run() 在飞 (每 session 只允许一个 in-flight)。
        """
        if self._run_active:
            raise RuntimeError(f"dsh session {self._session_id}: run() already in flight")
        self._run_active = True
        # 本轮状态: 起点 (turn 号) 由 turn/start 定; reason 由匹配 turn 的 turn/end 定。
        # done 置位后不再收帧 — 消费循环可能在本任务醒来解绑前抢跑到队尾下一个 turn。
        state: dict[str, Any] = {"turn": None, "reason": None, "events": [], "done": False}
        turn_started = asyncio.Event()
        turn_ended = asyncio.Event()

        async def _collect(event: SessionEvent) -> None:
            if state["done"]:
                return
            if state["turn"] is None:
                start = TurnStart.from_session_event(event)
                if start is None:
                    return
                state["turn"] = start.turn
                turn_started.set()
            state["events"].append(event)
            end = TurnEnd.from_session_event(event)
            if end is not None and end.turn == state["turn"]:
                state["reason"] = end.reason
                state["done"] = True
                turn_ended.set()

        remove = self.on_session_event(WILDCARD_EVENT, _collect)
        try:
            await self.prompt(content=_normalize_content(content), mode=mode, client_timezone=client_timezone)
            if timeout is None:
                await turn_started.wait()
                await turn_ended.wait()
            else:
                await asyncio.wait_for(turn_started.wait(), timeout)
                await asyncio.wait_for(turn_ended.wait(), timeout)
        finally:
            remove()
            self._run_active = False
        reason = state["reason"]
        return DshRunResult(
            session_id=self._session_id,
            final_response=_final_response(state["events"]),
            finish_reason=reason.kind if reason is not None else "",
            events=state["events"],
        )

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
        """按帧 type 分派. 运行态镜像 + session/event 事件分派 (on_session_event 注册回调)."""
        if frame.type == "host/session-status":
            self._set_running(frame.running)
        elif frame.type == "host/session-added":
            if frame.cwd is not None:
                self._cwd = frame.cwd
            if frame.agentPreset is not None:
                self._agent_preset = frame.agentPreset
        elif frame.type == "session/event":
            event = frame.event
            if event is None:
                return
            await self._dispatch_session_event(event)

    async def _dispatch_session_event(self, event: SessionEvent) -> None:
        """按事件名分派到 on_session_event* 注册的分派闭包 (阻塞消费, 逐 handler await).

        闭包内部已捕获 model_cls + callback, 完成 from_session_event 判别或原样透传.
        handler 异常隔离记录, 不影响后续 handler 与消费循环.
        精确名 handler 先派发, 再派发 WILDCARD_EVENT ("*") 的 catch-all — 全量观测消费
        与定向消费并存, 互不干扰.
        """
        for name in (event.meta.type, WILDCARD_EVENT):
            handlers = self._event_handlers.get(name)
            if not handlers:
                continue
            for handler in list(handlers):
                try:
                    await handler(event)
                except Exception:
                    self._logger.exception(
                        "dsh session %s %s event handler failed", self._session_id, event.meta.type
                    )

    def _set_running(self, running: bool) -> None:
        """翻转运行态镜像事件: running ⇄ idle 互斥 set/clear."""
        self._dsh_running = running
        if running:
            self._running_event.set()
            self._idle_event.clear()
        else:
            self._idle_event.set()
            self._running_event.clear()

    async def _on_assistant_message(self, event: AssistantMessageEvent) -> None:
        """内部治理回调 (on_session_event_model 注册): assistant/message usage 累进累计量."""
        if event.usage is not None:
            await self._accumulate_usage(event.usage)

    async def _accumulate_usage(self, usage: TokenUsage) -> None:
        """把一步的 usage 累进会话累计量 (纯记账; 对外消费经 on_session_event* 订阅)."""
        total = self._token_usage
        total.inputTokens += usage.inputTokens or 0
        total.outputTokens += usage.outputTokens or 0
        if usage.cacheReadTokens is not None:
            total.cacheReadTokens = (total.cacheReadTokens or 0) + usage.cacheReadTokens
        if usage.cacheWriteTokens is not None:
            total.cacheWriteTokens = (total.cacheWriteTokens or 0) + usage.cacheWriteTokens
        if usage.reasoningTokens is not None:
            total.reasoningTokens = (total.reasoningTokens or 0) + usage.reasoningTokens

    # ---- 对外状态面 (ego 消费) ---- #

    @property
    def running(self) -> bool:
        """dsh agent 当前运行态 — host/session-status 推帧实时更新, 非本对象生命周期态."""
        return self._dsh_running

    @property
    def token_usage(self) -> TokenUsage:
        """会话累计 token 用量 — assistant/message usage 累加 (纯记账, 实时可读)."""
        return self._token_usage

    async def instruction(self) -> str | None:
        """当前全量 instruction — 从 plugin 路由现场组装 (与 request/header.system 同源).

        plugin 侧经 ctx.systemPrompt.assemble + renderPrompt 产出与模型实际看到的完全一致的
        system prompt. 真实读接口: 任意时刻 session 在即可读, 不依赖 mux 帧或 history 冷锚.
        """
        result = await self._client.plugin_call(
            _DOLORES_SESSION_INSTRUCTION, {"sessionId": self._session_id}
        )
        instruction = result.get("instruction")
        return instruction if isinstance(instruction, str) else None

    async def surface_messages(self) -> list[Message]:
        """全量 surface 消息列表 — 模型可见投影 (user/assistant/tool-result, 尊重 compact replace).

        与 history() 的 log 全量事件不同: surface 只投影三种 surface-eligible 事件, 按模型可见序,
        空 content 的 assistant/message 被剔除. 从 plugin 路由 (deriveMessages) 读.
        """
        result = await self._client.plugin_call(
            _DOLORES_SESSION_SURFACE, {"sessionId": self._session_id}
        )
        raw = result.get("messages") or []
        return [Message.model_validate(m) for m in raw]

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

    async def cwd(self, *, force: bool = False) -> str | None:
        """会话工作目录 — host/session-added 或 session.list 的 header.cwd (创建后不变)."""
        if not force and self._cwd is not None:
            return self._cwd
        await self._session_summary()
        return self._cwd

    async def agent_preset(self, *, force: bool = False) -> str | None:
        """会话的 agentPreset (运行模式) — 创建时定, 首 turn 后锁死. 冷锚经 session.list 拉."""
        if not force and self._agent_preset is not None:
            return self._agent_preset
        await self._session_summary()
        return self._agent_preset

    async def _session_summary(self) -> None:
        """拉 session.list 找本会话, 一次性填充 cwd / agent_preset (会话常量)."""
        value = await self._client.call(
            "session.list", sessions.SessionListParams(), sessions.SessionListValue,
        )
        for item in value.items:
            if item.sessionId == self._session_id:
                self._cwd = item.cwd
                self._agent_preset = item.agentPreset
                return

    async def when_running(self) -> None:
        """等到 agent 处于 running. 已在 running 则立即返回 (状态镜像, 非边沿触发)."""
        await self._running_event.wait()

    async def when_idle(self) -> None:
        """等到 agent 处于 idle. 已在 idle 则立即返回 (状态镜像, 非边沿触发)."""
        await self._idle_event.wait()

    def on_session_event(
        self,
        event_type: str,
        callback: Callable[[SessionEvent], Awaitable[None]],
    ) -> Callable[[], None]:
        """注册会话级 session 事件回调 (原始信封), 返回解绑函数.

        按事件名字符串分派, 回调收原始 ``SessionEvent`` 信封 (不重建模型) — 适合治理 /
        日志 / 全量观测面. 强类型消费用 :meth:`on_session_event_model`. 与
        TopicService.subscribe 同思路: 注册的都归一到分派闭包 (EventDispatcher).

        传 ``WILDCARD_EVENT`` ("*") 作 event_type 注册 catch-all — 每个 session/event 帧
        都派发, 不挑事件名 (全量观测面).
        """
        async def _dispatch(event: SessionEvent) -> None:
            await callback(event)

        return self._register_event_handler(event_type, _dispatch)

    def on_session_event_model(
        self,
        model_cls: type[E],
        callback: Callable[[E], Awaitable[None]],
    ) -> Callable[[], None]:
        """注册会话级 session 事件回调 (强类型模型), 返回解绑函数.

        按 ``model_cls.event_type()`` 事件名分派; 回调收 ``model_cls.from_session_event(event)``
        重建的强类型模型实例. 与 TopicService.subscribe_model 同思路.

        阻塞消费: dispatch 在消费 task 内逐 handler await, 不做并发; handler 异常隔离记录,
        不影响其它 handler 与消费循环. 与 launcher 的 on_mux_frame/on_host_frame
        同一注册-解绑模式.
        """
        async def _dispatch(event: SessionEvent) -> None:
            model = model_cls.from_session_event(event)
            if model is not None:
                await callback(model)

        return self._register_event_handler(model_cls.event_type(), _dispatch)

    def _register_event_handler(
        self,
        event_type: str,
        dispatcher: EventDispatcher,
    ) -> Callable[[], None]:
        """把分派闭包挂到事件名下, 返回解绑函数. 解绑后空 set 即摘掉键."""
        handlers = self._event_handlers.setdefault(event_type, set())
        handlers.add(dispatcher)

        def _remove() -> None:
            handlers.discard(dispatcher)
            if not handlers:
                self._event_handlers.pop(event_type, None)

        return _remove
