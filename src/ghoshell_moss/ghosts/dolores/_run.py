"""DoloresRun — the run object for a Dolores thinking transaction (async-with boundary + logos() stream).

``async with`` is the transaction boundary (enter opens / exit closes); logos() forwards the dsh
events a debugger needs to the observability surface (``moss-ghost run``'s output), so iterating a
run tells you what happened without spelunking the dsh log:

    async with ego.run_thinking(thinking) as run:
        async for event in run.logos():
            ...

Lifecycle contract:

- aenter: bind the session catch-all listener first (so no enter broadcast is lost), then start the enter task.
- aexit: cancel the enter task, unbind the listener, re-send thinking/exit (even if enter failed, with a
  fail-safe timeout), then abort on error.
- turn/end: 自然收线 (completed / max-tokens) 只结束本帧; 中断类 (其它 cause, 见 _turn_end_abort_reason)
  额外 ``Thinking.abort(reason)`` —— 在**入队前**落地, 不等消费 (消费可能正被 tool handler 占住),
  让 mindflow 在 thinking 退出时急停身体, 并让 attention 一起收掉 (人喊停 / 故障都该是"整个体系停").
- _events(): queue consumption; the poison pill carries only the enter error (the normal path ends via
  turn/end break in logos()).
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Union, AsyncGenerator
from typing_extensions import Self
from ghoshell_moss.core.blueprint.ghost import GhostEvent
from ghoshell_moss.core.blueprint.mindflow import ActionExitedException, Thinking
from ghoshell_moss.core.concepts.errors import InterpretError
from ghoshell_moss.contracts.logger import LoggerItf, get_moss_logger
from ghoshell_moss.deepseek_harness.types.session_events import (
    AssistantChunk,
    SessionEvent,
    ToolCallEvent,
    TurnEnd,
)

from ._ctml_stream import CtmlArgumentStream
from ._tools import (
    InterpretToolCall,
    ReactToolCall,
    ObserveToolCall,
    WaitNextToolCall,
    ShellStatusToolCall,
    ReasoningToolCall,
    ChannelFacadeToolCall,
    ToolCallResult,
)

# 流式 tool 集合: 这两个 tool 的参数经 tool-call-delta 逐字进 articulator, 边生成边执行.
_STREAMING_TOOLS = frozenset({InterpretToolCall.tool_name(), ReactToolCall.tool_name()})

# 只有 moss* 命名空间的 tool 归本 run 管. 会话流 (on_session_event("*")) 里还有 host 自己的内环工具
# (bash / read / write / edit / subagent ...), 它们由 dsh 侧自行结算 —— 我们既不该回答, 也不该报警告.
# 早先把它们也当"等我们回答的 tool"逐条 warning, 一次排查刷出几百行噪音, 把真正的 D39 信号
# (我们自己的 tool 没 handler) 埋掉了.
_DOLORES_TOOL_NAMESPACE_PREFIX = "moss"

# 转发到观测面的 dsh 事件 (moss-ghost run 的 output 直接可读, 迭代时不用反查 dsh log).
# 只取**完整**事件, 不取逐 token 的流: assistant/chunk 的 reasoning/text delta 体量无界
# (一次长思考几千条), 会把 output 面灌满. 留下的这几个每 turn/step 常数条, 各自也不与别处重复:
# turn/* 给层级, tool/* 是行动脊椎, assistant/message 是组装好的输出 + usage (对阈值 debug 有用).
_FORWARDED_DSH_EVENTS = frozenset({
    "turn/start",
    "turn/end",
    "tool/call",
    "tool/result",
    "assistant/message",
})

# event 名前缀: 观测面是共享流, 不带命名空间会和别的 ghost 事件撞名.
_DSH_EVENT_NAME_PREFIX = "dsh/"

if TYPE_CHECKING:
    from ._ego import DoloresEgo
    from ghoshell_moss.core.blueprint.shell_trajectory import MShellContextFacade

__all__ = ["DoloresRun"]

# poison sentinel: enqueued on enter-task error; the consumer raises _enter_error when it reads it.
# never enqueued on the normal path — logos() ends itself on turn/end; the pill only carries enter errors.
_POISON = object()

# turn/end 的 kind 分两类: 自然收线 (帧自己退出, 绝不 abort) 与 中断 (显式 abort, 带 reason).
# 旧注释写的是"turn/end 一律只做观测, 不驱动任何中断" —— 那是把一个方向的判据当成了通性:
# 它只对下面"我们自己的 hook cancel"那一类成立 (tool 结果带 cancel, 身体在返回前已 drain);
# UI 点停与运行时故障若也走自然退出, 身体会继续跑 (语音照旧播完), 故障则自续成环.
_TURN_END_END_KINDS = frozenset({"completed", "max-tokens"})
_TURN_END_INTERRUPT_KINDS = frozenset({"aborted", "interrupted", "blocked", "error"})

# 我们自己收轮时 plugin 用的 cause (dsh_plugin: ``stepAgent.cancel({kind:'hook', reason:'moss cancel_turn'})``).
# 判据必须挂在 cause 上, 不能挂在 turn 号上: moss_react / moss_wait_next / moss_reasoning 的结果**常态**
# 就带 ``cancel=True``, 所以"记下 turn 号 = 自己人"等于把每一轮都记成自己人 —— 人在这一轮里点的
# user abort 会被一起豁免. 2026-10-04 D04 实测: turn/end aborted/user 被吞, 语音照旧 41.3 秒播完.
_OUR_TURN_CANCEL_CAUSE_KINDS = frozenset({"hook"})


def _turn_end_abort_reason(end: TurnEnd) -> str | None:
    """turn/end → 需要显式退出 thinking 时的 reason; None 表示这次收线走自然路径.

    - ``completed`` / ``max-tokens``: 模型说完了或被截断 —— 帧自己退出, 身体不动.
    - ``aborted`` / ``interrupted`` / ``blocked``: 别人打断的 (dsh 界面点停 / 抢占 / plugin 拒绝运行
      这个动作) —— 显式 abort, 由 mindflow 在 thinking 退出时急停身体.
    - ``error``: 运行时故障 —— 同样显式 abort; 不这么做就会自续成环 (弱网 / 断连).
    - 例外: cause 是我们自己的 hook cancel (``kind='hook'``) —— 那是我们自己让 plugin 收的轮
      (moss_react / moss_wait_next / moss_reasoning), 身体在返回结果前已经 drain 过, 不算外部打断.
      判据只认 cause, 不认 turn 号 (理由见上面的常量注释).
    - 未知 kind: dsh 的 kind 是 merge-extensible, 不认识的按自然收线处理 (上游日志里仍留痕).
    """
    kind = end.reason.kind
    if kind in _TURN_END_END_KINDS or kind not in _TURN_END_INTERRUPT_KINDS:
        return None
    cause = end.reason.reason
    if kind != "error" and cause is not None and cause.kind in _OUR_TURN_CANCEL_CAUSE_KINDS:
        return None
    if cause is None:
        return f"turn/end {kind}"
    detail = f": {cause.reason}" if cause.reason else ""
    return f"turn/end {kind} ({cause.kind}{detail})"

_LogosDelta = str


class _VoidCtmlError(Exception):
    """这条 CTML 流作废 —— 传给 articulator 的 ``__aexit__``: 只让 action 出栈, 不 commit.

    为什么需要它: ``tool/call`` 的 arguments 没解析通过时 handler 不会被调用, 于是
    ``_handle_interpret`` 里的 ``pop`` / ``finish`` 都不会发生, 而 delta 早就开出了 articulator。
    那条 action 永远等不到 commit, 会占住 mindflow 的 action loop (见 ``DoloresRun._void_stream``)。

    为什么走"异常退出"而不是 commit: ``BaseArticulator.__aexit__`` 收到异常退出会落
    ``BaseAction.close_lifecycle()`` —— stop / compiled / observed 三个端点一起落地, 于是 action loop
    那次 ``wait_compiled()`` 被放行; 而正常退出 (commit) 会去等一个不会再有人生产的编译位, 把整条
    消费链拖住。用本模块自己的异常而不是 ActionExitedException, 是为了让这次退出在语义上读成
    "这一单没有被受理", 而不是"attention 已经 abort" —— 这里 attention 是好的, 要放弃的只是这一个
    action, 不是整帧。

    为什么"作废"而不是 commit: arguments 被拒 = 这一单没有被受理, 那份半截 CTML 只是流式预览,
    一个命令都还没执行 (执行发生在 commit 之后)。commit 它等于执行模型没有承诺过的东西。
    """


class _StreamedCtml:
    """One streaming CTML tool call (``moss_interpret`` / ``moss_react``) as it streams in.

    The decoder reverses the JSON escaping of ``{"ctml": "…"}`` chunk by chunk; the decoded CTML is
    sent straight to the call's articulator, so the model's output reaches the shell while it is still
    being generated. ``stream.value`` is the exact prefix decoded so far — the handler reconciles
    against the parsed ``tool/call`` argument to cover a shape-mismatch / whole-argument fallback.

    Lifecycle: this object OWNS the articulator's async-with boundary. ``__aenter__`` opens it when
    the stream is created; ``finish`` flushes the decoded tail, exits the boundary, then waits for the
    requested node (compiled / observed). The boundary is the data — a stream region begins when the
    first delta arrives and ends when it is closed — never a bare ``send`` + ``wait_compiled`` pair,
    which leaves ``__aexit__`` (commit + settle) unreached.
    """

    def __init__(self, call_id: str, articulator) -> None:
        self.call_id = call_id
        self.stream = CtmlArgumentStream()
        self.articulator = articulator
        self._entered = False
        self._closed = False

    async def __aenter__(self) -> Self:
        if not self._entered:
            self._entered = True
            await self.articulator.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._closed:
            return None
        self._closed = True
        if self._entered:
            await self.articulator.__aexit__(exc_type, exc_val, exc_tb)
        return None

    async def feed(self, delta: str | None) -> _LogosDelta | None:
        if not self._entered:
            await self.__aenter__()
        text = self.stream.add(delta)
        if text:
            await self.articulator.send(text)
            return text
        return None

    async def finish(self, tail: str = "", wait: str = "compiled") -> None:
        """Close the region: flush any decoded tail, exit the boundary, then wait for the requested node.

        ``tail`` carries the reconciliation remainder (``call.ctml`` beyond the decoded prefix) for a
        whole-argument / shape-mismatch call that streamed nothing. ``wait`` is the settle node:

        - ``"compiled"`` — wait for the CTML to compile only (``moss_react``).
        - ``"observed"`` — wait for every ``always_observe`` command to finish (``moss_interpret``).

        :raise InterpretError: the CTML failed to compile. ``__aexit__`` settles but does not raise it,
        so the check is re-run here — ``wait_compiled`` is safe to call after the boundary (commit and
        approve are both idempotent) and is what surfaces the interpret error.
        """
        if tail:
            # ``tail`` is already-decoded CTML (the reconciliation remainder of call.ctml), so it goes
            # straight to the articulator — it is NOT a raw argumentsDelta and must not re-enter add().
            if not self._entered:
                await self.__aenter__()
            await self.articulator.send(tail)
        await self.__aexit__(None, None, None)
        if wait == "observed":
            await self.articulator.wait_observed(raise_interpret_error=True)
        else:
            await self.articulator.wait_compiled(raise_interpret_error=True)


class DoloresRun:
    """The run object for a Dolores thinking transaction — async-with boundary + logos() stream.

    Lifecycle contract is in the module docstring. Constructed from: ego (DoloresEgo, the narrow
    bridge to session/enter/exit) + thinking + thinking_event (the ego's "transaction running" event,
    set/cleared by run enter/exit, read by the ego self-wake gate). Dependencies go through the
    public interface.
    """

    def __init__(
            self,
            ego: "DoloresEgo",
            thinking: "Thinking",
            thinking_event: asyncio.Event,
            facade: "MShellContextFacade",
            logger: LoggerItf | None = None,
    ) -> None:
        self._ego = ego
        self._thinking = thinking
        self._facade = facade
        # logger 由调用方逐层传下来 (matrix/shell → ghost → ego → run): 它是"当前 node 的 logger",
        # 每层重建一个 get_moss_logger() 等于把日志从 node 语境里摘出去. None 只是兜底.
        self._logger = logger or get_moss_logger()
        self._queue: "asyncio.Queue[Any]" = asyncio.Queue()
        self._dispose_listener: "Callable[[], None] | None" = None
        self._enter_task: "asyncio.Task[None] | None" = None
        # 本帧的 thinking 配对序 — __aenter__ 就铸好, enter/exit 带同一个 (见 _ego.next_thinking_seq).
        self._thinking_seq: int = 0
        self._enter_error: Exception | None = None
        self._thinking_event: asyncio.Event = thinking_event
        # logos() single-consumption guard — a run has at most one logos stream (more would split the queue).
        self._logos_started = False
        # turn/end 只结算一次: 文本循环内与循环外各有一个 call site, 同一 event 会被看见两次.
        self._turn_end_noted = False
        # 我们自己让 plugin 切的 turn (tool 结果带 cancel=True): 那一轮的 turn/end 不是"被打断".
        # 只用于日志 (谁先动手), 不参与判定 —— 判据看 cause.
        self._expected_cancel_turn: int | None = None
        # 中断类 abort 只落地一次 (入队前落地, 消费者侧兜一次).
        self._interrupt_aborted = False
        # 本轮声明了新档位 → 收尾时要自己提交一条 need_observe, 驱动下一帧 (见 __aexit__).
        self._need_switch_thinking_effort: str | None = None

    # ── transaction boundary ─────────────────────────────────────────

    async def __aenter__(self) -> Self:
        """Open the transaction. Bind the listener first (so no enter broadcast is lost), then start the enter task."""
        # 序在起 enter 任务**之前**铸: enter 会被取消, 而 exit 必须能带上同一个序.
        self._thinking_seq = self._ego.next_thinking_seq()
        self._thinking_event.set()
        # 上一有效 run 决定的 compact 由 runtime 在进入本 run 之前执行 (见 Dolores.think): 它同时要开
        # 新 epoch 让 memory 走 recap 槽, 两个动作必须按序发生在请求之前.
        self._dispose_listener = self._ego.session.on_session_event("*", self._on_event)
        self._enter_task = asyncio.create_task(self._drive_enter())
        self._ego.logger.info(
            "Dolores ego run entered: seq=%s thinking=%s moment_index=%s previous_moment_id=%s "
            "need_observe=%s inputs=%s percepts=%s",
            self._thinking_seq,
            self._thinking.attention.id,
            self._thinking.moment.index,
            self._thinking.moment.last_moment_id(),
            self._thinking.observer.need_observe(),
            len(list(self._thinking.moment.inputs_messages())),
            len(list(self._thinking.moment.percepts_messages())),
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close the transaction. Cancel the enter task → unbind → re-send exit → abort (on error)."""
        self._thinking_event.clear()
        # 本 run 是否有效 (无异常且无中断 abort) 只在**收尾**落地 —— 供 compact 判定 (下一 run 才执行).
        self._ego.on_run_exit(valid=exc_type is None and not self._interrupt_aborted)
        task = self._enter_task
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        if self._dispose_listener is not None:
            self._dispose_listener()
        # re-send exit — even if enter failed (to clean up plugin-side state), blocking with a fail-safe timeout.
        await self._ego.exit_thinking(self._thinking_seq)
        # 观测探针 (D15 档位切换排查): 这三样决定"下一帧由谁签发".
        self._logger.info(
            "run exit: exc=%s enter_error=%s effort_marker=%s",
            type(exc_val).__name__ if exc_val is not None else 'None',
            self._enter_error, self._need_switch_thinking_effort,
        )
        if isinstance(exc_val, asyncio.CancelledError):
            self._logger.info("run exit: cancelled — no effort pushed, no echo declared")
            return None
        reason = exc_val if exc_val is not None else self._enter_error
        if reason is not None:
            self._logger.info("run exit: aborting this frame (%s) — echo is not declared here", reason)
            self._thinking.abort(reason)
        elif self._need_switch_thinking_effort is not None:
            # 等身体跑完再收线: 记账工具不能比它所作用的世界先结束.
            self._logger.info("run exit: draining actions before declaring the effort switch")
            await self._thinking.wait_actions_done()
            self._logger.info("run exit: actions drained — pushing effort %s", self._need_switch_thinking_effort)
            # 帧结束前最后写下的两件事: 推档位 (官方 selectModel), 再补 need_observe 回声驱动
            # 下一帧. 入口写下的标记会被这一轮任何一次 observe() 换容器带走, 只有这里写才算数.
            await self._push_effort_officially(self._need_switch_thinking_effort)
            self._thinking.observer.add_echoes([], need_observe=True)
            moment = self._thinking.moment
            self._logger.info(
                "run exit: echo declared — need_observe=%s moment_index=%s attention=%s "
                "previous_moment_id=%s inputs=%s percepts=%s",
                self._thinking.observer.need_observe(),
                moment.index,
                self._thinking.attention.id,
                moment.last_moment_id(),
                len(list(moment.inputs_messages())),
                len(list(moment.percepts_messages())),
            )
        # 无论走哪条路都清掉: 错误 / 取消路径不下发档位, 也不留一个悬着的声明.
        self._need_switch_thinking_effort = None
        return None

    # ── event stream ─────────────────────────────────────────────────

    async def _events(self) -> "AsyncGenerator[SessionEvent]":
        """Pull raw session events from the queue. The poison pill raises _enter_error; the normal path ends via logos() aclose."""
        while True:
            item = await self._queue.get()
            if item is _POISON:
                if self._enter_error is not None:
                    raise self._enter_error
                return
            yield item

    async def _handle_tool_use_event(self, event: ToolCallEvent, streams: dict[str, _StreamedCtml]) -> None:
        """tool/call dispatch — discriminate by name and route to the typed tool.

        The two streaming tools (interpret / react) have their CTML already arriving via the delta
        stream (see logos()); their handlers flush the reconciliation tail and settle the articulator.
        The rest run_tool → ToolCallResult, returned via tool-result RPC.
        """
        if not event.name.startswith(_DOLORES_TOOL_NAMESPACE_PREFIX):
            # 不是我们的 tool: 静默略过 —— host 内环工具由 dsh 自己结算, 我们连痕都不必留.
            return
        result = await InterpretToolCall.run_tool(
            event, lambda call: self._handle_interpret(call, streams), logger=self._logger,
        )
        if result is not None:
            await self._void_stream(event, streams)
            await self._dispatch_tool_result(result)
            return
        result = await ReactToolCall.run_tool(
            event, lambda call: self._handle_react(call, streams), logger=self._logger,
        )
        if result is not None:
            await self._void_stream(event, streams)
            await self._dispatch_tool_result(result)
            return
        result = await ObserveToolCall.run_tool(event, self._handle_observe, logger=self._logger)
        if result is not None:
            await self._dispatch_tool_result(result)
            return
        result = await WaitNextToolCall.run_tool(event, self._handle_wait_next, logger=self._logger)
        if result is not None:
            await self._dispatch_tool_result(result)
            return
        result = await ShellStatusToolCall.run_tool(event, self._handle_shell_status, logger=self._logger)
        if result is not None:
            await self._dispatch_tool_result(result)
            return
        result = await ChannelFacadeToolCall.run_tool(event, self._handle_channel_facade, logger=self._logger)
        if result is not None:
            await self._dispatch_tool_result(result)
            return
        result = await ReasoningToolCall.run_tool(event, self._handle_reasoning, logger=self._logger)
        if result is not None:
            await self._dispatch_tool_result(result)
            return
        # 名字是我们的 (moss*), 却没有任何 handler 认领 — nothing is dispatched, so the call never gets
        # an answer
        # (the plugin can only settle it when the turn ends) and the log says nothing at all. That is
        # D39's shape again: an action is promised and no mechanism performs it, silently. Leave a
        # trace, or this exists only as "that call felt like it never happened" on the model side.
        self._logger.warning(
            "tool call %s (%s) matched no dolores tool — dropped unanswered",
            event.callId, event.name,
        )

    async def _void_stream(self, event: ToolCallEvent, streams: dict[str, _StreamedCtml]) -> None:
        """handler 没认领这次流式 tool/call —— 当场作废 delta 开出来的那条 CTML 流.

        ``pop`` 到东西时只有一种可能: 参数没解析通过 (``run_tool`` 的 ValidationError 分支),
        handler 根本没被调用。handler 正常跑完时 ``_handle_interpret`` / ``_handle_react`` 已经把
        stream 取走了, 这里 pop 不到, 是 no-op。

        不作废的后果 (2026-10-13 线上): articulator 永不 commit → mindflow 的 action loop 停在
        那条 action 上 → 下一条 ``moss_interpret`` 的 handler 永久阻塞 → ``logos()`` 停止消费
        session 事件 → turn/end 收不到 → 整帧冻结, 只能重启 host。
        """
        stream = streams.pop(event.callId, None)
        if stream is None:
            return
        self._logger.info(
            "streaming tool call %s (%s) never reached its handler — voiding its CTML stream %s "
            "(the arguments were rejected, so what it decoded must not execute)",
            event.callId, event.name, stream.call_id,
        )
        void = _VoidCtmlError(f"{event.name} {event.callId}: arguments rejected")
        await stream.__aexit__(type(void), void, None)

    async def _handle_interpret(
            self,
            call: InterpretToolCall,
            streams: dict[str, _StreamedCtml],
    ) -> ToolCallResult:
        """moss_interpret handler — the CTML already streamed into its own articulator; settle it.

        The stream owns the articulator's async-with boundary (see ``_StreamedCtml``); this handler
        flushes the reconciliation tail and waits for **observed** — every ``always_observe`` command
        finished, but not all actions (non-observe commands keep running cross-frame). Then it signs
        the freshest moment and hands its ref back with the result. An InterpretError surfaces from
        the commit, caught here rather than by a bare ``wait_compiled``.
        """
        stream = streams.pop(call.tool_call_event.callId, None)
        if stream is None:
            stream = _StreamedCtml(call.tool_call_event.callId, self._thinking.articulator())
        if stream.stream.failed:
            # the arguments stream was corrupt (a malformed escape) — nothing to trust, cut the turn.
            # The model sees "ctml parse error" but no interpreter ever saw this CTML, so the shell's
            # own ERROR line never happens either: without this the log is empty on a cut turn.
            self._logger.warning(
                "CTML argument stream for call %s was corrupt (%s chars decoded) — turn cut",
                stream.call_id, len(stream.stream.value),
            )
            await stream.__aexit__(None, None, None)
            return ToolCallResult(
                call=call.tool_call_event,
                result="ctml parse error",
                cancel=True,
            )
        decoded = stream.stream.value
        tail = call.ctml[len(decoded):]
        try:
            await stream.finish(tail=tail, wait="observed")
        except InterpretError:
            return ToolCallResult(
                call=call.tool_call_event,
                result="ctml syntax error",
                cancel=True,
            )
        # The observed commands' results land as task-done events the moment each task finishes (see
        # ctml_shell's add_done_callback), so observing now carries them; non-observe commands keep
        # running cross-frame by the clear_after_exit=False protocol.
        #
        # Refresh the warm layer *before* signing the moment: the commands just awaited may have
        # changed channel state (a resolve, a commit), and metas are otherwise refreshed only at the
        # next interpreter dispatch — so without this the moment carries the world as it was *before*
        # the action and the model perceives its own effect one frame late. Same step moss_observe
        # already takes.
        await self._facade.shell.refresh_metas(timeout=5.0, stale_time=1.0)
        moment = self._thinking.observe()
        moment_ref = f"{self._thinking.observer.epoch.index}-{moment.index}"
        return ToolCallResult(
            call=call.tool_call_event,
            result={"moment_ref": moment_ref},
            moment=moment,
        )

    async def _handle_react(
            self,
            call: ReactToolCall,
            streams: dict[str, _StreamedCtml],
    ) -> ToolCallResult:
        """moss_react handler — settle compile, wait for the actions, then cut the turn.

        Shares the streamed parse with ``moss_interpret`` but signs no moment. It waits for **compiled**
        (the CTML is valid), then ``wait_actions_done`` so the reaction's commands actually finish
        before the turn is cut — without it, cancel=True cuts the turn with the react's own actions
        still in flight.
        """
        stream = streams.pop(call.tool_call_event.callId, None)
        if stream is None:
            stream = _StreamedCtml(call.tool_call_event.callId, self._thinking.articulator())
        if stream.stream.failed:
            self._logger.warning(
                "CTML argument stream for call %s was corrupt (%s chars decoded) — turn cut",
                stream.call_id, len(stream.stream.value),
            )
            await stream.__aexit__(None, None, None)
            return ToolCallResult(
                call=call.tool_call_event,
                result="ctml parse error",
                cancel=True,
            )
        decoded = stream.stream.value
        tail = call.ctml[len(decoded):]
        try:
            await stream.finish(tail=tail, wait="compiled")
        except InterpretError:
            return ToolCallResult(
                call=call.tool_call_event,
                result="ctml syntax error",
                cancel=True,
            )
        await self._thinking.wait_actions_done()
        return ToolCallResult(
            call=call.tool_call_event,
            result="reacted",
            cancel=True,
        )

    async def _handle_observe(self, call: ObserveToolCall) -> ToolCallResult:
        """moss_observe handler — (optionally) interrupt, wait for all actions to finish, refresh
        metas, then observe the freshest moment.

        ``interrupt=true`` first replans with an empty 'clear' interpreter to cancel the current plan,
        waits for that cancellation to land (``wait_action_done`` on the replan articulator — the same
        as ``thinking.wait_actions_done``), then waits for all actions. Then refresh metas (awaited —
        the moment's facade must be fresh) and observe.
        """
        if call.interrupt:
            async with self._thinking.articulator(replan=True) as articulator:
                await articulator.wait_action_done()
        await self._thinking.wait_actions_done()
        await self._facade.shell.refresh_metas(timeout=5.0, stale_time=1.0)
        moment = self._thinking.observe()
        moment_ref = f"{self._thinking.observer.epoch.index}-{moment.index}"
        return ToolCallResult(
            call=call.tool_call_event,
            result={"moment_ref": moment_ref},
            moment=moment,
        )

    async def _handle_shell_status(self, call: ShellStatusToolCall) -> str:
        """shell_status handler — observe the Shell running status now; returns the status description, produces no moment."""
        return self._facade.status().description()

    async def _handle_channel_facade(self, call: ChannelFacadeToolCall) -> str:
        """moss_channel_facade handler — recursive lists channels under the path prefix; otherwise one channel's full surface."""
        if not call.recursive:
            text = self._facade.get_channel_full_facade(call.channel_path)
            return text or f"no such channel: {call.channel_path!r}"
        from ghoshell_common.helpers import yaml_pretty_dump

        prefix = call.channel_path
        kv = {
            path: meta.description or "(no desc)"
            for path, meta in self._facade.channel_metas(available_only=True).items()
            if path.startswith(prefix)
        }
        return yaml_pretty_dump(kv)

    async def _handle_wait_next(self, call: WaitNextToolCall) -> ToolCallResult:
        """moss_wait_next handler — wait for all actions to finish, then yield the turn.

        cancel=True: this result is what replaces the final answer. The plugin cuts the turn the moment
        it lands, so no further step runs; the turn/end that follows is what ends this thinking
        transaction (MOSS never cancels the turn itself).
        """
        await self._thinking.wait_actions_done()
        return ToolCallResult(
            call=call.tool_call_event,
            result="yielded",
            cancel=True,
        )

    async def _handle_reasoning(self, call: ReasoningToolCall) -> ToolCallResult:
        """moss_reasoning handler — mark a thinking-depth switch, then end this turn.

        Marks it, and does nothing else: the depth is pushed and the next frame is driven at run
        exit, never here (see ``__aexit__``). Returns cancel=True, so the result replaces the final
        answer and no further step of this turn runs.

        The marker carries a value, not a flag: 'which depth to switch to' is state, and state
        survives the ``observe()`` calls this turn does on its way out.

        ``add_echoes`` lives on the frame's **observer** (a ``Moments``), not on ``Thinking`` — the
        latter has no such method. Calling it on the thinking raised AttributeError, which vanished
        into ``run_tool``'s error path (result=None, cancel dropped), which is how this tool silently
        stopped ending turns. Guarded by ``test_tool_surface_matches_production_thinking``.
        """
        # default_thinking_effort 是插件那条私有路 (下一帧 enter 携带这个值).
        self._ego.default_thinking_effort = call.effort
        self._need_switch_thinking_effort = call.effort
        return ToolCallResult(
            call=call.tool_call_event,
            result={"effort": call.effort},
            cancel=True,
        )

    async def _push_effort_officially(self, effort: str) -> None:
        """Declare the depth a second time, through DSH's official ``session/selectModel``.

        **Why two paths, for one stage.** The plugin's private pending (ego declaration →
        ``thinking/enter``'s effort → ``declareEffort`` → request-header injection) writes the
        *agent's* selection. The DSH session projection never sees that, so the web UI cannot
        show it — the human changes a depth and the interface stays on the old one. That is
        exactly the report this change answers.

        ``session/selectModel`` is the mechanism the UI itself uses (``/model`` and the
        composer both go through it), so the projection (``lastUsed`` / ``pending``) moves and
        the UI syncs. Both paths carry the *same* value, so a frame in which only one has
        landed is still correct — but they must not both stay: once this one is verified live,
        the private path should be deleted, not kept as a fallback.

        **Known upstream side effect** (measured 2026-10-15): ``selectModel`` also rewrites the
        *default* model profile, in the background and in submission order. So a depth declared
        here becomes the default for sessions created later. That is dsh's behaviour, not ours,
        and the request carries no session-only switch — recorded here so the next reader does
        not mistake it for something we did.

        **Never fatal.** This rides a bookkeeping tool: a failed RPC must not break the turn, so
        every failure degrades to the plugin path and says so in the log.
        """
        if not effort:
            return
        try:
            session = self._ego.session
        except RuntimeError:
            return  # not started yet — nothing to steer

        route = self._ego.model_route
        if route is None:
            # No request observed yet, so the session has no route of its own to preserve;
            # the catalog default is the only thing on offer and it is what this session
            # will start from anyway. Safe *only* here — see model_route's docstring.
            try:
                current = await session.model_selection()
                route = (current.provider, current.model)
            except Exception:
                self._logger.warning(
                    "moss_reasoning: no provider/model to name for session/selectModel — "
                    "effort went out on the plugin path only, so the web UI will not show it"
                )
                return

        provider, model = route
        try:
            await session.select_model(provider=provider, model=model, reasoning_effort=effort)
        except Exception:
            self._logger.exception(
                "moss_reasoning: session/selectModel failed for effort=%s — the effort still went "
                "out on the plugin path, but the web UI will not show this change",
                effort,
            )

    async def _dispatch_tool_result(self, result: ToolCallResult) -> None:
        """Return a ToolCallResult to the plugin via tool-result RPC: result unlocks the tool, moment injects.

        moment_id is taken explicitly from result["moment_ref"] (fetch_next_moment's structured result), not re-derived.

        The moment travels in the *same* ``{context, inputs, moment_id}`` protocol that ``thinking/enter``
        uses. It used to travel as a context-only slice here, which was a silent loss channel: the moment
        has already drained its percepts, so dropping ``inputs`` destroyed them — a buffered impulse riding
        the next observation could be eaten by a mid-thought ``moss_interpret`` and never reach the model.

        A failed RPC is **absorbed, never fatal**: the plugin settles pending tools itself when a thinking turn
        ends (interrupted result + settled-call tombstone), so a late result for an already-settled call is a
        normal race, not an error. Letting it propagate would kill the whole logos() stream — i.e. one late
        moment would burn the entire turn. Drop the result and keep thinking.
        """
        moment_payload = None
        if result.moment is not None and isinstance(result.result, dict):
            moment_ref = result.result.get("moment_ref")
            if moment_ref is not None:
                moment_payload = self._ego.moment_payload(result.moment, moment_ref)
        if result.cancel:
            # 这一轮的 turn/end 会由 plugin 在下一个 pre-step 切出来 (cancel_turn): 记下 turn 号,
            # 让 _note_turn_end 能把它与"外部打断"区分开.
            self._expected_cancel_turn = result.call.turn
        try:
            await self._ego.rpc_tool_result(
                result.call.callId, result.result, moment_payload, cancel=result.cancel,
                turn=result.call.turn,
            )
        except Exception:
            self._logger.warning(
                "tool-result RPC dropped (call %s already settled plugin-side); result discarded",
                result.call.callId,
            )

    async def logos(self) -> "AsyncIterator[Union[GhostEvent, str]]":
        """Consume the event stream, dispatch tool calls, forward debug events, end on turn/end.

        The final answer is plain text that is never parsed into CTML — the ghost acts through tool
        calls. ``moss_interpret`` and ``moss_react`` are the two streaming tools: their
        ``tool-call-delta`` chunks are fed into a per-call articulator as they arrive, so the CTML
        reaches the shell while the model is still generating. Single consumption: a run has at most
        one logos stream. Ending (turn/end) is internal — the consumer needs no break.

        Yields one ``GhostEvent`` per dsh event in ``_FORWARDED_DSH_EVENTS``, or logos delta in arrival order. The
        raw envelope rides in ``payload`` untouched (``to_dict()``), so a dsh upgrade that adds
        envelope fields reaches the surface without changing this code.
        """
        if self._logos_started:
            raise RuntimeError("DoloresRun.logos() can only be consumed once")
        self._logos_started = True
        events = self._events()
        streams: dict[str, _StreamedCtml] = {}
        try:
            while True:
                event = await anext(events)
                # 转发先于 turn/end 判定: 那个判定会让本函数 return, turn/end 必须在它之前 yield 出去.
                if event.meta.type in _FORWARDED_DSH_EVENTS:
                    yield GhostEvent(
                        event=f"{_DSH_EVENT_NAME_PREFIX}{event.meta.type}",
                        payload=event.to_dict(),
                    )
                if self._note_turn_end(event):
                    return
                if chunk := AssistantChunk.from_session_event(event):
                    logos_delta = await self._handle_ctml_delta(chunk.chunk, streams)
                    if logos_delta is not None:
                        yield logos_delta
                    continue
                if tool := ToolCallEvent.from_session_event(event):
                    await self._handle_tool_use_event(tool, streams)
        finally:
            await events.aclose()
            await self._close_leftover_streams(streams)

    async def _close_leftover_streams(self, streams: dict[str, _StreamedCtml]) -> None:
        """Settle CTML streams the turn never finished — a stream opened by a delta whose tool/call
        event never arrived (interrupted generation, a cut turn).

        Such a stream is a real failure mode that used to be perfectly silent: the articulator's
        boundary was opened (``__aenter__``) and never closed, so nothing committed, nothing settled,
        nothing logged. Closing is the correct cleanup — the turn is over, no delta can still feed it —
        and an interrupted turn (abort) legitimately leaves a stream behind, so this is INFO, not a
        warning: the log line aids debugging a cut turn, not alarming.

        Closing goes through the **interrupted** exit, never the normal commit-and-wait one: nobody
        will feed this region any more, so waiting for its compile node would wait on a value another
        task is supposed to produce. The close runs in ``logos()``'s ``finally``, so a close that
        never returns wedges the whole run rather than just this stream.
        """
        leftover = list(streams.values())
        streams.clear()
        for stream in leftover:
            self._logger.info(
                "CTML stream %s never settled (no tool/call arrived for it) — closing its articulator",
                stream.call_id,
            )
            try:
                # 按中断退出关边界, 而不是走正常收线: 这条 region 被这一轮丢下了 (tool/call 永不会到),
                # 没有人会再喂它 —— 正常路径会 commit 后等编译位, 而那是别处在生产的节点, 等下去
                # 这次关闭就永不返回; 它在 logos 的 finally 里, 拖住的是整条消费链.
                await stream.__aexit__(
                    ActionExitedException,
                    ActionExitedException(f"leftover CTML stream {stream.call_id} abandoned by a cut turn"),
                    None,
                )
            except Exception:
                self._logger.exception("closing leftover CTML stream %s failed", stream.call_id)

    async def _handle_ctml_delta(
            self, chunk, streams: dict[str, _StreamedCtml]) -> _LogosDelta | None:
        """tool-call-delta → feed a streaming tool's argument stream into its own articulator.

        The tool name rides on the first delta; later deltas carry only the call id + argumentsDelta,
        so we key by call id and create the stream lazily on the first sight of a streaming tool's name
        (``_STREAMING_TOOLS``). The decoded CTML is not yielded to the broadcast — execution goes
        through the articulator.
        """
        if chunk.type != 'tool-call-delta':
            return
        call_id = chunk.id
        if call_id is None:
            return
        stream = streams.get(call_id)
        if stream is None:
            if chunk.name not in _STREAMING_TOOLS:
                return
            stream = _StreamedCtml(call_id, self._thinking.articulator())
            streams[call_id] = stream
        return await stream.feed(chunk.argumentsDelta)

    def _note_turn_end(self, event: SessionEvent) -> bool:
        """turn/end → 收线本帧 (返回该 event 是否为 turn/end); 中断类在入队前 abort.

        本帧的退出始终走自然路径 —— logos() 在此 return → run 收线 → articulate 返回 → 帧自己结束;
        但"这次收线是自然还是被打断"必须分类, 见 ``_turn_end_abort_reason``:

        - 自然 (completed / max-tokens): 什么都不做, attention 留着 (need_observe 的回声帧循环
          ``while not attention.is_aborted() and need_observe()`` 要求它活着, 自续帧靠它).
        - 中断 (其它 cause): abort 并带 reason, 级联到 attention, 回声帧循环随之停止 (人喊停 / 故障
          都该是"整个体系停"), 而 reason 落到 mindflow 的 thinking 退出点, 由那里急停身体
          (``mindflow_in_shell._run_thinking`` 的 finally). **这一步不在这里做** —— 见 _predispatch_abort.
        - 我们自己的 hook cancel 例外 (见 ``_turn_end_abort_reason``): 那一类已经 drain 过.

        reason 是调用方自己的话 (对齐 ``Future.cancel(msg)``), 不当词汇表用; 日志里只带 kind (+ cause).
        """
        end = TurnEnd.from_session_event(event)
        if end is None:
            return False
        if not self._turn_end_noted:
            self._turn_end_noted = True
            cause = end.reason.reason
            self._logger.info(
                "turn %s ended: %s",
                end.turn,
                end.reason.kind if cause is None else f"{end.reason.kind}/{cause.kind}",
            )
        # 中断类本该早在入队时落地 (_predispatch_abort); 这里兜一次, 防有人在队列上直投事件.
        self._predispatch_abort(event)
        return True

    # ── internals ────────────────────────────────────────────────────

    async def _on_event(self, event: "SessionEvent") -> None:
        """catch-all callback: 入队前处理, 再入队 (async-wrapped — the on_session_event consumer awaits)."""
        self._predispatch_abort(event)
        self._queue.put_nowait(event)

    def _predispatch_abort(self, event: "SessionEvent") -> None:
        """turn/end 的中断类在**入队前**落地 (同步, 幂等).

        不能等消费: 消费是单线程顺序的 (logos() 里 ``await self._handle_tool_use_event(...)``),
        一个 tool handler 可能正占着循环 —— ``_handle_react`` 的 ``wait_actions_done()`` 就是在等身体
        把动作跑完, **而人点停要停的正是那具身体**. 2026-10-04 D04 实测: dsh 侧已把 moss_react 正确
        中断, 我们这边 turn/end 却排在队列里 41 秒, 等语音自己播完才被消费 —— 那时它已经没有可停的
        东西了. ``Attention.abort`` 是同步立即生效的, 所以这里能直接调; abort 级联下去, 那个 handler
        也很快返回 (动作被取消 → wait_actions_done 满足).
        """
        if self._interrupt_aborted:
            return
        end = TurnEnd.from_session_event(event)
        if end is None:
            return
        reason = _turn_end_abort_reason(end)
        if reason is None:
            return
        cause = end.reason.reason
        if (self._expected_cancel_turn == end.turn
                and (cause is None or cause.kind not in _OUR_TURN_CANCEL_CAUSE_KINDS)):
            # 我们约好的切轮还没到, 别人先动手了 (人点了停 / 运行时故障) —— 值得单独留痕.
            self._logger.info(
                "turn %s was cut by %s while our own cancel_turn was still pending",
                end.turn, "no cause" if cause is None else cause.kind,
            )
        self._interrupt_aborted = True
        self._logger.info(
            "turn %s ended by interruption (%s) — aborting this thinking frame before the consumer runs",
            end.turn, reason,
        )
        self._thinking.abort(reason)

    async def _drive_enter(self) -> None:
        """enter task: thinking/enter RPC. Enqueues the poison pill only on error.

        The normal path never enqueues the pill — the turn is run by dsh, live-moment events stream in
        via the catch-all listener, and the consumer ends on turn/end. The pill only pins a terminal
        marker on enter error, so _events() can raise _enter_error to the consumer. Enqueuing it on the
        normal path would end the event stream before the model produces any logos — the enter RPC
        returns before the model has generated a frame.
        """
        try:
            await self._ego.enter_thinking(self._thinking, self._thinking_seq)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            # 失败经 poison pill 交给消费者, 一路都在"抛", 但没有任何一处写日志: 整轮被 abort,
            # 而日志里查不到原因. 与 exit_thinking 对称, 这里也留痕.
            self._logger.exception("thinking/enter failed — the turn will be aborted")
            self._enter_error = error
            self._queue.put_nowait(_POISON)
