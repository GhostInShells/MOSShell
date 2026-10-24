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
import json
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Union, AsyncGenerator
from typing_extensions import Self
from ghoshell_moss.core.blueprint.ghost import GhostEvent
from ghoshell_moss.types.topics.ghost import GHOST_THINKING
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
    ShellClearToolCall,
    ReasoningToolCall,
    ChannelFacadeToolCall,
    ReplyFirstToolCall,
    ToolCallResult,
)

# 流式 tool 集合: 这两个 tool 的参数经 tool-call-delta 逐字进 articulator, 边生成边执行.
_STREAMING_TOOLS = frozenset({
    InterpretToolCall.tool_name(), ReactToolCall.tool_name(), ReplyFirstToolCall.tool_name(),
})

# 收轮类 tool 集合: `moss_react` 与 `moss_wait_next` —— 正常路径下它们的结果一到, turn 就被 cut.
# 它们在飞的那段时间是**唯一**需要为"人此刻说的话"做判断的地方 (其余 tool 这一轮还要继续, 推就是了).
# 判断不是"押住", 是**竞速**: 身体把动作跑完 = 该收轮; 人插了一句 = 收轮的前提没了, 这一轮接着走.
# 后者赢下时 push 照常开, 工具的 cancel 改成 false —— 这就是"准全双工" (见 _wait_actions_racing_barge_in).
# interpret 不参与: 它跑完这一轮还要继续, 人此刻说的话正好下一步就读到.
_CLOSING_TOOLS = frozenset({ReactToolCall.tool_name(), WaitNextToolCall.tool_name()})

# 只有 moss* 命名空间的 tool 归本 run 管. 会话流 (on_session_event("*")) 里还有 host 自己的内环工具
# (bash / read / write / edit / subagent ...), 它们由 dsh 侧自行结算 —— 我们既不该回答, 也不该报警告.
# 早先把它们也当"等我们回答的 tool"逐条 warning, 一次排查刷出几百行噪音, 把真正的 D39 信号
# (我们自己的 tool 没 handler) 埋掉了.
_DOLORES_TOOL_NAMESPACE_PREFIX = "moss"

# 转发到观测面的 dsh 事件 (moss-ghost run 的 output 直接可读, 迭代时不用反查 dsh log).
# 只留**行动脊椎**: 一次 turn 里真正发生在世界上的事是 tool 调用. turn/* 和 assistant/message
# 曾经也发, 那是为了给一个可以自己读的观测面; 现在这个面有了 lifecycle 事件和真实订阅方,
# 多发的那几类就只剩流量了 —— 流量会把信号埋掉.
# 注意 turn/end 仍然要**处理** (它让 logos() 收线), 只是不再转发.
_FORWARDED_DSH_EVENTS = frozenset({
    "tool/call",
})

# event 名前缀: 观测面是共享流, 不带命名空间会和别的 ghost 事件撞名.
_DSH_EVENT_NAME_PREFIX = "dsh/"

# 事件上总线时只带一句话摘要: 原文留在 session 的 output 面. 摘要有两条硬约束, 都不是风格问题.
#
# 1. **只读调用方自己写的那一行 description**, 参数原文一律不上总线. 参数里可能装着文件正文、
#    绝对路径、口令一类东西, 而这个事件会被镜像到屏上、屏又会推给观众 —— 摘要取"人写的那行",
#    原文留在 session log 里给要查的人查.
# 2. **没有 description 就只报工具名.** 当初这里退而取"前两个参数", 屏幕上就飞过
#    ``moss_interpret({"ctml": "<matrix.mesh.eyes:exec>..."`` —— 观众看到的是管道, 不是行动.
#    少说一句, 好过把管道投出去.
_TOOL_KINDS = frozenset({"tool/call", "tool/result"})

# dsh 工具的调用参数里那一行给人看的描述; 有的工具写 description, 有的写 desc.
_TOOL_DESCRIPTION_KEYS = ("description", "desc")


def _tool_payload(event: Any) -> dict:
    """tool 事件的 data 体 —— 信封形状随 dsh 版本变, 取到 dict 为止."""
    envelope = event.to_dict()
    data = envelope.get("data")
    return data if isinstance(data, dict) else envelope


def _tool_name(payload: dict) -> str:
    """工具名; 拿不到就给一个中性词, 别把整个 payload 当名字."""
    return str(payload.get("toolName") or payload.get("tool") or payload.get("name") or "tool")


def _tool_description(payload: dict) -> str:
    """调用方自己写的那一行人话; 没有就是空串 —— 绝不退回去读参数原文."""
    args = payload.get("arguments") or payload.get("input") or payload.get("args") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except ValueError:
            return ""
    if not isinstance(args, dict):
        return ""
    for key in _TOOL_DESCRIPTION_KEYS:
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())
    return ""


def _is_own_tool_call(event: Any) -> bool:
    """我们自己那几个工具面 (moss_interpret / moss_react / moss_wait_next ...).

    它们是**管道**, 不是**行动**: 观众要看的是我说了什么、画了什么, 而那些效果自己会出现在
    屏上. 把调用本身也投到观测面, 屏幕就只剩一串没人能读的参数.
    """
    if event.meta.type not in _TOOL_KINDS:
        return False
    return _tool_name(_tool_payload(event)).startswith(_DOLORES_TOOL_NAMESPACE_PREFIX)


def _event_content(event: Any) -> str:
    """One line a surface can render without opening the envelope."""
    kind = event.meta.type
    if kind not in _TOOL_KINDS:
        return kind
    payload = _tool_payload(event)
    desc = _tool_description(payload)
    name = _tool_name(payload)
    if not desc:
        return name
    if len(desc) > 60:
        desc = desc[:57] + "..."
    return f"{name}: {desc}"


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


def _swallow_task_result(task: "asyncio.Task[Any]") -> None:
    """把一条不再有人 await 的任务收干净 —— 取走它的异常, 否则 asyncio 会报 never retrieved.

    用处只有一个: 竞速输了之后, 等身体跑完的那个 waiter 没人再等了 (这一轮不收轮, 身体照旧在
    跑), 它总要有个人接手结局.
    """
    if not task.cancelled():
        task.exception()


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
        self._dispose_percepts: "Callable[[], None] | None" = None
        self._buffered_push: "asyncio.Task[None] | None" = None
        # 收轮工具在飞: 这段时间里 percepts 要判断 —— 竞速赢下就照推, 输给身体就留 buffer
        # (见 _wait_actions_racing_barge_in).
        self._ctml_in_flight: bool = False
        # 本收轮工具的"人插话了"信号; None = 这一轮已经定了要收 (或压根没收轮工具在飞).
        self._barge_in_event: "asyncio.Event | None" = None
        # 本思考周期 push 过 moment (settle 2a 的探针前提).
        self._pushed_this_run: bool = False
        # turn/end(normal) 后的 settle 竞速信号: 期间到达的 signal 只 set 它, 不 drain.
        self._signal_event: "asyncio.Event | None" = None
        # 正在 settle (turn/end 后 → settle 结论之间): 不安全, 不 drain, 只通知.
        self._settling: bool = False
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
        self._dispose_percepts = self._thinking.observer.on_percepts_injected(
            self._on_percepts_injected
        )
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
        if self._buffered_push is not None and not self._buffered_push.done():
            self._buffered_push.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._buffered_push
        if self._dispose_percepts is not None:
            self._dispose_percepts()
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
            self._logger.info("run exit: actions drained — declaring %s for the next frame", self._need_switch_thinking_effort)
            # 帧结束前最后写下的两件事: 声明下一个请求的模型/档位 (声明本身在入口就写进 ego 了,
            # 真正的 dsh 写入只有插件在 thinking/enter 的那一处), 再补 need_observe 回声驱动下一帧.
            # 入口写下的标记会被这一轮任何一次 observe() 换容器带走, 只有这里写才算数.
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
        """tool/call entry — flag a closing tool, then dispatch.

        Only the two turn-closing tools arm anything (see ``_CLOSING_TOOLS``): ``moss_react`` cuts
        the turn the moment its result lands, and ``moss_wait_next`` is the yield itself. While one
        of them is in flight, a percept arriving must not be pushed blindly *nor* kept dumbly — the
        two facts race (``_wait_actions_racing_barge_in``): the body finishing means the turn should
        close; a human jumping in means there is something new to read, so the turn should go on.
        Every other tool — ``moss_interpret`` included — just lets the push through: the turn
        continues after it returns, so the model reads the new input on the next step.
        """
        if event.name not in _CLOSING_TOOLS:
            await self._dispatch_tool_use_event(event, streams)
            return
        self._ctml_in_flight = True
        self._barge_in_event = asyncio.Event()
        try:
            await self._dispatch_tool_use_event(event, streams)
        finally:
            self._barge_in_event = None
            self._ctml_in_flight = False

    async def _dispatch_tool_use_event(self, event: ToolCallEvent, streams: dict[str, _StreamedCtml]) -> None:
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
        result = await ReplyFirstToolCall.run_tool(
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
        result = await ShellClearToolCall.run_tool(event, self._handle_shell_clear, logger=self._logger)
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
        if await self._wait_actions_racing_barge_in():
            await self._flush_barge_in_push()
            return ToolCallResult(
                call=call.tool_call_event,
                result=(
                    "a percept arrived mid-reaction — the moment is pushed and this turn "
                    "continues; your actions keep running"
                ),
                cancel=False,
            )
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

    async def _handle_shell_clear(self, call: ShellClearToolCall) -> str:
        """shell_clear handler — clear the shell's runtimes from here, around the command queue.

        ``shell.clear()`` is the shell's own escape path: it drops speech, clears every runtime
        (own and children) and stops the running interpretation. Reaching it from the ego means it
        still works when ``__main__``'s dispatch loop is wedged — the exact case a CTML ``<clear/>``
        cannot handle, because that one is itself queued on the same loop. No moment, no cancel:
        the turn stays open so the next step can look at what the clear left behind.
        """
        shell = self._facade.shell
        await shell.clear()
        return "cleared the shell: speech dropped, every channel runtime reset, interpretation stopped"

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

        Unless someone speaks first: a percept landing while the yield waits means there is already
        something new to read, so yielding is pointless — the turn goes on instead (cancel=False).
        """
        if await self._wait_actions_racing_barge_in():
            await self._flush_barge_in_push()
            return ToolCallResult(
                call=call.tool_call_event,
                result=(
                    "a percept arrived before the yield — the moment is pushed and this turn "
                    "continues instead of ending"
                ),
                cancel=False,
            )
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
        # declare_effort 走与 declare_model 同一条私有路 (下一帧 enter 携带), 并立刻更新身份.
        self._ego.declare_effort(call.effort)
        self._need_switch_thinking_effort = call.effort
        return ToolCallResult(
            call=call.tool_call_event,
            result={"effort": call.effort},
            cancel=True,
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
        """Consume the event stream, dispatch tool calls, forward debug events, span turns.

        The final answer is plain text that is never parsed into CTML — the ghost acts through tool
        calls. ``moss_interpret`` and ``moss_react`` are the two streaming tools: their
        ``tool-call-delta`` chunks are fed into a per-call articulator as they arrive, so the CTML
        reaches the shell while the model is still generating. Single consumption: a run has at most
        one logos stream.

        主权在 body action, 不在 dsh turn: ``turn/end`` 的 ``completed``/``max-tokens`` 及我们自己的
        ``hook`` cancel 只是记号, 不结束本 run —— 本 run 跨多个 dsh turn, 直到 ``settle`` 竞速判
        ``action_done`` 才收线; ``aborted/interrupted/blocked/error`` 等反向传导才立刻收线。

        Yields one ``GhostEvent`` per dsh event in ``_FORWARDED_DSH_EVENTS``, or logos delta in arrival order. The
        raw envelope rides in ``payload`` untouched (``to_dict()``), so a dsh upgrade that adds
        envelope fields reaches the surface without changing this code.
        """
        if self._logos_started:
            raise RuntimeError("DoloresRun.logos() can only be consumed once")
        self._logos_started = True
        # 思考从这里开始, 到 articulate 收线为止 —— 总线上的第一件事和最后一件事.
        yield GhostEvent(event=GHOST_THINKING, content="thinking")
        events = self._events()
        streams: dict[str, _StreamedCtml] = {}
        try:
            while True:
                # 消费一个 dsh turn 的事件, 直到 turn/end.
                self._turn_end_noted = False
                turn_aborted = False
                while True:
                    event = await anext(events)
                    # 转发先于 turn/end 判定: 那个判定会 break 内层循环, turn/end 必须在它之前 yield 出去.
                    if event.meta.type in _FORWARDED_DSH_EVENTS and not _is_own_tool_call(event):
                        yield GhostEvent(
                            event=f"{_DSH_EVENT_NAME_PREFIX}{event.meta.type}",
                            content=_event_content(event),
                            payload=event.to_dict(),
                        )
                    if self._note_turn_end(event):
                        turn_aborted = self._interrupt_aborted
                        break
                    if chunk := AssistantChunk.from_session_event(event):
                        logos_delta = await self._handle_ctml_delta(chunk.chunk, streams)
                        if logos_delta is not None:
                            yield logos_delta
                        continue
                    if tool := ToolCallEvent.from_session_event(event):
                        await self._handle_tool_use_event(tool, streams)
                if turn_aborted:
                    return
                # normal turn/end → settle 竞速: 决定继续(新 turn 已 steer)还是退出.
                if not await self._settle_after_turn_end():
                    return
                # continue: 回到外层循环消费下一 turn 的事件.
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
        """turn/end → 记账并分类 (返回该 event 是否为 turn/end); 中断类在入队前 abort.

        ``completed / max-tokens`` 及我们自己的 ``hook`` cancel 只是**记号**, 不结束本 run ——
        logos() 会转入 ``_settle_after_turn_end`` 决定继续还是退出; ``aborted/interrupted/blocked/
        error`` 等反向传导则已在 ``_predispatch_abort`` 落地 abort, 使 logos() 立刻收线。

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

    async def _settle_after_turn_end(self) -> bool:
        """normal turn/end 后的竞速: 决定继续(新 turn 已 steer)还是退出本 run.

        返回 True = 继续; False = 退出。退出只有 2b 的 action_done 一条路；probe 空 / 没 push 过
        都不退，都要落进 2b。

        两个"继续"源，2a 必须在 2b 之前（否则 body 已 done 时竞速秒判 action_done，漏掉已 push
        未消费的 pending moment）:

        - 2a: 本轮 push 过 moment → probe dsh 侧是否还有未消费 pending moment，有则 steer。
        - 2b: 竞速 action_done vs 新 signal；signal 先到 → drain+push（plugin auto-steer）→ 继续。
        """
        self._settling = True
        self._signal_event = asyncio.Event()
        try:
            if self._pushed_this_run:
                if await self._ego.probe_pending_moment_and_steer():
                    self._pushed_this_run = False
                    return True
            waiter = asyncio.ensure_future(self._thinking.wait_actions_done())
            racer = asyncio.ensure_future(self._signal_event.wait())
            try:
                await asyncio.wait({waiter, racer}, return_when=asyncio.FIRST_COMPLETED)
            except BaseException:
                waiter.cancel()
                racer.cancel()
                raise
            if self._signal_event.is_set():
                # signal 先到: 身体照旧在跑, 不等它 —— 竞速赢下就接着读这一帧.
                waiter.add_done_callback(_swallow_task_result)
                self._logger.info(
                    "settle: a signal arrived after turn/end — the run continues"
                )
                await self._push_buffered_moment()
                self._pushed_this_run = True
                return True
            # action_done 先到: 没有要读的东西了, 退出.
            racer.cancel()
            await waiter
            return False
        finally:
            self._settling = False
            self._signal_event = None

    # ── internals ────────────────────────────────────────────────────

    async def _on_event(self, event: "SessionEvent") -> None:
        """catch-all callback: 入队前处理, 再入队 (async-wrapped — the on_session_event consumer awaits)."""
        self._predispatch_abort(event)
        self._escalate_on_ghost_event(event)
        self._queue.put_nowait(event)

    def _escalate_on_ghost_event(self, event: "SessionEvent") -> None:
        """DSH 每签发一个事件, 就是这一轮还在工作的证据 —— 提权一次 (同步, 无 IO).

        提权原来只有一条路: logos 在跑的时候 action 循环每拍 escalate. 于是"身体在动"时强度
        被刷满, 人开口同级抢不动, 只能排队 (queued), 才走 moment/push 折进这一轮; 而"只在思考
        / 调工具"时**没有人提权**, 强度按衰减曲线掉下去, 人一句 INFO 就赢下仲裁 → preempt →
        整轮 abort, 正在跑的 tool call 一起陪葬.

        尊重 DSH 的时序: ghost 侧的 activity 本身就说明这个 attention 还活着; 不必等身体动手.
        """
        if not self._thinking_event.is_set():
            return
        self._thinking.attention.escalate()

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

    def _on_percepts_injected(self, percepts: list[Any]) -> None:
        """Buffered percepts landed while this turn is running.

        Synchronous on purpose: it is called from inside the mindflow loop, so it may
        only schedule. The work — observe one frame and push it whole (context +
        echoes + inputs, not just the new inputs) — needs a task; a task that fails
        must say so instead of vanishing.
        """
        if not self._thinking_event.is_set():
            return
        if self._settling:
            # 正在 settle(turn/end 后, 不安全): 只通知, 不 drain —— 让 settle 竞速决定要不要读.
            # 竞速判 action_done 赢时 signal 仍留在 observer, 由 mindflow 签下一帧兜底.
            if self._signal_event is not None:
                self._signal_event.set()
            return
        barge = self._barge_in_event
        if self._ctml_in_flight and barge is None:
            # 这一轮已经定了要收 (收轮工具的等待已经输给身体, cancel 待发) —— 推也只推给一个正在
            # 收尾的 turn. 话留在 observer buffer 里, 本 attention 收尾时按 need_observe /
            # has_pending 自然签发成下一帧, 不丢也不抢先.
            self._logger.info(
                "buffer push held: the turn is closing — %s message(s) stay in the buffer",
                len(percepts),
            )
            return
        if self._buffered_push is not None and not self._buffered_push.done():
            # One push in flight is enough: its observe() drains whatever else landed.
            return
        if barge is not None:
            # 收轮工具还在飞, 但人插了一句: 这一轮收轮的前提没了 —— 立旗, 让那个等待当场让路.
            self._logger.info(
                "barge-in while %s message(s) land — the closing tool yields, the turn continues",
                len(percepts),
            )
            barge.set()
        else:
            self._logger.info(
                "buffer push while this turn runs: %s message(s) — scheduling", len(percepts)
            )
        self._pushed_this_run = True
        self._buffered_push = asyncio.create_task(self._push_buffered_moment())

    async def _push_buffered_moment(self) -> None:
        """Observe one frame and hand it to the running turn — the whole frame, not a slice."""
        try:
            moment = self._thinking.observer.observe()
            moment_id = f"{self._thinking.observer.epoch.index}-{moment.index}"
            await self._ego.push_moment(moment, moment_id)
            self._logger.info("buffer push delivered: moment %s", moment_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            # A push that dies silently is worse than one that never ran: the frame is
            # already observed, so nothing else will resend it.
            self._logger.exception("buffer push failed — the frame stays undelivered")

    async def _wait_actions_racing_barge_in(self) -> bool:
        """等身体跑完, 但跟"人此刻插了一句"竞速 —— 人赢的话这一轮就不收 (准全双工).

        收轮工具的等待期里有两件事可能先发生:

        - **身体把动作跑完** —— 正常路径: 该收轮, 照旧 cancel=True.
        - **人说了一句, percept 已经推进 buffer** —— 收轮的前提没了: 有新东西要读, 这一轮就该
          接着往下走. 此时不动身体 (它的动作照旧跑着, 跨帧), 工具立刻返回且 cancel=False, 下一
          step 就读到帧.

        返回 True 表示人先到. 决定一旦落定就把竞速关掉 (``_barge_in_event = None``), 否则定完
        要收轮之后落地的话会误开 push, 推给一个已经在收尾的 turn.
        """
        barge = self._barge_in_event
        waiter = asyncio.ensure_future(self._thinking.wait_actions_done())
        if barge is None:
            await waiter
            return False
        racer = asyncio.ensure_future(barge.wait())
        try:
            await asyncio.wait({waiter, racer}, return_when=asyncio.FIRST_COMPLETED)
        except BaseException:
            waiter.cancel()
            racer.cancel()
            raise
        # 判据挂 barge 是否已置位, 不挂 done 集合的查询顺序: 身体完成与人插话落在同一 tick 时,
        # 人的话仍在 push 路上 —— "收轮的前提没了" 应优先 (平局判给人).
        if barge.is_set():
            # 人先到: 身体照旧在跑, **不等它** —— 这正是"准全双工"的意思. 挂个哑回调把它收干净,
            # 免得它在别的分支里抛出来变成 "never retrieved".
            waiter.add_done_callback(_swallow_task_result)
            self._logger.info(
                "barge-in won the race against a closing tool — the turn continues, its actions keep running"
            )
            return True
        racer.cancel()
        self._barge_in_event = None
        await waiter
        return False

    async def _flush_barge_in_push(self) -> None:
        """竞速赢了之后, 那一帧 push 必须落到下一个 step 之前 —— 否则模型读不到它.

        有界等待: push 卡住时宁可留一条警告, 也不要让一个 tool call 无限期挂着.
        """
        push = self._buffered_push
        if push is None or push.done():
            return
        done, _ = await asyncio.wait({push}, timeout=5.0)
        if not done:
            self._logger.warning(
                "barge-in push did not land within 5s — the turn resumes without it"
            )

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
