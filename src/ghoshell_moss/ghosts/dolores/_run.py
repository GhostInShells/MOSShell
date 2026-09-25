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
- _events(): queue consumption; the poison pill carries only the enter error (the normal path ends via
  turn/end break in logos()).
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable
from typing_extensions import Self
from ghoshell_moss.core.blueprint.ghost import GhostEvent
from ghoshell_moss.core.blueprint.mindflow import Thinking
from ghoshell_moss.core.concepts.errors import InterpretError
from ghoshell_moss.contracts.logger import get_moss_logger
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

_logger = get_moss_logger()

# 流式 tool 集合: 这两个 tool 的参数经 tool-call-delta 逐字进 articulator, 边生成边执行.
_STREAMING_TOOLS = frozenset({InterpretToolCall.tool_name(), ReactToolCall.tool_name()})

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

# turn/end 的 reason.kind 只用于观测记录 (debug 时能看出这轮怎么收的), 不驱动任何中断.
# 曾经按 kind 调 Thinking.abort 来"退帧 + 停身体", 那是错的: abort 会冒泡到 attention,
# 把 attention 一起杀掉, 而 need_observe 驱动的回声帧循环要求 attention 活着 —— 自续帧因此丢失.


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

    async def feed(self, delta: str | None) -> None:
        if not self._entered:
            await self.__aenter__()
        text = self.stream.add(delta)
        if text:
            await self.articulator.send(text)

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
    ) -> None:
        self._ego = ego
        self._thinking = thinking
        self._facade = facade
        self._queue: "asyncio.Queue[Any]" = asyncio.Queue()
        self._dispose_listener: "Callable[[], None] | None" = None
        self._enter_task: "asyncio.Task[None] | None" = None
        self._enter_error: Exception | None = None
        self._thinking_event: asyncio.Event = thinking_event
        # logos() single-consumption guard — a run has at most one logos stream (more would split the queue).
        self._logos_started = False
        # turn/end 只结算一次: 文本循环内与循环外各有一个 call site, 同一 event 会被看见两次.
        self._turn_end_noted = False

    # ── transaction boundary ─────────────────────────────────────────

    async def __aenter__(self) -> Self:
        """Open the transaction. Bind the listener first (so no enter broadcast is lost), then start the enter task."""
        self._thinking_event.set()
        self._dispose_listener = self._ego.session.on_session_event("*", self._on_event)
        self._enter_task = asyncio.create_task(self._drive_enter())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close the transaction. Cancel the enter task → unbind → re-send exit → abort (on error)."""
        self._thinking_event.clear()
        task = self._enter_task
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        if self._dispose_listener is not None:
            self._dispose_listener()
        # re-send exit — even if enter failed (to clean up plugin-side state), blocking with a fail-safe timeout.
        await self._ego.exit_thinking()
        if isinstance(exc_val, asyncio.CancelledError):
            return None
        reason = exc_val if exc_val is not None else self._enter_error
        if reason is not None:
            self._thinking.abort(reason)
        return None

    # ── event stream ─────────────────────────────────────────────────

    async def _events(self) -> "AsyncIterator[SessionEvent]":
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
        result = await InterpretToolCall.run_tool(
            event, lambda call: self._handle_interpret(call, streams),
        )
        if result is not None:
            await self._dispatch_tool_result(result)
            return
        result = await ReactToolCall.run_tool(
            event, lambda call: self._handle_react(call, streams),
        )
        if result is not None:
            await self._dispatch_tool_result(result)
            return
        result = await ObserveToolCall.run_tool(event, self._handle_observe)
        if result is not None:
            await self._dispatch_tool_result(result)
            return
        result = await WaitNextToolCall.run_tool(event, self._handle_wait_next)
        if result is not None:
            await self._dispatch_tool_result(result)
            return
        result = await ShellStatusToolCall.run_tool(event, self._handle_shell_status)
        if result is not None:
            await self._dispatch_tool_result(result)
            return
        result = await ChannelFacadeToolCall.run_tool(event, self._handle_channel_facade)
        if result is not None:
            await self._dispatch_tool_result(result)
            return
        result = await ReasoningToolCall.run_tool(event, self._handle_reasoning)
        if result is not None:
            await self._dispatch_tool_result(result)
            return

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
        """moss_react handler — fire-and-await: settle compile only, then cut the turn.

        Shares the streamed parse with ``moss_interpret`` but waits only for **compiled** — no moment
        is signed, no actions awaited. The CTML's commands keep running cross-frame.
        """
        stream = streams.pop(call.tool_call_event.callId, None)
        if stream is None:
            stream = _StreamedCtml(call.tool_call_event.callId, self._thinking.articulator())
        if stream.stream.failed:
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
        """moss_reasoning handler — declare a one-shot thinking depth, then sign a new thinking frame.

        Sets the ego's unconsumed default effort (consumed on the next enter), marks need_observe so
        the next thinking frame starts now, and cuts the current turn. The depth applies exactly once
        and never fights the dsh/UI-held depth afterwards.
        """
        self._ego.default_thinking_effort = call.effort
        self._thinking.add_echoes(observe=True)
        return ToolCallResult(
            call=call.tool_call_event,
            result={"effort": call.effort},
            cancel=True,
        )

    async def _dispatch_tool_result(self, result: ToolCallResult) -> None:
        """Return a ToolCallResult to the plugin via tool-result RPC: result unlocks the tool, moment injects.

        moment_id is taken explicitly from result["moment_ref"] (fetch_next_moment's structured result), not re-derived.

        A failed RPC is **absorbed, never fatal**: the plugin settles pending tools itself when a thinking turn
        ends (interrupted result + settled-call tombstone), so a late result for an already-settled call is a
        normal race, not an error. Letting it propagate would kill the whole logos() stream — i.e. one late
        moment would burn the entire turn. Drop the result and keep thinking.
        """
        moment_parts = None
        if result.moment is not None and isinstance(result.result, dict):
            moment_ref = result.result.get("moment_ref")
            if moment_ref is not None:
                moment_parts = self._ego.moment_context_parts(result.moment, moment_ref)
        try:
            await self._ego.rpc_tool_result(
                result.call.callId, result.result, moment_parts, cancel=result.cancel,
                turn=result.call.turn,
            )
        except Exception:
            _logger.warning(
                "tool-result RPC dropped (call %s already settled plugin-side); result discarded",
                result.call.callId,
            )

    async def logos(self) -> "AsyncIterator[GhostEvent]":
        """Consume the event stream, dispatch tool calls, forward debug events, end on turn/end.

        The final answer is plain text that is never parsed into CTML — the ghost acts through tool
        calls. ``moss_interpret`` and ``moss_react`` are the two streaming tools: their
        ``tool-call-delta`` chunks are fed into a per-call articulator as they arrive, so the CTML
        reaches the shell while the model is still generating. Single consumption: a run has at most
        one logos stream. Ending (turn/end) is internal — the consumer needs no break.

        Yields one ``GhostEvent`` per dsh event in ``_FORWARDED_DSH_EVENTS``, in arrival order. The
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
                    await self._handle_ctml_delta(chunk.chunk, streams)
                    continue
                if tool := ToolCallEvent.from_session_event(event):
                    await self._handle_tool_use_event(tool, streams)
        finally:
            await events.aclose()

    async def _handle_ctml_delta(self, chunk, streams: dict[str, _StreamedCtml]) -> None:
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
        await stream.feed(chunk.argumentsDelta)

    def _note_turn_end(self, event: SessionEvent) -> bool:
        """turn/end → 收线本帧; 返回该 event 是否为 turn/end.

        只退出当前 thinking 帧, **绝不冒泡到 attention**: 本帧的退出走自然路径 —— logos()
        在此返回 → run 收线 → articulate 返回 → 帧自己结束. 冒泡 (``Thinking.abort``) 会把
        attention 一起杀掉, 而 need_observe 驱动的回声帧循环
        (``while not attention.is_aborted() and need_observe()``) 要求 attention 活着, 自续帧
        因此丢失. run 退出时同时 cancel dsh —— 双向对齐, 看谁先到, 不互相反复 cancel.

        reason 只做观测记录 (debug 时能看出这轮怎么收的), 原样带 kind (+ cause), 不归一成散文.
        """
        end = TurnEnd.from_session_event(event)
        if end is None:
            return False
        if not self._turn_end_noted:
            self._turn_end_noted = True
            cause = end.reason.reason
            _logger.info(
                "turn %s ended: %s",
                end.turn,
                end.reason.kind if cause is None else f"{end.reason.kind}/{cause.kind}",
            )
        return True

    # ── internals ────────────────────────────────────────────────────

    async def _on_event(self, event: "SessionEvent") -> None:
        """catch-all callback: enqueue (async-wrapped — the on_session_event consumer awaits)."""
        self._queue.put_nowait(event)

    async def _drive_enter(self) -> None:
        """enter task: thinking/enter RPC. Enqueues the poison pill only on error.

        The normal path never enqueues the pill — the turn is run by dsh, live-moment events stream in
        via the catch-all listener, and the consumer ends on turn/end. The pill only pins a terminal
        marker on enter error, so _events() can raise _enter_error to the consumer. Enqueuing it on the
        normal path would end the event stream before the model produces any logos — the enter RPC
        returns before the model has generated a frame.
        """
        try:
            await self._ego.enter_thinking(self._thinking)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._enter_error = error
            self._queue.put_nowait(_POISON)
