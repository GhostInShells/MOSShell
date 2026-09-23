"""DoloresRun — the run object for a Dolores thinking transaction (async-with boundary + logos() stream).

``async with`` is the transaction boundary (enter opens / exit closes); logos() yields logos deltas
extracted from the raw session events:

    async with ego.run_thinking(thinking) as run:
        async for delta in run.logos():
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
    CtmlAppendToolCall,
    WaitActionDoneToolCall,
    WaitNextMomentToolCall,
    ShellStatusToolCall,
    ReasoningToolCall,
    ChannelFacadeToolCall,
    ReactToolCall,
    DefineReactsToolCall,
    ToolCallResult,
)

_logger = get_moss_logger()

if TYPE_CHECKING:
    from ._ego import DoloresEgo
    from ghoshell_moss.core.blueprint.shell_trajectory import MShellContextFacade

__all__ = ["DoloresRun"]

# poison sentinel: enqueued on enter-task error; the consumer raises _enter_error when it reads it.
# never enqueued on the normal path — logos() ends itself on turn/end; the pill only carries enter errors.
_POISON = object()

# turn/end 的 reason.kind → 是否打断本轮 thinking. aborted = turn 被外部掐掉 (人按停 / 新输入抢占 /
# 我们自己 exit); error / max-tokens = 没跑完. 三者都要退帧循环 + 停身体, 下一帧再拿到 <stop_reason>.
# interrupted **故意不在内**: dsh 已把 pending tool 结算成 interrupted, MOSS 照常轮转 —
# 它与 aborted 同属"没跑完", 不能顺手一起打断. blocked 尚未观测, 同 interrupted 处理.
_TURN_END_ABORT = frozenset({"aborted", "error", "max-tokens"})


class _StreamedCtml:
    """One ``moss_ctml_append`` call as it streams in: the argument decoder + its own articulator.

    The decoder reverses the JSON escaping of ``{"ctml": "…"}`` chunk by chunk; the decoded CTML is
    sent straight to the call's articulator, so the model's output reaches the shell while it is still
    being generated. ``stream.value`` is the exact prefix decoded so far — the handler reconciles
    against the parsed ``tool/call`` argument to cover a shape-mismatch / whole-argument fallback.
    """

    def __init__(self, call_id: str, articulator) -> None:
        self.call_id = call_id
        self.stream = CtmlArgumentStream()
        self.articulator = articulator

    async def feed(self, delta: str | None) -> None:
        text = self.stream.add(delta)
        if text:
            await self.articulator.send(text)


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

        ctml_append's CTML already arrived via the stream (see logos()); its handler only waits for
        compile and returns the outcome. wait_action_done / wait_next_moment / shell_status →
        run_tool produces a ToolCallResult, returned via tool-result RPC. moss_reasoning
        (declaration) → records the default effort on the ego (applied next round), no tool-result.
        """
        result = await CtmlAppendToolCall.run_tool(
            event, lambda call: self._handle_ctml_append(call, streams),
        )
        if result is not None:
            await self._dispatch_tool_result(result)
            return
        result = await WaitActionDoneToolCall.run_tool(event, self._handle_wait_action_done)
        if result is not None:
            await self._dispatch_tool_result(result)
            return
        result = await WaitNextMomentToolCall.run_tool(event, self._handle_wait_next_moment)
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
        result = await ReactToolCall.run_tool(event, self._handle_react)
        if result is not None:
            await self._dispatch_tool_result(result)
            return
        result = await DefineReactsToolCall.run_tool(event, self._handle_define_reacts)
        if result is not None:
            await self._dispatch_tool_result(result)
            return
        if (call := ReasoningToolCall.from_tool_call(event)) is not None:
            self._ego.default_effort = call.effort

    async def _handle_ctml_append(
            self,
            call: CtmlAppendToolCall,
            streams: dict[str, _StreamedCtml],
    ) -> ToolCallResult:
        """ctml_append handler — the CTML already streamed into its own articulator; wait for compile.

        ``wait_compiled(raise_interpret_error=True)`` turns an InterpretError into a one-line "ctml
        syntax error" result + cancel: continuing on a broken output is pointless, and the detail
        arrives in the next round's echoes. The stream's decoded prefix is reconciled against the
        parsed argument — a shape-mismatch or whole-argument call streamed nothing, so the missing
        tail is sent here as one shot.
        """
        stream = streams.pop(call.tool_call_event.callId, None)
        if stream is None:
            stream = _StreamedCtml(call.tool_call_event.callId, self._thinking.articulator())
        if stream.stream.failed:
            # the arguments stream was corrupt (a malformed escape) — nothing to trust, cut the turn.
            return ToolCallResult(
                call=call.tool_call_event,
                result="ctml parse error",
                cancel=True,
            )
        decoded = stream.stream.value
        if decoded != call.ctml:
            await stream.articulator.send(call.ctml[len(decoded):])
        try:
            await stream.articulator.wait_compiled(raise_interpret_error=True)
        except InterpretError:
            return ToolCallResult(
                call=call.tool_call_event,
                result="ctml syntax error",
                cancel=True,
            )
        return ToolCallResult(
            call=call.tool_call_event,
            result="compiled",
        )

    async def _handle_wait_action_done(self, call: WaitActionDoneToolCall) -> ToolCallResult:
        """wait_action_done handler — (optionally) replan, wait for the actions to finish, refresh
        metas, then observe the freshest moment.

        ``replan`` is None (just wait) or a CTML string that first replans — a fresh 'clear'
        interpreter with that CTML (empty string = replan with nothing). A replan CTML that fails to
        compile marks ``observe`` (so the next thinking frame carries the error echo) and cuts the
        turn with "ctml syntax error" + cancel. ``timeout`` is -1 (wait without bound) or a positive
        bound that gives up early but still observes.
        """
        if call.replan is not None:
            async with self._thinking.articulator(replan=True) as articulator:
                if call.replan:
                    await articulator.send(call.replan)
                try:
                    await articulator.wait_compiled(raise_interpret_error=True)
                except InterpretError:
                    # replan ctml failed to compile — issue the next thinking frame so the model sees
                    # the error in the next moment's echoes, then cut this turn.
                    self._thinking.add_echoes(observe=True)
                    return ToolCallResult(
                        call=call.tool_call_event,
                        result="ctml syntax error",
                        cancel=True,
                    )
        if call.timeout < 0:
            await self._thinking.wait_actions_done()
        else:
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._thinking.wait_actions_done(), timeout=call.timeout)
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

    async def _handle_define_reacts(self, call: DefineReactsToolCall) -> ToolCallResult:
        """moss_define_reacts handler — merge/overwrite reacts into the in-memory table."""
        store = self._ego.react_store
        if store is None:
            return ToolCallResult(call=call.tool_call_event, result="react unavailable")
        try:
            defined = store.define(call.reacts)
        except ValueError as error:
            return ToolCallResult(
                call=call.tool_call_event,
                result=f"react define error: {error}",
            )
        return ToolCallResult(call=call.tool_call_event, result={"defined": defined})

    async def _handle_react(self, call: ReactToolCall) -> ToolCallResult:
        """moss_react handler — resolve the char's template, substitute args, stream the CTML out.

        A fast reply: ``template % args`` becomes CTML, appended through a fresh articulator (one
        whole shot, not streamed deltas — unlike ctml_append). ``wait_next_moment`` waits for the
        actions to finish and cuts the turn (the say-then-done shape). The result is the resolved
        CTML itself, so the model can reconcile what it actually sent.
        """
        store = self._ego.react_store
        if store is None:
            return ToolCallResult(call=call.tool_call_event, result="react unavailable")
        try:
            ctml = store.render(call.char, call.args)
        except KeyError:
            return ToolCallResult(
                call=call.tool_call_event,
                result=f"no react defined for {call.char!r}",
            )
        except ValueError as error:
            return ToolCallResult(
                call=call.tool_call_event,
                result=f"react arg error: {error}",
            )
        articulator = self._thinking.articulator()
        await articulator.send(ctml)
        try:
            await articulator.wait_compiled(raise_interpret_error=True)
        except InterpretError:
            return ToolCallResult(
                call=call.tool_call_event,
                result="ctml syntax error",
                cancel=True,
            )
        if call.wait_next_moment:
            await self._thinking.wait_actions_done()
            return ToolCallResult(call=call.tool_call_event, result=ctml, cancel=True)
        return ToolCallResult(call=call.tool_call_event, result=ctml)

    async def _handle_wait_next_moment(self, call: WaitNextMomentToolCall) -> ToolCallResult:
        """wait_next_moment handler — wait for all actions to finish, then yield the turn.

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
            )
        except Exception:
            _logger.warning(
                "tool-result RPC dropped (call %s already settled plugin-side); result discarded",
                result.call.callId,
            )

    async def logos(self) -> "AsyncIterator[str]":
        """Consume the event stream, dispatch tool calls, end on turn/end.

        The final answer is plain text that is never parsed into CTML — the ghost acts through tool
        calls. ``moss_ctml_append`` is the one streaming tool: its ``tool-call-delta`` chunks are fed
        into a per-call articulator as they arrive, so the CTML reaches the shell while the model is
        still generating. Single consumption: a run has at most one logos stream. Ending (turn/end)
        is internal — the consumer needs no break.
        """
        if self._logos_started:
            raise RuntimeError("DoloresRun.logos() can only be consumed once")
        self._logos_started = True
        events = self._events()
        streams: dict[str, _StreamedCtml] = {}
        try:
            while True:
                event = await anext(events)
                if self._note_turn_end(event):
                    return
                if chunk := AssistantChunk.from_session_event(event):
                    await self._handle_ctml_delta(chunk.chunk, streams)
                    continue
                if tool := ToolCallEvent.from_session_event(event):
                    await self._handle_tool_use_event(tool, streams)
        finally:
            await events.aclose()
        if False:  # 保持 async generator 语法: plain text 放弃治理后无实际 logos.
            yield ""

    async def _handle_ctml_delta(self, chunk, streams: dict[str, _StreamedCtml]) -> None:
        """tool-call-delta → feed the ctml tool's argument stream into its own articulator.

        The tool name rides on the first delta; later deltas carry only the call id + argumentsDelta,
        so we key by call id and create the stream lazily on the first sight of the ctml tool's name.
        The decoded CTML is not yielded to the broadcast — execution goes through the articulator.
        """
        if chunk.type != 'tool-call-delta':
            return
        call_id = chunk.id
        if call_id is None:
            return
        stream = streams.get(call_id)
        if stream is None:
            if chunk.name != CtmlAppendToolCall.tool_name():
                return
            stream = _StreamedCtml(call_id, self._thinking.articulator())
            streams[call_id] = stream
        await stream.feed(chunk.argumentsDelta)

    def _note_turn_end(self, event: SessionEvent) -> bool:
        """turn/end → 按 reason.kind 决定是否打断本轮 thinking; 返回该 event 是否为 turn/end.

        aborted / error / max-tokens → abort: 帧循环退出 (attention abort), 身体停
        (action loop 的 _abort_clear 会 shell.clear), 下一帧经 previous.stop_reason 看到原因.
        completed → 正常收线; interrupted → 不 abort, 照常轮转.

        reason 字符串原样带 kind (+ cause), 归一成散文会让模型读不到是 error 还是 max-tokens.
        """
        end = TurnEnd.from_session_event(event)
        if end is None:
            return False
        if not self._turn_end_noted:
            self._turn_end_noted = True
            if end.reason.kind in _TURN_END_ABORT:
                cause = end.reason.reason
                self._thinking.abort(
                    end.reason.kind if cause is None else f"{end.reason.kind}/{cause.kind}"
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
