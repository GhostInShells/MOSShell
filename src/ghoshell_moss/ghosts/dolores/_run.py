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
from ghoshell_moss.contracts.logger import get_moss_logger
from ghoshell_moss.deepseek_harness.types.session_events import SessionEvent, ToolCallEvent, TurnEnd

from ._tools import (
    CtmlAppendToolCall,
    WaitNextMomentToolCall,
    ObserveStatusToolCall,
    ReasoningToolCall,
    ChannelsToolCall,
    ChannelFacadeToolCall,
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

    async def _handle_tool_use_event(self, event: ToolCallEvent) -> None:
        """tool/call dispatch — discriminate by name and route to the typed tool.

        ctml_append / wait_next_moment / observe_status → run_tool produces a ToolCallResult,
        returned via tool-result RPC. moss_reasoning (declaration) → records the default effort on
        the ego (applied next round), no tool-result.
        """
        result = await CtmlAppendToolCall.run_tool(event, self._handle_ctml_append)
        if result is not None:
            await self._dispatch_tool_result(result)
            return
        result = await WaitNextMomentToolCall.run_tool(event, self._handle_wait_next_moment)
        if result is not None:
            await self._dispatch_tool_result(result)
            return
        result = await ObserveStatusToolCall.run_tool(event, self._handle_observe_status)
        if result is not None:
            await self._dispatch_tool_result(result)
            return
        result = await ChannelsToolCall.run_tool(event, self._handle_channels)
        if result is not None:
            await self._dispatch_tool_result(result)
            return
        result = await ChannelFacadeToolCall.run_tool(event, self._handle_channel_facade)
        if result is not None:
            await self._dispatch_tool_result(result)
            return
        if (call := ReasoningToolCall.from_tool_call(event)) is not None:
            self._ego.default_effort = call.effort

    async def _handle_ctml_append(self, call: CtmlAppendToolCall) -> ToolCallResult | str:
        """ctml_append handler — append CTML mid-thought (segmented tool call).

        wait_done=False → wait only for compile and return the Shell status (no moment).
        wait_done=True → wait for the actions to finish, observe the freshest moment, return
        {moment_ref} and inject its context.
        """
        async with self._thinking.articulator(replan=call.replan) as articulator:
            await articulator.send(call.ctml)
            if call.wait_done:
                await articulator.wait_action_done()
                await self._facade.shell.refresh_metas(timeout=5.0, stale_time=1.0)
                moment = self._thinking.observe()
                moment_ref = f"{self._thinking.observer.epoch.index}-{moment.index}"
                return ToolCallResult(
                    call=call.tool_call_event,
                    result={"moment_ref": moment_ref},
                    moment=moment,
                )
            await articulator.wait_compiled()
            return self._facade.status().description()

    async def _handle_observe_status(self, call: ObserveStatusToolCall) -> str:
        """observe_status handler — observe Shell running status for replan; returns the status description, produces no moment."""
        return self._facade.status().description()

    async def _handle_channels(self, call: ChannelsToolCall) -> str:
        """moss_channels handler — the channel catalog (path → description), for discovery or debugging."""
        return self._facade.channels_description()

    async def _handle_channel_facade(self, call: ChannelFacadeToolCall) -> str:
        """moss_channel_facade handler — one channel's full operating surface; unknown path is reported, not raised."""
        text = self._facade.get_channel_full_facade(call.path)
        if not text:
            return f"no such channel: {call.path!r}"
        return text

    async def _handle_wait_next_moment(self, call: WaitNextMomentToolCall) -> str:
        """wait_next_moment handler — wait for all actions to finish, then yield the turn.

        Returns "yielded": the plugin cancels the turn after this tool returns, so the next moment
        wakes the ghost (nothing more to output in a voice/body interaction).
        """
        await self._thinking.wait_actions_done()
        return "yielded"

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
            await self._ego.rpc_tool_result(result.call.callId, result.result, moment_parts)
        except Exception:
            _logger.warning(
                "tool-result RPC dropped (call %s already settled plugin-side); result discarded",
                result.call.callId,
            )

    async def logos(self) -> "AsyncIterator[str]":
        """Consume the event stream, dispatch tool calls, end on turn/end.

        Plain text is not governed here: the final answer is plain text that is never parsed into
        CTML — the ghost acts through tool calls (moss_ctml_append etc.), not through the stream.
        Single consumption: a run has at most one logos stream. Ending (turn/end) is internal —
        the consumer needs no break.
        """
        if self._logos_started:
            raise RuntimeError("DoloresRun.logos() can only be consumed once")
        self._logos_started = True
        events = self._events()
        try:
            while True:
                event = await anext(events)
                if self._note_turn_end(event):
                    return
                if tool := ToolCallEvent.from_session_event(event):
                    await self._handle_tool_use_event(tool)
        finally:
            await events.aclose()
        if False:  # 保持 async generator 语法: plain text 放弃治理后无实际 logos.
            yield ""

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
