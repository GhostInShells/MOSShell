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

yield tool (moss_wait_next_moment): when the consumer recognizes tool/call = moss_wait_next_moment it
breaks and sets run.yielded → on exit the plugin does not cancel (the tool stays pending, unlocked by
the next enter's moment). Moment production belongs to the normal mindflow loop, not the run.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable
from typing_extensions import Self
from ghoshell_moss.core.blueprint.mindflow import Thinking, Articulator
from ghoshell_moss.deepseek_harness.types.session_events import SessionEvent, ToolCallEvent, AssistantChunk

from ._tools import FetchNextMomentToolCall, WaitNextMomentToolCall, InterleavedCtmlToolCall, ToolCallResult

if TYPE_CHECKING:
    from ._ego import DoloresEgo
    from ghoshell_moss.core.blueprint.shell_trajectory import MShellContextFacade

__all__ = ["DoloresRun"]

# poison sentinel: enqueued on enter-task error; the consumer raises _enter_error when it reads it.
# never enqueued on the normal path — logos() ends itself on turn/end; the pill only carries enter errors.
_POISON = object()


def _get_text_chunk(event: SessionEvent) -> str | None:
    if assistant_chunk := AssistantChunk.from_session_event(event):
        if assistant_chunk.chunk.type == 'text-delta':
            return assistant_chunk.chunk.text
    return None


_LogosDelta = str


def _close_marker(open_marker: str) -> str:
    """Derive a wrap's close marker from its open marker: `<|X|>` → `</|X|>`."""
    return open_marker[:1] + '/' + open_marker[1:]


class _CtmlParser:
    """Split a text stream into logos (CTML) and plain (markdown) regions.

    CTML-first: the stream starts in logos. ``markdown_quoter`` (``<|Markdown|>``) opens a
    plain region and its close marker (``</|Markdown|>``) closes it — a proper wrap with an
    explicit close tag. Text inside the wrap is dropped; everything else is logos and is
    forwarded to the articulator.

    A partial marker is buffered across chunk boundaries; a partial match that fails is
    flushed as content (logos when inside logos, dropped otherwise).
    """

    def __init__(
            self,
            articulator: Articulator,
            markdown_quoter: str = '<|Markdown|>',
    ) -> None:
        self._articulator = articulator
        self._in_logos = True
        # markers that switch region: _to_plain_chars (seen in logos) opens the markdown
        # region; _to_logos_chars (seen in plain) is its close marker.
        self._to_plain_chars = list(markdown_quoter)
        self._to_logos_chars = list(_close_marker(markdown_quoter))
        self._buffer = ''

    async def add(self, text: str) -> _LogosDelta:
        logos = ''
        for char in text:
            delta = self._add_char(char)
            if delta is not None:
                logos += delta
        if logos:
            await self._articulator.send(logos)
        return logos

    def _add_char(self, char: str) -> _LogosDelta | None:
        marker_chars = self._to_plain_chars if self._in_logos else self._to_logos_chars
        if self._buffer == '':
            if marker_chars and char == marker_chars[0]:
                self._buffer = char
                return None
            return char if self._in_logos else None
        index = len(self._buffer)
        if index < len(marker_chars) and char == marker_chars[index]:
            self._buffer += char
            if len(self._buffer) == len(marker_chars):
                self._in_logos = not self._in_logos
                self._buffer = ''
            return None
        if self._in_logos:
            out = self._buffer + char
            self._buffer = ''
            return out
        self._buffer = ''
        return None

    async def __aenter__(self) -> Self:
        await self._articulator.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._in_logos and self._buffer:
            await self._articulator.send(self._buffer)
            self._buffer = ''

        await self._articulator.wait_action_done()
        await self._articulator.__aexit__(exc_type, exc_val, exc_tb)
        return None


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
        # yield marker: set True when the consumer recognizes tool/call == wait_next_moment and breaks;
        # __aexit__ passes it to exit_thinking(yielded=...) — never cancel on yield.
        self.yielded = False
        # logos() single-consumption guard — a run has at most one logos stream (more would split the queue).
        self._logos_started = False

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
        # yielded marker: whether this break is a yield; the plugin decides whether to cancel.
        await self._ego.exit_thinking(yielded=self.yielded)
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

        fetch_next_moment / interleaved_ctml → run_tool produces a ToolCallResult, returned via tool-result RPC.
        wait_next_moment (yield) → sets self.yielded (logos() breaks on it), no tool-result.
        """
        result = await FetchNextMomentToolCall.run_tool(event, self._handle_fetch_next_moment)
        if result is not None:
            await self._dispatch_tool_result(result)
            return
        result = await InterleavedCtmlToolCall.run_tool(event, self._handle_interleaved_ctml)
        if result is not None:
            await self._dispatch_tool_result(result)
            return
        if WaitNextMomentToolCall.from_tool_call(event) is not None:
            self.yielded = True

    async def _handle_fetch_next_moment(self, call: FetchNextMomentToolCall) -> ToolCallResult:
        """fetch_next_moment handler — produce a moment, return a structured {moment_ref} result, carry the moment."""
        if call.wait_actions_done:
            await self._thinking.wait_actions_done()
        if call.refresh_meta:
            await self._facade.shell.refresh_metas(timeout=5.0, stale_time=1.0)

        moment = self._thinking.observe()
        moment_ref = f"{self._thinking.observer.epoch.index}-{moment.index}"
        return ToolCallResult(
            call=call.tool_call_event,
            result={"moment_ref": moment_ref},
            moment=moment,
        )

    async def _handle_interleaved_ctml(self, call: InterleavedCtmlToolCall) -> str:
        """interleaved_ctml handler — emit CTML mid-thought, thinking ahead of behavior (interleaved).

        refresh_meta: refresh shell meta before execution.
        wait_done: true → wait_action_done, false → wait_compiled (thinking ahead).
        """
        if call.refresh_meta:
            await self._facade.shell.refresh_metas(timeout=5.0, stale_time=1.0)
        async with self._thinking.articulator(replan=False, wait_action_done=call.wait_done) as articulator:
            await articulator.send(call.ctml)
            if call.wait_done:
                await articulator.wait_action_done()
            else:
                await articulator.wait_compiled()
        return "ok"

    async def _dispatch_tool_result(self, result: ToolCallResult) -> None:
        """Return a ToolCallResult to the plugin via tool-result RPC: result unlocks the tool, moment injects.

        moment_id is taken explicitly from result["moment_ref"] (fetch_next_moment's structured result), not re-derived.
        """
        moment_parts = None
        if result.moment is not None and isinstance(result.result, dict):
            moment_ref = result.result.get("moment_ref")
            if moment_ref is not None:
                moment_parts = self._ego.moment_context_parts(result.moment, moment_ref)
        await self._ego.rpc_tool_result(result.call.callId, result.result, moment_parts)

    async def logos(self) -> "AsyncIterator[str]":
        """Consume the event stream, extract logos deltas (split by the <|CTML|> marker), end on turn/end.

        Single consumption: a run has at most one logos stream. Tool and other non-text events are
        left empty (see _handle_tool_use_event). Ending (turn/end) is internal here — the consumer
        needs no break.
        """
        if self._logos_started:
            raise RuntimeError("DoloresRun.logos() can only be consumed once")
        self._logos_started = True
        events = self._events()
        try:
            while True:
                event = await anext(events)
                text = _get_text_chunk(event)
                if text is not None:
                    parser = _CtmlParser(
                        self._thinking.articulator(replan=False, wait_action_done=True),
                    )
                    async with parser:
                        while text is not None:
                            delta = await parser.add(text)
                            if delta:
                                yield delta
                            event = await anext(events)
                            text = _get_text_chunk(event)
                # non-text event: tool left empty, turn/end ends the stream.
                if event.meta.type == "turn/end":
                    return
                if tool := ToolCallEvent.from_session_event(event):
                    await self._handle_tool_use_event(tool)
                    if self.yielded:
                        return
        finally:
            await events.aclose()

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
