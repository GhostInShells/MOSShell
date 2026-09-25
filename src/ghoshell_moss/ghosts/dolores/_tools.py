"""Typed tool-call models for Dolores, aligned with the dsh plugin's defineTool.

Each tool class discriminates by name via its own ``from_tool_call(event)``: a name mismatch returns
None; a match parses ``json.loads(arguments)`` into strongly-typed fields. ``callId`` (dsh camelCase)
is moved to ``call_id`` (snake_case).

The fifth-round surface is seven tools (see ``dolores-tool-surface.md``): two streaming CTML tools
(``moss_interpret`` / ``moss_react``), one observe tool, one yield tool, and three non-waiting
self-inspection / declaration tools.
"""

from abc import ABC, abstractmethod
from typing import Optional, Callable, Awaitable
from pydantic import BaseModel, Field, ValidationError
from typing_extensions import Self

from ghoshell_moss.deepseek_harness.types.session_events import ToolCallEvent
from ghoshell_moss.core.blueprint.moment import Moment
from ghoshell_moss.contracts.logger import LoggerItf, get_moss_logger

__all__ = [
    "InterpretToolCall",
    "ReactToolCall",
    "ObserveToolCall",
    "WaitNextToolCall",
    "ShellStatusToolCall",
    "ReasoningToolCall",
    "ChannelFacadeToolCall",
]

_ResultType = dict | list | str | None


class ToolCallResult(BaseModel):
    """Raw data structure returned to the plugin via the tool-result RPC.

    Every tool that waits on MOSS returns through this protocol: the result unlocks the pending call,
    the moment (if any) is injected, and ``cancel`` optionally cuts the turn right after.
    """

    call: ToolCallEvent
    result: _ResultType = Field(
        description="the value returned to the model."
    )
    error: str | None = Field(
        default=None,
        description="parameter failure.",
    )
    moment: Optional[Moment] = Field(
        default=None,
        description=(
            "the moment carried back; after the plugin resolves the tool's call id, the moment is "
            "injected. Moment injection only goes through the <moment> (context) slot; inputs go "
            "through thinking/enter as steer."
        ),
    )
    cancel: bool = Field(
        default=False,
        description=(
            "when true, the plugin cancels the turn as soon as this result unlocks the call: the model "
            "still receives the full result, then the next step is cut. This is how a turn ends without "
            "a final answer (moss_wait_next / moss_react / moss_reasoning); the turn/end that follows is "
            "what ends the thinking transaction, so MOSS never cancels the turn on its own."
        ),
    )


class ToolCallParameter(BaseModel, ABC):
    tool_call_event: ToolCallEvent | None = Field(
        default=None,
        description="the original tool call; set after construction.",
    )

    @classmethod
    @abstractmethod
    def tool_name(cls) -> str:
        ...

    @classmethod
    def from_tool_call(cls, event: ToolCallEvent) -> Self | None:
        """Build from a tool-call event; None when the name doesn't match.

        :raise ValidationError: argument parsing failed.
        """
        if event.name != cls.tool_name():
            return None
        parameter = cls.model_validate_json(event.arguments or "{}")
        parameter.tool_call_event = event
        return parameter

    @classmethod
    async def run_tool(
            cls,
            event: ToolCallEvent,
            handler: Callable[[Self], Awaitable[_ResultType | ToolCallResult]],
            logger: LoggerItf | None = None,
    ) -> ToolCallResult | None:
        """Run one typed tool's handler and normalize the outcome.

        ``logger`` is the caller's logger — the one threaded down from matrix/shell through the ghost
        and the ego (see ``DoloresRun``). None is a last-resort fallback for tests; a module-level
        ``get_moss_logger()`` here would take every swallowed exception out of the node's log context.
        """
        log = logger or get_moss_logger()
        try:
            call = cls.from_tool_call(event)
            if call is None:
                return None
        except ValidationError as error:
            # The model sees this error string, but nothing was logged — a tool whose arguments do
            # not parse is a real failure and must leave a trace beyond the model's own surface.
            log.warning("tool %s arguments failed to parse (%s): %s", event.name, error, event.arguments)
            return ToolCallResult(
                call=event,
                result=None,
                error="invalid tool parameter",
            )

        try:
            result = await handler(call)
            if isinstance(result, ToolCallResult):
                return result
            return call.new_tool_call_result(result)

        except Exception as e:
            # 一次抛出 = 一行日志. 这里的静默曾经很贵: moss_reasoning 的生产路径抛
            # AttributeError (Thinking 上没有 add_echoes), 被吞成 result=None —— cancel 一起丢,
            # 本轮永不结束, 而模型只看到一个 null. 任何被吞掉的异常都必须留痕.
            log.exception("tool %s handler failed", event.name)
            return ToolCallResult(
                call=event,
                result=None,
                error=str(e),
            )

    def new_tool_call_result(self, result: _ResultType) -> ToolCallResult:
        return ToolCallResult(
            call=self.tool_call_event,
            result=result,
            # moment is attached by the caller.
            moment=None,
        )


class InterpretToolCall(ToolCallParameter):
    """moss_interpret — append CTML mid-thought, streamed into its own articulator, wait for observed.

    The single ``ctml`` argument is decoded from the tool-call delta stream (see _ctml_stream.py), so
    the CTML reaches the shell while the model is still generating. The handler waits for every
    ``always_observe`` command in the CTML to finish (not all actions — non-observe commands keep
    running cross-frame), then signs and returns the freshest moment. On an interpret error it returns
    "ctml syntax error" + cancel.
    """

    ctml: str = Field(default="", description="the CTML command to execute.")

    @classmethod
    def tool_name(cls) -> str:
        return "moss_interpret"


class ReactToolCall(ToolCallParameter):
    """moss_react — fire a fast reaction, streamed into its own articulator.

    The CTML streams in and compiles, then the actions it started run to completion, then the turn is
    cut (cancel). No moment is signed. The counterpart of ``moss_interpret`` (which waits for observed
    and signs a moment instead).
    """

    ctml: str = Field(default="", description="the CTML command to execute.")

    @classmethod
    def tool_name(cls) -> str:
        return "moss_react"


class ObserveToolCall(ToolCallParameter):
    """moss_observe — wait for all actions to finish, then sign the freshest moment.

    ``interrupt=true`` first replans with a fresh 'clear' interpreter (empty CTML) to cancel the
    current plan, waits for that cancellation to land, then waits for all actions; ``interrupt=false``
    just waits. Returns ``{moment_ref}`` and carries the moment for context injection.
    """

    interrupt: bool = Field(
        default=False,
        description="true cancels the current plan (a fresh 'clear' replan) before waiting.",
    )

    @classmethod
    def tool_name(cls) -> str:
        return "moss_observe"


class WaitNextToolCall(ToolCallParameter):
    """moss_wait_next — wait for all actions to finish, then yield the turn.

    Returns "yielded": the turn is cancelled after this tool returns, so the next moment wakes you.
    Use it in a voice/body interaction instead of emitting empty text nobody will read.
    """

    @classmethod
    def tool_name(cls) -> str:
        return "moss_wait_next"


class ShellStatusToolCall(ToolCallParameter):
    """moss_shell_status — observe the Shell running status now, for the thinking that precedes acting; returns the status description, produces no moment."""

    @classmethod
    def tool_name(cls) -> str:
        return "moss_shell_status"


class ChannelFacadeToolCall(ToolCallParameter):
    """moss_channel_facade — read the channel operating surface.

    ``recursive=True`` (default) lists every channel under ``channel_path`` (a prefix; empty = all),
    which replaces the former moss_channels; ``recursive=False`` reads one channel's full surface.
    Self-inspection: normally the surface is already in your context; reach for this when the surface
    changed underneath you or when you are debugging.
    """

    channel_path: str = Field(default="", description="the channel path (a prefix when recursive), e.g. 'ghost.frame'.")
    recursive: bool = Field(default=True, description="true lists channels under the path prefix; false reads one channel's full surface.")

    @classmethod
    def tool_name(cls) -> str:
        return "moss_channel_facade"


class ReasoningToolCall(ToolCallParameter):
    """moss_reasoning — declare your thinking depth (off/low/high/max).

    Records the declaration on the ego; the next thinking frame that is not already carrying its own
    effort hands it to the plugin, which writes it as a pending model selection. The depth applies
    from that turn on and stays in the durable request config until you or the UI change it again —
    it is a declaration, not a pin, so the session's canonical model selection stays authoritative.
    """

    effort: str = Field(default="", description="thinking depth: off/low/high/max.")

    @classmethod
    def tool_name(cls) -> str:
        return "moss_reasoning"
