"""Typed tool-call models for Dolores, aligned with the dsh plugin's defineTool.

Each tool class discriminates by name via its own ``from_tool_call(event)``: a name mismatch returns
None; a match parses ``json.loads(arguments)`` into strongly-typed fields. ``callId`` (dsh camelCase)
is moved to ``call_id`` (snake_case).
"""

from abc import ABC, abstractmethod
from typing import Optional, Callable, Awaitable
from pydantic import BaseModel, Field, ValidationError
from typing_extensions import Self

from ghoshell_moss.deepseek_harness.types.session_events import ToolCallEvent
from ghoshell_moss.core.blueprint.moment import Moment

from ._react import React

__all__ = [
    "CtmlAppendToolCall",
    "WaitActionDoneToolCall",
    "WaitNextMomentToolCall",
    "ShellStatusToolCall",
    "ReasoningToolCall",
    "ChannelFacadeToolCall",
    "ReactToolCall",
    "DefineReactsToolCall",
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
            "a final answer (wait_next_moment / react); the aborted turn/end is what ends the thinking "
            "transaction, so MOSS never cancels the turn on its own."
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
    ) -> ToolCallResult | None:
        try:
            call = cls.from_tool_call(event)
            if call is None:
                return None
        except ValidationError:
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


class CtmlAppendToolCall(ToolCallParameter):
    """moss_ctml_append — append CTML mid-thought, streamed into its own articulator.

    The single ``ctml`` argument is decoded from the tool-call delta stream (see _ctml_stream.py), so
    the CTML reaches the shell while the model is still generating. The handler only waits for compile;
    on an interpret error it returns "ctml syntax error" + cancel (detail arrives in the next echoes).
    """

    ctml: str = Field(default="", description="the CTML command to execute.")

    @classmethod
    def tool_name(cls) -> str:
        return "moss_ctml_append"


class WaitActionDoneToolCall(ToolCallParameter):
    """moss_wait_action_done — wait for all actions to finish, then observe the freshest moment.

    ``replan`` is None (just wait) or a CTML string that first replans (a fresh 'clear' interpreter
    replaces the plan) — an empty string replans with nothing, a non-empty one replans with that CTML.
    ``timeout=-1`` waits without bound; a positive timeout gives up early. Returns ``{moment_ref}``
    and carries the moment for context injection.
    """

    replan: str | None = Field(
        default=None,
        description="CTML to replan with before waiting; None means no replan (an empty string replans with nothing).",
    )
    timeout: float = Field(default=-1, description="seconds to wait; -1 waits without bound.")

    @classmethod
    def tool_name(cls) -> str:
        return "moss_wait_action_done"


class WaitNextMomentToolCall(ToolCallParameter):
    """moss_wait_next_moment — wait for all actions to finish, then yield the turn.

    Returns "yielded": the turn is cancelled after this tool returns, so the next moment wakes you.
    Use it in a voice/body interaction instead of emitting empty text nobody will read.
    """

    @classmethod
    def tool_name(cls) -> str:
        return "moss_wait_next_moment"


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
    """moss_reasoning — declare the default thinking depth (off/low/high/max).

    Pure declaration: the ego records it as its default effort and carries it on the next round's
    thinking/enter (reasoning_effort), applied at the turn boundary — not perStep. Produces no
    ToolCallResult (the plugin tool returns immediately).
    """

    effort: str = Field(default="", description="thinking depth: off/low/high/max.")

    @classmethod
    def tool_name(cls) -> str:
        return "moss_reasoning"


class ReactToolCall(ToolCallParameter):
    """moss_react — fire a runtime-defined react.

    ``char`` keys a react defined via ``moss_define_reacts`` (char → CTML template). ``args`` fills
    the template's ``%s`` slots to form the CTML, which executes. ``wait_next_moment=True`` (default)
    waits for it to finish, then ends the turn — say it, then done.
    """

    char: str = Field(description="the single-character react key.")
    args: list[str] | None = Field(
        default=None,
        description="positional args filling the template's %s slots, in order.",
    )
    wait_next_moment: bool = Field(
        default=True,
        description="wait for the CTML to finish, then end the turn.",
    )

    @classmethod
    def tool_name(cls) -> str:
        return "moss_react"


class DefineReactsToolCall(ToolCallParameter):
    """moss_define_reacts — bulk define reacts (char → CTML template with %s slots).

    Merge/overwrite, in-memory only. Returns the chars defined — the model defines what a scenario
    needs, when it needs it; no seed, no notice, no persistence.
    """

    reacts: list[React] = Field(description="list of {char, template}.")

    @classmethod
    def tool_name(cls) -> str:
        return "moss_define_reacts"
