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

__all__ = ["FetchNextMomentToolCall", "WaitNextMomentToolCall", "AppendCtmlToolCall"]

_ResultType = dict | list | str | None


class ToolCallResult(BaseModel):
    """Raw data structure returned to the plugin via the tool-result RPC.

    Maps into the plugin's tool-result interface. The yield tool (the dsh side yields the session
    awaiting the next moment) does not return through this protocol.
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


class FetchNextMomentToolCall(ToolCallParameter):
    """moss_fetch_next_moment — actively fetch the next frame: produce a moment, return {moment_ref} and inject context."""

    @classmethod
    def tool_name(cls) -> str:
        return "moss_fetch_next_moment"


class WaitNextMomentToolCall(ToolCallParameter):
    """moss_wait_next_moment (yield) — passively yield, block until the next moment.

    A control signal; produces no ToolCallResult (does not go through the tool-result RPC).
    """

    @classmethod
    def tool_name(cls) -> str:
        return "moss_wait_next_moment"


class AppendCtmlToolCall(ToolCallParameter):
    """moss_append_ctml — append CTML to execution, thinking ahead of behavior (interleaved).

    Produces no moment — producing a moment is fetch_next_moment's job; the two don't mix.
    """

    ctml: str = Field(default="", description="the CTML command to execute.")
    refresh_meta: bool = Field(default=False, description="refresh shell meta before execution.")
    wait_done: bool = Field(default=False, description="true waits for action done, false only for compile.")

    @classmethod
    def tool_name(cls) -> str:
        return "moss_append_ctml"
