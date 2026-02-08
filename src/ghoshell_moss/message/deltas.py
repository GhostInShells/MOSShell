from typing import Optional

from pydantic import Field

from .abcd import DeltaModel

__all__ = ["TextDelta"]


class TextDelta(DeltaModel):
    DELTA_TYPE = "text"

    content: str = Field(
        default="",
        description="The text of the delta",
    )


class FunctionCallDelta(DeltaModel):
    """
    function call 协议.
    """

    DELTA_TYPE = "function_call"

    call_id: Optional[str] = Field(default=None, description="caller 的 id, 用来 match openai 的 tool call 协议. ")
    name: str = Field(description="方法的名字.")
    arguments: str = Field(description="方法的参数. ")

    @classmethod
    def keyword(cls) -> str:
        return "function_call"
