from typing import Optional
from typing_extensions import Self

from pydantic import Field

from ghoshell_moss.message.abcd import ContentModel, DeltaModel, Delta

__all__ = ["FunctionCall", "FunctionOutput", "FunctionCallDelta"]


class FunctionCall(ContentModel):
    CONTENT_TYPE = "function_call"

    call_id: Optional[str] = Field(default=None, description="caller 的 id, 用来 match openai 的 tool call 协议. ")
    name: str = Field(description="方法的名字.")
    arguments: str = Field(description="方法的参数. ")

    @classmethod
    def from_delta(cls, delta: Delta | DeltaModel) -> Self | None:
        if isinstance(delta, Delta):
            model = FunctionCallDelta.from_delta(delta)
        else:
            model = delta

        if model and isinstance(model, FunctionCallDelta):
            return cls(
                call_id=model.call_id,
                name=model.name,
                arguments=model.arguments,
            )
        else:
            return None

    def buffer_delta(self, delta: Delta | DeltaModel) -> bool:
        if isinstance(delta, Delta):
            model = FunctionCallDelta.from_delta(delta)
        else:
            model = delta
        if model and isinstance(model, FunctionCallDelta):
            if model.call_id and model.call_id != self.call_id:
                return False
            if model.name and model.name != self.name:
                return False
            self.arguments += model.arguments
            return True
        return False


class FunctionOutput(ContentModel):
    CONTENT_TYPE = "function_output"

    call_id: Optional[str] = Field(default=None, description="caller 的 id, 用来 match openai 的 tool call 协议. ")
    name: Optional[str] = Field(default=None, description="方法的名字.")
    content: str = Field(default="", description="方法的返回值")

    @classmethod
    def from_delta(cls, delta: Delta | DeltaModel) -> Self | None:
        return None

    def buffer_delta(self, delta: Delta | DeltaModel) -> bool:
        return False


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
