from typing import Generic, TypeVar, Tuple, Type
from abc import ABC, abstractmethod
from typing_extensions import Self
from pydantic import BaseModel, Field
from ghoshell_moss.core.concepts.command import CommandMeta, Command
from anthropic.types import ToolParam
from anthropic.types import Message


class ToolMeta(BaseModel):
    """
    兼容工具调用的元信息描述.
    """

    name: str
    description: str
    strict: bool = Field(
        default=True,
        description="whether the tool is strictly or not",
    )
    parameters: dict = Field(
        description="the parameters json schema of the tool",
    )

    @classmethod
    def from_command_meta(cls, command_meta: CommandMeta, chan: str = "", *, strict: bool = False) -> Self | None:
        if command_meta.args_schema is None:
            return None
        name = Command.make_uniquename(chan, command_meta.name)
        return cls(
            name=name,
            description=command_meta.description,
            strict=strict,
            parameters=command_meta.args_schema,
        )

    def to_ai_function(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "strict": self.strict,
                "parameters": self.parameters,
            },
        }

    def to_openai_function_def(self) -> dict:
        from openai.types.shared_params import FunctionDefinition

        return FunctionDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
            strict=self.strict,
        )

    def to_anthropic_tool(self) -> ToolParam:
        return ToolParam(
            input_schema=self.parameters,
            name=self.name,
            description=self.description,
            allowed_callers=["direct"],
            defer_loading=True,
        )


R = TypeVar("R", bound=ToolMeta)


class Tool(Generic[R], ABC):
    """
    兼容工具调用.
    """

    @abstractmethod
    def meta(self) -> ToolMeta:
        """
        meta info about the tool.
        """
        pass

    @abstractmethod
    async def call(self, parameters: dict, *, call_id: str | None = None) -> R:
        """
        call and get result.
        :param parameters: the parameters match the parameters json schema of the tool meta
        :param call_id: id of the call
        """
        pass

    @abstractmethod
    async def call_for_messages(self, parameters: dict, *, call_id: str | None = None) -> list[Message]:
        """
        call and get message as result.
        """
        pass
