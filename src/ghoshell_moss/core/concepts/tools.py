from typing import Generic, TypeVar, Callable
from abc import ABC
from typing_extensions import Self
from pydantic import BaseModel, Field
from ghoshell_moss.core.concepts.command import CommandMeta, Command, CommandTask, BaseCommandTask
from ghoshell_moss.message import Message
from openai.types.shared_params import FunctionDefinition
from anthropic.types import ToolParam
from pydantic_ai import Tool as PydanticTool, ToolReturn

CommandTaskCallback = Callable[[CommandTask], None]


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
    parameters: dict[str, object] = Field(
        description="the parameters json schema of the tool",
    )

    @classmethod
    def from_command_meta(cls, command_meta: CommandMeta, chan: str = "", *, strict: bool = False) -> Self | None:
        if command_meta.json_schema is None:
            return None
        name = Command.make_uniquename(chan, command_meta.name)
        return cls(
            name=name,
            description=command_meta.description,
            strict=strict,
            parameters=command_meta.json_schema,
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

    def to_openai_function_def(self) -> FunctionDefinition:
        """
        to openai function definition.
        """
        parameters = self.parameters.copy()
        return FunctionDefinition(
            name=self.name,
            description=self.description,
            parameters=parameters,
            strict=self.strict,
        )

    def to_anthropic_tool_param(self) -> ToolParam:
        return ToolParam(
            input_schema=self.parameters,
            name=self.name,
            description=self.description,
            allowed_callers=["direct"],
            defer_loading=True,
        )


R = TypeVar("R", bound=ToolMeta)


class CommandAsTool(Generic[R]):
    """
    Wrap Command as Tool
    """

    def __init__(
            self,
            command: Command[R],
            *,
            task_callback: CommandTaskCallback | None = None,
            channel_path: str = '',
    ):
        self.channel_path = channel_path
        self.command = command
        self.task_callback = task_callback

    def meta(self) -> ToolMeta:
        """
        meta info about the tool.
        """
        return ToolMeta.from_command_meta(self.command.meta())

    async def task_call(self, args: list, kwargs: dict, *, call_id: str | None = None) -> tuple[R, list[Message]]:
        """
        call and get result with result and messages

        :param args: the arguments of the tool
        :param kwargs: the keyword arguments of the tool
        :param call_id: id of the call
        """
        task = self.create_task(args, kwargs, call_id=call_id)
        if self.task_callback is not None:
            self.task_callback(task)
            await task.wait(throw=True)
        else:
            await task.run()
        r = task.result()
        messages = task.task_result().as_messages(with_serialized_result=False)
        return r, messages

    async def call(self, *args, **kwargs) -> R:
        """
        execute the command and get result
        """
        if self.task_callback is not None:
            task = self.create_task(args, kwargs)
            return await task
        else:
            return await self.command(*args, **kwargs)

    async def call_with_tool_return(self, *args, **kwargs) -> ToolReturn:
        """
        return pydantic tool return.
        """
        r, messages = await self.task_call(*args, **kwargs)
        content = None
        if len(messages) > 0:
            content = []
            for m in messages:
                content.extend(m.as_contents())
        return ToolReturn(return_value=r, content=content if len(content) > 0 else None)

    def create_task(self, args: list | tuple, kwargs: dict, *, call_id: str | None = None) -> CommandTask:
        """
        create task from the arguments and keyword arguments
        """
        task = BaseCommandTask.from_command(
            self.command,
            chan_=self.channel_path,
            args=args,
            kwargs=kwargs,
            call_id=call_id,
        )
        return task

    def as_pydantic_tool(self) -> PydanticTool:
        """
        adapt into pydantic tool
        """
        meta = self.command.meta()
        return PydanticTool.from_schema(
            self.call,
            name=Command.make_uniquename(self.channel_path, meta.name),
            description=meta.description,
            json_schema=meta.json_schema,
            takes_ctx=False,
            sequential=True,
        )
