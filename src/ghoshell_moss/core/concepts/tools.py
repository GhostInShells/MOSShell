"""
Wrap the MOSS command system as tools for common Agent frameworks, for compatibility.
"""

import typing
from typing import Any, Generic, TypeVar, Callable
from typing_extensions import Self
from pydantic import BaseModel, Field
from ghoshell_moss.core.concepts.command import CommandMeta, Command, CommandTask, BaseCommandTask
from ghoshell_moss.message import Message

if typing.TYPE_CHECKING:
    # support but not depend
    from openai.types.shared_params import FunctionDefinition
    from anthropic.types import ToolParam
    from pydantic_ai import Tool as PydanticTool, ToolReturn

CommandTaskCallback = Callable[[CommandTask], None]


def simplify_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline every ``$ref`` and drop presentation noise — a schema a prompt can carry.

    Pydantic renders every nested model and every enum as a ``$ref`` into ``$defs``. That is
    right for a validator and wrong for a model-facing schema: a reader should not have to
    resolve references before seeing the shape. This walks the document once, replaces each
    reference with its target (the ``$ref``'s siblings, e.g. ``default`` or ``description``,
    merge over it), and removes ``title`` — pydantic writes one on every node and it says
    nothing the property name does not already say.

    Unions stay as pydantic writes them (``anyOf``): it is the most portable shape for a schema
    that may also ride in a tool declaration. Everything else is left untouched, so constraints
    (``pattern`` / ``minimum`` / ``format``) still reach the model.

    :raise ValueError: a ``$ref`` that does not resolve, or a recursive model. A recursive
        schema cannot be inlined, and a schema with a dangling reference is worse than a loud
        failure.
    """
    definitions = schema.get("$defs") or schema.get("definitions") or {}
    body = {key: value for key, value in schema.items() if key not in ("$defs", "definitions")}
    return _inline_schema(body, definitions, ())


def model_json_schema_simplified(model: type[BaseModel]) -> dict[str, Any]:
    """``BaseModel`` → a schema a model can read without resolving references.

    ``by_alias=True`` is pydantic's default and stated here on purpose: the wire name is what a
    caller must send back, so the schema has to show the alias, never the attribute name.
    """
    return simplify_json_schema(model.model_json_schema(by_alias=True))


def _inline_schema(node: Any, definitions: dict[str, Any], stack: tuple[str, ...]) -> Any:
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str):
            name = ref.rsplit("/", 1)[-1]
            if name in stack:
                raise ValueError(
                    f"cannot inline a recursive schema: $ref {ref!r} resolves to itself"
                )
            target = definitions.get(name)
            if not isinstance(target, dict):
                raise ValueError(f"schema $ref {ref!r} does not resolve")
            merged = {**target, **{key: value for key, value in node.items() if key != "$ref"}}
            return _inline_schema(merged, definitions, (*stack, name))
        return {
            key: _inline_schema(value, definitions, stack)
            for key, value in node.items()
            if key != "title"
        }
    if isinstance(node, list):
        return [_inline_schema(item, definitions, stack) for item in node]
    return node


class ToolMeta(BaseModel):
    """
    Meta information describing a tool call, for compatibility across tool-calling frameworks.
    """

    name: str
    description: str
    strict: bool = Field(
        default=True,
        description="whether the tool is strict or not",
    )
    parameters: dict[str, object] = Field(
        description="the parameters json schema of the tool",
    )

    @classmethod
    def from_command_meta(cls, command_meta: CommandMeta, chan: str = "", *, strict: bool = False) -> Self | None:
        if command_meta.json_schema is None:
            return None
        name = Command.make_unique_name(chan, command_meta.name)
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

    def to_openai_function_def(self) -> 'FunctionDefinition':
        """
        to openai function definition.
        """
        from openai.types.shared_params import FunctionDefinition
        parameters = self.parameters.copy()
        return FunctionDefinition(
            name=self.name,
            description=self.description,
            parameters=parameters,
            strict=self.strict,
        )

    def to_anthropic_tool_param(self) -> 'ToolParam':
        from ghoshell_moss.depends import depend_ghost
        depend_ghost()
        from anthropic.types import ToolParam
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

    async def call_with_tool_return(self, *args, **kwargs) -> "ToolReturn":
        """
        return pydantic tool return.
        """
        from ghoshell_moss.depends import depend_ghost
        depend_ghost()
        from pydantic_ai import ToolReturn
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

    def as_pydantic_tool(self) -> "PydanticTool":
        """
        adapt into pydantic tool
        """
        from ghoshell_moss.depends import depend_ghost
        depend_ghost()
        from pydantic_ai import Tool
        meta = self.command.meta()
        return Tool.from_schema(
            self.call,
            name=Command.make_unique_name(self.channel_path, meta.name),
            description=meta.description,
            json_schema=meta.json_schema,
            takes_ctx=False,
            sequential=True,
        )
