"""
将 Python 代码中的 Function/Method 封装反射成 MOSS 架构可以理解和调度的 Command 对象.

它包含:
1. 代码即提示词: 反射代码提供以代码形式描述的提示词.
2. 完整动态性: 提示词本身可以动态变更
3. Command Token: 让模型输出的 token 被标记上对应的命令作用域.
4. 通道参数: 定义 chunks__, ctml__ 等通道参数, 能分层做流式传输.
5. Command As Function: AI 看到的 Command 同时是一个 callable, 因此 AI 基于所见写的 python 代码也是可执行的.
6. Command Task: 基于时间是第一公民观点, 将 command 的调用进行传输, 在一个 Shell 调度体系里按时调用. 同时考虑线程安全.
7. 兼容性: Command 可以降级为 JSON Schema Function Call...
8. 运行结果管理: Command 的运行结果能转化为 Message, 从而被模型理解. 效果类似 Tool. 但 CTML 是流式的.
"""

import asyncio
import contextvars
import inspect
import logging
import threading
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import (
    Any,
    Generic,
    Literal,
    Optional,
    TypeVar,
    Union,
    ClassVar,
    Protocol,
    AsyncIterator, Callable, Coroutine, AsyncIterable, TypeAlias,
    Tuple,
)
from jsonargparse import ArgumentParser as JsonArgumentParser
from argparse import ArgumentParser
from ghoshell_moss.message import unique_id, format_timestamp
from ghoshell_common.helpers import Timeleft
from ghoshell_container import get_caller_info
from pydantic import BaseModel, Field, TypeAdapter, AwareDatetime
from pydantic.errors import PydanticInvalidForJsonSchema, PydanticSchemaGenerationError
from typing_extensions import Self

from ghoshell_moss.core.concepts.errors import CommandError, CommandErrorCode, InterpretError
from ghoshell_moss.core.helpers.asyncio_utils import ThreadSafeEvent, ThreadSafeFuture
from ghoshell_moss.core.helpers.func import parse_function_interface
from ghoshell_moss.contracts import get_moss_logger
from ghoshell_moss.message import Message, Text
import orjson as json
import contextlib
import datetime
import dateutil

__all__ = [
    "RESULT",
    "BaseCommandTask",
    "Command",
    "CommandUniqueName",
    "CommandDeltaArgName",
    "CommandDeltaArgType",
    "CommandDeltaArgName2TypeMap",
    "CommandError",
    "CommandErrorCode",
    "CommandMeta",
    "CommandTask",
    "CommandStackResult",
    "CommandTaskResult",
    "CommandTaskState",
    "CommandToken",
    "CommandTokenSeq",
    "CommandWrapper",
    "PyCommand", "CliCommand",
    "make_command_group",
    "CommandTaskContextVar",
    "ObserveError",
    "Observe",
    "CommandCtx",
    "TaskScope",
    "CommandFunc",
    "CommandTaskContextKey",
    "MacroResult",
]

RESULT = TypeVar("RESULT")


class CommandTaskState(str, Enum):
    """
    the state types of a CommandTask
    """

    created = "created"  # the command task is just created by interpreter or other
    queued = "queued"  # the command task is sent to shell runtime
    pending = "pending"  # the command task is pending in the channel runtime
    executing = "executing"
    failed = "failed"  # the task is failed
    done = "done"  # the task is resolved
    cancelled = "cancelled"  # the task is cancelled

    @classmethod
    def is_complete(cls, state: str | Self) -> bool:
        return state in (cls.done, cls.failed, cls.cancelled)

    @classmethod
    def is_stopped(cls, state: str | Self) -> bool:
        return state in (cls.cancelled, cls.failed)

    def __str__(self):
        return self.value


StringType = Union[str, Callable[[], str]]


class CommandTokenSeq(str, Enum):
    """
    Command Token 是指, 对大模型输出的 Token 进行标记, 标记它们属于哪一个 Command 调用.
    通过这种方式, 将大模型输出的 Tokens 流染色成 CommandToken 流, 从而可以被流式解释器去调度.

    以 CTML 语法举例: <foo>streaming tokens</foo>  就包含三个部分:
     - start: <foo>
     - deltas: streaming tokens
     - end: </foo>

    """

    START = "start"
    END = "end"
    DELTA = "delta"

    @classmethod
    def all(cls) -> set[str]:
        return {cls.START.value, cls.END.value, cls.DELTA.value}


class CommandToken(BaseModel):
    """
    将大模型流式输出的文本结果, 包装为流式的 Command Token 对象.
    整个 Command 的生命周期是: start -> ?[delta -> ... -> delta] -> end
    在生命周期中所有被包装的 token 都带有相同的 cid.
    """

    seq: Literal["start", "delta", "end"] = Field(description="tokens seq")
    type: Literal[""] = Field(default="", description="token type, default is text")

    name: str = Field(description="command name")
    chan: str = Field(default="", description="channel name")
    call_id: str | None = Field(default=None, description="生成 command 时对应的 call_id")

    order: int = Field(default=0, description="the output order of the command")
    cmd_idx: int = Field(default=0, description="command index of the stream")
    part_idx: int = Field(
        default=0, description="continuous part idx of the command. [start, delta, delta, end] are four parts e.g."
    )

    stream_id: Optional[str] = Field(default=None, description="the id of the stream the command belongs to")

    content: str = Field(default="", description="origin tokens that llm generates")
    args: Optional[list[Any]] = Field(default=None, description="command position arguments, only for start token")
    kwargs: Optional[dict[str, Any]] = Field(default=None, description="attributes, only for start token")

    def command_id(self) -> str:
        """
        each command is presented by many command tokens. all the command tokens share a same command id.
        """
        return f"{self.stream_id}-{self.cmd_idx}"

    def command_part_id(self) -> str:
        """
        the command tokens has many parts, each part has a unique id.
        Notice the delta part may be separated by the child command tokens, for example:
        <start> [<delta> ... <delta>] - child command tokens - [<delta> ... <delta>] <end>.

        the deltas before the child command and after the child command have the different part_id `n` and `n + 1`
        """
        return f"{self.stream_id}-{self.cmd_idx}-{self.part_idx}"

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True, exclude_defaults=True)

    def __str__(self):
        return self.content


CommandLogos = str
"""可以解释成 CommandToken 的字符串, 中文 '道', 同时是 言说/行动的道路/理性决策 等含义. 未找到等价技术术语. """


class CommandDeltaArgName(str, Enum):
    """
    Command 体系里的特殊通道参数.
    Command 可以定义特殊的入参名, 这种特殊的入参名支持接受模型流式传输的 tokens 来生成参数.
    以 CTML 语法举例:
        当一个函数定义为
        >>> async def foo(tokens__):
        ...
        模型用 CTML 对它的调用可能是 <foo>streaming delta tokens</foo>
        这其中的 `streaming delta tokens` 不是等组装完才解析, 而是会流式地解析, 最终合成为函数的真实入参.

    """

    # 解析结果, 传递给参数类型应该是 str.
    TEXT = "text__"

    # 通过 AsyncIterable[CommandToken] 传递 ctml 流.
    CTML = "ctml__"

    # 通过 AsyncIterable[str] 传递文本流.
    CHUNKS = "chunks__"

    JSON = "json__"

    TOKENS = "tokens__"

    @classmethod
    def all(cls) -> set[str]:
        return {cls.TEXT.value, cls.CTML.value, cls.TOKENS.value, cls.CHUNKS.value}


class CommandDeltaArgType:
    """
    支持的类型.
    """

    COMMAND_TOKEN_STREAM = AsyncIterator[CommandToken]
    TEXT_CHUNKS_STREAM = AsyncIterator[str]
    TEXT = str


CommandDeltaArgName2TypeMap = {
    CommandDeltaArgName.TEXT.value: CommandDeltaArgType.TEXT,
    CommandDeltaArgName.TOKENS.value: CommandDeltaArgType.COMMAND_TOKEN_STREAM,
    CommandDeltaArgName.CTML.value: CommandDeltaArgType.COMMAND_TOKEN_STREAM,
    CommandDeltaArgName.CHUNKS.value: CommandDeltaArgType.TEXT_CHUNKS_STREAM,
    CommandDeltaArgName.JSON.value: CommandDeltaArgType.TEXT,
}
"""
拥有不同的语义的 Delta 类型. 
如果一个 Command 函数的入参包含这种特定命名的参数, 它生成 Command Token 的 Delta 应该遵循相同的处理逻辑.
"""


class CommandMeta(BaseModel):
    """
    命令的元信息. 用这个信息, 可以还原出大模型看到的 Command.
    而 Command 真实的执行逻辑, 对于大模型而言并不重要.
    """

    name: str = Field(description="the name of the command")
    description: str = Field(default="", description="the description of the command")
    dynamic: bool = Field(default=False, description="whether this command is dynamic or not")
    macro: bool = Field(default=False, description="whether this command returns command token language")
    available: bool = Field(
        default=True,
        description="whether this command is available",
    )
    tags: list[str] = Field(default_factory=list, description="tags of the command")
    delta_arg: Optional[str] = Field(
        default=None,
        description="the delta arg type",
        json_schema_extra={"enum": CommandDeltaArgName.all()},
    )
    visible: bool = Field(
        default=True,
        description="whether this command is visible for model."
                    "if a command is invisible to model, means it is in protocol that model does not need to see again,"
                    "or a system backdoor that bypass model recognition.",
    )
    interface: str = Field(
        default="",
        description="大模型所看到的关于这个命令的 prompt. 类似于 FunctionCall 协议提供的 JSON Schema."
                    "但核心思想是 Code As Prompt."
                    "通常是一个 python async 函数的 signature. 形如:"
                    "```python"
                    "async def name(arg: typehint = default) -> return_type:"
                    "    ''' docstring '''"
                    "    pass"
                    "```",
    )
    json_schema: Optional[dict[str, Any]] = Field(
        default=None,
        description="the json schema. 兼容性实现.",
    )
    timeout: float | None = Field(
        default=None,
        description="command protocol level max timeout"
    )

    # --- advance options --- #

    call_soon: bool = Field(
        default=False,
        description="如果为 True, 它在进入 Channel 队列时, 就会立刻触发执行."
                    "如果是 None blocking, 则会立刻开始运行."
                    "如果是 Blocking, 意味着它会立刻清空整个队列自身, 但不代表清空子队列",
    )
    blocking: bool = Field(
        default=True,
        description="执行完成后, 后面的命令, 包括 blocking = None 的命令才会开始执行."
                    "blocking = False 的命令想要立刻执行, 也需要配合 call soon.",
    )
    priority: int = Field(
        default=0,
        description="命令的优先级, 主要用于相同优先级的命令. 遵循以下基本规则:"
                    "相同优先级的命令, 一个执行完了才能执行另一个. "
                    "如果下一个高优先级的命令入队, 前一个会被立刻取消. "
                    "如果优先级为负值, 任何新任务在排队, 都会被立刻取消.",
    )
    always_observe: bool = Field(
        default=False,
        description="if the command result shall always be observed or not",
    )


CommandUniqueName = str
_ChannelFullPath = str
_CommandName = str

CommandArgs = list | tuple
CommandKwargs = dict
CommandPartial = Callable[[...], Coroutine[None, None, tuple[CommandArgs, CommandKwargs]]]
CommandFunc: TypeAlias = Union[Callable[[...], Coroutine[None, None, Any]], Callable[[...], Any]]


class Command(Generic[RESULT], ABC):
    """
    对大模型可见的命令描述. 包含几个核心功能:
    大模型通常能很好地理解, 并且使用这个函数.

    这个 Command 本身还会被伪装成函数, 让大模型可以直接用代码的形式去调用它.
    Shell 也将支持一个直接执行代码的控制逻辑, 形如 <exec> ... </exec> 的方式, 用 asyncio 语法直接执行它所看到的 Command
    """

    @abstractmethod
    def name(self) -> str:
        pass

    @staticmethod
    def make_unique_name(chan: str, name: str) -> CommandUniqueName:
        prefix = chan + ":" if chan else ""
        return f"{prefix}{name}"

    @staticmethod
    def split_unique_name(name: str) -> tuple[str, str]:
        parts = name.split(":", 1)
        return (parts[0], parts[1]) if len(parts) == 2 else ("", parts[0])

    @staticmethod
    def is_magic_command(name: str) -> bool:
        """
        魔法函数默认由 channel 判断是否存在, 如何使用.
        非内核开发者不需要理解这个规则. 用于支持流式解释器的特殊语法.
        """
        # todo: command name pattern match
        return len(name) >= 5 and name.startswith("__") and name.endswith("__")

    @abstractmethod
    def is_available(self) -> bool:
        """
        是否是可用的.
        """
        pass

    @abstractmethod
    def is_dynamic(self) -> bool:
        """
        是否是需要更新的.
        """
        pass

    @abstractmethod
    def meta(self) -> CommandMeta:
        """
        返回 Command 的元信息.
        """
        pass

    @abstractmethod
    def refresh_meta(self) -> None:
        """
        更新 command 的元信息.
        如果是动态的 Command (interface 会变化) 则需要重新生成 meta. 否则不需要执行.
        """
        pass

    @property
    @abstractmethod
    def partial(self) -> Optional[CommandPartial]:
        """
        CommandTask 在执行前需要运行的逻辑, 对入参进行第一遍加工.
        默认在 command task 的 on_compiled 生命周期执行.
        """
        pass

    @abstractmethod
    async def __call__(self, *args, **kwargs) -> RESULT:
        """
        基于入参, 出参, 生成一个 CommandCall 交给调度器去执行.
        """
        pass


class CliCommand(Command, ABC):

    @abstractmethod
    def cli_usage(self) -> str:
        pass

    @abstractmethod
    def cli_help(self) -> str:
        pass

    @abstractmethod
    def cli_argument_parser(self) -> ArgumentParser:
        pass

    @abstractmethod
    async def cli(self, arguments: str | list[str]) -> RESULT | str:
        pass


class CommandCtx(Protocol):

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class CommandWrapper(Command[RESULT]):
    """
    快速包装一个临时的 Command 对象.
    """

    def __init__(
            self,
            meta: CommandMeta,
            func: Callable[..., Coroutine[Any, Any, RESULT]],
            available_fn: Callable[[], bool] | None = None,
            partial: CommandPartial | None = None,
            refresh: Callable[[], None] | None = None,
            meta_func: Callable[[], CommandMeta] | None = None,
            ctx_fn: Callable[[], CommandCtx] | None = None,
            dynamic: bool = False,
    ):
        self._func = func
        self._meta = meta
        self._available_fn = available_fn
        self._partial = partial
        self._refresh = refresh
        self._meta_func = meta_func
        self._ctx_fn = ctx_fn
        self._dynamic = dynamic

    @classmethod
    def wrap(
            cls,
            command: Command[RESULT],
            *,
            func: Callable[..., Coroutine[Any, Any, RESULT]] | None = None,
            meta: CommandMeta | None = None,
            ctx_fn: Callable[[], CommandCtx] | None = None,
    ) -> Command[RESULT]:

        if func is None:
            if isinstance(command, CommandWrapper):
                func = command._func
            else:
                func = command.__call__

        meta = meta or command.meta()
        return CommandWrapper(
            meta=meta,
            func=func,
            available_fn=command.is_available,
            partial=command.partial,
            refresh=command.refresh_meta,
            meta_func=command.meta,
            ctx_fn=ctx_fn,
            dynamic=command.is_dynamic(),
        )

    @property
    def func(self) -> Callable:
        return self._func

    @property
    def partial(self) -> Optional[CommandPartial]:
        return self._partial

    def name(self) -> str:
        return self._meta.name

    def is_dynamic(self) -> bool:
        return self._dynamic

    def is_available(self) -> bool:
        if self._available_fn is not None:
            with self._in_ctx():
                return self._meta.available and self._available_fn()
        return self._meta.available

    def meta(self) -> CommandMeta:
        if self._meta_func is not None:
            with self._in_ctx():
                return self._meta_func()
        return self._meta

    def refresh_meta(self) -> None:
        if self._refresh:
            with self._in_ctx():
                self._refresh()
        return None

    @contextlib.contextmanager
    def _in_ctx(self):
        if not self._ctx_fn:
            yield
            return
        _ctx = self._ctx_fn()
        with _ctx:
            yield

    async def __call__(self, *args, **kwargs) -> RESULT:
        with self._in_ctx():
            return await self._func(*args, **kwargs)


class _MockSystemError(Exception):
    def __init__(self, status, message: str | None = None) -> None:
        self.message = message or ''
        super().__init__(message)


class PyCommand(CliCommand):
    """
    将 python 的 Coroutine 函数封装成 Command
    通过反射获取 interface.

    推荐永远用 async def 函数去封装 PyCommand.
    这样才能定义一个可以 cancel 的生命周期.
    否则需要用特别 trick 的方式去理解. 比如 ChannelCtx.task().done()
    """

    def __init__(
            self,
            func: Callable[..., Coroutine[None, None, RESULT]] | Callable[..., RESULT],
            *,
            partial: CommandPartial | None = None,
            chan: Optional[str] = None,
            name: Optional[str] = None,
            available: Callable[[], bool] | None = None,
            interface: Optional[str | Callable[..., Coroutine[None, None, RESULT]]] = None,
            doc: Optional[StringType] = None,
            comments: Optional[StringType] = None,
            meta: Optional[CommandMeta] = None,
            tags: Optional[list[str]] = None,
            call_soon: bool = False,
            blocking: bool = True,
            always_observe: bool = False,
            priority: int = 0,
            delta_types: Optional[set] = None,
            with_json_schema: bool = False,
            timeout: Optional[float] = None,
            visible: bool = True,
            macro: bool = False,
    ):
        """
        :param func: origin coroutine function
        :param available: if given, determine if the command is available dynamically
        :param interface: if not given, will reflect the origin function signature to generate the interface.
                if given
                - str: instead of the real signature
                - async function: generate interface from it.
        :param doc: if given, will change the docstring of the function or generate one dynamically
        :param comments: if given, will add to the body of the function interface.
        :param meta: the defined command meta information. if none, will generate one dynamically
        :param tags: tag the command if someplace want to filter commands. the tags need to be unique and common.
        :param call_soon: the command will be called right after it is sent to the channel.
        :param blocking: blocking command will be called only when channel is idle, one at a time.
        :param priority: the priority of the command. see command meta
        :param always_observe: shall always observe the command result
        :param delta_types: don't set it if you do not know why
        :param visible: if the command is visible to model
        :param macro: if the command is macro, return command logos
        """
        self._chan = chan
        self._func_name = func.__name__
        self._name = name or self._func_name
        self._func = func
        self._func_itf = parse_function_interface(func)
        self._partial = partial
        if timeout is not None and timeout < 0:
            raise ValueError(f"timeout {timeout} is invalid")
        self._timeout = timeout or None
        self._is_coroutine_func = inspect.iscoroutinefunction(func)
        self._interface_or_fn: Optional[str] = None
        if interface:
            if inspect.iscoroutinefunction(interface):
                self._interface_or_fn = parse_function_interface(interface).to_interface()
            else:
                self._interface_or_fn = interface
        # dynamic method
        self._doc_or_fn = doc
        self._available_or_fn = available
        self._comments_or_fn = comments
        self._is_dynamic_itf = (
                callable(self._interface_or_fn) or callable(doc) or callable(available) or callable(comments)
        )
        self._call_soon = call_soon
        self._blocking = blocking
        self._tags = tags
        self._visible = visible
        self._meta = meta
        self._always_observe = always_observe
        self._json_arg_parser: JsonArgumentParser | None = None
        self._priority = priority
        self._delta_types = delta_types if delta_types is not None else list(CommandDeltaArgName2TypeMap.keys())
        self._with_json_schema = with_json_schema
        delta_arg = None
        for arg_name in self._func_itf.signature.parameters:
            if arg_name.endswith("__") or arg_name in self._delta_types:
                if delta_arg is not None:
                    raise AttributeError(f"function {func} has more than one delta arg {meta.delta_arg} and {arg_name}")
                delta_arg = arg_name
                # only first delta_arg type. and not allow more than 1
                break
        self._delta_arg = delta_arg
        self._macro = macro

    def name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return self._available_or_fn() if self._available_or_fn is not None else True

    def is_dynamic(self) -> bool:
        return self._is_dynamic_itf

    def refresh_meta(self) -> None:
        if self._is_dynamic_itf:
            # refresh only command is dynamic.
            self._meta = self._generate_meta()

    @property
    def partial(self) -> Optional[CommandPartial]:
        if self._partial is not None:
            return self._partial
        return None

    def cli_argument_parser(self) -> JsonArgumentParser:
        if self._json_arg_parser is None:
            self._json_arg_parser = JsonArgumentParser(prog=self._name)
            self._json_arg_parser.description = self.meta().description
            self._json_arg_parser.add_function_arguments(self._func, as_positional=True)
            setattr(self._json_arg_parser, 'exit', self._cli_exit)
        return self._json_arg_parser

    @staticmethod
    def _cli_exit(status: int = 0, message: str | None = None) -> None:
        raise _MockSystemError(status, message)

    def cli_help(self) -> str:
        return self.cli_argument_parser().format_help()

    def cli_usage(self) -> str:
        return self.cli_argument_parser().format_usage()

    async def cli(self, arguments: str | list[str]) -> RESULT:
        import shlex
        import io
        from contextlib import redirect_stdout, redirect_stderr
        if isinstance(arguments, list):
            parts = arguments
        elif isinstance(arguments, str):
            parts = shlex.split(arguments)
        else:
            raise ValueError(f"argument must be str or list, `{arguments}` given")
        parser = self.cli_argument_parser()
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            with redirect_stderr(buffer):
                try:
                    cfg = parser.parse_args(parts, env=False)
                    r = await self.__call__(**cfg.as_dict())
                except _MockSystemError as e:
                    r = e.message or None
                if r is None:
                    if value := buffer.getvalue():
                        return value
                return r

    def _generate_meta(self) -> CommandMeta:
        meta = CommandMeta(name=self._name)
        doc = self._unwrap_string_type(self._doc_or_fn, "")
        meta.interface = self._gen_interface(meta.name, doc)
        meta.available = self.is_available()
        meta.delta_arg = self._delta_arg
        meta.call_soon = self._call_soon
        meta.tags = self._tags or []
        meta.blocking = self._blocking
        meta.always_observe = self._always_observe
        meta.timeout = self._timeout
        meta.visible = self._visible
        docstring = doc or self._func_itf.docstring
        meta.description = docstring.splitlines()[0] if docstring else ''
        # 标记 meta 是否是动态变更的.
        meta.dynamic = self._is_dynamic_itf
        meta.priority = self._priority
        meta.macro = self._macro

        if self._with_json_schema and self._func is not None:
            try:
                adapter = TypeAdapter(self._func)
                schema = adapter.json_schema()
                meta.json_schema = schema or dict(type="object")
            except (TypeError, PydanticInvalidForJsonSchema, PydanticSchemaGenerationError) as e:
                get_moss_logger().info("failed to create json schema for %r: %s", self._func, e)

        return meta

    def meta(self) -> CommandMeta:
        if self._meta is None:
            self._meta = self._generate_meta()
        meta = self._meta.model_copy()
        meta.available = self.is_available()
        return meta

    @staticmethod
    def _unwrap_string_type(value: StringType | None, default: Optional[str]) -> str:
        if value is None:
            return ""
        elif callable(value):
            return value()
        return value or default or ""

    def _gen_interface(self, name: str, doc: str) -> str:
        if self._interface_or_fn is not None:
            r = self._unwrap_string_type(self._interface_or_fn, None)
            return r
        comments = self._unwrap_string_type(self._comments_or_fn, None)
        func_itf = self._func_itf

        return func_itf.to_interface(
            name=name,
            doc=doc,
            comments=comments,
        )

    def parse_kwargs(self, *args, **kwargs) -> tuple[tuple, dict[str, Any]]:
        real_args, real_kwargs = self._func_itf.prepare_kwargs(*args, **kwargs)
        return real_args, real_kwargs

    async def __call__(self, *args, **kwargs) -> RESULT:
        try:
            real_args, real_kwargs = self.parse_kwargs(*args, **kwargs)
        except Exception as e:
            raise ValueError(f"command parse args failed: %s", e)

        if self._is_coroutine_func:
            return await self._func(*real_args, **real_kwargs)
        else:
            task = asyncio.to_thread(self._func, *real_args, **real_kwargs)
            return await task

    def __prompt__(self) -> str:
        return self.meta().interface


CommandTaskContextVar = contextvars.ContextVar("moss.ctx.CommandTask")


class Observe(BaseModel):
    """
    Command 的特殊返回值, 当 Command 返回这一结构时, 会立刻中断 Shell Interpreter 的返回值.
    """

    messages: list[Message] = Field(
        default_factory=list, description="ghoshell_moss.core.concepts.command:CommandTask 的特殊返回值类型."
    )

    @classmethod
    def new(cls, value: str) -> Self:
        return Observe(messages=[Message.new(tag="observe").with_content(value)])


class ObserveError(Exception):
    """
    一种将观察数据作为中断抛出的语法糖, 方便中断复杂 command 逻辑.
    """

    def __init__(self, message: str = '') -> None:
        self.message = message
        super().__init__(message)

    def as_messages(self) -> list[Message]:
        if self.message:
            return [Message.new(tag='observe').with_content(self.message)]
        return []

    def as_observe(self) -> Observe:
        return Observe.new(self.message)


class MacroResult(BaseModel):
    """Command 可以显式声明返回值是 macro 展开后的结果. 让给模型的数据和给解释器的数据不一样"""
    logos: CommandLogos = Field(description="macro 返回的 command logos")
    result: str | None = Field(default=None, description="展示给模型的数据, 为 None 时直接展示 CommandLogos")

    def display(self) -> str:
        return self.result if self.result is not None else self.logos


class CommandTaskResult(BaseModel):
    """
    Command Task 的标准返回值.
    1. 它持有函数的返回值. 这个值可以是任意类型. 但如果不可序列化的话, 就无法跨进程正确传输数据结构.
    2. 它可以添加 outputs 消息体, 意味着 AI 侧需要使用它发送消息.
    3. 它可以添加 messages 消息体, 作为可查看的消息给大模型.
    4. 它返回一个 operator 算子. 如果这个算子符合 Agent / Ghost 的协议的话,
    """

    result: Any | None = Field(
        default=None,
        description="command 的真实返回值",
    )
    serialized: bool = Field(
        default=False,
        description='result is serialized',
    )
    caller: str | None = Field(
        default=None, description="生成 CommandTask 的 caller name. 通常不用设置. 在 resolve 时自动添加."
    )
    from_macro_id: str | None = Field(
        default=None,
        description="生成 command task 的 macro id",
    )
    macro_id: str | None = Field(
        default=None,
        description="当前 Command Task 自身的 macro id",
    )
    logos: CommandLogos | None = Field(
        default=None,
        description="如果是通过 macro Result 构建的, logos 会独立传输"
    )
    output: list[Message] = Field(
        default_factory=list,
        description="对外部输出的消息体, 通常不用设置 role / name. 模型本身看不到这部分消息. "
    )
    messages: list[Message] = Field(
        default_factory=list,
        description="给大模型查看, 但不对外输出的消息体. "
                    "通常用于 multi-agent 等场景, 才返回包含 role, name 的消息体. 否则应该由 Agent 负责配置.",
    )
    observe: bool = Field(
        default=False,
        description="默认的 interpreter 交互协议. 当 Interpreter 生成的 Task 返回一个 observe==True 的结果时,"
                    "Interpreter 应该停止运行逻辑, 取消后续所有的命令. ",
    )
    created: AwareDatetime = Field(
        default_factory=lambda: datetime.datetime.now(dateutil.tz.gettz()),
        description="记录创建时间",
    )

    @classmethod
    def new(cls, value: Any) -> 'CommandTaskResult':
        if isinstance(value, CommandTaskResult):
            return cls.from_serializable(value)
        elif isinstance(value, Observe):
            return cls.from_observe(value)
        elif isinstance(value, MacroResult):
            return cls(
                result=value.result,
                logos=value.logos,
            )
        else:
            return cls(
                result=value,
            )

    @classmethod
    def from_observe(cls, observe: "Observe") -> Self:
        """create task result from Observe instance"""
        return cls(
            # 不用 deepcopy
            messages=observe.messages.copy(),
            observe=True,
        )

    def macro_result(self) -> MacroResult | None:
        if self.logos is None:
            return None
        return MacroResult(logos=self.logos, result=self.result)

    def to_observe(self) -> Observe | None:
        """ to Observe object if self is from Observe instance"""
        if self.observe:
            return Observe(messages=self.messages.copy() if len(self.messages) > 0 else [])
        return None

    def to_raw_result(self) -> Any:
        if macro_result := self.macro_result():
            return macro_result
        elif observe := self.to_observe():
            return observe
        else:
            return self.result

    def serializable_copy(self) -> Self:
        """return a copy that serializable"""
        result = self.model_copy()
        serialized_result, ok = self.serialize_result()
        if ok:
            result.result = serialized_result
            result.serialized = True
        return result

    @classmethod
    def from_serializable(cls, value: Self | None) -> Self:
        if value is None:
            return None
        elif not isinstance(value.result, str):
            return value
        elif not value.serialized:
            return value
        content = value.result
        try:
            result = json.loads(content)
        except (json.JSONDecodeError, ValueError):
            result = content
        return value.model_copy(update={"result": result})

    def serialize_result(self) -> tuple[Any, bool]:
        """serialize the result field"""
        if self.result is None:
            return None, False
        if isinstance(self.result, str):
            return self.result, False
        try:
            serialized_content = json.dumps(self.result).decode("utf-8")
        except (ValueError, TypeError):
            serialized_content = repr(self.result)
        return serialized_content, True

    def is_empty_result(self) -> bool:
        # 空 logos (返回了空串的 void 宏) 视为空结果: 不展开、不显示.
        return (self.logos is None or self.logos == "") and self.result is None and len(self.messages) == 0

    def as_messages(
            self,
            *,
            name: str | None = None,
            with_serialized_result: bool = True,
    ) -> list[Message]:
        """
        生成可以被模型观察的消息体.
        首先目前主流模型的约定, 不支持 system/assistant 等角色持有图片等类型的 content. 而定义这种 content 可以让 Command 返回多模态.
        然后, 主流模型支持的函数调用返回是 FunctionCall 协议. 基本都不支持异步返回, 必须同步阻塞调用.
        Anthropic 消息协议更可怕, 不支持 role.
        所以要在现有的协议基础上支持异步的, 多个 command 返回的 command result, 就考虑用最基础的类型, 字符串 xml 包裹.
        """
        attrs = {'at': format_timestamp(self.created)}
        name = name or self.caller or None
        if name and self.from_macro_id:
            name = f"macro:{self.from_macro_id}#{name}"
        if self.macro_id is not None:
            attrs['macro_id'] = self.macro_id
        if self.is_empty_result():
            return []
        result_message = None

        # 先把结果序列化.
        if self.logos is not None:
            result_message = Message.new(tag='result').with_content(
                self.result if self.result is not None else self.logos,
            )
        elif with_serialized_result and self.result is not None:
            # 保留 name.
            serialized_content, ok = self.serialize_result()
            if serialized_content:
                result_message = Message.new(tag='result').with_content(serialized_content)

        messages = []
        if result_message is not None and not result_message.is_empty():
            messages.append(result_message)
        # only merge messages. not output messages which is not for ai.
        for message in self.messages:
            if message.is_empty():
                continue
            # 不再合并.
            messages.append(message)
        return [
            Message.new(
                tag='command', name=name, timestamp=False, attributes=attrs,
            ).with_messages(*messages),
        ]
        # return messages

    def join_result(self, *results: Self | Observe) -> None:
        """
        合并多个 result.
        """
        for result in results:
            _result = result
            if isinstance(_result, Observe):
                _result = CommandTaskResult.from_observe(_result)

            if _result.observe:
                # observe 关键字传染.
                self.observe = True

            # output 合并.
            if len(_result.output) > 0:
                self.output.extend(_result.output)
            # message 合并.
            messages = _result.as_messages()
            if len(messages) > 0:
                self.messages.extend(messages)


CommandTaskContextKey = str


class CommandTask(Generic[RESULT], ABC):
    """
    线程安全的 Command Task 对象. 相当于重新实现一遍 asyncio.Task 类似的功能.
    有区别的部分:
    1. 建立全局唯一的 cid, 方便在双工通讯中赋值.
    2. **必须实现线程安全**, 因为通讯可能是在多线程里.
    3. 包含 debug 需要的 state, trace 等信息.
    4. 保留命令的元信息, 包括入参等.
    5. 不是立刻启动, 而是被 channel 调度时才运行.
    6. 兼容 json rpc 协议, 方便跨进程通讯.
    7. 可复制, 复制后可重入, 方便做循环.
    """

    instances_count: ClassVar[int] = 0

    def __init__(
            self,
            *,
            chan: str,
            meta: CommandMeta,
            func: Callable[..., Coroutine[None, None, RESULT]] | None,
            partial: CommandPartial | None = None,
            tokens: str,
            args: list,
            kwargs: dict[str, Any],
            cid: str | None = None,
            # 必须是可序列化对象.
            context: dict[CommandTaskContextKey, Any] | None = None,
            call_id: str | int | None = None,
            timeout: float | None = None,
            scope_id: str | None = None,
    ) -> None:
        self.chan = chan
        # command id
        self.cid: str = cid or unique_id()
        self.tokens: str = tokens
        self.args: list = list(args)
        self.kwargs: dict[str, Any] = kwargs
        self.state: str = "created"
        self.meta = meta
        self.func = func
        self.progress: str = ''
        self.partial = partial
        self.errcode: Optional[int] = None
        self.errmsg: Optional[str] = None
        self.context = context or {}
        self.errcode: int = 0
        self.errmsg: Optional[str] = None
        self.scope_id = scope_id
        if timeout is not None and timeout < 0:
            raise ValueError(f"timeout {timeout} is invalid")
        self.timeout: float | None = timeout or meta.timeout or None
        # --- debug --- #
        self.last_trace: tuple[str, float] = ('', 0.0)
        self.trace: dict[str, float] = {
            "created": time.time(),
        }
        self.send_through: list[str] = [""]
        self.exec_chan: Optional[str] = None
        """记录 task 在哪个 channel 被运行. """

        # 编译检查阶段.
        self.on_compiled_task: Optional[asyncio.Task] = None
        self.done_at: Optional[str] = None
        """最后产生结果的 fail/cancel/resolve 函数被调用的代码位置."""
        self.call_id: str = str(call_id) if call_id is not None else ""
        self.macro_id: str | None = None
        self.from_macro_id: str | None = None
        CommandTask.instances_count += 1

    def __del__(self):
        CommandTask.instances_count -= 1

    @abstractmethod
    def set_progress(self, progress: str) -> None:
        pass

    @abstractmethod
    def on_progress_callback(self, callback: Callable[['CommandTask'], None]) -> None:
        pass

    def set_command(self, command: Command) -> None:
        self.func = command.__call__
        self.meta = command.meta
        self.partial = command.partial

    def is_magical(self) -> bool:
        """未完成创建的魔法 command task. 非内核开发者不需要理解其规则. """
        return Command.is_magic_command(self.meta.name)

    def is_bare_task(self) -> bool:
        """是否没有注入执行函数. 非内核开发者不需要理解其规则. """
        return self.func is None

    def caller_name(self) -> str:
        """
        用三元信息标定一个调用名.
        """
        parts = []
        if self.chan:
            parts.append(self.chan)
        parts.append(self.meta.name)
        if self.call_id:
            parts.append(self.call_id)
        return ":".join(parts)

    def compiled(self) -> bool:
        return self.partial is None or self.on_compiled_task is not None

    def on_compiled(self, loop: asyncio.AbstractEventLoop = None) -> None:
        """
        约定的 command task 预先加工参数的周期.
        一个 command 只会执行一次.
        """
        if self.on_compiled_task is None and self.partial is not None:
            loop = loop or asyncio.get_running_loop()
            self.on_compiled_task = loop.create_task(self.partial(*self.args, **self.kwargs))

    @abstractmethod
    def result(self, throw: bool = True) -> Optional[RESULT]:
        """
        返回 task 的结果, 可以选择是否抛出异常. 这点和 Future 不一样.
        """
        pass

    @abstractmethod
    def done(self) -> bool:
        """
        if the command is done (cancelled, done, failed)
        """
        pass

    def success(self) -> bool:
        return self.done() and self.state == "done" and self.errcode == 0

    def observe(self) -> bool:
        result = self.task_result()
        if result:
            return result.observe
        return False

    def cancelled(self) -> bool:
        return self.done() and self.state == "cancelled"

    @abstractmethod
    def add_done_callback(self, fn: Callable[[Self], None]):
        pass

    @abstractmethod
    def remove_done_callback(self, fn: Callable[[Self], None]):
        pass

    @abstractmethod
    def clear(self) -> None:
        """清空运行结果."""
        pass

    @abstractmethod
    def cancel(self, reason: str = "") -> None:
        """
        cancel the command if running.
        """
        pass

    @abstractmethod
    def set_state(self, state: CommandTaskState | str) -> None:
        """
        set the state of the command with time
        """
        pass

    @abstractmethod
    def fail(self, error: Exception | str) -> None:
        """
        fail the task with error.
        """
        pass

    def is_failed(self) -> bool:
        return self.done() and self.errcode != 0

    def is_critical_failed(self) -> bool:
        return self.done() and self.errcode != 0 and CommandErrorCode.is_critical(self.errcode)

    @abstractmethod
    def resolve(self, result: RESULT | CommandTaskResult | Observe) -> None:
        """
        resolve the result of the task if it is running.
        可以接受 CommandTaskResult 对象. 设置成 result 的应该是 CommandTaskResult 的 result
        """
        pass

    @abstractmethod
    def task_result(self) -> Optional[CommandTaskResult]:
        """
        task 未完成时返回 None. 否则生成 CommandTaskResult 对象.
        这是专门为 CommandTask 设计的对象.

        对于 AI 所看见的上下文而言, command 的返回值是 result()
        对于 Agent / Ghost 工程而言, command 的返回值其实是这个 CommandTaskResult.
        其中 observe 为 True 表示需要观察一次结果.

        通常有三种方式可以让 observe 为 True:
        1. command 返回 command task result 本身, 其中 observe 为 True
        2. 出现了严重异常, 所以需要 observe
        3. command 返回了一个 Observe 对象.

        :return: None 是 task 本身没有执行完毕. 否则一定返回 result.
        """
        pass

    def raise_exception(self) -> None:
        """
        返回存在的异常.
        """
        exp = self.exception()
        if exp is not None:
            raise exp

    @abstractmethod
    def exception(self) -> Optional[Exception]:
        pass

    @abstractmethod
    async def wait(
            self,
            *,
            throw: bool = True,
            timeout: float | None = None,
    ) -> Optional[RESULT]:
        """
        async wait the task to be done thread-safe
        :raise TimeoutError: if the task is not done until timeout
        :raise CancelledError: if the task is cancelled
        :raise CommandError: if the command failed and already be wrapped
        :raise ObserveError: if the command return Observe
        """
        pass

    @abstractmethod
    def wait_sync(self, *, throw: bool = True, timeout: float | None = None) -> Optional[RESULT]:
        """
        wait the command to be done in the current thread (blocking). thread-safe.
        """
        pass

    async def dry_run(self) -> RESULT:
        """无状态的运行逻辑"""
        # if not prepared
        self.on_compiled()
        if self.func is None:
            return None
        if self.on_compiled_task is not None:
            args, kwargs = await self.on_compiled_task
        else:
            args, kwargs = self.args, self.kwargs
        r = await self.func(*args, **kwargs)
        return r

    async def dry_run_with_timeout(self, timeout: float | None = None) -> RESULT:
        timeout = timeout or self.timeout
        if timeout is not None and timeout > 0:
            return await asyncio.wait_for(self.dry_run(), timeout=timeout)
        else:
            return await self.dry_run()

    async def run(self) -> RESULT:
        """
        典型的案例展示如何使用一个 command task. 有状态的运行逻辑.
        实际在链路中通常运行的是 dry run.
        """
        if self.done():
            self.raise_exception()
            return self.result()

        set_token = CommandTaskContextVar.set(self)
        try:
            dry_run_task = asyncio.create_task(self.dry_run_with_timeout())
            wait_task_done_by_outside = asyncio.create_task(self.wait(throw=False))
            # resolve 生效, wait 就会立刻生效.
            # 否则 wait 先生效, 也一定会触发 cancel, 确保 resolve task 被 wait 了, 而且执行过 cancel.
            done, pending = await asyncio.wait(
                [dry_run_task, wait_task_done_by_outside],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            if dry_run_task in done:
                result = await dry_run_task
                self.resolve(result)
            else:
                result = None
                self.raise_exception()
            return result

        except asyncio.CancelledError:
            if not self.done():
                self.cancel(reason="command execution canceled")
            raise
        except Exception as e:
            if not self.done():
                self.fail(e)
            raise
        finally:
            CommandTaskContextVar.reset(set_token)
            if not self.done():
                self.cancel()

    def __await__(self):
        """
        等待 task 执行结束, 但和 asyncio.Task 不同, 这里不会真的执行 task 的 run 逻辑
        它仍然要被别的地方 (比如 ChannelRuntime) 执行完后 resolve 才能解除阻塞.
        这是 Command 体系跨进程的本质决定的.
        """
        if self.done():
            async def _already_done():
                return self.result(throw=True)

            return _already_done().__await__()

        async def _wait_done():
            await self.wait(throw=True)
            return self.result(throw=True)

        return _wait_done().__await__()

    def __repr__(self):
        tokens = self.tokens
        if len(tokens) > 50:
            tokens = f"{tokens[:50]}..."
        return (
            f"<CommandTask chan=`{self.chan}` name=`{self.meta.name}` call_id=`{self.call_id}``"
            f"args=`{self.args}` kwargs=`{str(self.kwargs)}`"
            f"cid=`{self.cid}` "
            f"state=`{self.state}` done_at=`{self.done_at}` exec_chan=`{self.exec_chan}` "
            f"errcode=`{self.errcode}` errmsg=`{self.errmsg}` "
            f"send_through=`{self.send_through}` "
            f">{tokens}</CommandTask>"
        )


class BaseCommandTask(Generic[RESULT], CommandTask[RESULT]):
    """
    大模型的输出被转化成 CmdToken 后, 再通过执行器生成的运行时对象.
    实现一个跨线程安全的等待机制.
    """

    def __init__(
            self,
            *,
            chan: str,
            meta: CommandMeta,
            func: Callable[..., Coroutine[None, None, RESULT]] | None,
            tokens: str,
            args: list,
            kwargs: dict[str, Any],
            cid: str | None = None,
            context: dict[str, Any] | None = None,
            call_id: str | int | None = None,
            partial: CommandPartial | None = None,
            timeout: float | None = None,
            scope_id: str | None = None,
    ) -> None:
        super().__init__(
            chan=chan,
            meta=meta,
            func=func,
            tokens=tokens,
            args=args,
            kwargs=kwargs,
            cid=cid,
            context=context,
            call_id=call_id,
            partial=partial,
            timeout=timeout,
            scope_id=scope_id,
        )
        self.__result: Optional[RESULT] = None
        self.__done_event: ThreadSafeEvent = ThreadSafeEvent()
        self.__done_lock = threading.Lock()
        self.__done_callbacks = set()
        self.__task_result: Optional[CommandTaskResult] = None
        self.__progress_callbacks: set[Callable[[CommandTask], None]] = set()

    def result(self, throw: bool = True) -> Optional[RESULT]:
        if throw:
            self.raise_exception()
        if self.done() and self.__result is None and self.errcode == 0:
            return self.task_result().to_observe()
        return self.__result

    def add_done_callback(self, fn: Callable[[CommandTask], None]):
        if not self.__done_event.is_set():
            self.__done_callbacks.add(fn)

    def remove_done_callback(self, fn: Callable[[CommandTask], None]):
        self.__done_callbacks.discard(fn)

    def set_progress(self, progress: str) -> None:
        self.progress = progress
        invalid = None
        if len(self.__progress_callbacks) > 0:
            for callback in self.__progress_callbacks:
                try:
                    callback(self)
                except Exception as e:
                    logging.exception(f"Error in Command {self.caller_name()}  progress callback: {e}")
                    if invalid is None:
                        invalid = []
                    invalid.append(callback)
            if invalid is not None and len(invalid) > 0:
                for c in invalid:
                    self.__progress_callbacks.remove(c)

    def on_progress_callback(self, callback: Callable[['CommandTask'], None]) -> None:
        self.__progress_callbacks.add(callback)

    def copy(self, cid: str = "") -> Self:
        """ copy 过的 task 不是同一个 task. """
        cid = cid or unique_id()
        return BaseCommandTask(
            chan=self.chan,
            cid=cid,
            meta=self.meta.model_copy(),
            scope_id=self.scope_id,
            func=self.func,
            tokens=self.tokens,
            args=self.args,
            kwargs=self.kwargs,
            context=self.context,
            call_id=self.call_id,
        )

    @classmethod
    def from_command(
            cls,
            command_: Command[RESULT],
            chan_: str = "",
            tokens_: str = "",
            args: tuple | list | None = None,
            kwargs: dict | None = None,
            cid: str | None = None,
            call_id: str | int | None = None,
            context: dict[str, Any] | None = None,
            scope_id: str | None = None,
    ) -> "BaseCommandTask":
        return cls(
            chan=chan_,
            meta=command_.meta(),
            func=command_.__call__,
            tokens=tokens_,
            args=list(args) if args is not None else [],
            kwargs=kwargs if kwargs is not None else {},
            partial=command_.partial,
            cid=cid,
            call_id=call_id,
            context=context,
            scope_id=scope_id,
        )

    def done(self) -> bool:
        """
        命令已经结束.
        """
        return self.__done_event.is_set()

    def cancel(self, reason: str = ""):
        """
        停止命令.
        """
        self._set_result(None, "cancelled", CommandErrorCode.CANCELLED, reason)

    def clear(self) -> None:
        self.__result = None
        self.__done_event.clear()
        self.errcode = 0
        self.errmsg = None

    def set_state(self, state: CommandTaskState | str) -> None:
        with self.__done_lock:
            if self.__done_event.is_set():
                return None
            if isinstance(state, CommandTaskState):
                state = state.value
            if state in self.trace:
                # 只设置一次.
                return None
            self.state = state
            now = round(time.time(), 4)
            self.last_trace = (self.state, now)
            self.trace[self.state] = now
            return None

    def _set_result(
            self,
            result: Optional[RESULT],
            state: CommandTaskState | str,
            errcode: int,
            errmsg: Optional[str],
            done_at: Optional[str] = None,
    ) -> bool:
        with self.__done_lock:
            if self.__done_event.is_set():
                return False
            done_at = done_at or get_caller_info(3)
            self.__result = result
            self.errcode = errcode
            self.errmsg = errmsg
            self.done_at = done_at
            self.__done_event.set()
            self.state = str(state)
            self.trace[self.state] = time.time()
            self.func = None
            self.partial = None
            self._real_args = None
            self._real_kwargs = None
            if self.on_compiled_task is not None and not self.on_compiled_task.done():
                # cancel compile task also.
                self.on_compiled_task.cancel()
            # 运行结束的回调.
            if len(self.__done_callbacks) > 0:
                for done_callback in self.__done_callbacks:
                    try:
                        done_callback(self)
                    except Exception as e:
                        logging.exception("CommandTask done callback failed: %r", e)
                        continue
            # 避免互相持有.
            self.__done_callbacks.clear()
            self.__progress_callbacks.clear()
            return True

    def fail(self, error: Exception | str) -> None:
        if not self.__done_event.is_set():
            if isinstance(error, ObserveError):
                self.__task_result = CommandTaskResult(
                    caller=self.caller_name(),
                    messages=error.as_messages(),
                    observe=True,
                )
                self._set_result(None, "failed", CommandErrorCode.OBSERVE, error.message)
                return

            elif isinstance(error, str):
                errmsg = error
                errcode = CommandErrorCode.UNKNOWN_ERROR.value
            elif isinstance(error, CommandError):
                errcode = error.code
                errmsg = error.message
            elif isinstance(error, InterpretError):
                # 解释器解析异常: 归口 INTERPRET_ERROR, 不要降到 UNKNOWN_ERROR.
                errcode = CommandErrorCode.INTERPRET_ERROR.value
                errmsg = error.message
            elif isinstance(error, asyncio.CancelledError):
                errcode = CommandErrorCode.CANCELLED.value
                errmsg = "cancelled"
            elif isinstance(error, asyncio.TimeoutError):
                errcode = CommandErrorCode.TIMEOUT.value
                errmsg = "timeout"
            elif isinstance(error, TimeoutError):
                errcode = CommandErrorCode.TIMEOUT.value
                errmsg = "timeout"
            elif isinstance(error, Exception):
                errcode = CommandErrorCode.UNKNOWN_ERROR.value
                # 忽略回调.
                errmsg = str(error)
            else:
                errcode = 0
                errmsg = ""
            self._set_result(
                None,
                "cancelled" if CommandErrorCode.is_cancelled(errcode) else "failed",
                errcode,
                errmsg,
            )

    def resolve(self, result: RESULT | CommandTaskResult | Observe | MacroResult) -> None:
        if self.__done_event.is_set():
            return
        task_result = CommandTaskResult.new(value=result)
        #  必须设置 caller name.
        task_result.caller = self.caller_name()
        task_result.from_macro_id = self.from_macro_id
        task_result.macro_id = self.macro_id
        self.__task_result = task_result
        self._set_result(task_result.to_raw_result(), "done", 0, None)

    def task_result(self) -> Optional[CommandTaskResult]:
        if not self.__done_event.is_set():
            return None
        if self.__task_result is None:
            exp = self.exception()
            # failed 以上级别的异常要记录.
            # cancel 不要. 因为 cancel 可能很多.
            if exp is not None and CommandErrorCode.is_failed(exp):
                item = Message.new().with_content("Error: %s" % exp)
                task_result = CommandTaskResult(
                    caller=self.caller_name(),
                    messages=[
                        item,
                    ],
                )
                self.__task_result = task_result
            else:
                # 返回空对象.
                self.__task_result = CommandTaskResult()
        # command 可以约定 always observe, 这样不用特地返回 Observe 对象.
        self.__task_result.observe = self.__task_result.observe or self.meta.always_observe
        return self.__task_result

    def exception(self) -> Optional[Exception]:
        if self.errcode is None or self.errcode == 0:
            return None
        else:
            return CommandError(self.errcode, self.errmsg or "", at_line=self.done_at)

    async def wait(
            self,
            *,
            throw: bool = True,
            timeout: float | None = None,
    ) -> Optional[RESULT]:
        """
        等待命令被执行完毕. 但不会主动运行这个任务. 仅仅是等待.
        Command Task 的 Await done 要求跨线程安全.
        :param throw: 如果为 True, 有异常, 或者有 observe == True 都会抛出异常.
        :param timeout: 等待的超时时间, 并不是 task 自身的异常时间.
        :raise CommandError: task 自身的异常.
        """
        if self.__done_event.is_set():
            if throw:
                self.raise_exception()
            return self.__result
        if timeout is not None:
            await asyncio.wait_for(self.__done_event.wait(), timeout=timeout)
        else:
            await self.__done_event.wait()
        if throw:
            if self.errcode != 0:
                raise CommandError(self.errcode, self.errmsg or "", at_line=self.done_at)
        return self.__result

    def wait_sync(self, *, throw: bool = True, timeout: float | None = None) -> Optional[RESULT]:
        """
        线程的 wait.
        """
        if not self.__done_event.wait_sync():
            raise TimeoutError(f"wait timeout: {timeout}")
        if throw:
            self.raise_exception()
        return self.__result


class TaskScope:
    """
    为 task 准备的几种标准的 wait 机制.
    """
    default_until = "flow"

    def __init__(
            self,
            *,
            channel: str = '',
            until: Literal['flow', 'all', 'any'] = 'flow',
            timeout: float | None = None,
            strict: bool = False,
    ) -> None:
        self.tasks: set[CommandTask] = set()
        self.timeout = timeout
        self.until = until
        self.channel = channel
        self._done_event = ThreadSafeEvent()
        self._compiled_event = ThreadSafeEvent()
        self._tick_task: asyncio.Future | None = None
        self._strict = strict

    def add(self, task: CommandTask) -> None:
        if self._done_event.is_set():
            task.cancel("group already done")
        self.tasks.add(task)
        if not task.done():
            task.add_done_callback(self.callback)

    def compiled(self):
        self._compiled_event.set()
        # 完成 compiled 的时候已经过期了.
        if self._done_event.is_set():
            for task in self.tasks:
                task.cancel()

    def callback(self, task: CommandTask) -> None:
        if task not in self.tasks:
            return
        if task.done():
            if self.until == 'any':
                if self._compiled_event.is_set():
                    self.cancel("other task finished")
                else:
                    self._done_event.set()
                return

    def cancel(self, reason: str = "") -> None:
        if len(self.tasks) == 0:
            return
        if self._tick_task is not None:
            self._tick_task.cancel(reason)
        tasks = self.tasks.copy()
        self.tasks.clear()
        for task in tasks:
            if not task.done():
                task.cancel(reason)

    def tick(self) -> asyncio.Future[None]:
        """
        开始异步的 timeout 计数.
        """
        if self.timeout is None:
            return asyncio.create_task(self._noop())
        if self._tick_task is not None:
            return self._tick_task
        self._tick_task = asyncio.shield(self._cancel_after_timeout(self.timeout))
        return self._tick_task

    async def _noop(self) -> None:
        pass

    async def _cancel_after_timeout(self, timeout: float) -> None:
        """
        cancel after timeout.
        """
        if timeout <= 0.0:
            return
        await self._compiled_event.wait()
        await asyncio.sleep(timeout)
        self.cancel("timeout")

    async def wait(self):
        self.compiled()
        wait_tasks: list[CommandTask] = []
        for task in self.tasks:
            if self.until == 'flow':
                if self.channel == task.chan:
                    wait_tasks.append(task)
            else:
                wait_tasks.append(task)
        if len(wait_tasks) == 0:
            if self.until == 'flow' and not self._strict:
                # 容错逻辑. 没有传入任何命令时,
                wait_tasks = list(self.tasks)
        if len(wait_tasks) > 0:
            await asyncio.gather(*[t.wait(throw=False) for t in wait_tasks])


class CommandStackResult:
    """
    特殊的数据结构, 用来标记一个 task 序列, 也可以由 task 返回.
    当 Command 返回这个数据结构时, Runtime 应该要依次执行其生成的子 tasks, 最后回调它的 callback 函数.
    这个方法是用来实现 Command 原语的关键功能, 通过 task 栈的方式提供递归的栈生成.
    """

    def __init__(
            self,
            iterator: AsyncIterable[CommandTask] | list[CommandTask],
            callback: Callable[[list[CommandTask]], Coroutine[None, None, Any]] = None,
            timeout: float | None = None,
    ) -> None:
        if isinstance(iterator, list):

            async def generate():
                for item in iterator:
                    yield item

            self._iterator = generate()
        else:
            self._iterator = aiter(iterator)
        self._generated = []
        self._on_callback = callback
        self._iterator_done = asyncio.Event()
        self._timeleft = Timeleft(timeout) if timeout is not None and timeout > 0.0 else None
        self._exception = None
        self._wait_timeout_task: asyncio.Task | None = None
        self._wait_owner_done: asyncio.Task | None = None

    async def __aenter__(self) -> Self:
        self._wait_timeout_task = asyncio.create_task(self._wait_timeout())
        return self

    def _on_task_done(self, task: CommandTask) -> None:
        # 基础规则, 如果触发了 observe 就退出.
        if task.observe():
            self._iterator_done.set()

    async def _wait_timeout(self):
        if self._timeleft is not None:
            await asyncio.sleep(self._timeleft.left())
            self._iterator_done.set()
            # 超时后生成出来的也全部超时.
            for task in self._generated:
                task.cancel("timeout")

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._iterator_done.set()
        if exc_val is not None:
            # 退出时如果发生了异常, 则必须要清空所有未完成任务.
            self._exception = exc_val
            for task in self._generated:
                if not task.done():
                    task.fail(exc_val)
        if self._wait_timeout_task is not None and not self._wait_timeout_task.done():
            self._wait_timeout_task.cancel()

    async def callback(self, owner: CommandTask) -> Self | None:
        """
        回调 owner.
        """
        if owner.done():
            return
        if self._exception is not None:
            owner.fail(self._exception)
            return
        if self._on_callback and callable(self._on_callback):
            # 如果是回调函数, 则用回调函数决定 task.
            result = await self._on_callback(self._generated)
            if isinstance(result, CommandStackResult):
                # but not resolve
                return result
            owner.resolve(result)
            return None
        else:
            owner.resolve(None)
            return None

    def generated(self) -> list[CommandTask]:
        return self._generated.copy()

    def __aiter__(self) -> AsyncIterator[CommandTask]:
        return self

    async def __anext__(self) -> CommandTask:
        if self._iterator_done.is_set():
            raise StopAsyncIteration
        try:
            item = await self._iterator.__anext__()
            item.add_done_callback(self._on_task_done)
        except StopAsyncIteration:
            self._iterator_done.set()
            raise StopAsyncIteration
        self._generated.append(item)
        return item


def make_command_group(
        chan: str,
        *commands: Command,
        groups: dict[str, dict[str, Command] | None] = None,
) -> dict[str, dict[str, Command]]:
    """
    command 分组的基本逻辑. ChannelPath: {command_name: command}
    """
    result = groups or {}
    for command in commands:
        meta = command.meta()
        if chan not in result:
            result[chan] = {}
        result[chan][meta.name] = command
    return result
