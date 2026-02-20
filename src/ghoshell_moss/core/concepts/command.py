import asyncio
import contextvars
import inspect
import logging
import threading
import time
import traceback
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable, Coroutine, Iterable
from enum import Enum
from typing import (
    Any,
    Generic,
    Literal,
    Optional,
    TypeVar,
    Union,
)

from ghoshell_common.helpers import uuid
from ghoshell_container import get_caller_info
from pydantic import BaseModel, Field
from typing_extensions import Self

from ghoshell_moss.core.concepts.errors import CommandError, CommandErrorCode
from ghoshell_moss.core.helpers.asyncio_utils import ThreadSafeEvent
from ghoshell_moss.core.helpers.func import parse_function_interface

__all__ = [
    "RESULT",
    "BaseCommandTask",
    "CancelAfterOthersTask",
    "Command",
    "CommandUniqueName",
    "CommandDeltaType",
    "CommandDeltaTypeMap",
    "CommandError",
    "CommandErrorCode",
    "CommandMeta",
    "CommandTask",
    "CommandResultStack",
    "CommandTaskState",
    "CommandToken",
    "CommandTokenType",
    "CommandType",
    "CommandWrapper",
    "PyCommand",
    "make_command_group",
    'CommandTaskContextVar',
]

RESULT = TypeVar("RESULT")


class CommandTaskState(str, Enum):
    """
    the state types of a CommandTask
    """

    created = "created"  # the command task is just created by interpreter or other
    queued = "queued"  # the command task is sent to shell runtime
    pending = "pending"  # the command task is pending in the channel runtime
    running = "running"  # the task is running
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


class CommandDeltaType(str, Enum):
    """
    Command 可以定义特殊的入参名, 这种特殊的入参名支持接受模型流式传输的 tokens 来生成参数.
    以 CTML 语法举例:
        当一个函数定义为
        >>> async def foo(tokens__):
        ...
        模型用 CTML 对它的调用可能是 <foo>streaming delta tokens</foo>
        这其中的 `streaming delta tokens` 不是等组装完才解析, 而是会流式地解析, 最终合成为函数的真实入参.

    todo: 命名过于费解, 需要改动.
    todo: 考虑支持 ctml__, tokens__, text__, json__ 等几种预设的语法规则.
    """

    TEXT = "text__"
    TOKENS = "tokens__"
    CTML = "ctml__"

    @classmethod
    def all(cls) -> set[str]:
        return {cls.TEXT.value, cls.TOKENS.value}


CommandDeltaTypeMap = {
    CommandDeltaType.TEXT.value: "the deltas are text string",
    CommandDeltaType.CTML.value: "the deltas are ctml string",
}
"""
拥有不同的语义的 Delta 类型. 
如果一个 Command 函数的入参包含这种特定命名的参数, 它生成 Command Token 的 Delta 应该遵循相同的处理逻辑.
"""


class CommandType(str, Enum):
    """
    Command 的基础类型, 用来在调用大模型前, 根据情况筛选不同类型的 Command.
    """

    FUNCTION = ""
    """函数, 需要一段时间执行, 执行完后结束. 其值为空, 降低传输成本. """

    PROMPTER = "prompter"
    """
    返回一个字符串, 可以用来生成 prompt. 是构成 PML (prompter markdown language) 语法的核心函数. 
    PML 指一段 XML 风格的函数调用, 作为模板语法, 将所有函数返回的字符串结果拼到模板中, 生成一个动态的 Prompt. 
    
    Agent 可以同时看到自己某块上下文的 PML + prompt,  它通过暴露出来的函数修改 PML, 就可以修改自己的 prompt. 
    从而达到认知的自治.  
    """

    PRIMITIVE = "primitive"
    """
    控制原语类型. 
    """

    @classmethod
    def all(cls) -> set[str]:
        return {
            cls.FUNCTION.value,
            cls.PROMPTER.value,
            cls.PRIMITIVE.value,
        }


class CommandTokenType(str, Enum):
    """
    Command Token 是指, 对大模型输出的 Token 进行标记, 标记它们属于哪一个 Command 调用.
    通过这种方式, 将大模型输出的 Tokens 流染色成 CommandToken 流, 从而可以被流式解释器去调度.

    以 CTML 语法举例: <foo>streaming tokens</foo>  就包含三个部分:
     - start: <foo>
     - deltas: streaming tokens
     - end: </foo>

    # todo: 考虑更名为 CommandTokenSeq . 因为从 type 的角度看, 未来双工模型输出多模态, delta 可能有 文本/音频/图片/视频 等.
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
    type: Literal[''] = Field(default="", description="token type, default is text")

    name: str = Field(description="command name")
    chan: str = Field(default="", description="channel name")
    call_id: Optional[int] = Field(None, description="生成 command 时对应的 call_id")

    order: int = Field(default=0, description="the output order of the command")
    cmd_idx: int = Field(description="command index of the stream")
    part_idx: int = Field(description="continuous part idx of the command. "
                                      "[start, delta, delta, end] are four parts e.g.")

    stream_id: Optional[str] = Field(description="the id of the stream the command belongs to")

    content: str = Field(description="origin tokens that llm generates")
    kwargs: Optional[dict[str, Any]] = Field(default=None, description="attributes, only for command start")

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


class CommandMeta(BaseModel):
    """
    命令的元信息. 用这个信息, 可以还原出大模型看到的 Command.
    而 Command 真实的执行逻辑, 对于大模型而言并不重要.
    """

    name: str = Field(description="the name of the command")
    chan: str = Field(default="", description="the channel name that the command belongs to")
    dynamic: bool = Field(default=False, description="whether this command is dynamic or not")
    available: bool = Field(
        default=True,
        description="whether this command is available",
    )
    type: str = Field(
        default=CommandType.FUNCTION.value,
        description="",
        json_schema_extra={"enum": CommandType.all()},
    )
    tags: list[str] = Field(default_factory=list, description="tags of the command")
    delta_arg: Optional[str] = Field(
        default=None,
        description="the delta arg type",
        json_schema_extra={"enum": CommandDeltaType.all()},
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
    args_schema: Optional[dict[str, Any]] = Field(
        default=None,
        description="the json schema. 兼容性实现.",
    )

    # --- advance options --- #

    call_soon: bool = Field(
        default=False,
        description="if true, this command is called soon when append to the channel",
    )
    blocking: bool = Field(
        default=True,
        description="whether this command block the channel. if block + call soon, will clear the channel first",
    )
    interruptable: bool = Field(
        default=False,
        description="interruptable command task will be cancelled when next blocking task is pending",
    )


CommandUniqueName = str
_ChannelFullPath = str
_CommandName = str


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
    def make_uniquename(chan: str, name: str) -> CommandUniqueName:
        prefix = chan + ":" if chan else ""
        return f"{prefix}{name}"

    @staticmethod
    def split_uniquename(name: str) -> tuple[str, str]:
        parts = name.split(":", 1)
        return (parts[0], parts[1]) if len(parts) == 2 else ("", parts[0])

    @abstractmethod
    def is_available(self) -> bool:
        """
        是否是可用的.
        """
        pass

    @abstractmethod
    def meta(self) -> CommandMeta:
        """
        返回 Command 的元信息.
        """
        pass

    @abstractmethod
    async def refresh_meta(self) -> None:
        """
        更新 command 的元信息.
        如果是动态的 Command (interface 会变化) 则需要重新生成 meta. 否则不需要执行.
        """
        pass

    @abstractmethod
    async def __call__(self, *args, **kwargs) -> RESULT:
        """
        基于入参, 出参, 生成一个 CommandCall 交给调度器去执行.
        """
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
            ctx: contextvars.Context | None = None,
    ):
        self._func = func
        self._meta = meta
        self._ctx = ctx
        self._available_fn = available_fn

    @classmethod
    def wrap(
            cls,
            command: Command[RESULT],
            *,
            func: Callable[..., Coroutine[Any, Any, RESULT]] | None = None,
            ctx: contextvars.Context | None = None,
            meta: CommandMeta | None = None,
    ) -> Command[RESULT]:

        if func is None:
            if isinstance(command, CommandWrapper):
                func = command._func
            else:
                func = command.__call__

        return CommandWrapper(
            meta=meta or command.meta(),
            func=func,
            ctx=ctx,
            available_fn=command.is_available,
        )

    @property
    def func(self) -> Callable:
        return self._func

    def name(self) -> str:
        return self._meta.name

    def is_available(self) -> bool:
        if self._available_fn is not None:
            return self._meta.available and self._available_fn()
        return self._meta.available

    def meta(self) -> CommandMeta:
        return self._meta

    async def refresh_meta(self) -> None:
        return None

    async def __call__(self, *args, **kwargs) -> RESULT:
        if self._ctx:
            return await self._ctx.run(self._func, *args, **kwargs)
        return await self._func(*args, **kwargs)


class PyCommand(Generic[RESULT], Command[RESULT]):
    """
    将 python 的 Coroutine 函数封装成 Command
    通过反射获取 interface.

    Example of how to implement a Command
    """

    def __init__(
            self,
            func: Callable[..., Coroutine[None, None, RESULT]] | Callable[..., RESULT],
            *,
            chan: Optional[str] = None,
            name: Optional[str] = None,
            available: Callable[[], bool] | None = None,
            interface: Optional[StringType] = None,
            doc: Optional[StringType] = None,
            comments: Optional[StringType] = None,
            meta: Optional[CommandMeta] = None,
            tags: Optional[list[str]] = None,
            # todo: 思考这两个 feature 是否有更合理的定义方式.
            call_soon: bool = False,
            blocking: bool = True,
            delta_types: Optional[set] = None
    ):
        """
        :param func: origin coroutine function
        :param available: if given, determine if the command is available dynamically
        :param interface: if not given, will reflect the origin function signature to generate the interface.
        :param doc: if given, will change the docstring of the function or generate one dynamically
        :param comments: if given, will add to the body of the function interface.
        :param meta: the defined command meta information. if none, will generate one dynamically
        :param tags: tag the command if someplace want to filter commands. the tags need to be unique and common.
        :param call_soon: the command will be called right after it is sent to the channel.
        :param blocking: blocking command will be called only when channel is idle, one at a time.
        """
        self._chan = chan
        self._func_name = func.__name__
        self._name = name or self._func_name
        self._func = func
        self._func_itf = parse_function_interface(func)
        self._is_coroutine_func = inspect.iscoroutinefunction(func)
        # dynamic method
        self._interface_or_fn = interface
        self._doc_or_fn = doc
        self._available_or_fn = available
        self._comments_or_fn = comments
        self._is_dynamic_itf = callable(interface) or callable(doc) or callable(available) or callable(comments)
        self._call_soon = call_soon
        self._blocking = blocking
        self._tags = tags
        self._meta = meta
        self._delta_types = delta_types if delta_types is not None else CommandDeltaType.all()
        delta_arg = None
        for arg_name in self._func_itf.signature.parameters:
            if arg_name in self._delta_types:
                if delta_arg is not None:
                    raise AttributeError(f"function {func} has more than one delta arg {meta.delta_arg} and {arg_name}")
                delta_arg = arg_name
                # only first delta_arg type. and not allow more than 1
                break
        self._delta_arg = delta_arg

    def name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return self._available_or_fn() if self._available_or_fn is not None else True

    async def refresh_meta(self) -> None:
        if self._is_dynamic_itf:
            self._meta = await asyncio.to_thread(self._generate_meta)

    def _generate_meta(self) -> CommandMeta:
        meta = CommandMeta(name=self._name)
        meta.chan = self._chan or ""
        doc = self._unwrap_string_type(self._doc_or_fn, "")
        meta.interface = self._gen_interface(meta.name, doc)
        meta.available = self.is_available()
        meta.delta_arg = self._delta_arg
        meta.call_soon = self._call_soon
        meta.tags = self._tags or []
        meta.blocking = self._blocking
        # 标记 meta 是否是动态变更的.
        meta.dynamic = self._is_dynamic_itf
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
            r = self._interface_or_fn()
            return r
        comments = self._unwrap_string_type(self._comments_or_fn, None)
        func_itf = self._func_itf

        return func_itf.to_interface(
            name=name,
            doc=doc,
            comments=comments,
        )

    def parse_kwargs(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        real_kwargs = self._func_itf.prepare_kwargs(*args, **kwargs)
        return real_kwargs

    async def __call__(self, *args, **kwargs) -> RESULT:
        try:
            real_kwargs = self.parse_kwargs(*args, **kwargs)
        except Exception as e:
            raise ValueError(f"command parse args failed: %s", e)

        if self._is_coroutine_func:
            return await self._func(**real_kwargs)
        else:
            task = asyncio.to_thread(self._func, **real_kwargs)
            return await task


CommandTaskContextVar = contextvars.ContextVar("moss.ctx.CommandTask")


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

    IDX_ARG = "_idx"

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
    ) -> None:
        self.chan = chan
        self.cid: str = cid or uuid()
        self.tokens: str = tokens
        self.args: list = list(args)
        self.kwargs: dict[str, Any] = kwargs
        self.state: str = "created"
        self.meta = meta
        self.func = func
        self.errcode: Optional[int] = None
        self.errmsg: Optional[str] = None
        self.context = context or {}
        self.errcode: int = 0
        self.errmsg: Optional[str] = None
        self.last_trace: tuple[str, float] = ("", 0.0)
        """ command task 在 shell 执行的 task 中的排序. 传入这个参数本身没有意义. 最终都以 Shell 的定义为准. """
        if self.IDX_ARG in self.kwargs:
            del self.kwargs[self.IDX_ARG]

        # --- debug --- #
        self.trace: dict[str, float] = {
            "created": time.time(),
        }
        self.send_through: list[str] = ['']
        self.exec_chan: Optional[str] = None
        """记录 task 在哪个 channel 被运行. """

        self.done_at: Optional[str] = None
        """最后产生结果的 fail/cancel/resolve 函数被调用的代码位置."""
        self.call_id: str = str(call_id) if call_id is not None else ""

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

    @abstractmethod
    def resolve(self, result: RESULT) -> None:
        """
        resolve the result of the task if it is running.
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
        """
        pass

    @abstractmethod
    def copy(self, cid: str = "") -> Self:
        """
        返回一个状态清空的 command task, 一定会生成新的 cid.
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
        if self.func is None:
            return None
        r = await self.func(*self.args, **self.kwargs)
        return r

    async def run(self) -> RESULT:
        """典型的案例如何使用一个 command task. 有状态的运行逻辑."""
        if self.done():
            self.raise_exception()
            return self.result()

        if self.func is None:
            # func 为 none 的情况下, 完全依赖外部运行赋值.
            return await self.wait(throw=True)

        try:
            # todo: ctx 接下来统一交给 CommandTaskCtx 管理.
            ctx = contextvars.copy_context()
            CommandTaskContextVar.set(self)
            dry_run_cor = ctx.run(self.dry_run)
            dry_run = asyncio.create_task(dry_run_cor)
            wait = asyncio.create_task(self.wait())
            # resolve 生效, wait 就会立刻生效.
            # 否则 wait 先生效, 也一定会触发 cancel, 确保 resolve task 被 wait 了, 而且执行过 cancel.
            done, pending = await asyncio.wait([dry_run, wait], return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            if dry_run in done:
                result = await dry_run
                self.resolve(result)
            else:
                self.raise_exception()
            return self.result()

        except asyncio.CancelledError:
            if not self.done():
                self.cancel(reason="canceled")
            raise
        except Exception as e:
            if not self.done():
                self.fail(e)
            raise
        finally:
            if not self.done():
                self.cancel()

    def __await__(self):
        def generator():
            while not self.done():
                yield
            return self.result()

        return generator()

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
    TODO: refact with asyncio.Future?
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
        )
        self._result: Optional[RESULT] = None
        self._done_event: ThreadSafeEvent = ThreadSafeEvent()
        self._done_lock = threading.Lock()
        self._done_callbacks = set()

    def result(self, throw: bool = True) -> Optional[RESULT]:
        if throw:
            self.raise_exception()
        return self._result

    def add_done_callback(self, fn: Callable[[CommandTask], None]):
        self._done_callbacks.add(fn)

    def remove_done_callback(self, fn: Callable[[CommandTask], None]):
        if fn in self._done_callbacks:
            self._done_callbacks.remove(fn)

    def copy(self, cid: str = "") -> Self:
        cid = cid or uuid()
        return BaseCommandTask(
            chan=self.chan,
            cid=cid,
            meta=self.meta.model_copy(),
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
            args: tuple | None = None,
            kwargs: dict | None = None,
    ) -> "BaseCommandTask":
        return cls(
            chan=chan_,
            meta=command_.meta(),
            func=command_.__call__,
            tokens=tokens_,
            args=list(args) if args is not None else [],
            kwargs=kwargs if kwargs is not None else {},
        )

    def done(self) -> bool:
        """
        命令已经结束.
        """
        return self._done_event.is_set()

    def cancel(self, reason: str = ""):
        """
        停止命令.
        """
        self._set_result(None, "cancelled", CommandErrorCode.CANCELLED, reason)

    def clear(self) -> None:
        self._result = None
        self._done_event.clear()
        self.errcode = 0
        self.errmsg = None

    def set_state(self, state: CommandTaskState | str) -> None:
        with self._done_lock:
            if self._done_event.is_set():
                return None
            self.state = str(state)
            now = round(time.time(), 4)
            self.last_trace = (self.state, now)
            self.trace[self.state] = now

    def _set_result(
            self,
            result: Optional[RESULT],
            state: CommandTaskState | str,
            errcode: int,
            errmsg: Optional[str],
            done_at: Optional[str] = None,
    ) -> bool:
        with self._done_lock:
            if self._done_event.is_set():
                return False
            done_at = done_at or get_caller_info(3)
            self._result = result
            self.errcode = errcode
            self.errmsg = errmsg
            self.done_at = done_at
            self._done_event.set()
            self.state = str(state)
            self.trace[self.state] = time.time()
            # 运行结束的回调.
            if len(self._done_callbacks) > 0:
                for done_callback in self._done_callbacks:
                    try:
                        done_callback(self)
                    except Exception as e:
                        logging.exception("CommandTask done callback failed")
                        continue
            return True

    def fail(self, error: Exception | str) -> None:
        if not self._done_event.is_set():
            if isinstance(error, str):
                errmsg = error
                errcode = CommandErrorCode.UNKNOWN_ERROR.value
            elif isinstance(error, CommandError):
                errcode = error.code
                errmsg = error.message
            elif isinstance(error, asyncio.CancelledError):
                errcode = CommandErrorCode.CANCELLED.value
                errmsg = ""
            elif isinstance(error, Exception):
                errcode = CommandErrorCode.UNKNOWN_ERROR.value
                # 忽略回调.
                errmsg = str(error)
            else:
                errcode = 0
                errmsg = ""
            self._set_result(
                None,
                "cancelled" if errcode == CommandErrorCode.CANCELLED.value else "failed",
                errcode,
                errmsg,
            )

    def resolve(self, result: RESULT) -> None:
        if not self._done_event.is_set():
            self._set_result(result, "done", 0, None)

    def exception(self) -> Optional[Exception]:
        if self.errcode is None or self.errcode == 0:
            return None
        else:
            return CommandError(self.errcode, self.errmsg or "")

    async def wait(
            self,
            *,
            throw: bool = True,
            timeout: float | None = None,
    ) -> Optional[RESULT]:
        """
        等待命令被执行完毕. 但不会主动运行这个任务. 仅仅是等待.
        Command Task 的 Await done 要求跨线程安全.
        """
        try:
            if self._done_event.is_set():
                if throw:
                    self.raise_exception()
                return self._result
            if timeout is not None:
                await asyncio.wait_for(self._done_event.wait(), timeout=timeout)
            else:
                await self._done_event.wait()
            if throw and self.errcode != 0:
                raise CommandError(self.errcode, self.errmsg or "")
            return self._result
        except asyncio.CancelledError:
            pass

    def wait_sync(self, *, throw: bool = True, timeout: float | None = None) -> Optional[RESULT]:
        """
        线程的 wait.
        """
        if not self._done_event.wait_sync():
            raise TimeoutError(f"wait timeout: {timeout}")
        if throw:
            self.raise_exception()
        return self._result


class WaitDoneTask(BaseCommandTask):
    """
    等待其它任务完成.
    """

    def __init__(
            self,
            tasks: Iterable[CommandTask],
            after: Optional[Callable[[], Coroutine[None, None, RESULT]]] = None,
    ) -> None:
        meta = CommandMeta(
            name="_wait_done",
            chan="",
            type=CommandType.PRIMITIVE.value,
        )

        async def wait_done() -> Optional[RESULT]:
            await asyncio.gather(*[t.wait() for t in tasks])
            if after is not None:
                return await after()
            return None

        super().__init__(
            meta=meta,
            func=wait_done,
            tokens="",
            args=[],
            kwargs={},
        )


class CancelAfterOthersTask(BaseCommandTask[None]):
    """
    等待其它任务完成后, cancel 当前任务.
    """

    def __init__(
            self,
            current: CommandTask,
            *tasks: CommandTask,
            tokens: str = "",
    ) -> None:
        meta = CommandMeta(
            name="cancel_" + current.meta.name,
            chan=current.chan,
            type=CommandType.PRIMITIVE.value,
            block=False,
            call_soon=True,
        )

        async def wait_done_then_cancel() -> Optional[None]:
            waiting = list(tasks)
            if not current.done() and len(waiting) > 0:
                await asyncio.gather(*[t.wait() for t in tasks])
            if not current.done():
                # todo
                current.cancel()
                await current.wait()

        super().__init__(
            chan=current.chan,
            meta=meta,
            func=wait_done_then_cancel,
            tokens=tokens,
            args=[],
            kwargs={},
        )


class CommandResultStack:
    """
    特殊的数据结构, 用来标记一个 task 序列, 也可以由 task 返回.
    """

    def __init__(
            self,
            iterator: AsyncIterator[CommandTask] | list[CommandTask],
            callback: Callable[[list[CommandTask]], Coroutine[None, None, Any]] = None,
    ) -> None:
        self._iterator = iterator
        self._on_callback = callback
        self._generated = []

    async def callback(self, owner: CommandTask) -> None:
        """
        回调 owner.
        """
        if self._on_callback and callable(self._on_callback):
            # 如果是回调函数, 则用回调函数决定 task.
            result = await self._on_callback(self._generated)
            owner.resolve(result)
        else:
            owner.resolve(None)

    def generated(self) -> list[CommandTask]:
        return self._generated.copy()

    def __aiter__(self) -> AsyncIterator[CommandTask]:
        return self

    async def __anext__(self) -> CommandTask:
        if isinstance(self._iterator, list):
            if len(self._iterator) == 0:
                raise StopAsyncIteration
            item = self._iterator.pop(0)
            self._generated.append(item)
            return item
        else:
            item = await self._iterator.__anext__()
            self._generated.append(item)
            return item

    def __str__(self):
        return ""


def make_command_group(*commands: Command) -> dict[str, dict[str, Command]]:
    result = {}
    for command in commands:
        meta = command.meta()
        chan = meta.chan
        if chan not in result:
            result[chan] = {}
        result[chan][meta.name] = command
    return result
