import asyncio
import threading
from abc import ABC, abstractmethod
from typing import (
    TypedDict, Literal, Optional, Dict, Any, Awaitable, List, Generic, TypeVar, Tuple, Callable, Coroutine, Union,
    is_typeddict,
)
from ghoshell_common.helpers import uuid, generate_import_path
from ghoshell_moss.helpers.func import parse_function_interface, awaitable_caller
from .errors import CommandError
from pydantic import BaseModel, Field
from enum import Enum
import inspect
import time

RESULT = TypeVar("RESULT")

CommandState = Literal['created', 'queued', 'pending', 'running', 'failed', 'done', 'cancelled']
StringType = Union[str, Callable[[], Coroutine[None, None, str]]]

CommandDeltaType = dict(
    text_="the delta is text string",
    json_="the delta is in json format",
    yaml_="the delta is in yaml format",
    markdown_="the delta is in markdown format",
    python_="the delta is python code",
    commands_="the delta are commands, transport as Iterable[CommandToken]",
)
"""
拥有不同的语义的 Delta 类型. 如果一个 Command 的入参包含这些类型, 它生成 Command Token 的 Delta 应该遵循相同逻辑.
"""


class CommandType(str, Enum):
    FUNCTION = "function"
    """功能, 需要一段时间执行, 执行完后结束. """

    POLICY = "policy"
    """状态变更函数. 会改变 Command 所属 Channel 的运行策略, 立刻生效. 但只有 run_policy (没有其它命令阻塞时) 才会执行. """

    PROMPTER = "prompter"
    """返回一个字符串, 用来生成 prompt. 仅当 Agent 自主生成 prompt 时才要用它. 结合 pml """

    META = "meta"
    """AI 进入元控制状态, 可以自我修改时, 才能使用的函数"""

    CONTROL = "control"
    """通常只面向人类开放的控制函数. 人类可以通过一个 AI 作为 interface 去控制它. """

    @classmethod
    def all(cls) -> set[str]:
        return {
            cls.FUNCTION.value,
            cls.POLICY.value,
            cls.PROMPTER.value,
            cls.META.value,
            cls.CONTROL.value,
        }


class CommandToken(BaseModel):
    """
    将大模型流式输出的文本结果, 包装为流式的 Command Token 对象.
    整个 Command 的生命周期是: start -> ?[delta -> ... -> delta] -> end
    在生命周期中所有被包装的 token 都带有相同的 cid.

    * start: 携带 command 的参数信息.
    * delta: 表示这个 command 所接受到的流式输入.
    * stop: 表示一个 command 已经结束.
    """
    type: Literal['start', 'delta', 'end'] = Field(description="tokens type")

    chan: Optional[str] = Field(default=None, description="the channel name that the command belongs to ")

    name: str = Field(description="command name")

    order: int = Field(default=0, description="the output order of the command")

    cmd_idx: int = Field(description="command index of the stream")

    part_idx: int = Field(description="continuous part idx of the command. start, delta, delta, end are four parts")

    stream_id: Optional[str] = Field(description="the id of the stream the command belongs to")

    content: str = Field(description="origin tokens that llm generates")

    kwargs: Optional[Dict[str, Any]] = Field(default=None, description="attributes, only for command start")

    def unique_id(self) -> str:
        return f"{self.stream_id}-{self.cmd_idx}-{self.part_idx}"


class CommandMeta(BaseModel):
    """
    命令的原始信息.
    """
    name: str = Field(default="", description="the name of the command")
    chan: str = Field(default="", description="the channel name that the command belongs to")
    description: str = Field(default="", description="the doc of the command")
    available: bool = Field(default=True, description="whether this command is available")
    type: CommandType = Field(
        default=CommandType.FUNCTION.value,
        description="",
        json_schema_extra=dict(enum=CommandType.all()),
    )
    delta_arg: Optional[str] = Field(default=None, description="the delta arg type")
    call_soon: bool = Field(default=False)
    block: bool = Field(default=True)
    interface: str = Field(
        default="",
        description="大模型所看到的关于这个命令的 prompt. 类似于 FunctionCall 协议提供的 JSON Schema."
                    "但核心思想是 Code As Prompt."
                    "通常是一个 python async 函数的 signature. 形如:"
                    "```python"
                    "async def name(arg: typehint = default) -> return_type:"
                    "    ''' docstring '''"
                    "    pass"
                    "```"
    )
    args_schema: Optional[Dict[str, Any]] = Field(default=None, description="the json schema. 兼容性实现.")


class Command(Generic[RESULT], ABC):
    """
    对大模型可见的命令描述. 包含几个核心功能:
    大模型通常能很好地理解, 并且使用这个函数.

    这个 Command 本身还会被伪装成函数, 让大模型可以直接用代码的形式去调用它.
    Shell 也将支持一个直接执行代码的控制逻辑, 形如 <exec> ... </exec> 的方式, 用 asyncio 语法直接执行它所看到的 Command
    """

    @abstractmethod
    def parse_kwargs(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def meta(self) -> CommandMeta:
        """
        返回 Command 的元信息.
        """
        pass

    @abstractmethod
    async def __call__(self, *args, **kwargs) -> RESULT:
        """
        基于入参, 出参, 生成一个 CommandCall 交给调度器去执行.
        """
        pass


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
            name: Optional[str] = None,
            available: Callable[[], Coroutine[None, None, bool]] | None = None,
            interface: Optional[StringType] = None,
            doc: Optional[StringType] = None,
            comments: Optional[StringType] = None,
            meta: Optional[CommandMeta] = None,
    ):
        """
        :param func: origin coroutine function
        :param meta: the defined command meta information
        :param available: if given, determine if the command is available dynamically
        :param interface: if not given, will reflect the origin function signature to generate the interface.
        :param doc: if given, will change the docstring of the function or generate one dynamically
        :param comments: if given, will add to the body of the function interface.
        """
        self._name = name or func.__name__
        self._func = func
        self._func_itf = parse_function_interface(func)
        self._is_coroutine_func = inspect.iscoroutinefunction(func)
        # dynamic method
        self._dynamic = callable(doc) or callable(available) or callable(comments) or callable(interface)
        self._interface_fn = awaitable_caller(interface, default="") if interface is not None else None
        self._doc_fn = awaitable_caller(doc, default=self._func_itf.docstring or "")
        self._available_fn = awaitable_caller(available, default=True)
        self._comments_fn = awaitable_caller(comments, default="")
        # cached meta
        self._meta = meta or CommandMeta()
        self._cached_meta: Optional[CommandMeta] = None
        for arg_name in self._func_itf.signature.parameters.keys():
            if arg_name in CommandDeltaType:
                if self._meta.delta_arg is not None:
                    raise AttributeError(f"function {func} has more than one delta arg {meta.delta_arg} and {arg_name}")
                self._meta.delta_arg = arg_name
                # only first delta_arg type. and not allow more than 1
                break

    async def meta(self) -> CommandMeta:
        if self._cached_meta is not None:
            return self._cached_meta.model_copy()

        meta = self._meta.model_copy()
        meta.description = await self._doc_fn()
        meta.name = meta.name or self._name
        meta.available = await self._available_fn()
        meta.interface = await self._gen_interface(meta.name, meta.description)

        if not self._dynamic:
            self._cached_meta = meta
        return meta

    async def _gen_interface(self, name: str, doc: str) -> str:
        if self._interface_fn is not None:
            r = await self._interface_fn()
            return r
        comments = ""
        if self._comments_fn is not None:
            comments = await self._comments_fn()

        func_itf = self._func_itf

        return func_itf.to_interface(
            name=name,
            doc=doc,
            comments=comments,
        )

    def parse_kwargs(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        real_kwargs = self._func_itf.prepare_kwargs(*args, **kwargs)
        return real_kwargs

    async def __call__(self, *args, **kwargs) -> RESULT:
        if self._is_coroutine_func:
            return await self._func(*args, **kwargs)
        else:
            task = asyncio.to_thread(self._func, *args, **kwargs)
            return await task


class CommandTask(Generic[RESULT], ABC):
    """
    thread-safe future object for command execution.
    1. cancel / fail the task thread-safe, cancel a task outside the loop.
    2. thread-safe wait the result of the task.
    """

    cid: str
    tokens: str
    args: List
    kwargs: Dict[str, Any]
    state: CommandState
    errcode: int = 0
    errmsg: Optional[str] = None
    trace: Dict[CommandState, float] = {}
    result: Optional[RESULT] = None

    @abstractmethod
    def is_done(self) -> bool:
        """
        if the command is done (cancelled, done, failed)
        """
        pass

    @abstractmethod
    def cancel(self, reason: str = "") -> None:
        """
        cancel the command if running.
        """
        pass

    def set_state(self, state: CommandState) -> None:
        """
        set the state of the command with time
        """
        self.state = state
        self.trace[state] = time.time()

    @abstractmethod
    def fail(self, error: Exception | str) -> None:
        """
        fail the task with error.
        """
        pass

    @abstractmethod
    def resolve(self, result: RESULT) -> None:
        """
        resolve the result of the task if it is running.
        """
        pass

    @abstractmethod
    async def wait_done(
            self,
            *,
            throw: bool = False,
            timeout: float | None = None,
    ) -> Optional[RESULT]:
        """
        async wait the task to be done thread-safe
        :raise TimeoutError: if the task is not done until timeout
        :raise CancelledError: if the task is cancelled
        :raise CommandError: if the command failed and already be wrapped
        """
        pass

    def wait(self, *, throw: bool = False, timeout: float | None = None) -> Optional[RESULT]:
        """
        wait the command to be done in the current thread (blocking). thread-safe.
        """
        pass


class BasicCommandTask(Generic[RESULT], CommandTask[RESULT]):
    """
    大模型的输出被转化成 CmdToken 后, 再通过执行器生成的运行时对象.
    实现一个跨线程安全的等待机制.
    """

    def __init__(
            self,
            *,
            command: Command[RESULT],
            tokens: str,
            args: list,
            kwargs: Dict[str, Any],
            cid: str | None = None,
    ) -> None:
        self.cid: str = cid or uuid()
        self.tokens: str = tokens
        self.args: List = list(args)
        self.kwargs: Dict[str, Any] = kwargs
        self.real_kwargs: Dict[str, Any] = command.parse_kwargs(*self.args, **self.kwargs)
        self.state: CommandState = "created"
        self.command: Command = command
        self.errcode: int = 0
        self.errmsg: Optional[str] = None
        self.trace: Dict[CommandState, float] = {}
        self.result: Optional[RESULT] = None
        self._done_event: threading.Event = threading.Event()
        self._done_lock = threading.Lock()
        self._done = False
        self._awaits: List[Tuple[asyncio.AbstractEventLoop, asyncio.Event]] = []

    def is_done(self) -> bool:
        """
        命令已经结束.
        """
        return self._done_event.is_set()

    def cancel(self, reason: str = ""):
        """
        停止命令.
        """
        self._set_result(None, 'cancelled', 1, reason)

    def _add_await(self, loop: asyncio.AbstractEventLoop, event: asyncio.Event):
        with self._done_lock:
            if self._done_event.is_set():
                loop.call_soon_threadsafe(event.set)
            else:
                self._awaits.append((loop, event))

    def _set_result(self, result: Optional[RESULT], state: CommandState, errcode: int, errmsg: Optional[str]) -> bool:
        with self._done_lock:
            if self._done_event.is_set():
                return False
            self.result = result
            self.errcode = errcode
            self.errmsg = errmsg
            self._done_event.set()
            awaits = self._awaits.copy()
            self._awaits.clear()
            for loop, event in awaits:
                loop.call_soon_threadsafe(event.set)
            self.set_state(state)
            return True

    @abstractmethod
    def fail(self, error: Exception | str) -> None:
        if not self._done_event.is_set():
            if isinstance(error, str):
                errmsg = error
                errcode = -1
            elif isinstance(error, CommandError):
                errcode = error.code
                errmsg = error.message
            elif isinstance(error, asyncio.CancelledError):
                errcode = 1
                errmsg = str(error)
            elif isinstance(error, Exception):
                errcode = -1
                errmsg = str(error)
            else:
                errcode = 0
                errmsg = ""
            self._set_result(None, errcode, errmsg)
            self.set_state("failed")

    @abstractmethod
    def resolve(self, result: RESULT) -> None:
        if not self._done_event.is_set():
            self._set_result(result, 'done', 0, None)

    def raise_error(self) -> None:
        if self.errcode is None or self.errcode == 0:
            return None
        elif self.errcode == 1:
            raise asyncio.CancelledError(self.errmsg)
        else:
            raise CommandError(self.errcode, self.errmsg or "")

    async def wait_done(
            self,
            *,
            throw: bool = False,
            timeout: float | None = None,
    ) -> Optional[RESULT]:
        """
        等待命令被执行完毕. 但不会主动运行这个任务. 仅仅是等待.
        Command Task 的 Await done 要求跨线程安全.
        """
        if self._done_event.is_set():
            if throw:
                self.raise_error()
            return self.result

        loop = asyncio.get_running_loop()
        event = asyncio.Event()
        self._add_await(loop, event)

        await asyncio.wait_for(event.wait(), timeout=timeout)
        if throw:
            self.raise_error()
        return self.result

    def wait(self, *, throw: bool = False, timeout: float | None = None) -> Optional[RESULT]:
        """
        线程的 wait.
        """
        self._done_event.wait(timeout=timeout)
        if throw:
            self.raise_error()
        return self.result


class CommandTaskSeq:

    def __init__(self, *tasks: BasicCommandTask) -> None:
        self.tasks = tasks

    def __iter__(self):
        return iter(self.tasks)
