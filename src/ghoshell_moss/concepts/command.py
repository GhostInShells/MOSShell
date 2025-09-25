import asyncio
import threading
from abc import ABC, abstractmethod
from typing import (
    TypedDict, Literal, Optional, Dict, Any, Awaitable, List, Generic, TypeVar, Tuple, Callable, Coroutine, Union,
    is_typeddict, Protocol, Iterable,
)
from typing_extensions import Self
from ghoshell_common.helpers import uuid, generate_import_path
from ghoshell_moss.helpers.func import parse_function_interface
from ghoshell_moss.helpers.event import ThreadSafeEvent
from .errors import CommandError
from pydantic import BaseModel, Field
from enum import Enum
import traceback
import inspect
import time

RESULT = TypeVar("RESULT")

CommandTaskStateType = Literal['created', 'queued', 'pending', 'running', 'failed', 'done', 'cancelled']


class CommandTaskState(str, Enum):
    CREATED = "created"
    QUEUED = "queued"
    PENDING = "pending"
    RUNNING = "running"
    FAILED = "failed"
    DONE = "done"
    CANCELLED = "cancelled"


StringType = Union[str, Callable[[], str]]


class CommandDeltaType(str, Enum):
    TEXT = "text__"
    TOKENS = "tokens__"

    @classmethod
    def all(cls) -> set[str]:
        return {cls.TEXT.value, cls.TOKENS.value}


CommandDeltaTypeMap = {
    CommandDeltaType.TEXT.value: "the deltas are text string",
    CommandDeltaType.TOKENS.value: "the delta are commands, transporting as Iterable[CommandToken]",
}
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


class CommandTokenType(str, Enum):
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

    * start: 携带 command 的参数信息.
    * delta: 表示这个 command 所接受到的流式输入.
    * stop: 表示一个 command 已经结束.
    """
    type: Literal['start', 'delta', 'end'] = Field(description="tokens type")

    name: str = Field(description="command name")

    order: int = Field(default=0, description="the output order of the command")

    cmd_idx: int = Field(description="command index of the stream")

    part_idx: int = Field(description="continuous part idx of the command. start, delta, delta, end are four parts")

    stream_id: Optional[str] = Field(description="the id of the stream the command belongs to")

    content: str = Field(description="origin tokens that llm generates")

    kwargs: Optional[Dict[str, Any]] = Field(default=None, description="attributes, only for command start")

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

    def __str__(self):
        return self.content


class CommandMeta(BaseModel):
    """
    命令的原始信息.
    """
    name: str = Field(
        description="the name of the command"
    )
    chan: str = Field(
        default="",
        description="the channel name that the command belongs to"
    )
    description: str = Field(
        default="",
        description="the doc of the command",
    )
    available: bool = Field(
        default=True,
        description="whether this command is available",
    )
    type: CommandType = Field(
        default=CommandType.FUNCTION.value,
        description="",
        json_schema_extra=dict(enum=CommandType.all()),
    )
    tags: List[str] = Field(default_factory=list, description="tags of the command")
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
                    "```"
    )
    args_schema: Optional[Dict[str, Any]] = Field(
        default=None,
        description="the json schema. 兼容性实现.",
    )

    # --- advance options --- #

    call_soon: bool = Field(
        default=False,
        description="if true, this command is called soon when append to the channel",
    )
    block: bool = Field(
        default=True,
        description="whether this command block the channel. if block + call soon, will clear the channel first",
    )


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

    @abstractmethod
    def is_available(self) -> bool:
        pass

    @abstractmethod
    def meta(self) -> CommandMeta:
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
            chan: str = "",
            name: Optional[str] = None,
            available: Callable[[], bool] | None = None,
            interface: Optional[StringType] = None,
            doc: Optional[StringType] = None,
            comments: Optional[StringType] = None,
            meta: Optional[CommandMeta] = None,
            tags: Optional[List[str]] = None,
            call_soon: bool = False,
            block: bool = True,
    ):
        """
        :param func: origin coroutine function
        :param meta: the defined command meta information
        :param available: if given, determine if the command is available dynamically
        :param interface: if not given, will reflect the origin function signature to generate the interface.
        :param doc: if given, will change the docstring of the function or generate one dynamically
        :param comments: if given, will add to the body of the function interface.
        """
        self._chan = chan
        self._func_name = func.__name__
        if not name:
            name = self._func_name
        fullname = self.make_fullname(self._chan, name)
        self._name = fullname
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
        self._block = block
        self._tags = tags
        self._meta = meta
        delta_arg = None
        for arg_name in self._func_itf.signature.parameters.keys():
            if arg_name in CommandDeltaTypeMap:
                if delta_arg is not None:
                    raise AttributeError(f"function {func} has more than one delta arg {meta.delta_arg} and {arg_name}")
                delta_arg = arg_name
                # only first delta_arg type. and not allow more than 1
                break
        self._delta_arg = delta_arg

    @classmethod
    def make_fullname(cls, chan: str, name: str) -> str:
        return f"{chan}_{name}" if chan else name

    def name(self) -> str:
        if self._meta is not None:
            return self._meta.name
        return self._name

    def is_available(self) -> bool:
        return self._available_or_fn() if self._available_or_fn is not None else True

    def meta(self) -> CommandMeta:
        if self._meta is not None:
            meta = self._meta.model_copy()
            meta.available = self.is_available()
            return meta

        meta = CommandMeta(name=self._name)
        meta.description = self._unwrap_string_type(self._doc_or_fn, meta.description)
        meta.interface = self._gen_interface(meta.name, meta.description)
        meta.available = self.is_available()
        meta.delta_arg = self._delta_arg
        meta.call_soon = self._call_soon
        meta.tags = self._tags or []
        meta.block = self._block

        if not self._is_dynamic_itf:
            self._meta = meta
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

    def parse_kwargs(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        real_kwargs = self._func_itf.prepare_kwargs(*args, **kwargs)
        return real_kwargs

    async def __call__(self, *args, **kwargs) -> RESULT:
        real_kwargs = self.parse_kwargs(*args, **kwargs)
        if self._is_coroutine_func:
            return await self._func(**real_kwargs)
        else:
            task = asyncio.to_thread(self._func, **real_kwargs)
            return await task


class CommandTask(Generic[RESULT], ABC):
    """
    线程安全的 Command Task 对象. 相当于重新实现一遍 asyncio.Task 类似的功能.
    有区别的原理:
    1. 建立全局唯一的 cid, 方便在双工通讯中赋值.
    2. 必须实现线程安全, 因为通讯可能是在多线程里.
    3. 包含 debug 需要的 state, trace 等信息.
    4. 保留命令的元信息, 包括入参等.
    5. 不是立刻启动, 而是被 channel 调度时才运行.
    6. 兼容 json rpc 协议, 方便跨进程通讯.
    7. 可复制, 复制后可重入, 方便做循环.
    """

    cid: str
    tokens: str
    args: List
    kwargs: Dict[str, Any]
    state: CommandTaskStateType
    errcode: int = 0
    errmsg: Optional[str] = None
    trace: Dict[CommandTaskStateType, float] = {}
    result: Optional[RESULT] = None
    meta: CommandMeta
    func: Callable[..., Coroutine[None, None, RESULT]]

    @abstractmethod
    def done(self) -> bool:
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

    def set_state(self, state: CommandTaskStateType) -> None:
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
    async def run(self, *args, **kwargs) -> RESULT:
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

    def __repr__(self):
        return f"<CommandTask name=`{self.meta.name}` cid=`{self.cid}` tokens=`{self.tokens}` args=`{self.args}` kwargs=`{self.kwargs}`>"


class BaseCommandTask(Generic[RESULT], CommandTask[RESULT]):
    """
    大模型的输出被转化成 CmdToken 后, 再通过执行器生成的运行时对象.
    实现一个跨线程安全的等待机制.
    """

    def __init__(
            self,
            *,
            meta: CommandMeta,
            func: Callable[..., Coroutine[None, None, RESULT]],
            tokens: str,
            args: list,
            kwargs: Dict[str, Any],
            cid: str | None = None,
    ) -> None:
        self.cid: str = cid or uuid()
        self.tokens: str = tokens
        self.args: List = list(args)
        self.kwargs: Dict[str, Any] = kwargs
        self.state: CommandTaskStateType = "created"
        self.meta = meta
        self.func = func
        self.errcode: int = 0
        self.errmsg: Optional[str] = None
        self.trace: Dict[CommandTaskStateType, float] = {
            "created": time.time(),
        }
        self.result: Optional[RESULT] = None
        self._done_event: ThreadSafeEvent = ThreadSafeEvent()
        self._done_lock = threading.Lock()

    def copy(self, cid: str = "") -> Self:
        cid = cid or uuid()
        return BaseCommandTask(
            cid=cid,
            meta=self.meta.model_copy(),
            func=self.func,
            tokens=self.tokens,
            args=self.args,
            kwargs=self.kwargs,
        )

    @classmethod
    def from_command(cls, command_: Command[RESULT], *args, tokens_: str = "", **kwargs) -> "BaseCommandTask":
        return cls(
            meta=command_.meta(),
            func=command_.__call__,
            tokens=tokens_,
            args=list(args),
            kwargs=kwargs,
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
        self._set_result(None, 'cancelled', CommandError.CANCEL_CODE, reason)

    def _set_result(
            self,
            result: Optional[RESULT],
            state: CommandTaskStateType,
            errcode: int,
            errmsg: Optional[str],
    ) -> bool:
        with self._done_lock:
            if self._done_event.is_set():
                return False
            self.result = result
            self.errcode = errcode
            self.errmsg = errmsg
            self._done_event.set()
            self.set_state(state)
            return True

    def fail(self, error: Exception | str) -> None:
        import sys
        if not self._done_event.is_set():
            if isinstance(error, str):
                errmsg = error
                errcode = CommandError.UNKNOWN_CODE
            elif isinstance(error, CommandError):
                errcode = error.code
                errmsg = error.message
            elif isinstance(error, asyncio.CancelledError):
                errcode = CommandError.CANCEL_CODE
                errmsg = "".join(traceback.format_exception(error, limit=3))
            elif isinstance(error, Exception):
                errcode = CommandError.UNKNOWN_CODE
                errmsg = "".join(traceback.format_exception(error, limit=3))
            else:
                errcode = 0
                errmsg = ""
            self._set_result(None, "failed", errcode, errmsg)

    def resolve(self, result: RESULT) -> None:
        if not self._done_event.is_set():
            self._set_result(result, 'done', 0, None)

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
        if self._done_event.is_set():
            if throw:
                self.raise_exception()
            return self.result

        await asyncio.wait_for(self._done_event.wait(), timeout=timeout)
        if throw:
            self.raise_exception()
        return self.result

    def wait_sync(self, *, throw: bool = True, timeout: float | None = None) -> Optional[RESULT]:
        """
        线程的 wait.
        """
        if not self._done_event.wait_sync():
            raise TimeoutError(f"wait timeout: {timeout}")
        if throw:
            self.raise_exception()
        return self.result

    async def run(self) -> RESULT:
        """典型的案例如何使用一个 command task """
        if self.done():
            self.raise_exception()
            return self.result

        async def resolve() -> None:
            r = await self.func(*self.args, **self.kwargs)
            self.resolve(r)

        async def wait() -> None:
            await self.wait()
            self.raise_exception()

        try:
            result_task = asyncio.create_task(resolve())
            wait_task = asyncio.create_task(wait())
            await asyncio.gather(result_task, wait_task)
            return self.result

        except asyncio.CancelledError as e:
            if not self.done():
                self.cancel(reason=str(e))
            raise
        except Exception as e:
            if not self.done():
                self.fail(e)
            raise
        finally:
            if not self.done():
                self.cancel()


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
            type=CommandType.CONTROL.value,
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
            chan=current.meta.chan,
            type=CommandType.CONTROL.value,
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
            meta=meta,
            func=wait_done_then_cancel,
            tokens=tokens,
            args=[],
            kwargs={},
        )


class CommandTaskSeq:
    """特殊的数据结构, 用来标记一个 task 序列, 也可以由 task 返回. """

    def __init__(self, *tasks: BaseCommandTask) -> None:
        self.tasks = tasks

    def __iter__(self):
        return iter(self.tasks)

    def __str__(self):
        return "".join([t.tokens for t in self.tasks])
