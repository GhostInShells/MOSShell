from abc import ABC, abstractmethod
from typing import TypedDict, Literal, Optional, Dict, Any, Awaitable, List, Generic, TypeVar
from ghoshell_common.helpers import uuid
from queue import Queue
from pydantic import BaseModel, Field
from enum import Enum
import time

RESULT = TypeVar("RESULT")

CommandState = Literal['created', 'queued', 'pending', 'running', 'failed', 'done', 'cancelled']

CommandDeltaType = Literal['_text', '_json', '_xml', '_yaml', '_markdown', '_python', '_stream']


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


class CommandToken(TypedDict):
    """
    将大模型流式输出的文本结果, 包装为流式的 Command Token 对象.
    整个 Command 的生命周期是: start -> ?[delta -> ... -> delta] -> end
    在生命周期中所有被包装的 token 都带有相同的 cid.

    * start: 携带 command 的参数信息.
    * delta: 表示这个 command 所接受到的流式输入.
    * stop: 表示一个 command 已经结束.
    """

    name: str
    """command name"""

    chan: Optional[str]
    """the channel name that the command belongs to """

    cid: str
    """command unique id"""

    type: Literal['start', 'delta', 'end']
    """tokens type"""

    content: str
    """origin tokens that llm generates"""

    kwargs: Optional[Dict[str, Any]]
    """attributes, only for command start"""


def cmd_start(content: str, chan: Optional[str], name: str, kwargs: Dict[str, Any], cid: str = "") -> CommandToken:
    cid = cid or uuid()
    return CommandToken(
        name=name,
        chan=chan,
        type="start",
        kwargs=kwargs,
        content=content,
        cid=cid,
    )


def cmd_end(content: str, chan: Optional[str], name: str, cid: str) -> CommandToken:
    return CommandToken(
        name=name,
        chan=chan,
        type="end",
        content=content,
        cid=cid,
        kwargs=None
    )


def cmd_delta(content: str, chan: Optional[str], name: str, cid: str) -> CommandToken:
    return CommandToken(
        name=name,
        chan=chan,
        type="delta",
        content=content,
        cid=cid,
        kwargs=None,
    )


class CommandMeta(BaseModel):
    """
    命令的原始信息.
    """
    name: str = Field(description="the name of the command")
    chan: str = Field(description="the channel name that the command belongs to")
    doc: str = Field(default="", description="the doc of the command")
    type: CommandType = Field(description="")
    delta_arg: Optional[CommandDeltaType] = Field(default=None, description="the delta arg type")
    interface: str = Field(
        description="大模型所看到的关于这个命令的 prompt. 类似于 FunctionCall 协议提供的 JSON Schema."
                    "但核心思想是 Code As Prompt."
                    "通常是一个 python async 函数的 signature. 形如:"
                    "```python"
                    "async def name(arg: typehint = default) -> return_type:"
                    "    ''' docstring '''"
                    "    pass"
                    "```"
    )


class CommandTask(ABC, Awaitable[RESULT]):
    """
    大模型的输出被转化成 CmdToken 后, 再通过执行器生成的运行时对象.
    """
    cid: str
    args: List[Any]
    kwargs: Dict[str, Any]
    name: str
    chan: str
    meta: CommandMeta
    state: CommandState = "created"
    trace: Dict[CommandState, float] = {}
    none_block: bool = False
    errcode: Optional[int] = None
    errmsg: Optional[str] = None
    result: Optional[RESULT] = None

    @abstractmethod
    def is_done(self) -> bool:
        """
        命令已经结束.
        """
        pass

    @abstractmethod
    def cancel(self, reason: str = ""):
        """
        停止命令.
        """
        pass

    def set_state(self, state: CommandState) -> None:
        self.state = state
        self.trace[state] = time.time()

    @abstractmethod
    def fail(self, error: Exception | str) -> None:
        pass

    @abstractmethod
    def resolve(self, result: RESULT) -> None:
        pass

    @abstractmethod
    async def wait_done(
            self,
            timeout: float | None = None,
    ) -> None:
        """
        等待命令被执行完毕. 但不会主动运行这个任务. 仅仅是等待.
        """
        pass


class TaskStack:

    def __init__(self, *tasks: CommandTask) -> None:
        self.tasks = tasks


class Command(Generic[RESULT], ABC):
    """
    对大模型可见的命令描述. 包含几个核心功能:
    大模型通常能很好地理解, 并且使用这个函数.

    这个 Command 本身还会被伪装成函数, 让大模型可以直接用代码的形式去调用它.
    Shell 也将支持一个直接执行代码的控制逻辑, 形如 <exec> ... </exec> 的方式, 用 asyncio 语法直接执行它所看到的 Command
    """

    @abstractmethod
    def meta(self) -> CommandMeta:
        """
        返回 Command 的元信息.
        """
        pass

    @abstractmethod
    def __prompt__(self) -> str:
        """
        大模型所看到的关于这个命令的 prompt. 类似于 FunctionCall 协议提供的 JSON Schema.
        但核心思想是 Code As Prompt.

        通常是一个 python async 函数的 signature. 形如:
        ```python
        async def name(arg: typehint = default) -> return_type:
            ''' docstring '''
            pass
        ```
        """
        pass

    @abstractmethod
    async def __call__(self, *args, **kwargs) -> RESULT:
        """
        基于入参, 出参, 生成一个 CommandCall 交给调度器去执行.
        """
        pass
