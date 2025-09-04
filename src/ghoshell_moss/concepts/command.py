from abc import ABC, abstractmethod
from typing import TypedDict, Literal, Optional, Dict, Any, Awaitable, List, Generic, TypeVar
from ghoshell_common.helpers import uuid
from queue import Queue
import time
import asyncio

RESULT = TypeVar("RESULT")


class CommandToken(TypedDict):
    """
    将大模型流式输出的文本结果, 包装为流式的 Command Token 对象.
    整个 Command 的生命周期是: start -> ?[delta -> ... -> delta] -> stop
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

    type: Literal['init', 'delta', 'end']
    """tokens type"""

    text: str
    """origin tokens that llm generates"""

    attrs: Optional[Dict[str, Any]]
    """attributes, only for command start"""


class CommandTokenStream:
    """
    thread level iterable command token stream
    """

    def __init__(self):
        self.tokens: Queue[CommandToken | None] = Queue()
        self.stopped = False

    def append(self, token: CommandToken | None):
        if self.stopped:
            return
        if token is None:
            self.stopped = True
        self.tokens.put_nowait(token)

    def __iter__(self):
        return self

    def __next__(self):
        if self.stopped:
            raise StopIteration()
        result = self.tokens.get(block=True)
        if result is None:
            raise StopIteration()
        return result


def cmd_init(tokens: str, chan: Optional[str], name: str, attrs: Dict[str, Any], cid: str = "") -> CommandToken:
    cid = cid or uuid()
    return CommandToken(
        name=name,
        chan=chan,
        type="init",
        attrs=attrs,
        text=tokens,
        cid=cid,
    )


def cmd_end(tokens: str, chan: Optional[str], name: str, cid: str) -> CommandToken:
    return CommandToken(
        name=name,
        chan=chan,
        type="end",
        text=tokens,
        cid=cid,
        attrs=None
    )


def cmd_delta(tokens: str, chan: Optional[str], name: str, cid: str) -> CommandToken:
    return CommandToken(
        name=name,
        chan=chan,
        type="delta",
        text=tokens,
        cid=cid,
        attrs=None,
    )


CommandState = Literal['created', 'queued', 'pending', 'running', 'stopped']


class CommandTask(ABC, Awaitable[RESULT]):
    """
    大模型的输出被转化成 CmdToken 后, 再通过执行器生成的运行时对象.
    """

    name: str
    """command name"""

    chan: str
    """the command belongs to"""

    cid: str
    """command unique id"""

    args: List[Any]
    """调用时的入参. """

    kwargs: Dict[str, Any]
    """调用时的 keyword 入参. """

    state: CommandState = 'created'

    trace: Dict[CommandState, float] = {}
    """运行生命周期的时间点记录. """

    #  --- result of the command --- #
    code: int = 0
    error: Optional[str] = None
    data: Optional[RESULT] = None

    @abstractmethod
    def is_done(self) -> bool:
        """
        命令已经结束.
        """
        pass

    def set_state(self, state: CommandState):
        self.state = state
        self.trace[state] = time.time()

    @abstractmethod
    def cancel(self):
        """
        停止命令.
        """
        pass

    @abstractmethod
    async def send(self, delta: str) -> None:
        """
        向命令发送流式的输入. 如果它需要的话.
        """
        pass

    async def wait_until_done(
            self,
            timeout: float | None = None,
    ) -> RESULT:
        """
        等待命令被执行完毕. 但不会主动运行这个任务. 仅仅是等待.
        如果定义了 Timeout, 会在 Timeout 后抛出异常.
        """
        pass

    @abstractmethod
    def __await__(self) -> RESULT:
        """
        运行并等待
        """
        pass


class Command(ABC):
    """
    对大模型可见的命令描述. 包含几个核心功能:
    大模型通常能很好地理解, 并且使用这个函数.

    这个 Command 本身还会被伪装成函数, 让大模型可以直接用代码的形式去调用它.
    """

    name: str
    """命令的名字"""

    chan: str
    """命令所属的 channel. 通常 channel:name 可以构成一个全局唯一的 Command. """

    type: Literal['function', 'policy', 'meta', 'control']
    """
    命令的基础类型: 
    - function: 功能, 需要一段时间执行, 执行完后结束. 
    - policy:   状态变更函数. 会改变 Command 所属 Channel 的运行策略, 立刻生效. 
                Channel 在没有 Function 执行时, 会持续执行 policy. 
    - meta:     meta-agent 可以通过 meta 类型命令, 修改这个 channel, 比如创建新的函数. 不对普通 agent 暴露.   
    - control:  control 类型的命令对 channel 有最高控制权限, 通常只向人类进行开放.  
    """

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
    def __call__(self, *args, **kwargs) -> CommandTask:
        """
        基于入参, 出参, 生成一个 CommandCall 交给调度器去执行.
        """
        pass
