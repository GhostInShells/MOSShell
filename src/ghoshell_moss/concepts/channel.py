import asyncio
import contextvars
import threading
from abc import ABC, abstractmethod
from typing import (
    Iterable, Optional, Union, Callable, Coroutine, List, Type, TypeVar, Dict, Any,
    Protocol, AsyncIterator
)
from typing_extensions import Self
from ghoshell_moss.concepts.command import Command, CommandMeta, CommandTask
from ghoshell_container import IoCContainer, INSTANCE, Provider, BINDING
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

__all__ = [
    'FunctionCommand', 'LifecycleFunction', 'PrompterCommand', 'StringType',
    'ChannelMeta', 'Channel', 'ChannelServer', 'ChannelClient',
]

FunctionCommand = Union[Callable[..., Coroutine], Callable[..., Any]]
"""通常要求是异步函数, 如果是同步函数的话, 会卸载到线程池运行"""

LifecycleFunction = Union[Callable[..., Coroutine[None, None, None]], Callable[..., None]]

PrompterCommand = Union[Callable[..., Coroutine[None, None, str]], Callable[..., str]]

StringType = Union[str, Callable[[], str]]

R = TypeVar('R')


class ChannelMeta(BaseModel):
    """
    Channel 的元信息数据.
    可以用来 mock 一个 channel.
    """
    name: str = Field(description="The name of the channel.")
    available: bool = Field(description="Whether the channel is available.")
    description: str = Field(description="The description of the channel.")
    commands: List[CommandMeta] = Field(default_factory=list, description="The list of commands.")
    children: List[str] = Field(default_factory=list, description="the children channel names")
    # stats: List[State] = Field(default_factory=list, description="The list of state objects.")
    # instruction: str = Field(default="", description="The instruction of the channel.")
    # context: str = Field(default="", description="the runtime context of the channel.")


class ChannelClient(Protocol):
    """
    channel 的运行时方法.
    只有在 channel.start 之后才可使用.
    用于控制 channel 的所有能力.
    """

    container: IoCContainer
    """
    运行时 IoC 容器.
    """

    id: str
    """unique id of the channel client instance"""

    @abstractmethod
    def is_running(self) -> bool:
        pass

    @abstractmethod
    def meta(self, no_cache: bool = False) -> ChannelMeta:
        """
        返回 Channel 自身的 Meta.
        可能是动态更新的.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        当前 Channel Runtime 是否可用.
        """
        pass

    @abstractmethod
    def commands(self, available_only: bool = True) -> Dict[str, Command]:
        """
        返回所有 commands.
        不递归.
        """
        pass

    @abstractmethod
    def get_command(self, name: str) -> Optional[Command]:
        """
        查找一个 command.
        不递归.
        """
        pass

    @abstractmethod
    async def execute(self, task: CommandTask[R]) -> R:
        """
        在 channel 自带的上下文中执行一个 task.
        不递归.
        """
        pass

    @abstractmethod
    def update(self) -> ChannelMeta:
        """
        结合所有上下文, 触发更新当前 Channel.
        """
        pass

    @abstractmethod
    async def policy_run(self) -> None:
        """
        回归 policy 运行. 通常在一个队列里没有 function 在运行中时, 会运行 policy. 同时 none-block 的函数也不会中断 policy 运行.
        不会递归执行.
        """
        pass

    @abstractmethod
    async def policy_pause(self) -> None:
        """
        接受到了新的命令, 要中断 policy
        不会递归执行.
        """
        pass

    @abstractmethod
    async def clear(self) -> None:
        """
        当清空命令被触发的时候.
        不会递归执行.
        """
        pass

    @abstractmethod
    async def start(self) -> None:
        """
        启动 Channel 运行.
        注意: 会递归启动所有的子 channel. 这是因为子 channel 通常会和父 channel 共享通信通道.
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        关闭当前 Runtime. 同时阻塞销毁资源直到结束.
        注意, 会递归执行, 关闭所有的子 channel.
        """
        pass

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


class Builder(ABC):

    @abstractmethod
    def with_description(self) -> Callable[[StringType], StringType]:
        """
        注册一个全局唯一的函数, 用来动态生成 description.
        """
        pass

    @abstractmethod
    def with_available(self) -> Callable[[Callable[[], bool]], Callable[[], bool]]:
        pass

    @abstractmethod
    def command(
            self,
            *,
            name: str = "",
            chan: str | None = None,
            doc: Optional[StringType] = None,
            comments: Optional[StringType] = None,
            tags: Optional[List[str]] = None,
            interface: Optional[StringType] = None,
            available: Optional[Callable[[], bool]] = None,
            # --- 高级参数 --- #
            block: Optional[bool] = None,
            call_soon: bool = False,
    ) -> Callable[[FunctionCommand], FunctionCommand]:
        """
        返回 decorator 将一个函数注册到当前 Channel 里.
        对于 Channel 而言, Function 通常是会有运行时间的. 阻塞的命令, Channel 会一个一个执行.

        :param name: 改写这个函数的名称.
        :param chan: 设置这个命令所属的 channel.
        :param doc: 获取函数的描述, 可以使用动态函数.
        :param comments: 改写函数的 body 部分, 用注释形式提供的字符串. 每行前会自动添加 '#'. 不用手动添加.
        :param interface: 大模型看到的函数代码形式. 一旦定义了这个, doc, name, comments 就都会失效.
                          通常是
                          async def foo(...) -> ...:
                            '''docstring'''
                            # comments
                            pass
        :param tags: 标记函数的分类. 可以用来做筛选, 如果有这个逻辑的话.
        :param block: 这个函数是否会阻塞 channel. 默认都会阻塞.
        :param available: 通过函数定义这个命令是否 available.
        :param call_soon: 决定这个函数进入轨道后, 会第一时间执行 (不等待调度), 还是等待排队执行到自身时.
                          如果是 block + call_soon, 会先清空队列.
        """
        pass

    @abstractmethod
    def on_policy_run(self, run_policy: LifecycleFunction) -> LifecycleFunction:
        """
        注册一个函数, 当 Channel 运行 policy 时, 会执行这个函数.
        """
        pass

    @abstractmethod
    def on_policy_pause(self, pause_policy: LifecycleFunction) -> LifecycleFunction:
        """
        policy 回调.
        """
        pass

    @abstractmethod
    def on_clear(self, clear_func: LifecycleFunction) -> LifecycleFunction:
        """
        清空
        """
        pass

    @abstractmethod
    def on_start_up(self, start_func: LifecycleFunction) -> LifecycleFunction:
        """
        启动时执行的回调.
        """
        pass

    @abstractmethod
    def on_stop(self, stop_func: LifecycleFunction) -> LifecycleFunction:
        """
        关闭时的回调.
        """
        pass

    @abstractmethod
    def with_providers(self, *providers: Provider) -> Self:
        """
        提供依赖的注册能力. runtime.container 将持有这些依赖.
        register default providers for the contracts
        """
        pass

    @abstractmethod
    def with_contracts(self, *contracts: Type) -> Self:
        """
        声明 IoC 容器需要的依赖. 如果启动时传入的 IoC 容器没有注册这些依赖, 则启动本身会报错, 抛出异常.
        """
        pass

    @abstractmethod
    def with_binding(self, contract: Type[INSTANCE], binding: Optional[BINDING] = None) -> Self:
        """
        register default bindings for the given contract.
        """
        pass


ChannelContextVar = contextvars.ContextVar('MOSShell_Channel')


class Channel(ABC):
    """
    Shell 可以使用的命令通道.
    """

    @abstractmethod
    def name(self) -> str:
        """
        channel 的名字.
        """
        pass

    def set_context_var(self) -> None:
        ChannelContextVar.set(self)

    @classmethod
    def get_from_context(cls) -> Optional[Self]:
        try:
            return ChannelContextVar.get()
        except LookupError:
            return None

    @property
    @abstractmethod
    def client(self) -> ChannelClient:
        """
        Channel 在 bootstrap 之后返回的运行时.
        :raise RuntimeError: Channel 没有运行
        """
        pass

    # --- children --- #

    @abstractmethod
    def with_children(self, *children: "Channel", parent: Optional[str] = None) -> Self:
        """
        添加子 Channel 到当前 Channel. 形成树状关系.
        :raise LookupError: 如果出现重名会发出这个异常.
        """
        pass

    @abstractmethod
    def new_child(self, name: str) -> Self:
        """
        生成一个子 channel.
        :raise NotImplementError: 没有实现的话.
        """
        pass

    @abstractmethod
    def children(self) -> Dict[str, "Channel"]:
        """
        返回所有已注册的子 Channel.
        """
        pass

    @abstractmethod
    def descendants(self) -> Dict[str, "Channel"]:
        """
        返回所有的子孙 Channel, 先序遍历.
        """
        pass

    @abstractmethod
    def get_channel(self, name: str) -> Optional[Self]:
        """
        使用 channel 名从树中获取一个 Channel 对象. 包括自身.
        """
        pass

    # --- lifecycle --- #

    @abstractmethod
    def is_running(self) -> bool:
        """
        自身是不是 running 状态, 如果是, 则可以拿到 client
        """
        pass

    @abstractmethod
    def bootstrap(self, container: Optional[IoCContainer] = None, depth: int = 0) -> "ChannelClient":
        """
        传入一个父容器, 启动 Channel. 同时生成 Runtime.
        真正运行的是 channel runtime.
        """
        pass

    @property
    @abstractmethod
    def build(self) -> Builder:
        """
        用来快速包装各种函数.
        """
        pass


class ChannelServer(ABC):
    """
    将 Channel 包装成一个 Server, 可以被上层的 Client 调用.
    上层的 Client 将通过通讯协议, 还原出 Client 树, 但这个 Client 树里所有子 channel 都通过 Server 的通讯协议来传递.
    从而形成链式的封装关系, 在不同进程里还原出树形的架构.

    举例:
    ReverseWebsocketClient => ReverseWebsocketServer => ZMQClient => ZMQServer ... => Client
    """
    container: IoCContainer
    loop: asyncio.AbstractEventLoop | None = None
    channel: Channel | None = None

    async def arun_until_closed(self, channel: Channel) -> None:
        """
        运行 Client 服务.
        """
        self.loop = asyncio.get_running_loop()
        self.channel = channel
        task = await self.create_server_task(channel)
        await self.wait_closed()
        if not task.done():
            task.cancel()
        await task

    @abstractmethod
    async def create_server_task(self, channel: Channel) -> asyncio.Task:
        pass

    @abstractmethod
    async def wait_closed(self) -> None:
        pass

    @abstractmethod
    async def aclose(self) -> None:
        """
        主动关闭 server.
        """
        pass

    def run_until_closed(self, channel: Channel) -> None:
        """
        展示如何同步运行.
        """
        asyncio.run(self.arun_until_closed(channel))

    def run_in_thread(self, channel: Channel) -> None:
        """
        展示如何在多线程中异步运行.
        """
        thread = threading.Thread(target=self.run_until_closed, args=(channel,), daemon=True)
        thread.start()

    def close(self) -> None:
        self.loop.run_until_complete(self.aclose())

    @asynccontextmanager
    async def run(self, channel: Channel) -> AsyncIterator[Self]:
        """
        支持 with statement 的运行方式.
        """
        running = asyncio.create_task(self.arun_until_closed(channel=channel))
        yield self
        await self.aclose()
        running.cancel()
        await running
