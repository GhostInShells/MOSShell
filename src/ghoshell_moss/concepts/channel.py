import asyncio
import contextvars
import threading
from abc import ABC, abstractmethod
from typing import (
    Iterable, Optional, Union, Callable, Coroutine, List, Type, TypeVar, Dict, Any,
    Protocol, AsyncIterator, Tuple
)
from typing_extensions import Self
from ghoshell_moss.concepts.command import Command, CommandMeta, CommandTask, BaseCommandTask
from ghoshell_moss.concepts.states import StateStore, StateModel
from ghoshell_container import IoCContainer, INSTANCE, Provider, BINDING, set_container
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

__all__ = [
    'CommandFunction', 'LifecycleFunction', 'PrompterFunction', 'StringType',
    'ChannelMeta', 'Channel', 'ChannelServer', 'ChannelClient',
    'Builder',
    'R',
    'ChannelPaths', 'ChannelFullPath',
]

ChannelFullPath = str
ChannelPaths = List[str]

CommandFunction = Union[Callable[..., Coroutine], Callable[..., Any]]
"""通常要求是异步函数, 如果是同步函数的话, 会卸载到线程池运行"""

LifecycleFunction = Union[Callable[..., Coroutine[None, None, None]], Callable[..., None]]

PrompterFunction = Union[Callable[..., Coroutine[None, None, str]], Callable[..., str]]

StringType = Union[str, Callable[[], str]]

R = TypeVar('R')


class ChannelMeta(BaseModel):
    """
    Channel 的元信息数据.
    可以用来 mock 一个 channel.
    """
    name: str = Field(description="The origin name of the channel, kind like python module name.")
    channel_id: str = Field(default="", description="The ID of the channel.")
    available: bool = Field(default=True, description="Whether the channel is available.")
    description: str = Field(default="", description="The description of the channel.")
    commands: List[CommandMeta] = Field(default_factory=list, description="The list of commands.")
    children: List[str] = Field(default_factory=list, description="the children channel names")
    # states: List[State] = Field(default_factory=list, description="The list of state objects.")
    # instruction: str = Field(default="", description="The instruction of the channel.")
    # context: str = Field(default="", description="the runtime context of the channel.")


class ChannelClient(ABC):
    """
    channel 的运行时方法.
    只有在 channel.start 之后才可使用.
    用于控制 channel 的所有能力.
    """

    def __init__(
            self,
            id: str,  # unique id of the channel client instance
            container: IoCContainer,  # 运行的 ioc 容器.
    ):
        self.id = id
        self.container = container

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def is_running(self) -> bool:
        """
        是否已经启动了.
        """
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
        当前 Channel Client 是否可用.
        当一个 Client 是 running 状态下, 仍然可能会有被暂停等因素导致它暂时不能用.
        """
        pass

    @abstractmethod
    def commands(self, available_only: bool = True) -> Dict[str, Command]:
        """
        返回所有 commands. 注意, 只返回 Channel 自身的 Command.
        """
        pass

    @abstractmethod
    def get_command(self, name: str) -> Optional[Command]:
        """
        查找一个 command. 只返回自身的 command.
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

    @property
    @abstractmethod
    def state_store(self) -> StateStore:
        """
        返回当前 Channel 的状态存储.
        """
        pass


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
    def state_model(self) -> Callable[[Type[StateModel]], StateModel]:
        """
        注册一个状态模型.
        """
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
            return_command: bool = False,
    ) -> Callable[[CommandFunction], CommandFunction | Command]:
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
        :param return_command: 为真的话, 返回的是一个兼容的 Command 对象.
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

    @staticmethod
    def join_channel_path(parent: ChannelFullPath, name: str) -> ChannelFullPath:
        """连接两个 channel 的标准语法. """
        if parent:
            return f'{parent}.{name}'
        return name

    @staticmethod
    def split_channel_path_to_names(channel_path: ChannelFullPath) -> ChannelPaths:
        if not channel_path:
            return []
        return channel_path.split('.')

    def set_context_var(self) -> None:
        """与 get from context 配套使用, 可以在 Command 运行时拿到 Channel 本身. """
        ChannelContextVar.set(self)

    @staticmethod
    def get_from_context() -> Optional["Channel"]:
        """在 Command 内部调用这个函数, 可以拿到运行它的 channel. """
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
    def import_channels(self, *children: "Channel") -> Self:
        """
        添加子 Channel 到当前 Channel. 形成树状关系.
        效果可以比较 python 的 import module_name
        """
        pass

    @abstractmethod
    def new_child(self, name: str) -> Self:
        """
        生成一个子 channel 并返回它.
        :raise NotImplementError: 没有实现的话.
        """
        pass

    @abstractmethod
    def children(self) -> Dict[str, "Channel"]:
        """
        返回所有已注册的子 Channel.
        """
        pass

    def descendants(self, prefix: str = "") -> Dict[str, "Channel"]:
        """
        返回所有的子孙 Channel, 先序遍历.
        其中的 key 是 channel 的路径关系.
        每次都要动态构建, 有性能成本.
        """
        descendants: Dict[str, "Channel"] = {}
        children = self.children()
        if len(children) == 0:
            return descendants
        for child_name, child in children.items():
            child_path = Channel.join_channel_path(prefix, child_name)
            descendants[child_path] = child
            for descendant_full_path, descendant in child.descendants(child_path).items():
                # join descendant name with parent name
                descendants[descendant_full_path] = descendant
        return descendants

    def all_channels(self) -> Dict[str, "Channel"]:
        """
        语法糖, 返回所有的 channel, 包含自身.
        """
        descendants = self.descendants()
        descendants[""] = self
        return descendants

    def get_channel(self, channel_path: str) -> Optional[Self]:
        """
        使用 channel 名从树中获取一个 Channel 对象. 包括自身.
        """
        if channel_path == "":
            return self

        channel_path = Channel.split_channel_path_to_names(channel_path)
        return self.recursive_find_sub_channel(self, channel_path)

    @classmethod
    def recursive_find_sub_channel(cls, root: "Channel", channel_path: List[str]) -> Optional["Channel"]:
        names_count = len(channel_path)
        if names_count == 0:
            return None
        first = channel_path[0]
        children = root.children()
        if first not in children:
            return None
        new_root = children[first]
        if names_count == 1:
            return new_root
        return cls.recursive_find_sub_channel(new_root, channel_path[1:])

    # --- lifecycle --- #

    @abstractmethod
    def is_running(self) -> bool:
        """
        自身是不是 running 状态, 如果是, 则可以拿到 client
        """
        pass

    @abstractmethod
    def bootstrap(self, container: Optional[IoCContainer] = None) -> "ChannelClient":
        """
        传入一个 IoC 容器, 启动 Channel 的 Client. !不会递归启动所有的子 Channel.
        """
        pass

    @property
    @abstractmethod
    def build(self) -> Builder:
        """
        用来快速包装各种函数.
        """
        pass

    async def execute_task(self, task: CommandTask) -> Any:
        """运行一个 task 并且给它赋予当前 channel 的上下文. """
        if not self.is_running():
            raise RuntimeError(f"Channel {self.name()} not running")
        if task.done():
            task.raise_exception()
            return task.result()
        task.exec_chan = self.name()
        # 准备好 ctx. 包含 channel 的容器, 还有 command task 的 context 数据.
        ctx = contextvars.copy_context()
        self.set_context_var()
        # 将 container 也放入上下文中.
        set_container(self.client.container)
        task.set_context_var()
        ctx_ran_cor = ctx.run(task.dry_run)
        # 创建一个可以被 cancel 的 task.
        run_execution = asyncio.create_task(ctx_ran_cor)
        # 这个 task 是不是在运行出结果之前, 外部已经结束了.
        wait_outside_done = asyncio.create_task(task.wait(throw=False))
        done, pending = await asyncio.wait(
            [run_execution, wait_outside_done],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for t in pending:
            t.cancel()
        if task.done():
            task.raise_exception()
        return await run_execution

    def create_task(self, name: str, *args: Any, **kwargs: Any) -> CommandTask:
        """example to create channel task """
        command = self.client.get_command(name)
        if command is None:
            raise NotImplementedError(f'Channel {self.name()} has no command {name}')
        task = BaseCommandTask.from_command(command, *args, **kwargs)
        return task

    async def execute_command(self, command: Command, *args, **kwargs) -> Any:
        """basic example to execute command."""
        from ghoshell_moss.concepts.command import BaseCommandTask
        task = BaseCommandTask.from_command(command, *args, **kwargs)
        try:
            result = await self.execute_task(task)
            task.resolve(result)
            return result
        finally:
            if not task.done():
                task.cancel("task is executed but not done")


class ChannelServer(ABC):
    """
    将 Channel 包装成一个 Server, 可以被上层的 Client 调用.
    上层的 Client 将通过通讯协议, 还原出 Client 树, 但这个 Client 树里所有子 channel 都通过 Server 的通讯协议来传递.
    从而形成链式的封装关系, 在不同进程里还原出树形的架构.

    举例:
    ReverseWebsocketClient => ReverseWebsocketServer => ZMQClient => ZMQServer ... => Client
    """

    @abstractmethod
    async def arun(self, channel: Channel) -> None:
        """
        运行 Client 服务.
        """
        pass

    @abstractmethod
    async def wait_closed(self) -> None:
        """
        等待 server 运行到结束为止.
        """
        pass

    @abstractmethod
    async def aclose(self) -> None:
        """
        主动关闭 server.
        """
        pass

    @abstractmethod
    def is_running(self) -> bool:
        """
        判断这个实例是否在运行.
        """
        pass

    def run_until_closed(self, channel: Channel) -> None:
        """
        展示如何同步运行.
        """
        asyncio.run(self.arun_until_closed(channel))

    async def arun_until_closed(self, channel: Channel) -> None:
        """
        展示如何在 async 中持续运行到结束.
        """
        await self.arun(channel)
        await self.wait_closed()

    def run_in_thread(self, channel: Channel) -> None:
        """
        展示如何在多线程中异步运行, 非阻塞.
        """
        thread = threading.Thread(target=self.run_until_closed, args=(channel,), daemon=True)
        thread.start()

    @abstractmethod
    def close(self) -> None:
        """
        关闭当前 Server.
        """
        pass

    @asynccontextmanager
    async def run_in_ctx(self, channel: Channel) -> AsyncIterator[Self]:
        """
        支持 async with statement 的运行方式调用 channel server, 通常用于测试.
        """
        await self.arun(channel)
        yield self
        await self.aclose()
