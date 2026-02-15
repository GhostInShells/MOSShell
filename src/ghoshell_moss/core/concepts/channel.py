import asyncio
import contextvars
import threading
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable, Coroutine
from contextlib import asynccontextmanager
from typing import (
    Any,
    Optional,
    Protocol,
    TypeVar,
    Union,
    AsyncIterable,
)

from ghoshell_container import BINDING, INSTANCE, IoCContainer, Provider, set_container, Container, get_container
from pydantic import BaseModel, Field
from typing_extensions import Self

from ghoshell_moss.core.concepts.command import BaseCommandTask, Command, CommandMeta, CommandTask, CommandTaskStateType
from ghoshell_moss.core.concepts.states import StateModel, StateStore, State, BaseStateStore
from ghoshell_moss.core.helpers import ThreadSafeEvent
from ghoshell_moss.message import Message
from ghoshell_common.helpers import uuid
from ghoshell_common.contracts import LoggerItf
from .errors import CommandErrorCode, CommandError
import logging

__all__ = [
    "Builder",
    "Channel",
    "MutableChannel",
    "ChannelBroker",
    "Brokers",
    "ChannelFullPath",
    "BrokerOnTaskDone",
    "ChannelMeta",
    "ChannelPaths",
    "ChannelProvider",
    "ChannelCtx",
    "CommandFunction",
    "MessageFunction",
    "LifecycleFunction",
    "PrompterFunction",
    "StringType",
]

# 关于 Channel (中文名: 经络) :
#
# MOSS 架构的核心思想是 "面向模型的高级编程语言", 目的是定义一个类似 python 语法的编程语言给模型.
#
# 所以 Channel 可以理解为 python 中的 'module', 可以树形嵌套, 每个 channel 可以管理一批函数 (command).
#
# 同时在 "时间是第一公民" 的思想下, Channel 需要同时定义 "并行" 和 "阻塞" 的分发机制.
# 神经信号 (command call) 在运行时中的流向是从 父channel 流向 子channel.
#
#
# Channel 与 MCP/Skill 等类似思想最大的区别在于, 它需要:
# 1. 完全是实时动态的, 它的一切函数, 一切描述都随时可变.
# 2. 拥有独立的运行时, 可以单独运行一个图形界面或具身机器人.
# 3. 自动上下文同步, 大模型在每个思考的关键帧中, 自动从 channel 获得上下文消息.
# 4. 与 Shell 进行全双工实时通讯
#
# 可以把 Channel 理解为 AI 大模型上可以 - 任意插拔的, 顺序堆叠的, 自治的, 面向对象的 - 应用单元.
#
# todo: 目前 channel 的设计思想还没完全完成. 下一步还有 interface/extend/implementation 等面向对象的构建思路.
#
# 举个例子: 一个拥有人形控制能力的 AI, 向所有的人形肢体 (机器人/数字人) 发送 "挥手" 的指令, 实际上需要每个肢体都执行.
#
# 所以可以有 N 个人形肢体, 注册到同一个 channel interface 上.

ChannelFullPath = str
"""
在树形嵌套的 channel 结构中, 对一个具体 channel 进行寻址的方法.
完全对齐 python 的  a.b.c 寻址逻辑. 

同时它也描述了一个神经信号 (command call) 经过的路径, 比如从 a -> b -> c 执行.
"""

ChannelPaths = list[str]
"""字符串路径的数组表现形式. a.b.c -> ['a', 'b', 'c'] """

CommandFunction = Union[Callable[..., Coroutine], Callable[..., Any]]
"""
用于描述一个本地的 python 函数 (或者类的 method) 可以被注册到 Channel 中变成一个 command. 

通常要求是异步函数, 如果是同步函数的话, 会自动卸载到线程池运行 (asyncio.to_thread)
所有的 command function 都要考虑线程阻塞问题,  目前 moss 尚未实现多线程隔离 coroutine 的阻塞问题. 
"""

LifecycleFunction = Union[Callable[..., Coroutine[None, None, None]], Callable[..., None]]
"""
用于描述一个本地的 python 函数 (或者类的 method), 可以用来定义 channel 自身生命周期行为. 

一个 Channel 运行的生命周期设计是: 

- [on startup] : channel 启动时
- [idle] : 闲时, 没有任何命令输入
- [on command call]: 忙时, 执行某个 command call
- [on clear] : 强制要求清空所有命令
- [on disconnected]: channel 断连时
- [on close] : channel 关闭时 

举一个典型的例子: 数字人在执行动画 command 时, 运行轨迹动画; 执行完毕后, 没有命令输入时, 需要返回呼吸效果 (on_idle) 

这类运行时函数, 可以通过注册的方式定义到一个 channel 中. 
如果用编程语言的思想来理解, 这些函数类似于 python 的生命周期魔术方法:
- __init__
- __new__
- __del__
- __aenter__
- __aexit__

todo: alpha 版本生命周期定义得不完整, 预计在 beta 版本做一个整体的修复. 
"""

PrompterFunction = Union[Callable[..., Coroutine[None, None, str]], Callable[..., str]]
"""
可以生成 prompt 的函数类型. 它的返回值是一个字符串. 

为何这种函数从 command 中单独区分开来呢? 

因为它是最重要的大模型反身性控制工具, 让模型可以自己定义自己的 prompt. 
举个例子, 有一个字符串的 prompt 模板: 

>>> # persona
>>> <my_persona name="my_name">
>>> # behaviors
>>> <my_behavior name="my_name">

其中用 ctml 定义了 prompt 函数调用, 并行运行这些 prompt 函数, 拿到结果后可以拼成一个字符串,
这个字符串就是 AI 自治的某个 prompt 片段.

AI 的 meta 模式可以通过理解 prompt 函数的存在, 定义 prompt 模板, 生成 prompt 结果.

微软的 POML 就是类似的思路. 不过不需要那么复杂的数据结构嵌套, 用 prompt 函数 + 纯 python 代码即可自解释.    

todo: prompt function 体系尚未完成. 
"""

MessageFunction = Union[
    Callable[[], Coroutine[None, None, list[Message]]],
    Callable[[], list[Message]],
]
"""
一种可以注册到 Channel 中的函数, 也是最重要的一种函数. 

它可以定义这个 Channel 组件当前的上下文生成逻辑, 然后在模型思考的瞬间, 通过双工通讯提供给模型.

Agent 架构可以把 channel 有序排列, 然后自动拿到一个由很多个 channel context messages 堆叠出来的上下文.


通常上下文生成逻辑, 考虑 token 裁剪等问题, 需要和 agent 设计强耦合. 
而在 MOSS 架构中, 只需要引用一个现成的 channel, override 其中的 context message function, 
就可以定义新的上下文逻辑了. 
"""

StringType = Union[str, Callable[[], str]]


class ChannelMeta(BaseModel):
    """
    Channel 的元信息数据.
    可以用来 mock 一个 channel.
    """

    name: str = Field(description="The origin name of the channel, kind like python module name.")
    description: str = Field(default="", description="The description of the channel.")
    channel_id: str = Field(default="", description="The ID of the channel.")
    available: bool = Field(default=True, description="Whether the channel is available.")
    commands: list[CommandMeta] = Field(default_factory=list, description="The list of commands.")
    children: list[str] = Field(default_factory=list, description="the children channel names")

    # about instructions / context messages
    # ModelContext is built by many messages blocks, we believe the blocks should be :
    #  - instructions before conversation
    #  - conversation messages
    #  - dynamic context message before the inputs
    #  - inputs messages
    #  - [messages recalled by inputs]
    #  - [reasoning messages]
    #  - generated actions
    #
    # so channel as component of the AI Model context, shall provide instructions or context messages.

    instructions: list[Message] = Field(default_factory=list, description="the channel instructions messages")
    context: list[Message] = Field(default_factory=list, description="The channel context messages")

    dynamic: bool = Field(default=True, description="Whether the channel is dynamic, need refresh each time")


class Builder(ABC):
    """
    用来动态构建一个 Channel 的通用接口.
    目前主要用于 py channel.
    """

    # ---- decorators ---- #

    @abstractmethod
    def description(self) -> Callable[[StringType], StringType]:
        """
        注册一个全局唯一的函数, 用来动态生成 description.
        todo: 删除, 全部迁移到 instructions.
        """
        pass

    @abstractmethod
    def available(self) -> Callable[[Callable[[], bool]], Callable[[], bool]]:
        """
        注册一个函数, 用来标记 Channel 是否是 available 状态.
        todo: with 开头的不要用 decorator 形式 .
        """
        pass

    @abstractmethod
    def state_model(self, model: type[StateModel] | StateModel) -> type[StateModel] | StateModel:
        """
        注册一个状态模型.
        todo: 重做这个函数, 目前实现不符合预期.
        """
        pass

    @abstractmethod
    def context_messages(self, func: MessageFunction) -> MessageFunction:
        """
        注册一个上下文生成函数. 用来生成 channel 运行时动态的上下文.
        """
        pass

    @abstractmethod
    def instruction_messages(self, func: MessageFunction) -> MessageFunction:
        pass

    @abstractmethod
    def command(
            self,
            *,
            name: str = "",
            chan: str | None = None,
            doc: Optional[StringType] = None,
            comments: Optional[StringType] = None,
            tags: Optional[list[str]] = None,
            interface: Optional[StringType] = None,
            available: Optional[Callable[[], bool]] = None,
            # --- 高级参数 --- #
            blocking: Optional[bool] = None,
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
        :param blocking: 这个函数是否会阻塞 channel. 默认都会阻塞.
        :param available: 通过函数定义这个命令是否 available.
        :param call_soon: 决定这个函数进入轨道后, 会第一时间执行 (不等待调度), 还是等待排队执行到自身时.
                          如果是 block + call_soon, 会先清空队列.
        :param return_command: 为真的话, 返回的是一个兼容的 Command 对象.
        """
        pass

    @abstractmethod
    def on_idle(self, run_policy: LifecycleFunction) -> LifecycleFunction:
        """
        注册一个函数, 当 Channel 运行 policy 时, 会执行这个函数.
        """
        pass

    @abstractmethod
    def on_start_up(self, start_func: LifecycleFunction) -> LifecycleFunction:
        """
        启动时执行的回调.
        """
        pass

    @abstractmethod
    def on_close(self, stop_func: LifecycleFunction) -> LifecycleFunction:
        """
        关闭时的回调.
        """
        pass

    @abstractmethod
    def on_running(self, running_func: LifecycleFunction) -> LifecycleFunction:
        """
        整个开启时间运行的逻辑.
        注意, 这个函数不会和 idle / pause 冲突.
        """
        pass

    @abstractmethod
    def with_binding(self, contract: type[INSTANCE], binding: INSTANCE) -> Self:
        """
        register default bindings for the given contract.
        """
        pass

    # ---- builder method ---- #

    @abstractmethod
    def is_dynamic(self) -> bool:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass

    @abstractmethod
    async def get_context_message(self) -> list[Message]:
        pass

    @abstractmethod
    async def get_instruction_messages(self) -> list[Message]:
        pass

    @abstractmethod
    def commands(self) -> list[Command]:
        pass

    @abstractmethod
    def get_command(self, name: str) -> Command | None:
        pass

    @abstractmethod
    async def run_idling(self):
        pass

    @abstractmethod
    async def run_start_up(self) -> None:
        pass

    @abstractmethod
    async def run_close(self) -> None:
        pass

    @abstractmethod
    async def keep_running(self) -> None:
        pass

    @abstractmethod
    def update_container(self, container: IoCContainer) -> None:
        pass


_ChannelBrokerCtx = contextvars.ContextVar("MOSSChannelCtx.Broker")
_CommandTaskCtx = contextvars.ContextVar("MOSSChannelCtx.Task")


class ChannelCtx:
    """
    提供 Channel 相关的一些工具函数.
    """

    @classmethod
    def init(
            cls,
            broker: "ChannelBroker",
            task: CommandTask | None = None,
    ) -> None:
        _ChannelBrokerCtx.set(broker)
        if task is not None:
            _CommandTaskCtx.set(task)

    @classmethod
    def channel(cls) -> "Channel":
        broker = cls.broker()
        return broker.channel

    @classmethod
    def broker(cls) -> "ChannelBroker":
        return _ChannelBrokerCtx.get()

    @classmethod
    def task(cls) -> CommandTask | None:
        try:
            return _CommandTaskCtx.get()
        except LookupError:
            return None

    @classmethod
    def container(cls) -> IoCContainer:
        broker = cls.broker()
        return broker.container

    @classmethod
    def get_contract(cls, contract: type[INSTANCE]) -> INSTANCE:
        broker = cls.broker()
        return broker.container.force_fetch(contract)


class Brokers:
    """
    测试工具, 用来快速实例化一个 channel 树的所有 broker
    """

    def __init__(self, main: "Channel", container: IoCContainer, brokers: dict[str, "ChannelBroker"]):
        self.main = main
        self.container = container
        self.broker_map = brokers
        self._start = False
        self._close = False

    async def iter(self) -> AsyncIterable[tuple[ChannelFullPath, "ChannelBroker"]]:
        """
        动态获取 broker, 可能会临时初始化它们.
        """
        valid = set()
        for path, channel in self.main.all_channels().items():
            valid.add(path)
            if path in self.broker_map:
                yield path, self.broker_map.get(path)
            else:
                broker = channel.bootstrap(self.container)
                await broker.start()
                self.broker_map[path] = broker
                yield path, broker

        invalid = []
        for path in self.broker_map.keys():
            if path not in valid:
                invalid.append(path)

        # 关闭掉不对的 broker
        close_invalid = []
        for path in invalid:
            broker = self.broker_map.get(path)
            if broker is not None:
                del self.broker_map[path]
            close_invalid.append(broker.close())
        await asyncio.gather(*close_invalid)

    def get(self, path: ChannelFullPath) -> "ChannelBroker":
        broker = self.broker_map.get(path)
        if broker is None:
            raise LookupError(f'broker {path} not found')
        return broker

    def main_broker(self) -> "ChannelBroker":
        return self.get('')

    async def fetch(self, path: ChannelFullPath) -> Optional["ChannelBroker"]:
        channel = self.main.get_channel(path)
        broker = self.broker_map.get(path)
        if channel is None:
            if broker is not None:
                await broker.close()
                del self.broker_map[path]
            return None
        if broker is None:
            broker = channel.bootstrap(self.container)
            self.broker_map[path] = broker
            await broker.start()
        return broker

    @classmethod
    def new(cls, channel: "Channel", container: Optional[IoCContainer] = None) -> Self:
        container = container or get_container()
        brokers = {}
        for path, channel in channel.all_channels().items():
            brokers[path] = channel.bootstrap(container)

        return cls(channel, container, brokers)

    async def start(self):
        if self._start:
            return
        self._start = True
        start_all = []
        for broker in self.broker_map.values():
            start_all.append(asyncio.create_task(broker.start()))
        await asyncio.gather(*start_all)

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def close(self):
        if self._close:
            return
        self._close = True
        close_all = []
        for broker in self.broker_map.values():
            close_all.append(asyncio.create_task(broker.close()))
        await asyncio.gather(*close_all)

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


class Channel(ABC):
    """
    Shell 可以使用的命令通道.
    """

    @abstractmethod
    def name(self) -> str:
        """
        channel 的名字. 如果是主 channel, 默认为 ""
        非主 channel 不能为 ""
        """
        pass

    @abstractmethod
    def description(self) -> str:
        pass

    @staticmethod
    def join_channel_path(parent: ChannelFullPath, name: str) -> ChannelFullPath:
        """连接父子 channel 名称的标准语法. 作为全局的约束方式. """
        # todo: 校验 name 的类型, 不允许不合法的 name.
        if parent:
            return f"{parent}.{name}"
        return name

    @staticmethod
    def split_channel_path_to_names(channel_path: ChannelFullPath) -> ChannelPaths:
        """
        解析出 channel 名称轨迹的标准语法.
        """
        if not channel_path:
            return []
        return channel_path.split(".")

    @property
    @abstractmethod
    def broker(self) -> "ChannelBroker":
        """
        Channel 在 bootstrap 之后返回的运行时.
        :raise RuntimeError: Channel 没有运行
        # todo: 考虑彻底移除. 统一通过 CommandTaskCtx 去初始化或获取.
        """
        pass

    # --- children --- #

    @abstractmethod
    def children(self) -> dict[str, "Channel"]:
        """
        返回所有已注册的子 Channel.
        """
        pass

    def descendants(self, prefix: str = "") -> dict[str, "Channel"]:
        """
        返回所有的子孙 Channel, 先序遍历.
        其中的 key 是 channel 的路径关系.
        每次都要动态构建, 有性能成本.
        """
        descendants: dict[str, Channel] = {}
        children = self.children()
        if len(children) == 0:
            return descendants
        # 深度优先遍历.
        for child_name, child in children.items():
            child_path = Channel.join_channel_path(prefix, child_name)
            descendants[child_path] = child
            for descendant_full_path, descendant in child.descendants(child_path).items():
                # join descendant name with parent name
                descendants[descendant_full_path] = descendant
        return descendants

    def all_channels(self) -> dict[ChannelFullPath, "Channel"]:
        """
        语法糖, 返回所有的 channel, 包含自身.
        key 是以自身为起点的 channel path (相对路径), 用来发现原点.
        """
        descendants = {"": self}
        for path, channel in self.descendants().items():
            # 保持顺序.
            descendants[path] = channel
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
    def recursive_find_sub_channel(cls, root: "Channel", channel_path: list[str]) -> Optional["Channel"]:
        """
        从子孙节点中递归进行查找.
        """
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
        自身是不是 running 状态, 如果是, 则可以拿到 broker
        """
        pass

    @abstractmethod
    def bootstrap(self, container: Optional[IoCContainer] = None) -> "ChannelBroker":
        """
        传入一个 IoC 容器, 获取 Channel 的 broker 实例.
        """
        pass

    @asynccontextmanager
    def bootstrap_brokers(
            self,
            container: Optional[IoCContainer] = None,
    ) -> Brokers:
        """
        todo: 删除
        语法糖, 启动当前 Channel 和它所有的子节点.
        通常仅仅用于单元测试. 也是为了提示如何单独测试一个 Channel.
        """
        return Brokers.new(self, container)


class MutableChannel(Channel, ABC):
    """
    一个约定, 用来提示一些可构建的动态 Channel.
    """

    @abstractmethod
    def import_channels(self, *children: "Channel") -> Self:
        """
        添加子 Channel 到当前 Channel. 形成树状关系.
        效果可以比较 python 的 import module_name
        """
        pass

    @property
    @abstractmethod
    def build(self) -> Builder:
        """
        支持通过 Builder 动态构建一个 Channel.
        """
        pass


BrokerOnTaskDone = Callable[[CommandTask], None] | Callable[[CommandTask], Coroutine[None, None, None]]


class ChannelBroker(ABC):
    """
    channel 运行后提供出来的通用 API.
    只有在 channel.bootstrap 之后才可使用.
    用于控制 channel 的所有能力.

    如果用 "面向模型的高级编程语言" 角度看,
    可以把 channel broker 理解成 python 的 ModuleType 对象.
    """

    def __init__(
            self,
            *,
            channel: "Channel",
            container: IoCContainer | None = None,
            uid: str | None = None,
            logger: LoggerItf | None = None
    ):
        self._channel = channel
        self._name = channel.name()
        self._uid = uid or uuid()
        # 用不同的容器隔离依赖.
        self._container: IoCContainer | None = container

        self._starting = False
        self._started = False
        self._running_task: Optional[asyncio.Task] = None
        # 用线程安全的事件. 考虑到 broker 未来可能会跨线程被使用.
        self._closing_event = ThreadSafeEvent()
        self._closed_event = ThreadSafeEvent()

        self._loop: asyncio.AbstractEventLoop | None = None
        self._state_store: StateStore | None = None
        self._logger: LoggerItf | None = logger

        self._cached_meta: ChannelMeta | None = None
        # blocking lifecycle task 用来保证无论哪一层, 都不能有同时两个以上的生命周期任务在执行.
        self._blocking_lifecycle_task: asyncio.Task | None = None
        self._blocking_action_lock = asyncio.Lock()
        self._unblocking_tasks: set[asyncio.Task] = set()
        self._on_refresh_meta_callbacks: list[Callable[[ChannelMeta], Coroutine[None, None, None]]] = []
        self._task_done_callbacks: list[BrokerOnTaskDone] = []

    @property
    def channel(self) -> "Channel":
        return self._channel

    @property
    def states(self) -> StateStore:
        """
        返回当前 Channel 的状态存储.
        """
        if self._state_store is None:
            self._state_store = self._container.get(StateStore)
            if self._state_store is None:
                self._state_store = BaseStateStore(owner=self._uid)
        return self._state_store

    @property
    def logger(self) -> LoggerItf:
        if self._logger is None:
            self._logger = self.container.get(LoggerItf) or logging.getLogger("moss")
        return self._logger

    @property
    def container(self) -> IoCContainer:
        """
        broker 所持有的 ioc 容器.
        """
        if self._container is None:
            self._container = self.prepare_container(self._container)
        return self._container

    def prepare_container(self, container: IoCContainer | None) -> Container:
        return Container(name=f'moss.channel.{self._name}.{self._uid}.container', parent=container)

    @property
    def id(self) -> str:
        """
        broker 的唯一 id.
        """
        return self._uid

    @property
    def name(self) -> str:
        """
        对应的 channel name.
        """
        return self._name

    def meta(self) -> ChannelMeta:
        """
        返回 Channel 自身的 Meta.
        """
        if self._cached_meta is not None:
            return self._cached_meta
        if not self.is_running():
            raise RuntimeError(f"Channel {self.name} not running")
        else:
            raise RuntimeError(f"Channel {self.name} has not been started yet")

    def on_refresh_meta(self, callback: Callable[[ChannelMeta], Coroutine[None, None, None]]) -> None:
        self._on_refresh_meta_callbacks.append(callback)

    async def refresh_meta(
            self,
            callback: bool = True,
    ) -> None:
        """
        更新当前的 Channel Meta 信息. 用于支持被动拉取. 不会主动推送更新.
        """
        if not self._starting or self._closing_event.is_set():
            return
        ctx = contextvars.copy_context()
        # 生成时添加 ctx.
        ChannelCtx.init(self)
        meta = await ctx.run(self.generate_meta)
        self._cached_meta = meta
        self.logger.info(
            "[Channel %s %s] refreshed meta", self._name, self._uid,
        )
        # 创建异步的回调.
        if callback and self._on_refresh_meta_callbacks:
            for callback in self._on_refresh_meta_callbacks:
                _ = asyncio.create_task(callback(meta))

    @abstractmethod
    async def generate_meta(self) -> ChannelMeta:
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """
        判断一个 Broker 的连接与通讯是否正常。
        一个运行中的 Broker 不一定是正确连接的.
        """
        # 对于非通讯类的 channel, 比如 py-channel, 直接返回 True.
        pass

    @abstractmethod
    async def wait_connected(self) -> None:
        """
        等待 broker 到连接成功.
        """
        pass

    def is_running(self) -> bool:
        """
        是否已经启动了. 如果 Broker 被 close, is_running 为 false.
        """
        return self._started and not self._closing_event.is_set()

    def is_available(self) -> bool:
        """
        当前 Channel 对于使用者而言, 是否可用.
        当一个 Broker 是 running & connected 状态下, 仍然可能会因为种种原因临时被禁用.
        """
        return self.is_running() and self.is_connected() and self.meta().available

    @abstractmethod
    def commands(self, available_only: bool = True) -> dict[str, Command]:
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

    def on_task_done(self, callback: BrokerOnTaskDone) -> None:
        self._task_done_callbacks.append(callback)

    def _task_done_callback(self, task: CommandTask) -> None:
        import inspect
        if not self.is_running():
            return
        if len(self._task_done_callbacks) == 0:
            return
        for callback in self._task_done_callbacks:
            if inspect.iscoroutinefunction(callback):
                self._loop.create_task(callback(task))
            else:
                # 同步运行.
                callback(task)

    async def idle(self) -> None:
        """
        进入闲时状态.
        闲时状态指当前 Broker 及其 子 Channel 都没有 CommandTask 在运行的时候.
        """
        if not self.is_running():
            return
        try:
            await asyncio.sleep(0.0)
            await self._blocking_action_lock.acquire()
            if self._blocking_lifecycle_task is not None:
                # if not self._blocking_task.done()
                self._blocking_lifecycle_task.cancel()
                try:
                    await self._blocking_lifecycle_task
                except asyncio.CancelledError:
                    pass
                self._blocking_lifecycle_task = None

            ctx = contextvars.copy_context()
            ChannelCtx.init(self)
            on_idle_cor = ctx.run(self.on_idle)
            task = asyncio.create_task(on_idle_cor)
            self._blocking_lifecycle_task = task
        finally:
            self._blocking_action_lock.release()
            self.logger.info("[Channel %s %s] idling", self._name, self._uid)

    @abstractmethod
    async def on_idle(self) -> None:
        """
        进入闲时状态.
        闲时状态指当前 Broker 及其 子 Channel 都没有 CommandTask 在运行的时候.
        """
        pass

    def log_format(self) -> tuple[str, tuple]:
        """
        normalize log format

        >>> some_arg = 123
        >>> tmp, args = self.log_prefix()
        >>> self.logger.info(f"{tmp} the log info %s", *args, some_arg)
        """
        return "[Channel %s %s][%s]", (self._name, self._uid, self.__class__.__name__)

    async def clear(self) -> None:
        """
        当轨道命令被触发清空时候执行.
        """
        if not self._started or self._closed_event.is_set():
            return
        try:
            await asyncio.sleep(0.0)
            await self._blocking_action_lock.acquire()
            wait_done = []
            if self._blocking_lifecycle_task and not self._blocking_lifecycle_task.done():
                self._blocking_lifecycle_task.cancel()
                wait_done.append(self._blocking_lifecycle_task)
            self._blocking_lifecycle_task = None
            if len(self._unblocking_tasks) > 0:
                for t in self._unblocking_tasks:
                    if not t.done():
                        t.cancel()
                        wait_done.append(t)
            self._unblocking_tasks.clear()
            got = await asyncio.gather(*wait_done, return_exceptions=True)
            for _ in got:
                pass
            # 阻塞等待到清空结束.
            ctx = contextvars.copy_context()
            ChannelCtx.init(self)
            cor = ctx.run(self.on_clear)
            await cor
        finally:
            self._blocking_action_lock.release()
            self.logger.info("[Channel %s %s] cleared", self._name, self._uid)

    async def pause(self) -> None:
        """
        设置当前 Broker 为 pause 状态.
        pause 状态下 Channel Broker 应该要进入某种安全姿态.
        """
        if not self._started or self._closed_event.is_set():
            return
        # 先清空所有的运动.
        await self.clear()
        try:
            await asyncio.sleep(0.0)
            await self._blocking_action_lock.acquire()
            ctx = contextvars.copy_context()
            ChannelCtx.init(self)
            pause_cor = ctx.run(self.on_pause)
            self._blocking_lifecycle_task = asyncio.create_task(pause_cor)
        finally:
            self.logger.info("[Channel %s %s] is pausing", self._name, self._uid)
            self._blocking_action_lock.release()

    @abstractmethod
    async def on_pause(self) -> None:
        pass

    @abstractmethod
    async def on_clear(self) -> None:
        """
        当轨道命令被触发清空时候执行.
        """
        pass

    async def start(self) -> None:
        """
        启动 Channel Broker.
        通常用 with statement 或 async exit stack 去启动.
        只会启动当前 channel 自身.
        """
        if self._starting:
            return
        self._starting = True
        container = self.container
        # bootstrap container
        await asyncio.to_thread(container.bootstrap)
        # 启动 states 和 topics 模块.
        await self.states.start()
        await self.refresh_meta()
        self._started = True
        ctx = contextvars.copy_context()
        ChannelCtx.init(self)
        cor = ctx.run(self.on_start_up)
        self.logger.info(
            "[Channel %s %s] started", self._name, self._uid,
        )
        await cor
        self._running_task = asyncio.create_task(ctx.run(self.on_running))
        # set pause as default state.
        # make sure the shell runtime change it
        await self.pause()

    @abstractmethod
    async def on_start_up(self) -> None:
        pass

    async def wait_closing(self) -> None:
        await self._closing_event.wait()

    async def wait_closed(self) -> None:
        await self._closed_event.wait()

    async def close(self) -> None:
        """
        关闭当前 broker. 同时阻塞销毁资源直到结束.
        只会关闭当前 channel 的 broker.
        """
        if self._closing_event.is_set():
            return
        self._closing_event.set()
        try:
            self.logger.info(
                "[Channel %s %s] start to close", self._name, self._uid,
            )
            # 停止所有行为.
            await self.clear()
            if self._running_task and not self._running_task.done():
                self._running_task.cancel()
                try:
                    await self._running_task
                except asyncio.CancelledError:
                    pass
            self._running_task = None
            ctx = contextvars.copy_context()
            ChannelCtx.init(self)
            on_close_cor = ctx.run(self.on_close)
            # 等待运行全部结束.
            await on_close_cor
            # 关闭 state store.
            if self._state_store:
                await self._state_store.close()
            # 关闭容器运行.
            self.logger.info(
                "[Channel %s %s] prepare to shutdown", self._name, self._uid,
            )
            await asyncio.to_thread(self.container.shutdown)
        finally:
            self._closed_event.set()
            if self._logger:
                self._logger.info(
                    "[Channel %s %s] closed", self._name, self._uid,
                )
            # 做必要的清空.
            self.destroy()

    def destroy(self) -> None:
        self._container = None
        self._channel = None
        self._state_store = None
        self._logger = None
        self._on_refresh_meta_callbacks.clear()

    @abstractmethod
    async def on_close(self) -> None:
        pass

    @abstractmethod
    async def on_running(self) -> None:
        pass

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def execute_task(self, task: CommandTask) -> None:
        """
        在 Broker 中执行一个 command task. 会尽快返回, 由 Task 自身完成阻塞.
        """
        if task.done():
            return
        elif not self.is_running():
            self.logger.error(
                "[Channel %s %s] failed task %s: not running", self._name, self._uid, task.cid,
            )
            task.fail(CommandErrorCode.NOT_RUNNING.error(f"channel {self.name} not running"))
            return
        elif not self.is_connected():
            self.logger.info(
                "[Channel %s %s] failed task %s: not connected", self._name, self._uid, task.cid,
            )
            task.fail(CommandErrorCode.NOT_CONNECTED.error(f"channel {self.name} not connected"))
            return
        elif not self.is_available():
            self.logger.info(
                "[Channel %s %s] failed task %s: not available", self._name, self._uid, task.cid,
            )
            task.fail(CommandErrorCode.NOT_AVAILABLE.error(f"channel {self.name} not available"))
            return

        try:
            await asyncio.sleep(0)
            await self._blocking_action_lock.acquire()
            # 如果是阻塞类型的任务, 必须清空主要执行中的任务.
            task.set_state(CommandTaskStateType.executing)
            task.add_done_callback(self._task_done_callback)
            if task.meta.blocking:
                if self._blocking_lifecycle_task is not None:
                    if not self._blocking_lifecycle_task.done():
                        self._blocking_lifecycle_task.cancel()
                        try:
                            await self._blocking_lifecycle_task
                        except asyncio.CancelledError:
                            pass
                    self._blocking_lifecycle_task = None
                cor = self._ensure_task_done(task)
                blocking_task = asyncio.create_task(cor)
                self._blocking_lifecycle_task = blocking_task
            else:
                cor = self._ensure_task_done(task)
                unblocking_task = asyncio.create_task(cor)
                self._unblocking_tasks.add(unblocking_task)
        finally:
            self._blocking_action_lock.release()
            self.logger.info("[Channel %s %s] executing task %s", self._name, self._uid, task.cid)

    async def _ensure_task_done(self, task: CommandTask) -> None:
        if task.done():
            return

        # 准备执行.
        task.exec_chan = self.name
        try:
            await asyncio.sleep(0)
            # 在这里让出控制权, 保证 finally 一定被执行.
            self.logger.info("[Channel %s %s] start task %s", self._name, self._uid, task.cid)
            # 初始化函数运行上下文.
            ctx = contextvars.copy_context()
            ChannelCtx.init(self, task)
            # 使用 dry run 来管理生命周期.
            run_cor = ctx.run(task.dry_run)
            execution_task = asyncio.create_task(run_cor)
            task_done_outside = asyncio.create_task(task.wait(throw=False))
            done, pending = await asyncio.wait([execution_task, task_done_outside], return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
            # 为结果赋值.
            if not task.done():
                result = await execution_task
                task.resolve(result)
            self.logger.info("[Channel %s %s] resolved task %s", self._name, self._uid, task.cid)

        except Exception as e:
            self.logger.error("[Channel %s %s] task %s failed: %s", self._name, self._uid, task.cid, e)
            if not task.done():
                task.fail(e)
            raise
        finally:
            if not task.done():
                task.fail(CommandErrorCode.UNKNOWN_ERROR.error(f"task not done after execution"))
            self.logger.info(
                "[Channel %s %s] done task %s at state", self._name, self._uid, task.cid, task.state,
            )

    def create_command_task(self, name: str, *args: Any, **kwargs: Any) -> CommandTask:
        """example to create channel task"""
        command = self.get_command(name)
        if command is None:
            raise NotImplementedError(f"Channel {self.name} has no command {name}")
        task = BaseCommandTask.from_command(command, *args, **kwargs)
        return task

    async def execute_command(self, name: str, *args: Any, **kwargs: Any) -> Any:
        task = self.create_command_task(name, *args, **kwargs)
        await self.execute_task(task)
        return await task


class ChannelApp(Protocol):
    """
    简单定义一种有状态 Channel 的范式.
    基本思路是, 这个 App 运行的时候, 可以渲染图形界面或开启什么程序.
    同时它通过暴露一个 Channel, 使 App 可以和 Shell 进行通讯. 通过 Provider / Proxy 范式提供给 Shell 控制.

    对于未来的 AI App 而言, 假设其仍然为 MCV (model->controller->viewer) 架构, 模型扮演的应该是 Controller.
    而 Channel 就是用来取代 Controller, 和 AI 模型通讯的方式.

    新的 MCV 范式是:  data-model / AI-channel / human-viewer
    todo: 未完全定义清楚, 主要是生命周期问题.
    """

    @abstractmethod
    def as_channel(self) -> Channel:
        """
        返回一个 Channel 实例.
        """
        pass


ChannelProxy = Channel
"""
Channel Proxy 是一种特殊的 Channel, 它和 Channel Provider 成对出现. 
Provider 将本地的 Channel 以通讯协议的形式封装, 而 ChannelProxy 则用相同的通讯协议去还原这个 Channel. 
举例: ZmqChannelProvider.run(local_channel) => connection => ZmqChannelProxy, 这里的 ChannelProxy 对于模型而言和 local 一样.
"""


class ChannelProvider(ABC):
    """
    通过 Provider 运行一个 Local Channel, 提供通讯协议. 使用相同通讯协议的 Proxy 可以在远端还原出这个 Channel.

    从而形成链式的封装关系, 在不同进程里还原出树形的架构.
    Provider 和 Proxy 通常成对出现.
    """

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.aclose()

    @abstractmethod
    def brokers(self) -> Brokers:
        pass

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
    def wait_closed_sync(self) -> None:
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
