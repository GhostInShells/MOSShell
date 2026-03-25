import asyncio
import contextlib
import contextvars
import threading
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import (
    Any,
    Optional,
    Union,
    Callable,
    Coroutine,
    AsyncIterator,
    Awaitable,
    Generic,
    TypeVar,
)

from ghoshell_container import INSTANCE, IoCContainer, get_container
from pydantic import BaseModel, Field
from typing_extensions import Self

from ghoshell_moss.core.concepts.command import (
    BaseCommandTask,
    Command,
    CommandMeta,
    CommandTask,
    CommandTaskContextVar,
    CommandUniqueName,
)
from ghoshell_moss.core.concepts.errors import CommandErrorCode
from ghoshell_moss.core.concepts.states import StateStore
from ghoshell_moss.core.concepts.topic import (
    TopicService,
    TopicModel,
    Subscriber,
    Publisher,
    SubscribeKeep,
    Topic,
    TOPIC_MODEL,
)
from ghoshell_moss.message import Message
from ghoshell_common.contracts import LoggerItf

__all__ = [
    "Channel",
    "Builder",
    "MutableChannel",
    "TaskDoneCallback",
    "RefreshMetaCallback",
    "ChannelRuntime",
    "ChannelImportLib",
    "ChannelFullPath",
    "ChannelMeta",
    "ChannelPaths",
    "ChannelProvider",
    "ChannelCtx",
    "CommandFunction",
    "MessageFunction",
    "LifecycleFunction",
    "StringType",
    "ChannelInterface",
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
# 举个例子: 一个拥有人形控制能力的 AI, 向所有的人形肢体 (机器人/数字人) 发送 "挥手" 的指令, 实际上需要每个肢体都执行.
#
# 所以可以有 N 个人形肢体, 注册到同一个 channel interface 上.



class ChannelMeta(BaseModel):
    """
    Channel 的元信息数据.
    可以用来 mock 一个 channel.
    """

    name: str = Field(description="The origin name of the channel, kind like python module name.")
    description: str = Field(default="", description="The description of the channel.")
    failure: str = Field(default="", description="The failure status of the channel.")
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

    instruction: str = Field(default='', description="the channel instruction messages")
    context: list[Message] = Field(default_factory=list, description="The channel context messages")

    dynamic: bool = Field(default=True, description="Whether the channel is dynamic, need refresh each time")

    @classmethod
    def new_empty(cls, id: str, channel: "Channel") -> Self:
        return cls(
            name=channel.name(),
            description=channel.description(),
            dynamic=True,
            channel_id=id,
            available=False,
        )


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
"""

MessageFunction = Union[
    Callable[[], Coroutine[None, None, list[Message]]],
    Callable[[], list[Message]],
]
"""
可以生成消息体的函数. 这种函数注册到 Channel 中, 可以用来动态地生成 Context Messages 与 Instruction Messages.

AI 通过双工通讯, 在每个关键帧思考的瞬间, 提取对应的消息体替换到上下文中. 
"""

StringType = Union[
    str,
    Callable[[], str],
    Callable[[], Coroutine[None, None, str]],
]

LifecycleFunction = Union[Callable[..., Coroutine[None, None, None]], Callable[..., None]]
"""
用于描述一个本地的 python 函数 (或者类的 method), 可以用来定义 channel 自身生命周期行为. 

一个 Channel 运行的生命周期设计是: 

- [on startup] : channel 启动时
- [on idle] : 闲时, 没有任何命令输入
- [executing]: 忙时, 执行某个 command call
- [on clear] : 强制要求清空所有命令
- [on close] : channel 关闭时 

举一个典型的例子: 数字人在执行动画 command 时, 运行轨迹动画; 执行完毕后, 没有命令输入时, 需要返回呼吸效果 (on_idle) 

这类运行时函数, 可以通过注册的方式定义到一个 channel 中. 
如果用编程语言的思想来理解, 这些函数类似于 python 的生命周期魔术方法:
- __init__
- __aenter__
- __aexit__
"""


class Builder(ABC):
    """
    用来动态构建一个 Channel 的通用接口.
    """

    # ---- decorators ---- #

    @abstractmethod
    def available(self, func: Callable[[], bool]) -> Callable[[], bool]:
        """
        decorator
        注册一个函数, 用来动态生成整个 Channel 的 available 状态.
        Channel 每次刷新状态时, 都会从这个函数取值. 否则默认为 True.
        >>> async def building(chan: MutableChannel) -> None:
        >>>     chan.build.available(lambda: True)
        """
        pass

    @abstractmethod
    def context_messages(self, func: MessageFunction, reset: bool = False) -> MessageFunction:
        """
        decorator
        注册一个上下文生成函数. 用来生成 channel 运行时动态的上下文.
        这部分上下文会出现在模型上下文的 inputs 之前或之后.

        当 channel 每次刷新后, 都会通过它生成动态的上下文消息体.
        >>> async def building(chan: MutableChannel) -> None:
        >>>     async def context() -> list[Message]:
        >>>         return [
        >>>             Message.new(role="system").with_content("dynamic information")
        >>>         ]
        >>>     chan.build.context_messages(context)
        """
        pass

    @abstractmethod
    def instruction(self, func: StringType) -> StringType:
        """
        decorator
        注册一个上下文生成函数. 用来生成 channel 运行时的使用说明.
        这部分上下文会出现在模型交互历史之前, 靠近 system prompt.

        当 channel 每次刷新后, 都会通过它生成动态的 instructions.
        >>> def building(chan: MutableChannel) -> None:
        >>>     def instructions() -> str:
        >>>         return 'instructions'
        >>>     chan.build.instruction(instructions)
        """
        pass

    @abstractmethod
    def add_command(
            self,
            command: Command,
    ) -> None:
        """
        添加一个 Command 对象.
        """
        pass

    @abstractmethod
    def command(
            self,
            *,
            name: str = "",
            doc: Optional[StringType] = None,
            comments: Optional[StringType] = None,
            tags: Optional[list[str]] = None,
            interface: Optional[StringType | Callable[[...], Coroutine[None, None, Any]]] = None,
            available: Optional[Callable[[], bool]] = None,
            # --- 高级参数 --- #
            blocking: Optional[bool] = None,
            call_soon: bool = False,
            priority: int = 0,
            return_command: bool = False,
    ) -> Callable[[CommandFunction], CommandFunction | Command]:
        """
        decorator
        将一个 Python 函数或类的 method 注册到 Channel 上, 成为 Channel 的一个 Command.
        函数会自动反射出 signature, 作为给大模型查看的讯息.
        大模型只会看到函数的签名和注释, 不会看到原始代码.

        :param name: 不为空, 则改写这个函数的名称.
        :param doc: 重定义函数的docstring, 如果传入的是一个函数, 则会在每次刷新时, 动态调用这个函数, 生成它的 docstring.
        :param comments: 改写函数的 body 部分, 用注释形式提供的字符串. 每行前会自动添加 '#'. 不用手动添加.
        :param interface: 大模型看到的函数代码形式. 一旦定义了这个, doc, name, comments 就都会失效.
                支持三种传参方式:
                - str: 直接用字符串来定义模型看到的函数签名.
                    注意, 必须写成 Python Async 的形式.
                    async def foo(...) -> ...:
                      '''docstring'''
                      # comments
                - callalble[[], str]: 生成模型签名的函数
                - async function: 会反射这个 function 来生成一个模型签名的字符串.

        :param tags: 标记函数的分类. 可以让使用者用来过滤和筛选.
        :param available: 通过一个 Available 函数, 定义这个命令的状态. 当这个函数返回 False 时, Command 会动态地变成不可用.
                这种方式, 可以结合状态机逻辑, 动态定义一个 Channel 上的可用函数.
        :param blocking: 这个函数是否会阻塞 channel. 为 None 的话跟随 channel 的默认定义.
                blocking = True 类型的 Command 执行完毕前, 会阻塞后续 Command 执行, 通常是在机器人等需要时序规划的场景中.
                blocking = False 类型则会并发执行. 对于没有先后顺序的工具, 可以设置并行.
        :param call_soon: 决定这个函数进入轨道后, 会第一时间执行 (不等待调度), 还是等待排队执行到自身时.
                如果是 (blocking and call_soon) == True, 会在入队时立刻清空队列.

        :param priority: 命令优先级, <0 时, 有新的命令加入, 就会被自动取消. >0 时, 之前所有优先级比自己低的都会立刻取消.
                高级功能, 不理解的情况下请不要改动它.

        :param return_command: 为真的话, 返回的不是原函数, 而是一个可以视作该函数的 Command 对象. 通常用于测试.
        CommandFunction 最佳实践是:

        >>> # 原始函数是 async, 从而有能力根据真实运行的时间, 阻塞 Channel 后续命令.
        >>> # 有明确的类型约束, 类型约束也是 prompt 的一部分.
        >>> async def func(arg: type) -> Any:
        >>>     '''有清晰的说明'''
        >>>     from ghoshell_moss import ChannelCtx
        >>>     # 可以获取执行这个 command 的真实 runtime
        >>>     runtime = ChannelCtx.runtime()
        >>>     # 如果是被 CommandTask 触发的, 则上下文可以拿到 Task
        >>>     task = ChannelCtx.task()
        >>>     # 通过全局的 IoC 容器获取依赖, 可以拿到运行时的依赖注入.
        >>>     depend = ChannelCtx.get_contract(...)
        >>>     try
        >>>         # 执行逻辑, 不能有线程阻塞, 否则会阻塞全局.
        >>>         ...
        >>>     except asyncio.CancelledError:
        >>>         # 命令可以被调度层正常取消, 有取消的行为. 通常 AI 可以随时取消一个运行的 Command.
        >>>         ...
        >>>     except Exception as e:
        >>>         # 正确处理异常
        >>>         ...
        >>>     finally:
        >>>         # 有运行结束逻辑.
        >>>         ...
        """
        pass

    @abstractmethod
    def idle(self, func: LifecycleFunction) -> LifecycleFunction:
        """
        decorator
        注册一个生命周期函数, 当 Channel 运行 policy 时, 会执行这个函数.

        生命周期的最佳实践是:

        >>> # 原始函数是 async, 从而有能力根据真实运行的时间, 阻塞 Channel 后续命令.
        >>> async def func() -> None:
        >>>     from ghoshell_moss import ChannelCtx
        >>>     # 可以获取执行这个 command 的真实 runtime
        >>>     runtime = ChannelCtx.runtime()
        >>>     # 通过全局的 IoC 容器获取依赖, 可以拿到运行时的依赖注入.
        >>>     depend = ChannelCtx.get_contract(...)
        >>>     try
        >>>         # 执行逻辑, 不能有线程阻塞, 否则会阻塞全局.
        >>>         ...
        >>>     except asyncio.CancelledError:
        >>>         # 生命周期函数随时会被 Channel Runtime 调度取消
        >>>         ...
        >>>     except Exception as e:
        >>>         # 正确处理异常
        >>>         ...
        >>>     finally:
        >>>         # 有运行结束逻辑.
        >>>         ...
        """
        pass

    @abstractmethod
    def start_up(self, func: LifecycleFunction) -> LifecycleFunction:
        """
        启动时执行的生命周期函数
        """
        pass

    @abstractmethod
    def close(self, func: LifecycleFunction) -> LifecycleFunction:
        """
        关闭时执行的生命周期函数
        """
        pass

    @abstractmethod
    def running(self, func: LifecycleFunction) -> LifecycleFunction:
        """
        在整个 Channel Runtime is_running 时间里运行的逻辑. 只会被调用一次.
        注意, 这个函数和 idle / executing 是并行的.
        """
        pass

    @abstractmethod
    def with_binding(self, contract: type[INSTANCE], binding: INSTANCE) -> Self:
        """
        在运行之前, 注册 contract / instance 到全局的 IoC 容器中. 方便任何时候获取.
        """
        pass


ChannelRuntimeContextVar = contextvars.ContextVar("moss.ctx.Runtime")


class ChannelCtx:
    """
    在 Channel 的运行过程中, 方便一个 Command 或者 Lifecycle Function 可以拿到调用它的 Runtime.
    """

    def __init__(
            self,
            runtime: Optional["ChannelRuntime"] = None,
            task: Optional[CommandTask] = None,
    ):
        self._runtime = runtime
        self._task = task

    async def run(self, fn: Callable[..., Coroutine], *args, **kwargs) -> Any:
        """
        将指定的 Runtime 和 CommandTask 注入到一个函数的上下文中.
        """
        async with self.in_ctx():
            return await fn(*args, **kwargs)

    @classmethod
    def channel(cls) -> "Channel":
        """
        返回调用这个函数的 Channel.
        """
        runtime = cls.runtime()
        return runtime.channel

    @contextlib.asynccontextmanager
    async def in_ctx(self):
        runtime_token = None
        task_token = None
        if self._runtime:
            runtime_token = ChannelRuntimeContextVar.set(self._runtime)
        if self._task:
            task_token = CommandTaskContextVar.set(self._task)
        yield
        if runtime_token:
            ChannelRuntimeContextVar.reset(runtime_token)
        if task_token:
            CommandTaskContextVar.reset(task_token)

    @classmethod
    def runtime(cls) -> Optional["ChannelRuntime"]:
        """
        返回调用这个函数的 Runtime, 是一种元编程. 不理解的话不要轻易使用.
        """
        try:
            return ChannelRuntimeContextVar.get()
        except LookupError:
            return None

    @classmethod
    def task(cls) -> CommandTask | None:
        """
        返回触发一个 Command 运行的 CommandTask 对象.
        """
        try:
            return CommandTaskContextVar.get()
        except LookupError:
            return None

    @classmethod
    def container(cls) -> IoCContainer:
        """
        返回当前运行时里的 IoC 容器.
        """
        runtime = cls.runtime()
        if runtime:
            return runtime.container
        return get_container()

    @classmethod
    def get_contract(cls, contract: type[INSTANCE]) -> INSTANCE:
        """
        从 ioc 容器里获取一个实现.
        """
        runtime = cls.runtime()
        if runtime is None:
            raise CommandErrorCode.INVALID_USAGE.error(f"not running in channel ctx")

        item = runtime.container.get(contract)
        if item is None:
            raise CommandErrorCode.NOT_FOUND.error(f"contract {contract} not found")
        return item


class Channel(ABC):
    """
    MOSS 架构本质上想构建一种面向模型使用的高级编程语言.
    它能把跨越各个进程的能力 (主要是函数), 全部通过双工通讯的办法, 提供给 AI 大模型调用.

    对应编程语言 Python 的 Module,  在 Shell 架构中定义了 Channel (中文: 经络)
    """

    @abstractmethod
    def name(self) -> str:
        """
        channel 的名字. 和 Python 的 Module.__name__ 类似.
        全局应该只有一个主 Channel, 它可以是 __main__ .
        """
        pass

    @abstractmethod
    def id(self) -> str:
        """
        Channel 实例也只能用 id 来判断唯一性.
        """
        pass

    @abstractmethod
    def description(self) -> str:
        """
        Channel 的描述. 对于 AI 模型要理解 Channel, 需要看到每个 Channel 的 description.
        """
        pass

    @staticmethod
    def join_channel_path(parent: ChannelFullPath, name: str) -> ChannelFullPath:
        """连接父子 channel 名称的标准语法. 作为全局的约束方式."""
        # todo: 校验 name 的类型, 不允许不合法的 name.
        if parent:
            if not name:
                return parent
            return f"{parent}.{name}"
        return name

    @staticmethod
    def split_channel_path_to_names(channel_path: ChannelFullPath, limit: int = -1) -> ChannelPaths:
        """
        解析出 channel 名称轨迹的标准语法.
        """
        if not channel_path:
            return []
        return channel_path.split(".", limit)

    @abstractmethod
    def bootstrap(self, container: Optional[IoCContainer] = None) -> "ChannelRuntime":
        """
        传入一个 IoC 容器, 创建 Channel 的 Runtime 实例.
        """
        pass


class MutableChannel(Channel, ABC):
    """
    一个约定, 用来描述拥有动态构建能力的 Channel.
    """

    @abstractmethod
    def import_channels(self, *children: "Channel") -> Self:
        """
        添加子 Channel 到当前 Channel. 形成树状关系.
        效果可以比较 python 的 import module_name
        """
        pass

    # todo: 支持别名.
    # @abstractmethod
    # def from_channel_import(self, channel: "Channel", *imports: str | tuple[str, str]) -> Self:
    #     pass

    @property
    @abstractmethod
    def build(self) -> Builder:
        """
        支持通过 Builder 动态构建一个 Channel.
        """
        pass


ChannelInterface = dict[ChannelFullPath, ChannelMeta]
""" 用于描述一个 Channel 能够提供给 AI 的所有能力. """

TaskDoneCallback = Callable[[CommandTask], None] | Callable[[CommandTask], Coroutine[None, None, None]]
RefreshMetaCallback = Callable[[ChannelInterface], None] | Callable[[ChannelInterface], Coroutine[None, None, None]]


class ChannelRuntime(ABC):
    """
    Channel 具体能力的调用方式.
    是对 Channel 的实例化.
    设计思路上 Channel 类似 Python Module 的源代码.
    而 ChannelRuntime 相当于编译后的 ModuleType.

    使用 Runtime 抽象可以屏蔽 Channel 的具体实现, 同样可以用来兼容支持远程调用.

    >>> chan: Channel
    >>> con: IoCContainer
    >>> async def example(chan: Channel, con: IoCContainer):
    >>>     runtime = chan.bootstrap(con)
    >>>     async with runtime:
    >>>         ...

    为什么不叫 Client 呢? 因为 Channel 可能运行在 Client 和 Server 两侧. 它们会通过通讯被同构.
    """

    @property
    @abstractmethod
    def channel(self) -> "Channel":
        """
        Runtime 持有 Channel 本身. 类似实例持有源码.
        """
        pass

    @abstractmethod
    def sub_channels(self) -> dict[str, Channel]:
        """
        当前持有的子 Channel.
        """
        pass

    @property
    @abstractmethod
    def importlib(self) -> "ChannelImportLib":
        pass

    def topic_publisher(self) -> Publisher:
        """
        创建一个独立的 publisher 可以在链路中广播 topic.
        """
        return self.importlib.topics.publisher(
            creator=f"chan/{self.id}",
        )

    async def pub_topic(self, topic: TopicModel | Topic, topic_name: str = "") -> None:
        """
        发送一个 topic 到链路中, 其它监听的 channel 或者 shell 都能拿到这个事件.
        """
        await self.importlib.topics.pub(topic, name=topic_name, creator=f"chan/{self.id}")

    def topic_subscriber(
            self,
            model: type[TOPIC_MODEL],
            *,
            topic_name: str = "",
            maxsize: int = 0,
            keep: SubscribeKeep = "latest",
    ) -> Subscriber[TOPIC_MODEL]:
        """
        创建一个 Subscriber 来获取链路中的 Topic 广播.
        """
        return self.importlib.topics.subscribe_model(
            model=model,
            topic_name=topic_name,
            maxsize=maxsize,
            keep=keep,
        )

    async def fetch_sub_runtime(self, path: ChannelFullPath) -> Self | None:
        """
        在当前 Runtime 的上下文空间里, 寻找一个可能存在的子孙节点.
        """
        pass

    @property
    @abstractmethod
    def states(self) -> StateStore:
        """
        可以在多个 Channel 之间实现状态的共享.
        """
        pass

    @property
    @abstractmethod
    def logger(self) -> LoggerItf:
        """
        提供日志, 避免用户用 logging.getLogger 导致无法治理日志.
        """
        pass

    @property
    @abstractmethod
    def container(self) -> IoCContainer:
        """
        持有 IoC 容器用来解决复杂的调用依赖.
        """
        pass

    @property
    @abstractmethod
    def id(self) -> str:
        """
        runtime 的唯一 id.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        对应的 channel name.
        """
        pass

    @abstractmethod
    async def refresh_metas(
            self,
    ) -> None:
        """
        更新元信息. 是否递归需要每种 ChannelRuntime 自行决定.
        更新后从 metas 取到的值是给模型可以查阅的.
        """
        pass

    def own_meta(self) -> ChannelMeta:
        """
        获取当前 Channel 的元信息, 用来在远端同构出相同的 Channel.
        """
        return self.metas().get("")

    @abstractmethod
    def metas(self) -> dict[ChannelFullPath, ChannelMeta]:
        """
        返回当前模块自身的所有 meta 信息.
        dict 本身是有序的, 深度优先遍历.
        """
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """
        判断一个 Runtime 的连接与通讯是否正常。
        一个运行中的 Runtime 不一定是正确连接的.
        举例, Server 端的 ChannelRuntime 启动后, 可能并未连接到 Provider 端的 ChannelRuntime.
        """
        pass

    @abstractmethod
    def is_running(self) -> bool:
        """
        是否已经启动了. start < running < close
        它用来管理主要的生命周期.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        当前 Channel 对于使用者 (AI) 而言, 是否可用.
        当一个 Runtime 是 running & connected 状态下, 仍然可能会因为种种原因临时被禁用.
        """
        pass

    @abstractmethod
    def is_idle(self) -> bool:
        """
        判断是否进入到了闲时.
        """
        pass

    @abstractmethod
    async def wait_idle(self) -> None:
        """
        阻塞等待到闲时.
        """
        pass

    @abstractmethod
    async def wait_children_idled(self) -> None:
        pass

    @abstractmethod
    async def wait_connected(self) -> None:
        """
        等待 runtime 到连接成功.
        """
        pass

    @abstractmethod
    async def wait_closed(self) -> None:
        """
        等待 Runtime 彻底中断.
        """
        pass

    @abstractmethod
    async def wait_started(self) -> None:
        """
        阻塞等待到启动.
        """
        pass

    @abstractmethod
    def own_commands(self, available_only: bool = True) -> dict[str, Command]:
        """
        返回当前 ChannelRuntime 自身的 commands.
        key 是 command 在当前 Runtime 内部的唯一名字.
        """
        pass

    @abstractmethod
    def commands(self, available_only: bool = True) -> dict[ChannelFullPath, dict[str, Command]]:
        """
        列出所有的 commands.
        """
        pass

    @abstractmethod
    def get_command(self, name: CommandUniqueName) -> Optional[Command]:
        """
        使用 unique name 获取一个 command.
        """
        pass

    @abstractmethod
    async def clear(self) -> None:
        """
        清空当前 Runtime 所有的运行状态.
        """
        pass

    @abstractmethod
    async def clear_sub_channels(self) -> None:
        """
        清空当前 Runtime 所有子 channel 的 runtime
        """
        pass

    async def push_task(self, *tasks: CommandTask) -> None:
        """
        将一个 Task 推入到执行栈中. 阻塞到完成入栈为止. 仍然要在外侧 await.

        ChannelRuntime 运行的基本逻辑是:
        1. 一次只能运行一个阻塞 task
        2. none-blocking 的 task 不会阻塞, 但是可以被 clear.
        3. clear 会清空掉所有的运行状态.
        举例:
        >>> async def run_task(runtime: ChannelRuntime, t:CommandTask):
        >>>     await runtime.push_task(t)
        >>>     return await t
        """
        for task in tasks:
            paths = Channel.split_channel_path_to_names(task.chan)
            await self.push_task_with_paths(paths, task)

    @abstractmethod
    async def push_task_with_paths(self, paths: ChannelPaths, task: CommandTask) -> None:
        """
        按路径的方式分配 task. 在 runtime 中排列执行.
        """
        pass

    @abstractmethod
    def on_task_done(self, callback: TaskDoneCallback) -> None:
        """
        注册当 Task 运行结束后的回调.
        """
        pass

    def create_command_task(
            self,
            name: CommandUniqueName,
            *,
            args: tuple | None = None,
            kwargs: dict | None = None,
    ) -> CommandTask:
        """
        example to create channel task
        通过 Runtime 创建一个新的的 CommandTask.
        """
        command = self.get_command(name)
        if command is None:
            raise LookupError(f"Channel {self.name} has no command {name}")
        args = args or ()
        kwargs = kwargs or {}
        chan, command_name = Command.split_unique_name(name)
        task = BaseCommandTask.from_command(
            command,
            chan,
            args=args,
            kwargs=kwargs,
        )
        return task

    async def execute_command(
            self,
            name: CommandUniqueName,
            *,
            args: tuple | None = None,
            kwargs: dict | None = None,
    ) -> Any:
        """
        执行命令并且阻塞等待拿到结果.
        """
        task = self.create_command_task(name, args=args, kwargs=kwargs)
        await self.push_task(task)
        return await task

    @abstractmethod
    async def start(self) -> None:
        """
        启动 Runtime
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        关闭 Runtime.
        """
        pass

    @abstractmethod
    def close_sync(self) -> None:
        """
        同步关闭一个 Runtime.
        只有特殊情况下需要使用.
        """
        pass

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_val:
            self.logger.exception(exc_val)
        await self.close()


class ChannelImportLib(ABC):
    """
    在一个上下文中, 所有 ChannelRuntime 应该共享的 Importlib.
    用来避免一个 Channel 被多个 Channel 引用, 从而实例化出多个 Runtime.
    类似 python 的 __import__
    """

    @property
    @abstractmethod
    def main(self) -> ChannelRuntime:
        """
        实例化的起点 Channel. 类似 main.py
        """
        pass

    @abstractmethod
    def get_channel_runtime(self, channel: Channel) -> ChannelRuntime | None:
        """
        获取一个已经启动过的 Channel Runtime.
        """
        pass

    @abstractmethod
    async def get_or_create_channel_runtime(self, channel: Channel) -> ChannelRuntime | None:
        """
        获取一个 Channel Runtime, 如果没有启动的话就启动它.
        """
        pass

    @abstractmethod
    async def compile_channel(self, channel: Channel) -> ChannelRuntime | None:
        """ """
        pass

    @property
    @abstractmethod
    def logger(self) -> LoggerItf:
        """
        返回日志对象.
        """
        pass

    @property
    @abstractmethod
    def topics(self) -> TopicService:
        pass

    @abstractmethod
    def is_running(self) -> bool:
        """
        importlib 是否已经启动了.
        """
        pass

    @abstractmethod
    async def start(self) -> None:
        """
        启动.
        """
        pass

    def descendants(self, root: ChannelFullPath = "") -> dict[ChannelFullPath, ChannelRuntime]:
        root_runtime = self.recursively_find_runtime(self.main, root)
        if root_runtime is None:
            return {}
        return self.find_descendants(root_runtime.channel)

    def all(self, root: ChannelFullPath = "") -> dict[ChannelFullPath, ChannelRuntime]:
        root_runtime = self.recursively_find_runtime(self.main, root)
        if root_runtime is None:
            return {}
        all_runtimes = {"": root_runtime}
        for path, runtime in self.descendants(root).items():
            all_runtimes[path] = runtime
        return all_runtimes

    def find_descendants(
            self,
            channel: Channel,
            bloodline: set | None = None,
            depth: int = 0,
    ) -> dict[ChannelFullPath, ChannelRuntime]:
        """
        语法糖, 用来获取一个 Channel 所有的子孙 Channel. 如果成环就会抛出异常.
        """
        runtime = self.get_channel_runtime(channel)
        if runtime is None or not runtime.is_running():
            return {}
        result = {}
        bloodline = bloodline or set()
        if channel in bloodline:
            parent = [c.name for c in bloodline]
            raise RuntimeError(f"import loop of {channel.name()} id={channel.id()}, parent={parent}")
        bloodline.add(channel)
        for name, child in runtime.sub_channels().items():
            child_runtime = self.get_channel_runtime(child)
            result[name] = child_runtime
            if child_runtime is not None and child_runtime.is_running():
                descendants = self.find_descendants(child, bloodline, depth + 1)
                for path, descendant in descendants.items():
                    real_path = Channel.join_channel_path(name, path)
                    result[real_path] = descendant
        # 退栈.
        bloodline.remove(channel)
        return result

    def recursively_find_runtime(self, runtime: ChannelRuntime, path: ChannelFullPath) -> ChannelRuntime | None:
        if path == "":
            return runtime
        paths = Channel.split_channel_path_to_names(path, 1)
        child_name = paths[0]
        further_path = paths[1] if len(paths) > 1 else ""
        if child_name == "":
            return runtime
        child_channel = runtime.sub_channels().get(child_name)
        if child_channel is None:
            return None
        child_runtime = self.get_channel_runtime(child_channel)
        if child_runtime is None:
            return None
        return self.recursively_find_runtime(child_runtime, further_path)

    async def recursively_fetch_runtime(self, root: ChannelRuntime, paths: ChannelPaths) -> ChannelRuntime | None:
        if len(paths) == 0:
            return root
        child_name = paths[0]
        further_path = paths[1:]
        child = root.sub_channels().get(child_name)
        if child is None:
            return None
        child_runtime = await self.get_or_create_channel_runtime(child)
        return await self.recursively_fetch_runtime(child_runtime, further_path)

    @abstractmethod
    async def close(self) -> None:
        pass


# --- for develop --- #

class ChannelInterface(ABC):
    """
    ChannelApp 范式的可继承版本. 提供一种标准的 Channel 抽象设计策略.

    开发者实现一个 ChannelInterface 的 Abstract 类, 定义必要的函数 (Command 或生命周期函数)
    然后提前实现好 as_channel 函数.

    >>> class SomeChannelInterface(ChannelInterface):
    >>>      @abstractmethod
    >>>      def foo(self) -> int:
    >>>          pass
    >>>
    >>>      def as_channel(self, name, description) -> Channel:
    >>>           from ghoshell_moss import PyChannel
    >>>           channel = PyChannel(name=name, description=description)
    >>>           # 注册好 interface 上的函数.
    >>>           channel.build.command()(self.foo)
    >>>           return channel

    这样具体的实现就可以替换了.
    而且 ChannelInterface 本身也可以注册到容器中, 方便通过 IoC 容器来获取.


    >>> def build_channel(container: IoCContainer) -> Channel:
    >>>    return container.make(SomeChannelInterface).as_channel()

    也可以考虑类名就是 name, docstring 就是 description.
    这样未来 AI 创建一个 ChannelInterface 时, 有非常明确的要实现功能, 而且不需要去理解
    """

    @abstractmethod
    def as_channel(
            self,
            name: str = "",
            description: str = "",
    ) -> Channel:
        """
        子抽象类应该要实现这个函数.
        """
        pass


R = TypeVar("R")


class CommandExecutor(Generic[R]):
    """
    将 Command 包装成运行时对象.
    它被调用时, 实际上会把 CommandTask 发送给 ChannelRuntime.
    """

    def __init__(
            self,
            command: Command[R],
            runtime: ChannelRuntime,
            *,
            channel_path: ChannelFullPath = '',
    ):
        self._command = command
        self._runtime = runtime
        self._channel_path = channel_path

    async def execute(self, *args, **kwargs) -> CommandTask[R]:
        task = BaseCommandTask.from_command(
            command_=self._command,
            args=args,
            kwargs=kwargs,
            chan_=self._channel_path,
        )
        await self._runtime.push_task(task)
        return task

    async def __call__(self, *args, **kwargs) -> R:
        task = await self.execute(*args, **kwargs)
        return await task

    def __prompt__(self) -> str:
        return self._command.meta().interface


class ChannelExecutor:
    """
    可以用代码的方式理解和使用的 ChannelExecutor.
    todo: 想明白要怎么开发. push task 可能要改成同步的更简单.
    """

    def __init__(
            self,
            channel_path: ChannelFullPath,
            runtime: ChannelRuntime,
    ):
        self._runtime = runtime
        self._channel_path = channel_path

    def __getitem__(self, item: ChannelFullPath) -> Self:
        runtime = self._runtime.importlib.recursively_find_runtime(self._runtime, self._channel_path)
        if runtime is None:
            raise LookupError(f"Channel not found: {self._channel_path}")
        return ChannelExecutor(channel_path=item, runtime=runtime)

    def __getattr__(self, item: str) -> Callable[[...], Awaitable]:
        command = self._runtime.get_command(item)
        if command is None:
            raise AttributeError(f"Channel does not hav command: {item}")

        def wrapper(*args, **kwargs) -> Awaitable:
            task = BaseCommandTask.from_command(command, chan_=self._channel_path, args=args, kwargs=kwargs)
            return task

        return wrapper


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

    @property
    @abstractmethod
    def channel(self) -> Channel:
        pass

    @property
    @abstractmethod
    def runtime(self) -> ChannelRuntime:
        pass

    @abstractmethod
    async def wait_closed(self) -> None:
        """
        等待 provider 运行到结束为止.
        """
        pass

    @abstractmethod
    async def wait_stop(self) -> None:
        pass

    @abstractmethod
    def wait_closed_sync(self) -> None:
        """
        同步等待运行结束.
        """
        pass

    @abstractmethod
    async def aclose(self) -> None:
        """
        主动关闭
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
        async with self.arun(channel):
            await self.wait_stop()

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
    @abstractmethod
    async def arun(self, channel: Channel) -> AsyncIterator[Self]:
        """
        支持 async with statement 的运行方式启动一个 channel.
        """
        pass
