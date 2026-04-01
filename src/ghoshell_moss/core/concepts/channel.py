import asyncio
import contextlib
import contextvars
import threading
from abc import ABC, abstractmethod
from collections.abc import Awaitable
from contextlib import asynccontextmanager
from typing import (
    Any,
    Optional,
    Union,
    Callable,
    Coroutine,
    AsyncIterator,
)

from ghoshell_container import INSTANCE, IoCContainer, get_container
from pydantic import BaseModel, Field, AwareDatetime
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
from datetime import datetime
from dateutil import tz

__all__ = [
    "Channel",
    "TaskDoneCallback",
    "RefreshMetaCallback",
    "ChannelRuntime",
    "ChannelTree",
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

    name: str = Field(default='', description="The origin name of the channel, kind like python module name.")
    description: str = Field(default="", description="The description of the channel.")
    failure: str = Field(default="", description="The failure status of the channel.")
    channel_id: str = Field(default="", description="The ID of the channel.")
    available: bool = Field(default=True, description="Whether the channel is available.")
    commands: list[CommandMeta] = Field(default_factory=list, description="The list of commands.")
    states: dict[str, str] = Field(default_factory=dict, description="The states of the channel.")
    current_state: str = Field(default="", description="The current state of the channel.")
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
    virtual: bool = Field(default=False, description="Whether the channel is virtual")

    created: AwareDatetime = Field(
        default_factory= lambda: datetime.now(tz.gettz()),
        description="The channel meta creation time. "
    )

    @classmethod
    def new_empty(cls, id: str, channel: "Channel", failure: str = "") -> Self:
        return cls(
            name=channel.name(),
            description=channel.description(),
            dynamic=True,
            channel_id=id,
            available=False,
            failure=failure,
        )

    def marshal(self) -> str:
        return self.model_dump_json(indent=0, ensure_ascii=False, exclude_defaults=True)


ChannelFullPath = str
"""
在树形嵌套的 channel 结构中, 对一个具体 channel 进行寻址的方法.
完全对齐 python 的  a.b.c 寻址逻辑. 

同时它也描述了一个神经信号 (command call) 经过的路径, 比如从 a -> b -> c 执行.
"""

ChannelId = str
"""channel 实例需要有唯一 id"""

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

    async def run(self, fn: Callable[..., Awaitable[Any]], *args, **kwargs) -> Any:
        """
        将指定的 Runtime 和 CommandTask 注入到一个函数的上下文中.
        """
        with self.in_ctx():
            return await fn(*args, **kwargs)

    @classmethod
    def channel(cls) -> "Channel":
        """
        返回调用这个函数的 Channel.
        """
        runtime = cls.runtime()
        if runtime is None:
            raise CommandErrorCode.INVALID_USAGE.error(f"not running in channel ctx")
        return runtime.channel

    @contextlib.contextmanager
    def in_ctx(self):
        runtime_token = None
        task_token = None
        try:
            if self._runtime:
                runtime_token = ChannelRuntimeContextVar.set(self._runtime)
            if self._task:
                task_token = CommandTaskContextVar.set(self._task)
            yield
        finally:
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

    def virtual_sub_channels(self) -> dict[str, Channel]:
        """
        管理当前 Channel runtime 能拿到的动态子节点.
        """
        return {}

    @property
    @abstractmethod
    def tree(self) -> "ChannelTree":
        """
        channel tree shared by all channel runtime in the same scope (from main channel)
        """
        pass

    def topic_publisher(self) -> Publisher:
        """
        创建一个独立的 publisher 可以在链路中广播 topic.
        """
        return self.tree.topics.publisher(
            creator=f"channel/{self.id}",
        )

    async def pub_topic(self, topic: TopicModel | Topic, topic_name: str = "") -> None:
        """
        发送一个 topic 到链路中, 其它监听的 channel 或者 shell 都能拿到这个事件.
        """
        await self.tree.topics.pub(topic, name=topic_name, creator=f"channel/{self.id}")

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
        return self.tree.topics.subscribe_model(
            model=model,
            topic_name=topic_name,
            maxsize=maxsize,
            keep=keep,
        )

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

    def self_meta(self) -> ChannelMeta:
        """
        获取当前 Channel 的元信息, 用来在远端同构出相同的 Channel.
        """
        return self.metas().get("")

    def own_metas(self) -> dict[ChannelFullPath, ChannelMeta]:
        """
        返回当前 ChannelRuntime 持有的元信息. 通常只有自身的信息.
        但对于 Proxy 类型的 Channel 而言, 它同时代理了一个 Channel 树结构.
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
    async def refresh_own_metas(self) -> None:
        """
        刷新自身的 meta
        """
        pass

    @abstractmethod
    def own_commands(self, available_only: bool = True) -> dict[CommandUniqueName, Command]:
        """
        返回当前 ChannelRuntime 自身的 commands.
        key 是 command 在当前 Runtime 内部的唯一名字. 可以在 own_metas 中找到对应的存在.
        """
        pass

    @abstractmethod
    def has_own_command(self, name: CommandUniqueName) -> bool:
        """
        判断一个命令是否在当前 ChannelRuntime 内部持有.
        """
        pass

    @abstractmethod
    def get_own_command(self, name: CommandUniqueName) -> Optional[Command]:
        """
        获取自身持有的命令.
        """
        pass

    @abstractmethod
    async def clear_own(self) -> None:
        """
        清空自身的运行状态.
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

    @abstractmethod
    def create_asyncio_task(self, cor: Coroutine) -> asyncio.Task:
        """
        create asyncio task during runtime
        the task will be canceled if the runtime is closed.
        """
        pass

    async def execute_task(self, task: CommandTask) -> None:
        """
        simple way to execute task in runtime without queue logic.
        """
        if not self.is_running():
            task.fail(CommandErrorCode.NOT_RUNNING.error(f"Channel {self.name} is not running"))
        elif not self.is_connected():
            task.fail(CommandErrorCode.NOT_CONNECTED.error(f"Channel {self.name} is not connected"))
        try:
            with ChannelCtx(self, task).in_ctx():
                task.set_state('ex')
                # dry run 不会清空 task 状态.
                result = await task.dry_run()
                task.resolve(result)
        except Exception as e:
            task.fail(e)
        finally:
            if not task.done():
                task.cancel('unknown')

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
        不会执行.
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
    async def start(self) -> Self:
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

    # --- Channel tree recursive methods --- #

    def metas(self) -> dict[ChannelFullPath, ChannelMeta]:
        """
        返回当前模块自身的所有 meta 信息.
        dict 本身是有序的, 深度优先遍历.
        """
        return self.tree.metas(self.channel)

    def fetch_sub_runtime(self, path: ChannelFullPath) -> Self | None:
        """
        在当前 Runtime 的上下文空间里, 寻找一个可能存在的子孙节点.
        """
        return self.tree.get_runtime_by_path(path, self.channel)

    def refresh_metas(
            self,
    ) -> asyncio.Future[None]:
        """
        刷新 ChannelRuntime 树结构, 然后刷新包含自身在内的树节点元信息.
        """
        return self.tree.refresh(self.channel.id(), wait=True)

    async def clear(self) -> None:
        """
        清空当前 Runtime 所有的运行状态.
        """
        await self.tree.clear(self)

    async def clear_children(self) -> None:
        """
        清空当前 Runtime 所有子 channel 的 runtime
        """
        await self.tree.clear_children_runtimes(self.channel)

    def commands(self, available_only: bool = True) -> dict[ChannelFullPath, dict[str, Command]]:
        """
        列出所有的 commands.
        """
        # 递归逻辑统一通过 ChannelTree 实现. 保留 Runtime 接口
        return self.tree.commands(self.channel, available_only=available_only)

    def get_command(self, name: CommandUniqueName) -> Optional[Command]:
        """
        使用 unique name 获取一个 command.
        """
        # 递归逻辑统一通过 ChannelTree 实现. 保留 Runtime 接口
        return self.tree.get_command(self.channel, name)

    async def wait_children_idled(self) -> None:
        """
        wait sub channels idle
        """
        await self.tree.wait_channel_children_idle(self.channel)


class ChannelTree(ABC):
    """
    在一个上下文中, 所有 ChannelRuntime 应该共享的 tree.
    用来避免一个 Channel 被多个 Channel 引用, 从而实例化出多个 Runtime.
    保证 channel runtime 的唯一性同时, 管理父子关系.
    """

    @property
    @abstractmethod
    def main(self) -> ChannelRuntime:
        """
        实例化的起点 Channel. 类似 main.py
        """
        pass

    @abstractmethod
    def get_channel_runtime(self, channel: Channel, running: bool = False) -> ChannelRuntime | None:
        """
        获取一个已经启动过的 Channel Runtime.
        """
        pass

    async def wait_channel_children_idle(self, channel: Channel) -> None:
        """
        等待一个节点所有的子节点都 idle.
        如果目标节点的 runtime 不存在, 也会立刻返回.
        """
        children = self.get_children_runtimes(channel)
        if len(children) > 0:
            wait_all = []
            for child_name, runtime in children.items():
                wait_all.append(runtime.wait_idle())
            _ = await asyncio.gather(*wait_all, return_exceptions=True)
        return

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
        """
        持有所有 channel 共享的 topic service.
        """
        pass

    @abstractmethod
    def is_running(self) -> bool:
        """
        是否已经启动了.
        """
        pass

    @abstractmethod
    async def start(self) -> None:
        """
        启动.
        """
        pass

    def refresh_all(self) -> asyncio.Future[None]:
        return self.refresh(self.main.channel.id(), wait=True)

    @abstractmethod
    def refresh(self, id: ChannelId, wait: bool = False) -> asyncio.Future[None]:
        """
        更新一个 channel id 对应的整颗子树.
        同一时间每个 channel runtime 只会更新一次.
        """
        pass

    @abstractmethod
    def get_children_runtimes(self, channel: Channel) -> dict[str, "ChannelRuntime"]:
        """
        获取一个节点所有已经激活的子节点.
        """
        pass

    @abstractmethod
    def get_runtime_by_path(self, path: ChannelFullPath, root: Channel | None = None) -> ChannelRuntime | None:
        """
        基于路径查找一个 runtime.
        """
        pass

    async def clear(self, runtime: ChannelRuntime) -> None:
        """
        清空一个 runtime 和它所有的子节点.
        """
        if not runtime.is_running():
            return
        # 清空 runtime 自身.
        await runtime.clear_own()
        # 递归清空.
        await self.clear_children_runtimes(runtime.channel)
        self.logger.info("%r clear channel runtime %s, %s", self, runtime.name, runtime.id)

    async def clear_children_runtimes(self, channel: Channel) -> None:
        """
        根据 channel 清空其所有的子节点.
        """
        children = self.get_children_runtimes(channel)
        clearing = []
        for child_name, runtime in children.items():
            if runtime.is_running():
                clearing.append(self.clear(runtime))
        if len(clearing) > 0:
            done = await asyncio.gather(*clearing)
            for r in done:
                if isinstance(r, Exception):
                    self.logger.exception("%s clear child failed: %s", self, r)

    @abstractmethod
    def all(self, root: ChannelFullPath = "") -> dict[ChannelFullPath, ChannelRuntime]:
        """
        以 root 路径为根节点, 返回所有的运行中节点.
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        pass

    @abstractmethod
    def commands(self, channel: Channel, available_only: bool = True) -> dict[ChannelFullPath, dict[str, Command]]:
        """
        递归获取一个 channel 所有的子命令, 按路径完成分组.
        """
        pass

    @abstractmethod
    def get_command(self, channel: Channel, name: CommandUniqueName) -> Command | None:
        """
        递归查找单个命令.
        """
        pass

    @abstractmethod
    def metas(self, root: Channel | None = None) -> dict[ChannelFullPath, ChannelMeta]:
        """
        返回一个节点的所有在树中注册的子节点的 metas.
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
