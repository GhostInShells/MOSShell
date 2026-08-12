"""
Channel (中文名: 经络) : 流式解释器组织 树形/有状态/可流式控制 组件的抽象集合.
"""

import asyncio
import contextlib
import contextvars
import threading
from abc import ABC, abstractmethod
from collections.abc import Awaitable
from typing import (
    Any,
    Optional,
    Annotated,
    Callable,
    Coroutine, Literal,
    List, TYPE_CHECKING,
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
    Topic,
    TOPIC_MODEL,
)
from ghoshell_moss.message import Message
from ghoshell_common.contracts import LoggerItf
from datetime import datetime
from dateutil import tz

if TYPE_CHECKING:
    from PIL.Image import Image

__all__ = [
    "Channel", "ChannelState",
    "TaskDoneCallback",
    "ChannelRuntime",
    "ChannelTree",
    "ChannelFullPath",
    "ChannelMeta",
    "ChannelPaths",
    "ChannelProvider",
    "ChannelProxy",
    "ChannelCtx",
    "ChannelName",
    "ChannelNamePattern",
    # scope 语法
    "ChannelScope", "ChannelScopeType", "ChannelScopeDefaultType",
]


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
    modules: list[str] = Field(default_factory=list, description="Permanent capability module names (for debug).")
    proxy: bool = Field(default=False, description="Whether the channel is proxy, not local one.")

    # about instructions / context messages
    # ModelContext is built by many messages blocks, we believe the blocks should be :
    #  - instructions before conversation
    #  - memories
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
    help: str = Field(default="", description="Warm data — what this channel currently exposes. Rendered with command interfaces.")
    memory: list[Message] = Field(default_factory=list, description="The channel memory messages")

    dynamic: bool = Field(default=True, description="Whether the channel is dynamic, need refresh each time")
    virtual: bool = Field(default=False, description="Whether the channel is virtual")

    created: AwareDatetime = Field(
        default_factory=lambda: datetime.now(tz.gettz()),
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

    def to_dict(self) -> dict:
        return self.model_dump(exclude_none=True, exclude_defaults=True)

    def marshal(self) -> str:
        return self.model_dump_json(indent=0, ensure_ascii=False, exclude_defaults=True, exclude_none=True)


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

ChannelRuntimeContextVar = contextvars.ContextVar("moss.ctx.Runtime")

ChannelNamePattern = r'^[a-zA-Z_][a-zA-Z0-9_]*$'
ChannelName = Annotated[str, Field(pattern=ChannelNamePattern)]


class ChannelCtx:
    """
    在 Channel 的运行过程的 Command 或者 Lifecycle Function 可以使用的模块.
    通过 contextvars Context 来传递相关上下文.
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


class ChannelState(ABC):
    """
    Channel 的运行时状态, 用来快速构建一个 StateChannel.

    运行过程中要使用 IoC 容器, 可以通过 bootstrap 函数被调用时获取并持有依赖
    或者 channel_builder.CommandUtil.get_contract 在每个 command 和生命周期函数被调用时获取.
    """

    @abstractmethod
    def name(self) -> str:
        """
        return name of the state
        """
        pass

    @abstractmethod
    def description(self) -> str:
        """
        return description of the state
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        if the state is available
        """
        pass

    def is_dynamic(self) -> bool:
        """
        if the state is dynamic, need to refresh each time.
        """
        # 既然是有状态的 Channel, 默认是动态的.
        return True

    async def get_instruction(self) -> str:
        """
        return instruction provided by the state
        """
        return ''

    async def get_context_messages(self) -> "List[Message | str | Image]":
        """
        return the context messages from the state.
        """
        return []

    async def on_startup(self) -> None:
        """
        when channel startup.
        """
        return None

    async def on_close(self) -> None:
        """
        when channel close.
        """
        return None

    async def on_running(self) -> None:
        """
        when channel is running.
        """
        return None

    async def on_idle(self) -> None:
        """
        when channel is idle, all the commands are done and the children are idle as well
        """
        return None

    async def on_refresh_meta(self) -> None:
        """
        fires each refresh cycle, before the channel regenerates its own metas.

        async by design — the sync structure-refresh path (get_virtual_children,
        get_children) must return instantly.  Heavy work lives here instead:
        alive_cells queries, proxy cache rebuilds, any I/O-bound state sync.

        This hook and get_virtual_children are the two halves of a dynamic
        channel: on_refresh_meta updates the internal cache (async), and
        get_virtual_children returns it (sync).  One-cycle delay is expected.

        default noop.
        """
        return None

    @abstractmethod
    def own_commands(self) -> dict[str, Command]:
        """
        return the commands mapping by name
        """
        pass

    @abstractmethod
    def get_own_command(self, name: str) -> Command | None:
        """
        get a command by name
        """
        pass

    def bootstrap(self, container: IoCContainer) -> None:
        """
        register something to the container. or get some contracts from it.
        函数会被 ChannelRuntime 实例化后调用.
        """
        return

    def get_children(self) -> dict[ChannelName, 'Channel']:
        """
        return the sustain children channel
        """
        return {}

    def get_virtual_children(self) -> dict[ChannelName, 'Channel']:
        """
        return the virtual children that may be changed during runtime
        """
        return {}


class Channel(ABC):
    """
    MOSS 架构本质上想构建一种面向模型使用的高级编程语言.
    它能把跨越各个进程的能力 (主要是函数), 全部通过双工通讯的办法, 提供给 AI 大模型调用.

    对应编程语言 Python 的 Module,  在 Shell 架构中定义了 Channel (中文: 经络)
    Channel 实例本身应该是无副作用的, 只有在运行时通过 bootstrap 之后, 才会返回有作用的实例.
    它的唯一性通过 channel.id 判断, 而不是实例本身.
    """
    MAIN_CHANNEL_NAME = '__main__'
    PATH_SEPARATOR = '.'

    @abstractmethod
    def name(self) -> ChannelName:
        """
        channel 的名字. 和 Python 的 Module.__name__ 类似.
        全局应该只有一个主 Channel, 它可以是 __main__ .
        """
        pass

    @abstractmethod
    def id(self) -> str:
        """
        Channel 实例用 id 来判断唯一性, 会与 Runtime 绑定.
        """
        pass

    def __eq__(self, other):
        return self.id() == other.id() and self.name() == other.name()

    @abstractmethod
    def description(self) -> str:
        """
        Channel 的描述. 对于 AI 模型要理解 Channel, 需要看到每个 Channel 的 description.
        """
        pass

    @classmethod
    def join_channel_path(cls, parent: ChannelFullPath, *names: str) -> ChannelFullPath:
        """连接父子 channel 名称的标准语法. 作为全局的约束方式."""
        names = list(names)
        names_str = cls.PATH_SEPARATOR.join(names) if len(names) > 0 else ''
        if parent:
            if not names_str:
                return parent

            return f"{parent}.{names_str}"
        return names_str

    @classmethod
    def split_channel_path_to_names(cls, channel_path: ChannelFullPath, limit: int = -1) -> ChannelPaths:
        """
        解析出 channel 名称轨迹的标准语法.
        """
        if not channel_path:
            return []
        return channel_path.split(cls.PATH_SEPARATOR, limit)

    def bootstrap(self, container: Optional[IoCContainer] = None) -> "ChannelRuntime":
        """
        传入一个 IoC 容器, 创建 Channel 的 Runtime 实例.
        """
        if container is None:
            from ghoshell_container import Container
            container = Container(name="channel/{name}{id}".format(name=self.name(), id=self.id()))
        runtime_instance = self.materialize(container)
        if isinstance(runtime_instance, ChannelRuntime):
            return runtime_instance
        elif isinstance(runtime_instance, ChannelState):
            from ghoshell_moss.core.blueprint.states_channel import new_stateful_channel_from_main
            return new_stateful_channel_from_main(runtime_instance, id=self.id()).bootstrap(container)
        raise RuntimeError(f"invalid channel runtime instance: {runtime_instance}")

    @abstractmethod
    def materialize(self, container: IoCContainer) -> 'ChannelState | ChannelRuntime':
        pass

    @classmethod
    def validate_name(cls, name: str) -> bool:
        import regex as re
        return re.fullmatch(ChannelNamePattern, name) is not None


TaskDoneCallback = Callable[[CommandTask], None]

ChannelScopeType = Literal['flow', 'all', 'any']
ChannelScopeDefaultType = 'flow'


class ChannelScope(ABC):
    """
    Channel 作用域语法.
    用来管理同组作用域下所有的 CommandTask 生命周期.
    """

    @property
    @abstractmethod
    def scope_id(self) -> str:
        """作用域的唯一id, 通常就是 task.cid """
        pass

    @abstractmethod
    def add_task(self, task: CommandTask) -> CommandTask:
        """将一个 task 绑定到作用域. 作用域关闭时, 所有添加的 task 都会被关闭."""
        pass

    @abstractmethod
    def commit(self, task: CommandTask) -> CommandTask:
        """结束作用域的注册. 之后绑定到这个 scope 上的任务都会失败."""
        pass

    @abstractmethod
    def is_commited(self) -> bool:
        pass

    @abstractmethod
    def is_closed(self) -> bool:
        pass

    @abstractmethod
    async def tick(
            self,
            *,
            until: ChannelScopeType,
            timeout: float | None = None,
    ) -> None:
        """开始作用域的计时逻辑"""
        pass

    @abstractmethod
    async def wait_close(self) -> str | None:
        """等待作用域正常结束"""
        pass

    @abstractmethod
    def close(self, reason: str = '') -> None:
        """主动关闭作用域"""
        pass


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

    def get_child_channel(self, name: str) -> Optional[Channel]:
        child = self.sub_channels().get(name)
        if child is None:
            return self.virtual_sub_channels().get(name)
        return child

    @property
    @abstractmethod
    def tree(self) -> "ChannelTree":
        """
        channel tree shared by all channel runtime in the same scope (from main channel)
        """
        pass

    def topic_publisher(self, topic: type[TopicModel]) -> Publisher[TopicModel]:
        """
        创建一个独立的 publisher 可以在链路中广播 topic.
        """
        topic_name = topic
        if isinstance(topic, type):
            if issubclass(topic, TopicModel):
                topic_name = topic.default_topic_name()
            else:
                raise TypeError(f'topic {topic_name!r} is not a topic model')
        path = self.channel_path()
        return self.tree.topics.publisher(
            topic_name=topic_name,
            creator=f"channel/{path}",
        )

    def pub_topic(self, topic: TopicModel | Topic, topic_name: str = "") -> None:
        """
        发送一个 topic 到链路中, 其它监听的 channel 或者 shell 都能拿到这个事件.
        """
        self.tree.topics.pub(topic, name=topic_name, creator=f"channel/{self.id}")

    def topic_subscriber(
            self,
            model: type[TOPIC_MODEL],
            *,
            topic_name: str = "",
            maxsize: int = 0,
    ) -> Subscriber[TOPIC_MODEL]:
        """
        创建一个 Subscriber 来获取链路中的 Topic 广播.
        """
        return self.tree.topics.subscribe_model(
            model=model,
            topic_name=topic_name,
            maxsize=maxsize,
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
    def refresh_own_metas(self) -> asyncio.Future[None]:
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

    @abstractmethod
    def open_scope(
            self,
            task: CommandTask,
    ) -> None:
        """开启一个作用域. 返回处理后的 CommandTask. """
        pass

    @abstractmethod
    def get_active_scope(self, scope_id: str | None, pop: bool) -> ChannelScope | None:
        """获取作用域. scope_id 为None 的话返回最后一个 """
        pass

    @abstractmethod
    def commit_scope(self, task: CommandTask) -> None:
        """结束一个作用域的注册."""
        pass

    def push_task(self, *tasks: CommandTask) -> None:
        """
        将当前 ChannelRuntime 视作根节点, 将 task 推入 channel runtime 的执行栈.
        根节点唯一入口. 包含根节点自身特殊逻辑.
        这个函数的副作用, 不接受无序 tasks.
        """
        # 通过显式定义, 展示 CommandTask 体系处理的基本逻辑.
        is_running = self.is_running()
        for task in tasks:
            if not is_running:
                task.fail(CommandErrorCode.NOT_RUNNING.error('Channel Runtime not running'))
                continue
            # 作用域语法检查. 只有将 channel runtime 作为根路径使用时才会触发检查.
            # 对于一个 Channel 树的入口而言, 有责任创建作用域体系. 同一个进程内只有一个入口要管理整体作用域.
            # 跨进程通讯 (ChannelProxy) 的远程根节点入口会根据已有的标记完成重建.
            command_name = task.meta.name
            # ChannelRuntime 作用域语法.
            if command_name == self.__scope_enter__.__name__:
                # 开启一个新的作用域. 作用域关闭的话, 这个新作用域和它的所有task 都会关闭.
                # 使用 task id 作为作用域标记. 给后续节点做染色.
                # scope 体系是一个父子依赖的 stack, 父 scope 关闭时也会关闭子 stack.
                task.func = self.__scope_enter__
                self.open_scope(task)
            elif command_name == self.__scope_exit__.__name__:
                task.func = self.__scope_exit__
                self.commit_scope(task)
            elif last := self.get_active_scope(None, False):
                # 做标记.
                last.add_task(task)
                task.scope_id = last.scope_id
            else:
                # 无任何作用域信息时, 和默认规则保持完全一致.
                pass
            # 作用域语法可能直接开启, 或关闭了 task 生命周期.
            if task.done():
                continue
            paths = Channel.split_channel_path_to_names(task.chan)
            # 真正入执行栈.
            self.push_task_with_paths(paths, task)

    def push_task_with_paths(self, paths: ChannelPaths, task: CommandTask) -> None:
        """
        将一个 Task 推入到执行栈中.
        *** 是 ChannelRuntime 有序执行 Task 的唯一合法入口 ***
        :param paths: task 对于当前 ChannelRuntime 的相对路径, 为空表示为当前 ChannelRuntime.
        :param task: command task
        """
        if task.done():
            # 跳过已经 done 的不要入队.
            return
        if not self.is_connected():
            task.fail(CommandErrorCode.NOT_CONNECTED.error('Channel Runtime not connected'))
            return
        elif not self.is_available():
            task.fail(CommandErrorCode.NOT_AVAILABLE.error('Channel Runtime not available'))
            return
        # 对空函数做预处理, 允许传入 caller 本身.
        is_self_task = len(paths) == 0
        if is_self_task and task.is_bare_task():
            # 对 bare task 做预处理.
            own_command = self.get_own_command(task.meta.name)
            if own_command:
                # 优先用真实的 command, 包括魔法 command 来不足.
                task.set_command(own_command)
            elif task.is_magical():
                task = self.partial_bare_magical_task(task)
            else:
                task.fail(CommandErrorCode.NOT_FOUND.error(f'Command {task.caller_name()} not found'))
                return
        if task.done():
            return
        # 最终入队. 保证入队后有序消费.
        self._enqueue_task_with_paths(paths, task)

    @abstractmethod
    def _enqueue_task_with_paths(self, paths: ChannelPaths, task: CommandTask) -> None:
        """
        将相对路径的 task 入队.
        不应直接调用.
        """
        pass

    def partial_bare_magical_task(self, task: CommandTask) -> CommandTask:
        """
        基于约定的隐藏协议实现魔法命令的隐藏定义.

        通常和 System Prompt 配合, 不走 Command 体系. 保留复杂逻辑 hack 的可能性.
        """
        if not task.is_bare_task():
            return task
        command_name = task.meta.name
        # 使用约定的魔法函数.
        if command_name == self.__content__.__name__:
            task.func = self.__content__
        # 默认魔法函数继续传递.
        return task

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
        提示底层逻辑如何传入 ChannelCtx.
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
        不会执行, 需要执行可以用 execute task | execute command
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

    def execute_command(
            self,
            name: CommandUniqueName,
            *,
            args: tuple | None = None,
            kwargs: dict | None = None,
            timeout: float | None = None,
    ) -> Awaitable:
        """
        执行命令并且阻塞等待拿到结果. 通常用于调试.
        正确的路径是走 push task 做阻塞.
        """
        task = self.create_command_task(name, args=args, kwargs=kwargs)
        if timeout is not None:
            task.timeout = timeout
        self.push_task(task)
        return task

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

    def channel_path(self) -> ChannelFullPath | None:
        """
        return the channel path in the tree, or None means not registered yet (which is an unnormal issue)
        """
        return self.tree.get_channel_path(self.channel.id())

    # --- default magic methods, 为后来的胶水层图灵完备语法做准备 --- #

    @staticmethod
    async def __content__(chunks__=None) -> None | str:
        # 所有的 ChannelRuntime 均允许时序插入多端文本的 Command, 作为流式输入的基准函数.
        # 当 __content__ 魔法 Command 不存在时, ChannelRuntime 会用空函数兜底. 这里是空函数的标准形式.
        # 定义在 Channel Runtime 上提示这是系统级约定.
        return None

    async def __scope_enter__(
            self,
            *,
            timeout: float | None = None,
            until: ChannelScopeType = 'flow',
    ) -> None:
        """
        scope enter command
        """
        # 当 scope enter task 执行的时候, 正式开始为 Scope 计时.
        # scope 中所有 task 的交织关系是在 "编译器" 确定的, 正式运行则在 command 执行时计算.
        task = ChannelCtx.task()
        if task is None:
            return
        if not task.scope_id:
            return
        scope = self.get_active_scope(task.scope_id, False)
        if scope is not None:
            await scope.tick(timeout=timeout, until=until)
        return

    async def __scope_exit__(self) -> str | None:
        task = ChannelCtx.task()
        if task is None:
            return None
        if not task.scope_id:
            return None
        scope = self.get_active_scope(task.scope_id, True)
        if scope is not None:
            try:
                return await scope.wait_close()
            finally:
                if not scope.is_closed():
                    scope.close()
        return None


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

    @abstractmethod
    def get_channel_path(self, channel_id: str) -> ChannelFullPath | None:
        """从全局中重新定位当前 channel 的绝对路径."""
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

    @abstractmethod
    async def arun_until_closed(self, channel: Channel | ChannelRuntime) -> None:
        """
        展示如何在 async 中持续运行到结束.
        """
        pass

    def run_in_thread(self, channel: Channel) -> threading.Thread:
        """
        展示如何在多线程中异步运行, 非阻塞.
        """
        thread = threading.Thread(target=self.run_until_closed, args=(channel,), daemon=True)
        thread.start()
        return thread

    def on_proxy_event(self, callback: Callable[[Any], None]):
        """接受 proxy 发送事件的回调"""
        pass

    def on_error(self, callback: Callable[[Exception], None]):
        pass

    @abstractmethod
    def close(self) -> None:
        """
        关闭当前 Server.
        """
        pass

    @abstractmethod
    def arun(self, channel: Channel) -> contextlib.AbstractAsyncContextManager[Self]:
        """
        支持 async with statement 的运行方式启动一个 channel.
        """
        pass

    @abstractmethod
    def arun_channel_runtime(
            self,
            runtime: ChannelRuntime,
    ) -> contextlib.AbstractAsyncContextManager[Self]:
        pass

# MOSS 架构的核心思想是 "面向模型的高级编程语言", 目的是定义一个类似 python 语法的编程语言给模型.
#
# 所以 Channel 可以理解为 python 中的 'module', 可以树形嵌套, 每个 channel 可以管理一批函数 (command).
#
# 同时在 "时间是第一公民" 的思想下, Channel 需要同时定义 "并行" 和 "阻塞" 的分发机制.
# 神经信号 (command call) 在运行时中的流向是从 父channel 流向 子channel.
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
