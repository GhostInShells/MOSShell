"""
Channel (经络, "meridian"): an abstraction for components organized by a
streaming interpreter — tree-shaped, stateful, and stream-controllable.
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
    Meta data for a Channel.
    Can be used to mock a channel.
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

    # 温数据.
    notice: str = Field(default="",
                        description="Warm data — what this channel currently exposes. Rendered with command interfaces.")
    named_notices: dict[str, str] = Field(
        default_factory=dict,
        description="warm data with name, render in notice but per name rerender if diffed",
    )

    # memory 目前没有实装.
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
Addressing scheme for a concrete channel within the nested tree structure.
Fully aligned with Python's `a.b.c` addressing logic.

It also describes the path a nerve signal (command call) travels, e.g. executed from a -> b -> c.
"""

ChannelId = str
"""A channel instance is identified by a unique id."""

ChannelPaths = list[str]
"""Array representation of a string path. a.b.c -> ['a', 'b', 'c']"""

ChannelRuntimeContextVar = contextvars.ContextVar("moss.ctx.Runtime")

ChannelNamePattern = r'^[a-zA-Z_][a-zA-Z0-9_]*$'
ChannelName = Annotated[str, Field(pattern=ChannelNamePattern)]


class ChannelCtx:
    """
    Module usable by a Command or Lifecycle Function during a Channel's run.
    Passes the relevant context through a contextvars Context.
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
        Inject the given Runtime and CommandTask into a function's context.
        """
        with self.in_ctx():
            return await fn(*args, **kwargs)

    @classmethod
    def channel(cls) -> "Channel":
        """
        Return the Channel that is calling this function.
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
        Return the Runtime calling this function — a form of metaprogramming.
        Do not use this lightly unless you understand it.
        """
        try:
            return ChannelRuntimeContextVar.get()
        except LookupError:
            return None

    @classmethod
    def task(cls) -> CommandTask | None:
        """
        Return the CommandTask object that triggered a Command's run.
        """
        try:
            return CommandTaskContextVar.get()
        except LookupError:
            return None

    @classmethod
    def container(cls) -> IoCContainer:
        """
        Return the IoC container of the current runtime.
        """
        runtime = cls.runtime()
        if runtime:
            return runtime.container
        return get_container()

    @classmethod
    def get_contract(cls, contract: type[INSTANCE]) -> INSTANCE:
        """
        Get an implementation from the IoC container.
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
    Runtime state of a Channel, used to quickly build a StateChannel.

    To use the IoC container during runtime, either capture and hold dependencies
    when bootstrap is called, or fetch them per call via
    channel_builder.CommandUtil.get_contract in each command and lifecycle function.
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

    async def get_notice(self) -> str:
        """
        return warm notice text describing what this state currently exposes.

        distinct from ``get_context_messages`` (hot, per-frame) — notice is
        rendered with the command interface, only refreshed on change.
        """
        return ''

    async def get_named_notices(self) -> dict[str, str]:
        return {}

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
        Register something into the container, or get some contracts from it.
        Called after the ChannelRuntime is materialized.
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
    A Channel is the analog of a Python Module: a unit of capability exposed to the model.
    It exposes capabilities spanning processes (mainly functions) to the AI model
    through duplex communication.

    A Channel instance itself should be side-effect free; only after bootstrap at
    runtime does it return an instance with effects. Its uniqueness is determined by
    channel.id, not by the instance itself.
    """
    MAIN_CHANNEL_NAME = '__main__'
    PATH_SEPARATOR = '.'

    @abstractmethod
    def name(self) -> ChannelName:
        """
        The channel name, similar to Python's Module.__name__.
        There should be exactly one main Channel globally; it may be __main__.
        """
        pass

    @abstractmethod
    def id(self) -> str:
        """
        A Channel instance uses id to determine uniqueness; it is bound to the Runtime.
        """
        pass

    def __eq__(self, other):
        return self.id() == other.id() and self.name() == other.name()

    @abstractmethod
    def description(self) -> str:
        """
        Description of the Channel. For an AI model to understand a Channel,
        it needs to see each Channel's description.
        """
        pass

    @classmethod
    def join_channel_path(cls, parent: ChannelFullPath, *names: str) -> ChannelFullPath:
        """Standard syntax for joining parent/child channel names. A global convention."""
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
        Standard syntax for parsing a channel name path.
        """
        if not channel_path:
            return []
        return channel_path.split(cls.PATH_SEPARATOR, limit)

    def bootstrap(self, container: Optional[IoCContainer] = None) -> "ChannelRuntime":
        """
        Pass in an IoC container to create a Channel's Runtime instance.
        """
        if container is None:
            from ghoshell_container import Container
            container = Container(name="channel/{name}{id}".format(name=self.name(), id=self.id()))
        runtime_instance = self.materialize(container)
        if isinstance(runtime_instance, ChannelRuntime):
            return runtime_instance
        elif isinstance(runtime_instance, ChannelState):
            from ghoshell_moss.core.blueprint.states_channel import new_channel_from_state
            return new_channel_from_state(runtime_instance, id=self.id()).bootstrap(container)
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
    Channel scope syntax.
    Manages the lifecycle of all CommandTasks under the same scope group.
    """

    @property
    @abstractmethod
    def scope_id(self) -> str:
        """Unique id of the scope, usually task.cid."""
        pass

    @abstractmethod
    def add_task(self, task: CommandTask) -> CommandTask:
        """Bind a task to the scope. When the scope closes, all added tasks are closed too."""
        pass

    @abstractmethod
    def commit(self, task: CommandTask) -> CommandTask:
        """End the scope's registration. Tasks bound to this scope afterwards will fail."""
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
        """Start the scope's timing logic."""
        pass

    @abstractmethod
    async def wait_close(self) -> str | None:
        """Wait for the scope to end normally."""
        pass

    @abstractmethod
    def close(self, reason: str = '') -> None:
        """Actively close the scope."""
        pass


class ChannelRuntime(ABC):
    """
    The way to invoke a Channel's concrete capabilities.
    It is the materialization of a Channel.
    By design, a Channel is like the source code of a Python Module,
    while a ChannelRuntime is like the compiled ModuleType.

    Using the Runtime abstraction hides the Channel's concrete implementation,
    and can also be used to support remote invocation.

    >>> async def example(chan: Channel, con: IoCContainer):
    >>>     runtime = chan.bootstrap(con)
    >>>     async with runtime:
    >>>         ...

    Why not call it a Client? Because a Channel may run on both the Client and
    Server sides; they are made isomorphic through communication.
    """

    @property
    @abstractmethod
    def channel(self) -> "Channel":
        """
        The Runtime holds the Channel itself, like an instance holding its source.
        """
        pass

    @abstractmethod
    def sub_channels(self) -> dict[str, Channel]:
        """
        The child Channels currently held.
        """
        pass

    def virtual_sub_channels(self) -> dict[str, Channel]:
        """
        Manage the dynamic child nodes reachable from the current Channel runtime.
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
        Create an independent publisher that can broadcast a topic along the link.
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
        Publish a topic to the link; any listening channel or shell receives this event.
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
        Create a Subscriber to receive Topic broadcasts along the link.
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
        Provides logging, so users don't call logging.getLogger directly and make
        logs ungovernable.
        """
        pass

    @property
    @abstractmethod
    def container(self) -> IoCContainer:
        """
        Holds the IoC container for resolving complex call dependencies.
        """
        pass

    @property
    @abstractmethod
    def id(self) -> str:
        """
        Unique id of the runtime.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        The corresponding channel name.
        """
        pass

    def self_meta(self) -> ChannelMeta:
        """
        Get the current Channel's meta, used to reconstruct an identical Channel remotely.
        """
        return self.metas().get("")

    def own_metas(self) -> dict[ChannelFullPath, ChannelMeta]:
        """
        Return the meta held by the current ChannelRuntime. Usually only its own.
        For a Proxy-type Channel, however, it also proxies a whole Channel tree.
        """
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """
        Whether the Runtime's connection and communication are healthy.
        A running Runtime is not necessarily properly connected.
        For example, a Server-side ChannelRuntime may be started but not yet
        connected to the Provider-side ChannelRuntime.
        """
        pass

    @abstractmethod
    def is_running(self) -> bool:
        """
        Whether it has started. start < running < close.
        Used to manage the main lifecycle.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Whether the current Channel is available to the user (AI).
        Even when a Runtime is running & connected, it may be temporarily disabled
        for various reasons.
        """
        pass

    @abstractmethod
    def is_idle(self) -> bool:
        """
        Whether it has entered the idle state.
        """
        pass

    @abstractmethod
    async def wait_idle(self) -> None:
        """
        Block until idle.
        """
        pass

    @abstractmethod
    async def wait_connected(self) -> None:
        """
        Wait until the runtime is connected.
        """
        pass

    @abstractmethod
    async def wait_closed(self) -> None:
        """
        Wait until the Runtime is fully terminated.
        """
        pass

    @abstractmethod
    async def wait_started(self) -> None:
        """
        Block until started.
        """
        pass

    @abstractmethod
    def refresh_own_metas(self) -> asyncio.Future[None]:
        """
        Refresh its own meta.
        """
        pass

    @abstractmethod
    def own_commands(self, available_only: bool = True) -> dict[CommandUniqueName, Command]:
        """
        Return the current ChannelRuntime's own commands.
        The key is the command's unique name within this Runtime; the matching
        meta can be found in own_metas.
        """
        pass

    @abstractmethod
    def has_own_command(self, name: CommandUniqueName) -> bool:
        """
        Whether a command is held inside the current ChannelRuntime.
        """
        pass

    @abstractmethod
    def get_own_command(self, name: CommandUniqueName) -> Optional[Command]:
        """
        Get a command held by itself.
        """
        pass

    @abstractmethod
    async def clear_own(self) -> None:
        """
        Clear its own runtime state.
        """
        pass

    @abstractmethod
    def open_scope(
            self,
            task: CommandTask,
    ) -> None:
        """Open a scope for the given task."""
        pass

    @abstractmethod
    def get_active_scope(self, scope_id: str | None, pop: bool) -> ChannelScope | None:
        """Get a scope. If scope_id is None, returns the last one."""
        pass

    @abstractmethod
    def commit_scope(self, task: CommandTask) -> None:
        """End a scope's registration."""
        pass

    def push_task(self, *tasks: CommandTask) -> None:
        """
        Treat the current ChannelRuntime as the root node and push tasks onto the
        channel runtime's execution stack.
        The single entry point for the root node; includes the root node's own special logic.
        Side-effecting function; does not accept unordered tasks.
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
        Push a Task onto the execution stack.
        *** The only legal entry point for a ChannelRuntime to execute Tasks in order ***
        :param paths: task's relative path to the current ChannelRuntime; empty means the current ChannelRuntime.
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
        Enqueue a task with a relative path.
        Should not be called directly.
        """
        pass

    def partial_bare_magical_task(self, task: CommandTask) -> CommandTask:
        """
        Implement hidden definitions of magic commands via an agreed-upon hidden protocol.

        Usually works together with the System Prompt and bypasses the Command system;
        keeps the possibility of hacking in complex logic.
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
        Register a callback for when a Task finishes running.
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
        Simple way to execute a task in the runtime without queue logic.
        Shows how the low-level logic passes in a ChannelCtx.
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
        Example: create a channel task.
        Create a new CommandTask through the Runtime. It does not execute; to execute,
        use execute_task | execute_command.
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
        Execute a command and block until the result is available. Usually for debugging.
        The proper path is to block via push_task.
        """
        task = self.create_command_task(name, args=args, kwargs=kwargs)
        if timeout is not None:
            task.timeout = timeout
        self.push_task(task)
        return task

    @abstractmethod
    async def start(self) -> Self:
        """
        Start the Runtime.
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        Close the Runtime.
        """
        pass

    @abstractmethod
    def close_sync(self) -> None:
        """
        Synchronously close a Runtime.
        Only needed in special cases.
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
        Return all meta of the current module itself.
        The dict is ordered, in depth-first traversal order.
        """
        return self.tree.metas(self.channel)

    def fetch_sub_runtime(self, path: ChannelFullPath) -> Self | None:
        """
        Within the current Runtime's context space, find a possibly existing descendant node.
        """
        return self.tree.get_runtime_by_path(path, self.channel)

    def refresh_metas(
            self,
    ) -> asyncio.Future[None]:
        """
        Refresh the ChannelRuntime tree structure, then refresh the meta of tree
        nodes including itself.
        """
        return self.tree.refresh(self.channel.id(), wait=True)

    async def clear(self) -> None:
        """
        Clear all runtime state of the current Runtime.
        """
        await self.tree.clear(self)

    async def clear_children(self) -> None:
        """
        Clear the runtimes of all child channels of the current Runtime.
        """
        await self.tree.clear_children_runtimes(self.channel)

    def commands(self, available_only: bool = True) -> dict[ChannelFullPath, dict[str, Command]]:
        """
        List all commands.
        """
        # 递归逻辑统一通过 ChannelTree 实现. 保留 Runtime 接口
        return self.tree.commands(self.channel, available_only=available_only)

    def get_command(self, name: CommandUniqueName) -> Optional[Command]:
        """
        Get a command by its unique name.
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
    The tree that all ChannelRuntimes in one context should share.
    Prevents a Channel referenced by multiple Channels from materializing multiple Runtimes.
    Guarantees channel runtime uniqueness while managing parent/child relations.
    """

    @property
    @abstractmethod
    def main(self) -> ChannelRuntime:
        """
        The starting Channel of materialization, similar to main.py.
        """
        pass

    @abstractmethod
    def get_channel_runtime(self, channel: Channel, running: bool = False) -> ChannelRuntime | None:
        """
        Get a Channel Runtime that has already started.
        """
        pass

    async def wait_channel_children_idle(self, channel: Channel) -> None:
        """
        Wait until all child nodes of a node are idle.
        Returns immediately if the target node's runtime does not exist.
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
        Return the logger object.
        """
        pass

    @property
    @abstractmethod
    def topics(self) -> TopicService:
        """
        Holds the topic service shared by all channels.
        """
        pass

    @abstractmethod
    def is_running(self) -> bool:
        """
        Whether it has started.
        """
        pass

    @abstractmethod
    async def start(self) -> None:
        """
        Start.
        """
        pass

    def refresh_all(self) -> asyncio.Future[None]:
        return self.refresh(self.main.channel.id(), wait=True)

    @abstractmethod
    def refresh(self, id: ChannelId, wait: bool = False) -> asyncio.Future[None]:
        """
        Refresh the entire subtree for a channel id.
        Each channel runtime is refreshed only once at a time.
        """
        pass

    @abstractmethod
    def get_children_runtimes(self, channel: Channel) -> dict[str, "ChannelRuntime"]:
        """
        Get all activated child nodes of a node.
        """
        pass

    @abstractmethod
    def get_runtime_by_path(self, path: ChannelFullPath, root: Channel | None = None) -> ChannelRuntime | None:
        """
        Look up a runtime by path.
        """
        pass

    @abstractmethod
    def get_channel_path(self, channel_id: str) -> ChannelFullPath | None:
        """Relocate the current channel's absolute path from the global tree."""
        pass

    async def clear(self, runtime: ChannelRuntime) -> None:
        """
        Clear a runtime and all its child nodes.
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
        Clear all child nodes for a given channel.
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
        Return all running nodes, rooted at the root path.
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        pass

    @abstractmethod
    def commands(self, channel: Channel, available_only: bool = True) -> dict[ChannelFullPath, dict[str, Command]]:
        """
        Recursively get all sub-commands of a channel, grouped by path.
        """
        pass

    @abstractmethod
    def get_command(self, channel: Channel, name: CommandUniqueName) -> Command | None:
        """
        Recursively find a single command.
        """
        pass

    @abstractmethod
    def metas(self, root: Channel | None = None) -> dict[ChannelFullPath, ChannelMeta]:
        """
        Return the metas of all child nodes registered in the tree under a node.
        """
        pass


ChannelProxy = Channel
"""
A ChannelProxy is a special Channel that comes in a pair with a Channel Provider.
The Provider wraps a local Channel behind a communication protocol, and the
ChannelProxy reconstructs that Channel using the same protocol.
Example: ZmqChannelProvider.run(local_channel) => connection => ZmqChannelProxy;
to the model, the ChannelProxy here is identical to the local one.
"""


class ChannelProvider(ABC):
    """
    Run a Local Channel through a Provider and expose a communication protocol.
    A Proxy using the same protocol can reconstruct this Channel on a remote side.

    This forms a chained wrapping relation that reconstructs a tree-shaped
    architecture across processes. Provider and Proxy usually come in pairs.
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
        Wait until the provider finishes running.
        """
        pass

    @abstractmethod
    async def wait_stop(self) -> None:
        pass

    @abstractmethod
    def wait_closed_sync(self) -> None:
        """
        Synchronously wait until running finishes.
        """
        pass

    @abstractmethod
    async def aclose(self) -> None:
        """
        Actively close.
        """
        pass

    @abstractmethod
    def is_running(self) -> bool:
        """
        Whether this instance is running.
        """
        pass

    def run_until_closed(self, channel: Channel) -> None:
        """
        Example: run synchronously.
        """
        asyncio.run(self.arun_until_closed(channel))

    @abstractmethod
    async def arun_until_closed(self, channel: Channel | ChannelRuntime) -> None:
        """
        Example: run continuously until finished within async.
        """
        pass

    def run_in_thread(self, channel: Channel) -> threading.Thread:
        """
        Example: run asynchronously in a multithread, non-blocking.
        """
        thread = threading.Thread(target=self.run_until_closed, args=(channel,), daemon=True)
        thread.start()
        return thread

    def on_proxy_event(self, callback: Callable[[Any], None]):
        """Callback that receives events sent by the proxy."""
        pass

    def on_error(self, callback: Callable[[Exception], None]):
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Close the current Server.
        """
        pass

    @abstractmethod
    def arun(self, channel: Channel) -> contextlib.AbstractAsyncContextManager[Self]:
        """
        Start a channel via an async with statement.
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
