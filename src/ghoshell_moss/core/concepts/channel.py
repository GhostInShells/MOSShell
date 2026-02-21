import asyncio
import contextlib
import contextvars
import threading
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import (
    Any,
    Optional,
    Protocol,
    Union,
    Callable,
    Coroutine,
)

from ghoshell_container import INSTANCE, IoCContainer
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
from ghoshell_moss.core.concepts.states import StateModel, StateStore, State
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

    @classmethod
    def new_empty(cls, id: str, channel: "Channel") -> Self:
        return cls(
            name=channel.name(),
            description=channel.description(),
            dynamic=True,
            channel_id=id,
            available=False,
        )


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
    def default_states(self) -> list[State]:
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
    def idle(self, func: LifecycleFunction) -> LifecycleFunction:
        """
        注册一个函数, 当 Channel 运行 policy 时, 会执行这个函数.
        """
        pass

    @abstractmethod
    def start_up(self, func: LifecycleFunction) -> LifecycleFunction:
        """
        启动时执行的回调.
        """
        pass

    @abstractmethod
    def close(self, func: LifecycleFunction) -> LifecycleFunction:
        """
        关闭时的回调.
        """
        pass

    @abstractmethod
    def running(self, func: LifecycleFunction) -> LifecycleFunction:
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
    async def on_idle(self):
        pass

    @abstractmethod
    async def on_start_up(self) -> None:
        pass

    @abstractmethod
    async def on_close(self) -> None:
        pass

    @abstractmethod
    async def on_running(self) -> None:
        pass

    @abstractmethod
    def update_container(self, container: IoCContainer) -> None:
        pass


ChannelRuntimeContextVar = contextvars.ContextVar("moss.ctx.Runtime")


class ChannelCtx:
    """
    提供 Channel 相关的一些工具函数.
    """

    def __init__(
        self,
        runtime: Optional["ChannelRuntime"] = None,
        task: Optional[CommandTask] = None,
    ):
        self._runtime = runtime
        self._task = task

    async def run(self, fn: Callable[..., Coroutine], *args, **kwargs) -> Any:
        async with self.in_ctx():
            return await fn(*args, **kwargs)

    @classmethod
    def channel(cls) -> "Channel":
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
        try:
            return ChannelRuntimeContextVar.get()
        except LookupError:
            return None

    @classmethod
    def task(cls) -> CommandTask | None:
        try:
            return CommandTaskContextVar.get()
        except LookupError:
            return None

    @classmethod
    def container(cls) -> IoCContainer:
        runtime = cls.runtime()
        return runtime.container

    @classmethod
    def get_contract(cls, contract: type[INSTANCE]) -> INSTANCE:
        runtime = cls.runtime()
        return runtime.container.force_fetch(contract)


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
    def id(self) -> str:
        """
        Channel 实例也只能用 id 来判断唯一性.
        """
        pass

    @abstractmethod
    def description(self) -> str:
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
        传入一个 IoC 容器, 获取 Channel 的 runtime 实例.
        """
        pass


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


ChannelInterface = dict[ChannelFullPath, ChannelMeta]
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
    >>> runtime = chan.bootstrap(con)
    >>> async with runtime:
    >>>     ...

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
        chan, command_name = Command.split_uniquename(name)
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
    async def arun(self, channel: Channel) -> None:
        """
        支持 async with statement 的运行方式启动一个 channel.
        """
        pass
