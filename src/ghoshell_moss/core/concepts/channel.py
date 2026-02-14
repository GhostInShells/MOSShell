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
)

from ghoshell_container import BINDING, INSTANCE, IoCContainer, Provider, set_container
from pydantic import BaseModel, Field
from typing_extensions import Self

from ghoshell_moss.core.concepts.command import BaseCommandTask, Command, CommandMeta, CommandTask
from ghoshell_moss.core.concepts.states import StateModel, StateStore
from ghoshell_moss.message import Message

__all__ = [
    "Builder",
    "Channel",
    "MutableChannel",
    "ChannelBroker",
    "ChannelFullPath",
    "ChannelMeta",
    "ChannelPaths",
    "ChannelProvider",
    "ChannelUtils",
    "CommandFunction",
    "ContextMessageFunction",
    "LifecycleFunction",
    "PrompterFunction",
    "R",
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

ContextMessageFunction = Union[
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

R = TypeVar("R")


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


class ChannelBroker(ABC):
    """
    channel 运行后提供出来的通用 API.
    只有在 channel.bootstrap 之后才可使用.
    用于控制 channel 的所有能力.

    如果用 "面向模型的高级编程语言" 角度看,
    可以把 channel broker 理解成 python 的 ModuleType 对象.
    """

    @property
    @abstractmethod
    def container(self) -> IoCContainer:
        """
        broker 所持有的 ioc 容器.
        """
        pass

    @property
    @abstractmethod
    def id(self) -> str:
        """
        broker 的唯一 id.
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """
        对应的 channel name.
        """
        pass

    @abstractmethod
    def meta(self) -> ChannelMeta:
        """
        返回 Channel 自身的 Meta.
        """
        pass

    @abstractmethod
    async def refresh_meta(self) -> None:
        """
        更新当前的 Channel Meta 信息. 用于支持被动拉取. 不会主动推送更新.
        """
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """
        判断一个 Broker 的连接与通讯是否正常。
        一个运行中的 Broker 不一定是正确连接的.
        """
        # 对于非通讯类的 channel, 比如 py-channel, 直接返回 True.
        return True

    @abstractmethod
    async def wait_connected(self) -> None:
        """
        等待 broker 到连接成功.
        """
        pass

    @abstractmethod
    def is_running(self) -> bool:
        """
        是否已经启动了. 如果 Broker 被 close, is_running 为 false.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        当前 Channel 对于使用者而言, 是否可用.
        当一个 Broker 是 running & connected 状态下, 仍然可能会因为种种原因临时被禁用.
        """
        pass

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

    @abstractmethod
    async def on_idle(self) -> None:
        """
        进入闲时状态.
        闲时状态指当前 Broker 及其 子 Channel 都没有 CommandTask 在运行的时候.
        """
        pass

    @abstractmethod
    async def policy_pause(self) -> None:
        """
        接受到了新的命令, 要中断 policy
        不会递归执行.

        todo: policy pause 是一个错误的范式. 考虑 beta 版本移除.
        """
        pass

    @abstractmethod
    async def on_clear(self) -> None:
        """
        当轨道命令被触发清空时候执行.
        """
        pass

    async def on_disconnect(self) -> None:
        """
        todo: 将这个实现成正规的生命周期函数.
        """
        pass

    @abstractmethod
    async def start(self) -> None:
        """
        启动 Channel Broker.
        通常用 with statement 或 async exit stack 去启动.
        只会启动当前 channel 自身.
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        关闭当前 broker. 同时阻塞销毁资源直到结束.
        只会关闭当前 channel 的 broker.
        """
        pass

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    @property
    @abstractmethod
    def states(self) -> StateStore:
        """
        返回当前 Channel 的状态存储.

        todo: 现在的 state store 还是验证阶段.
        """
        pass


class Builder(ABC):
    """
    用来动态构建一个 Channel 的通用接口.
    目前主要用于 py channel.

    todo: decorator 风格没有统一, 同时有 with + decorator 两种语法习惯. 需要统一.
    """

    @abstractmethod
    def with_description(self) -> Callable[[StringType], StringType]:
        """
        注册一个全局唯一的函数, 用来动态生成 description.
        todo: with 开头的不要用 decorator 形式 . Deprecated: 不再用这种方式去变更, 让 description 不变.
        """
        pass

    @abstractmethod
    def with_available(self) -> Callable[[Callable[[], bool]], Callable[[], bool]]:
        """
        注册一个函数, 用来标记 Channel 是否是 available 状态.
        todo: with 开头的不要用 decorator 形式 .
        """
        pass

    @abstractmethod
    def state_model(self) -> Callable[[type[StateModel]], StateModel]:
        """
        注册一个状态模型.
        todo: 改成 with 开头的语法.
        """
        pass

    @abstractmethod
    def with_context_messages(self, func: ContextMessageFunction) -> Self:
        """
        注册一个上下文生成函数. 用来生成 channel 运行时动态的上下文.
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
    def on_policy_run(self, run_policy: LifecycleFunction) -> LifecycleFunction:
        """
        注册一个函数, 当 Channel 运行 policy 时, 会执行这个函数.
        """
        pass

    @abstractmethod
    def on_policy_pause(self, pause_policy: LifecycleFunction) -> LifecycleFunction:
        """
        policy 回调.
        todo: 考虑彻底移除.
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
        todo: 要统一考虑 channel 是否要用父子容器.
        """
        pass

    @abstractmethod
    def with_contracts(self, *contracts: type) -> Self:
        """
        声明 IoC 容器需要的依赖. 如果启动时传入的 IoC 容器没有注册这些依赖, 则启动本身会报错, 抛出异常.
        """
        pass

    @abstractmethod
    def with_binding(self, contract: type[INSTANCE], binding: Optional[BINDING] = None) -> Self:
        """
        register default bindings for the given contract.
        """
        pass


ChannelContextVar = contextvars.ContextVar("MOSShell_Channel")


class ChannelUtils:
    """
    提供 Channel 相关的一些工具函数.
    """

    @staticmethod
    def ctx_get_contract(contract: type[INSTANCE]) -> INSTANCE:
        """
        语法糖, 更快从上下文中获取
        """
        _chan = Channel.get_from_context()
        return _chan.get_contract(contract)


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

    def get_contract(self, contract: type[INSTANCE]) -> INSTANCE:
        """
        语法糖, 快速从 broker 里获取一个注册的实例.
        todo: 搬迁到 CommandTaskCtx 中. 然后禁止使用.
        """
        return self.broker.container.force_fetch(contract)

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

    def set_context_var(self) -> None:
        """
        与 get from context 配套使用, 可以在 Command 运行时拿到 Channel 本身.
        todo: 当 CommandTaskCtx 实现后, 彻底移除.
        """
        ChannelContextVar.set(self)

    @staticmethod
    def get_from_context() -> Optional["Channel"]:
        """
        在 Command 内部调用这个函数, 可以拿到运行它的 channel.
        todo: 考虑彻底移除. 这个范式过于耦合.
        """
        try:
            return ChannelContextVar.get()
        except LookupError:
            return None

    @property
    @abstractmethod
    def broker(self) -> ChannelBroker:
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
        for child_name, child in children.items():
            child_path = Channel.join_channel_path(prefix, child_name)
            descendants[child_path] = child
            for descendant_full_path, descendant in child.descendants(child_path).items():
                # join descendant name with parent name
                descendants[descendant_full_path] = descendant
        return descendants

    def all_channels(self) -> dict[str, "Channel"]:
        """
        语法糖, 返回所有的 channel, 包含自身.
        key 是以自身为起点的 channel path (相对路径), 用来发现原点.
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
    async def run_in_ctx(self, container: Optional[IoCContainer] = None) -> AsyncIterator["Channel"]:
        """
        语法糖, 启动当前 Channel 和它所有的子节点.
        通常仅仅用于单元测试. 也是为了提示如何单独测试一个 Channel.
        """

        async def recursive_start(_chan: Channel) -> None:
            await _chan.bootstrap(container).start()
            group_start = []
            for child in _chan.children().values():
                if not child.is_running():
                    group_start.append(recursive_start(child))
            await asyncio.gather(*group_start)

        async def recursive_close(_chan: Channel) -> None:
            children = _chan.children()
            if len(children) == 0:
                return
            group_stop = []
            for child in children.values():
                if not child.is_running():
                    group_stop.append(recursive_close(child))
            await asyncio.gather(*group_stop)
            if _chan.is_running():
                await _chan.broker.close()

        # 递归运行.
        await recursive_start(self)
        yield self
        await recursive_close(self)

    async def execute_task(self, task: CommandTask) -> Any:
        """
        运行一个 task 并且给它赋予当前 channel 到被运行函数的 context vars 中.
        todo: 彻底移除这个函数, 用 CommandTaskCtx 替代. 应该是 ChannelBroker 持有 Channel, 而不是相反.
        """
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
        set_container(self.broker.container)
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

    def create_command_task(self, name: str, *args: Any, **kwargs: Any) -> CommandTask:
        """example to create channel task"""
        command = self.broker.get_command(name)
        if command is None:
            raise NotImplementedError(f"Channel {self.name()} has no command {name}")
        task = BaseCommandTask.from_command(command, *args, **kwargs)
        return task

    async def execute_command(self, command: Command, *args, **kwargs) -> Any:
        """basic example to execute command."""
        from ghoshell_moss.core.concepts.command import BaseCommandTask

        task = BaseCommandTask.from_command(command, *args, **kwargs)
        try:
            result = await self.execute_task(task)
            task.resolve(result)
            return result
        finally:
            if not task.done():
                task.cancel("task is executed but not done")


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


_ChannelBrokerCtx = contextvars.ContextVar("MOSSCommandTaskCtx.Broker")
_CommandTaskCtx = contextvars.ContextVar("MOSSCommandTaskCtx.Task")


class CommandTaskCTX:
    """
    Command Task 运行时可以从执行上下文 contextvars 里拿到的数据.
    """

    @classmethod
    def init(
            cls,
            broker: ChannelBroker,
            task: CommandTask,
    ) -> None:
        _ChannelBrokerCtx.set(broker)
        _CommandTaskCtx.set(task)

    @classmethod
    def broker(cls) -> ChannelBroker:
        return _ChannelBrokerCtx.get()

    @classmethod
    def task(cls) -> CommandTask:
        return _CommandTaskCtx.get()

    @classmethod
    def container(cls) -> IoCContainer:
        broker = cls.broker()
        return broker.container

    @classmethod
    def get_contract(cls, contract: type[INSTANCE]) -> INSTANCE:
        broker = cls.broker()
        return broker.container.force_fetch(contract)


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
