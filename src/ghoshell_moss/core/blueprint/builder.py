# # Blueprint
# about how to build channel for MOSShell.
# the path of this module is ghoshell_moss.core.blueprint.builder

from abc import ABC, abstractmethod
from PIL import Image
from typing import Union, Callable, Coroutine, Any, Optional, TypeVar, AsyncIterable
from typing_extensions import Self
from ghoshell_moss.message import Message
from ghoshell_moss.core.concepts.command import Command
from ghoshell_moss.core.concepts.channel import Channel
from ghoshell_common.contracts import LoggerItf
import asyncio

__all__ = [
    "CommandFunction", "MessageFunction", "StringType", "LifecycleFunction",
    "Builder",
    "MutableChannel",
    "new_channel"
]

CommandFunction = Union[Callable[..., Coroutine], Callable[..., Any]]
"""
用于描述一个本地的 python 函数 (或者类的 method) 可以被注册到 Channel 中变成一个 command. 
"""

MessageFunction = Union[
    Callable[[], Coroutine[None, None, list[Message | str | Image.Image]]],
    Callable[[], list[Message]],
]
"""
可以生成消息体的函数. 这种函数注册到 Channel 中, 可以用来动态地生成 Context Messages 与 Memory Messages.
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
- [on close] : channel 关闭时 
- [on running] : start < running < close

举一个典型的例子: 数字人在执行动画 command 时, 运行轨迹动画; 执行完毕后, 没有命令输入时, 需要返回呼吸效果 (on_idle) 
"""

_ChannelName = str

INSTANCE = TypeVar("INSTANCE", bound=object)


class CommandCtx:
    """
    use it in command to get runtime ctx
    """

    @classmethod
    def get_contract(cls, contract: type[INSTANCE]) -> INSTANCE:
        """
        get contract from ioc Container.
        but you must know what contract is registered.
        """
        from ghoshell_moss.core.concepts.channel import ChannelCtx
        return ChannelCtx.get_contract(contract)

    @classmethod
    def logger(cls) -> LoggerItf:
        """返回日志, 只保留基础的记录函数. """
        from ghoshell_moss.core.concepts.channel import ChannelCtx
        return ChannelCtx.get_contract(LoggerItf)


def new_command(
        func: CommandFunction,
        *,
        name: str = "",
        doc: Optional[StringType] = None,
        comments: Optional[StringType] = None,
        interface: Optional[StringType | Callable[[...], Coroutine[None, None, Any]]] = None,
        available: Optional[Callable[[], bool]] = None,
        # --- 高级参数 --- #
        blocking: bool = True,
        call_soon: bool = False,
        priority: int = 0,
) -> Command:
    """
    定义一个 Command. 逻辑与 Builder.command 相同.
    """
    from ghoshell_moss.core.concepts.command import PyCommand
    return PyCommand(
        func=func,
        name=name,
        doc=doc,
        comments=comments,
        interface=interface,
        available=available,
        blocking=blocking,
        call_soon=call_soon,
        priority=priority,
    )


# special kind of content function
async def __content__(chunks__) -> None:
    pass


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
    def instruction(self, func: StringType) -> StringType:
        """
        decorator
        注册字符串或者函数, 用来生成当前 channel 提供的 instruction / system prompt. 只生成一次.

        Channel as Context Components 思想:
            直接将 Channel 作为上下文的组件, 提供模块化的上下文讯息.
            讯息应该足够简洁, 高效, 同时注意 token 用量. 具体裁剪和压缩由 Agent 工程决定.
            由于 Channel 持有的 Command 可以影响自身的运行时状态, 所以 Channel 提供了完整的上下文反身性.
            结合后续的 StatefulChannel 实现, 同时提供渐进式披露的能力.

        由 Channel 提供的 AI 上下文拓扑:
        - instructions (System Prompt)
        - memory messages
        - current conversation messages
        - context messages
        - new inputs

        注意! Channel 仅在特别有必要的时候, 才需要提供上下文讯息. 大部分 channel 完全不用提供.
        """
        pass

    @abstractmethod
    def context_messages(self, func: MessageFunction, reset: bool = False) -> MessageFunction:
        """
        decorator
        注册一个上下文生成函数. 用来生成 channel 运行时动态的上下文.
        举个例子, 如果是视觉模块, 则可以把当前瞬间看见的图片, 和视觉模块的简单描述作为 context messages.

        这部分上下文会出现在模型上下文的 inputs 之前或之后.
        当 channel 每次刷新后, 都会通过它生成动态的上下文消息体.
        通常只有具备感知功能的模块, 需要提供动态的 context messages.

        >>> async def building(chan: MutableChannel) -> None:
        >>>     async def context() -> list[Message]:
        >>>         return [
        >>>             Message.new(role="system").with_content("dynamic information")
        >>>         ]
        >>>     chan.build.context_messages(context)
        """
        pass

    def content_command(
            self,
            func: Callable[[AsyncIterable[str]], Coroutine[None, None, None]],
            doc: Optional[str] = None,
            override: bool = True,
    ) -> Command[None]:
        """
        register a special function for channel's content method.
        """
        from ghoshell_moss.core.ctml.v1_0_0.constants import CONTENT_COMMAND_NAME
        name = CONTENT_COMMAND_NAME or '__content__'
        return self.command(
            name=name,
            doc=doc,
            # use __content__ as interface, override the docstring if need.
            interface=__content__,
            override=override,
            return_command=True,
        )(func)

    @abstractmethod
    def add_command(
            self,
            command: Command,
            *,
            override: bool = True,
            name: Optional[str] = None,
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
            override: bool = True,
            # --- 高级参数 --- #
            blocking: bool = True,
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
                         Comments 最直接的用处是写使用的案例, 说明, 执行逻辑等. 辅助 AI 理解.

        :param interface: 大模型看到的函数代码形式. 一旦定义了这个, doc, name, comments 就都会失效.
                支持三种传参方式:
                - str: 直接用字符串来定义模型看到的函数签名.
                    注意, 必须写成 Python Async 的形式.
                    async def foo(...) -> ...:
                      '''docstring'''
                      # comments
                - callalble[[], str]: 生成模型签名的函数
                - async function: 直接反射这个 function, 来生成一个模型签名的字符串. 可以定义虚拟函数作为 interface.
        :param override: override existing one
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
        >>> # 参数和返回值有明确的类型约束, 类型约束也是 prompt 的一部分.
        >>> # 使用可序列化对象作为入参和出参
        >>> # 依赖线程安全的逻辑, 定义为 sync 函数.
        >>> async def func(arg: type) -> Any:
        >>>     '''有清晰的说明'''
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
        >>>     # 可以获取执行这个 command 的真实 runtime
        >>>     try
        >>>         # 通过全局的 IoC 容器获取依赖, 可以拿到运行时的依赖注入.
        >>>         contract = CommandCtx.get_contract(...)
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
    def startup(self, func: LifecycleFunction) -> LifecycleFunction:
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
    def with_binding(self, contract: type[INSTANCE], instance: INSTANCE) -> Self:
        """
        注册一个依赖, 在 Channel 实例化时完成注入, 不会污染其它 channel. 可以通过 CommandCtx.get_contract 获取.
        依赖注入完全是可选的, 可以通过模块实例化/全局工厂等替代.
        """
        pass

    @abstractmethod
    def with_factory(
            self,
            contract: type[INSTANCE],
            factory: Callable[[...], INSTANCE],
            *,
            singleton: bool = True,
            override: bool = False,
    ) -> Self:
        """
        注册一个依赖的工厂方法. 这个工厂方法如果有入参, 会被 IoC 容器自动注入执行.
        """
        pass

    @abstractmethod
    def import_channels(self, *children: Channel | tuple[Channel, _ChannelName]) -> Self:
        """
        add sustain channels to the channel.
        """
        pass


class MutableChannel(Channel, ABC):
    """
    一个约定, 用来描述拥有动态构建能力的 Channel.
    """

    def import_channels(self, *children: Channel | tuple[Channel, _ChannelName]) -> Self:
        """
        添加子 Channel 到当前 Channel. 形成树状关系.
        效果可以比较 python 的 import module as name
        """
        self.build.import_channels(*children)
        return self

    @property
    @abstractmethod
    def build(self) -> Builder:
        """
        支持通过 Builder 动态构建一个 Channel.
        """
        pass

    @abstractmethod
    def children(self) -> dict[_ChannelName, Channel]:
        """
        return all the static imported channel
        """
        pass

    @abstractmethod
    def virtual_children(self) -> dict[_ChannelName, Channel]:
        """
        return the virtual children channels
        """
        pass


class ChannelInterfaceExample(ABC):
    """
    一个 Channel 开发的范式的例子.
    通过独立的抽象类, 定义了若干个函数, 而这些函数通过 build 注册了依赖关系.
    这样, 可以把设计一个 Channel, 与实现它分成两个明确的步骤. 设计本身是独立的.
    """

    @abstractmethod
    async def example_command(self) -> str:
        """
        docstring
        """
        pass

    @abstractmethod
    def as_channel(self, name: str, description: str) -> Channel:
        channel = new_channel(name=name, description=description)
        # 注册自身的 command.
        channel.build.command(interface=ChannelInterfaceExample.example_command)(self.example_command)
        return channel


def new_channel(name: str, description: str = "") -> MutableChannel:
    """
    Create a new Mutable/Stateful Channel object with builder.
    Able to define all kinds of channels.
    Use this tool to build your own channel object.
    """
    from ghoshell_moss.core.py_channel import PyChannel
    return PyChannel(name=name, description=description)
