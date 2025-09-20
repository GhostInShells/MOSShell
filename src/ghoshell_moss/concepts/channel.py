from abc import ABC, abstractmethod
from typing import (
    Iterable, Optional, Union, Callable, Coroutine, List, Type, TypeVar, Dict, ClassVar, Any,
    AsyncIterable, Protocol, TYPE_CHECKING,
)
from typing_extensions import Self
from .command import Command, CommandMeta, CommandType
from .states import StateStore, State
from .topics import Topics
from ghoshell_container import IoCContainer, INSTANCE, Provider, BINDING
from pydantic import BaseModel, Field

FunctionCommand = Union[Callable[..., Coroutine], Callable[..., Any]]
"""通常要求是异步函数, 如果是同步函数的话, 会卸载到线程池运行"""

PolicyCommand = Union[Callable[..., Coroutine[None, None, None]], Callable[..., None]]

PrompterCommand = Union[Callable[..., Coroutine[None, None, str]], Callable[..., str]]

StringType = Union[str, Callable[..., str], Callable[..., Coroutine[None, None, str]]]

R = TypeVar('R')


class ChannelMeta(BaseModel):
    """
    Channel 的元信息数据.
    可以用来 mock 一个 channel.
    """
    name: str = Field(description="The name of the channel.")
    available: bool = Field(description="Whether the channel is available.")
    description: str = Field(description="The description of the channel.")
    stats: List[State] = Field(default_factory=list, description="The list of state objects.")
    commands: List[CommandMeta] = Field(default_factory=list, description="The list of commands.")
    children: List[str] = Field(default_factory=list, description="the children channel names")
    instruction: str = Field(default="", description="The instruction of the channel.")
    context: str = Field(default="", description="the runtime context of the channel.")


class ChannelRuntime(Protocol):
    """
    channel 的运行时方法. 只有在 channel.start 之后才可使用.
    """

    container: IoCContainer
    """
    运行时 IoC 容器.
    """

    states: StateStore
    """
    the states store
    """

    topics: Topics
    """
    运行时的 topics.
    """

    @abstractmethod
    def meta(self) -> ChannelMeta:
        """
        返回 Channel 自身的 Meta.
        可能是动态更新的.
        """
        pass

    @abstractmethod
    def available(self) -> bool:
        """
        当前 Channel Runtime 是否可用.
        """
        pass

    @abstractmethod
    def commands(self) -> Iterable[Command]:
        """
        返回所有 commands.
        """
        pass

    @abstractmethod
    def get_command(self, name: str) -> Optional[Command]:
        """
        查找一个 command.
        """
        pass

    @abstractmethod
    async def update(self) -> ChannelMeta:
        """
        结合所有上下文, 触发更新当前 Channel.
        """
        pass

    @abstractmethod
    async def set_available(self, available: bool):
        """
        修改当前 Channel 的 Available 状态.
        """
        pass

    @abstractmethod
    async def reset(self) -> None:
        """
        要求轨道重置到最初状态.
        """
        pass

    @abstractmethod
    async def run_policy(self) -> None:
        """
        回归 policy 运行. 通常在一个队列里没有 function 在运行中时, 会运行 policy. 同时 none-block 的函数也不会中断 policy 运行.
        """
        pass

    @abstractmethod
    async def start(self) -> None:
        """
        启动 Channel 运行.
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        关闭当前 Runtime. 同时阻塞销毁资源直到结束.
        """
        pass

    @abstractmethod
    async def __aenter__(self):
        await self.start()
        return self

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


class Channel(ABC):
    """
    Shell 可以使用的命令通道.
    """

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def description(self) -> str:
        pass

    @property
    @abstractmethod
    def runtime(self) -> ChannelRuntime:
        """
        Channel 在 bootstrap 之后返回的运行时.
        :raise RuntimeError: Channel 没有运行
        """
        pass

    # --- children --- #

    @abstractmethod
    def with_children(self, *children: "Channel") -> Self:
        """
        添加子 Channel 到当前 Channel. 形成树状关系.
        :raise KeyError: 如果出现重名会发出这个异常.
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
    def run(self, container: Optional[IoCContainer] = None) -> "ChannelRuntime":
        """
        传入一个父容器, 启动 Channel. 同时生成 Runtime.
        真正运行的是 channel runtime.
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        pass

    @abstractmethod
    def is_running(self) -> bool:
        """
        标注 Channel 已经开始运行了, 但没有终结.
        """
        pass

    @abstractmethod
    async def on_idle(self) -> None:
        pass

    @abstractmethod
    async def on_clear(self) -> None:
        pass


class ChannelDecorators(ABC):

    @abstractmethod
    def with_description(self, description: Callable[..., str]) -> Callable[..., str]:
        """
        注册一个全局唯一的函数, 用来动态生成 description.
        """
        pass

    @abstractmethod
    def with_command(
            self,
            command_type: CommandType,
            *,
            name: str = "",
            doc: Optional[StringType] = None,
            comments: Optional[StringType] = None,
            tags: Optional[List[str]] = None,
            interface: Optional[StringType] = None,
            # --- 高级参数 --- #
            block: bool = True,
            call_soon: bool = False,
    ) -> Callable[[FunctionCommand], FunctionCommand]:
        """
        返回 decorator 将一个函数注册到当前 Channel 里.
        对于 Channel 而言, Function 通常是会有运行时间的. 阻塞的命令, Channel 会一个一个执行.

        :param command_type: 命令的类型.
        :param name: 改写这个函数的名称.
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
        :param call_soon: 决定这个函数进入轨道后, 会第一时间执行 (不等待调度), 还是等待排队执行到自身时.
                          如果是 block + call_soon, 会先清空队列.
        """
        pass

    def with_function(
            self,
            *,
            name: str = "",
            doc: Optional[StringType] = None,
            interface: Optional[StringType] = None,
            comments: Optional[StringType] = None,
            tags: Optional[List[str]] = None,
            # --- 高级参数 --- #
            block: bool = True,
            call_soon: bool = False,
    ) -> Callable[[FunctionCommand], FunctionCommand]:
        return self.with_command(
            command_type=CommandType.FUNCTION,
            name=name,
            doc=doc,
            interface=interface,
            comments=comments,
            tags=tags,
            block=block,
            call_soon=call_soon,
        )

    def with_policy(
            self,
            name: str = "",
            doc: Optional[StringType] = None,
            comments: Optional[StringType] = None,
            tags: Optional[List[str]] = None,
            interface: Optional[StringType] = None,
    ) -> Callable[[PolicyCommand], PolicyCommand]:
        """
        返回 decorator 可以用来注册一个 policy 函数,
        """
        return self.with_command(
            command_type=CommandType.POLICY,
            name=name,
            doc=doc,
            interface=interface,
            comments=comments,
            tags=tags,
            block=True,
            call_soon=False,
        )

    @abstractmethod
    def on_run_policy(self, run_policy: Callable[[], Coroutine[None, None, None]]) -> None:
        """
        注册一个函数, 当 Channel 运行 policy 时, 会执行这个函数.
        """
        pass

    def with_prompter(
            self,
            *,
            name: str = "",
            doc: Optional[StringType] = None,
            comments: Optional[StringType] = None,
            tags: Optional[List[str]] = None,
            interface: Optional[StringType] = None,
    ) -> Callable[[PrompterCommand], PrompterCommand]:
        """
        返回 decorator, 注册一个可以生成 prompter 的函数, 通常只有 AI 在主动修改自己的 prompt 时 (元认知)
        才会使用这个函数.
        """
        return self.with_command(
            command_type=CommandType.PROMPTER,
            name=name,
            doc=doc,
            interface=interface,
            comments=comments,
            tags=tags,
            # 所有的 prompter 都应该是非阻塞函数.
            block=False,
            call_soon=False,
        )

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


if TYPE_CHECKING:
    async def prompter() -> str:
        return "hello"


    def with_prompter(func: PrompterCommand):
        pass


    with_prompter(prompter)
