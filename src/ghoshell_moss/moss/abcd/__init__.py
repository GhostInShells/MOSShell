from typing import Literal, Callable, Iterable
from typing_extensions import Self
from abc import ABC, abstractmethod

from .manifests import Manifests
from .matrix import Matrix
from .session import Session, ConversationItem
from .app import AppStore
from ghoshell_moss.core.concepts.shell import MOSShell
from ghoshell_moss.core.blueprint.states import PrimeChannel
from ghoshell_moss.message import Message
from ghoshell_container import IoCContainer
from pydantic import BaseModel, Field, AwareDatetime

RuntimeState = Literal['created', 'closed', 'idle', 'paused', 'looping', 'closing', 'startup']
'''
运行时的各种状态: 
created: 刚刚创建实例, 没有启动. 
startup: 启动过程中. 
idle: 没有输入也没有输出的闲置状态. 
looping: 在处理某个循环, 可能是对输入的响应, 或者执行某个命令. 
closing: 关闭中. 
closed: 已经关闭. 
'''


class MOSSToolSet(ABC):
    """
    将 MOSS runtime 包装成 tools, 希望可以被作为工具提供给别的框架.
    不过需要目标框架自行兼容输出协议.
    """

    @abstractmethod
    def moss_instruction(self) -> str:
        """
        返回所有的 instruction, 信息, 可以加入到 agent 的 instruction.
        """
        pass

    @abstractmethod
    def moss_dynamic_messages(self) -> list[Message]:
        """
        返回 moss 运行时的动态信息,
        包含组件的 interface, context messages 等等.
        不会返回最新的输入消息.
        """
        pass

    async def moss_exec(
            self,
            commands: str,
            call_soon: bool = True,
            observe: bool = True,
            with_dynamic: bool = True,
            priority: int = 0,
            on_ignore: Literal['buffer', 'drop'] = 'buffer',
    ) -> list[Message]:
        """
        向 MOSS 的运行时添加新的指令. 通常是 CTML.
        :param commands: 基于 ctml 语法提供的 command 字符串.
        :param call_soon: 如果为 True, 会立刻中断任何运行中的命令, 否则只是追加新的指令.
        :param observe: 为 True 的话, 阻塞到运行结束后, 拿到观察的结果. 包含命令的执行情况, 和新的输入. 为 False 的话会立刻返回.
        :param with_dynamic: 决定返回值里是否包含更新后的 moss dynamic 信息.
        :param priority: 注意力级别, 低于这个级别的输入事件不会中断行动.
        :param on_ignore: 被忽视的信息是否缓冲到上下文中.
        """
        pass

    @abstractmethod
    async def moss_observe(
            self,
            timeout: float | None = None,
            priority: int = 0,
            on_ignore: Literal['buffer', 'drop'] = 'buffer',
            with_dynamic: bool = True,
    ) -> list[Message]:
        """
        观察等待到 moss 运行状态变更.
        通常包含:
        1. 新的高优消息输入
        2. 当前有命令在执行, 并且已经执行完或发生了异常.
        3. 等待超时, 仍然返回最新的观察结果.

        :param timeout: 指定一个等待时间, 否则会持续等待到有任何事件为止.
        :param with_dynamic: 观察的结果里是否包含最新的 moss dynamic 信息.
        :param priority: 注意力级别, 低于这个级别的输入事件不会中断行动.
        :param on_ignore: 被忽视的信息是否缓冲到上下文中.
        """
        pass

    @abstractmethod
    async def moss_focus(
            self,
            priority: int = 0,
            on_ignore: Literal['buffer', 'drop'] = 'buffer',
            as_default: bool = False,
            timeout: float | None = None,
    ) -> str:
        """
        设置当前的注意力级别.
        :param priority: 设置优先级, 低于这个优先级的输入, 不会中断当前正在执行的任务.
        :param on_ignore: 决定低于优先级的输入如何处理, buffer 表示仍然保存到上下文; ignore 则彻底忽略.
        :param as_default: 是否作为默认的注意力状态.
        :param timeout: 如果设置了 timeout, 会在一定时间后回归默认的注意力状态.
        """
        pass

    @abstractmethod
    async def moss_interrupt(
            self,
    ) -> str:
        """
        立刻中断所有运行中的命令. 并且返回.
        """
        pass


class Snapshot(BaseModel):
    """
    当前运行状态的快照.
    """
    cursor: int = Field(
        description="当前快照的游标. 用于 ack. 每次获取 snapshot 都会得到一个新的快照, 没有 ack 的话不会清空其中的关键消息."
    )
    created_at: AwareDatetime = Field(
        description="当前快照的创建时间点. ",
    )
    runtime_state: RuntimeState = Field(
        description="运行时当前的状态",
    )
    focus_priority: int = Field(
        description="当前的注意力优先级",
    )
    ignore_method: Literal['buffer', 'drop'] = Field(
        description="当前的低优输入处理策略",
    )
    executed: list[Message] = Field(
        default_factory=list,
        description="最新运行逻辑中完成的部分, 和运行结果. "
    )
    status: list[Message] = Field(
        default_factory=list,
        description="当前的运行状态描述, 包含 state, executing, pending, focus level 等讯息. ",
    )
    moss_dynamic: list[Message] = Field(
        default_factory=list,
        description="运行时的动态信息, 包含组件的 interface 和 context messages 等. "
    )
    incomplete_inputs: dict[str, Message] = Field(
        default_factory=dict,
        description="拿到的输入消息, 不过没有完成, 是中间状态. 比如 asr 的分句. "
    )
    inputs: list[Message] = Field(
        default_factory=list,
        description="当前积累的输入"
    )

    def as_messages(self) -> Iterable[Message]:
        """
        生成一个消息集合, 通常是 Role == user 的一个消息总包.
        """
        yield from self.executed
        yield from self.status
        yield from self.moss_dynamic
        yield from self.incomplete_inputs.values()
        yield from self.inputs

    def as_conversation_item(self, **metadata) -> ConversationItem:
        return ConversationItem(
            role="user",
            metadata=metadata,
            messages=list(self.as_messages()),
        )


class IMossRuntime(ABC):
    """
    MOSS 架构的主运行时, 环境中的单例.
    """

    @property
    @abstractmethod
    def mode(self) -> str:
        """
        当前所处的模式.
        """
        pass

    @abstractmethod
    def as_toolset(self) -> MOSSToolSet:
        """
        提供作为工具的交互界面.
        本质上是对 MOSS Runtime 的封装.
        """
        pass

    @abstractmethod
    def is_running(self) -> bool:
        """
        是否在运行中.
        """
        pass

    @abstractmethod
    def snapshot(self, new: bool = False, ack: bool = False) -> Snapshot:
        """
        获取当前运行状态最新的关键帧.
        在没有 ack 的时候, 这个 snapshot 会停止更新.
        :param new: 如果 new 为 True, 则旧的 snapshot 会被废弃, 它无法被 ack.
        :param ack: 如果为 True, 则默认执行了 ack.
        """
        pass

    @abstractmethod
    def ack_snapshot(self, snapshot: Snapshot) -> bool:
        """
        snapshot 被实质地使用, 则通过 ack 通知它将被使用.
        产生的结果是其中的状态信息, 比如 inputs 等会被清除.
        """
        pass

    @abstractmethod
    def wait_closed_sync(self, timeout: float | None = None) -> bool:
        """
        同步阻塞.
        """
        pass

    @abstractmethod
    async def wait_closed(self) -> None:
        """
        异步阻塞到运行结束.
        """
        pass

    @abstractmethod
    def state(self) -> RuntimeState:
        """
        当前的运行状态.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        发送关闭信号, 中断 Runtime.
        """
        pass

    @abstractmethod
    def pause(self, toggle: bool = True) -> None:
        """
        pause the runtime immediately
        产生的效果: 停止所有运行中逻辑, 中断循环, clear & pause shell, 除非 unpause 否则不接受新命令.
        """
        pass

    @property
    @abstractmethod
    def container(self) -> IoCContainer:
        """
        运行时 ioc 容器.
        Runtime 相关所有单例都在里面.
        """
        pass

    def contracts(self) -> Iterable[type]:
        """
        返回 IoC 容器里绑定的所有对象.
        """
        return self.container.contracts(recursively=True)

    @property
    @abstractmethod
    def apps(self) -> AppStore:
        """
        管理 moss 架构下的 app 体系.
        可以启动/关闭 app.
        """
        pass

    @property
    @abstractmethod
    def shell(self) -> MOSShell:
        """
        全双工运行时.
        可以在它没启动时做一些操作.
        运行时可以直接通过它的 API 去控制 clear / pause 等操作.
        """
        pass

    @property
    def main_channel(self) -> PrimeChannel:
        """
        shell 的 main channel, 可以
        """
        return self.shell.main_channel

    @property
    @abstractmethod
    def matrix(self) -> Matrix:
        """
        MOSS 架构下, 多节点并行运行时的交互总线.
        """
        pass

    @property
    def session(self) -> Session:
        """
        runtime 当前所处的 Session.
        可以管理 input 和 output.

        这个函数缩短路径并声明它的存在.
        """
        return self.matrix.session

    def add_input(self, *messages: Message, priority: int = 0) -> None:
        """
        立刻添加新的输入到 Runtime 中.
        这些输入会发送给 on_output, 同时判断是否中断正在运行的 loop, 并且新起一个消费 inputs 的 loop.
        如果不能中断的话, 则会被 buffer 或丢弃.
        """
        pass

    def output(self, *items: ConversationItem) -> None:
        """
        输出 output item. 由于这是 moss 的 output, 所以里面其实包含 input.
        """
        return self.matrix.session.output(*items)

    def on_output(self, callback: Callable[[ConversationItem], None]):
        """
        接受 output item 并考虑渲染.
        """
        pass

    @abstractmethod
    async def __aenter__(self) -> Self:
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class IMossMode(ABC):
    """
    指定的运行模式.
    用来管理 MOSS Runtime 的运行时可发现资源.
    不使用 Mode 仍然可以启动 MOSS.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        模式的名称.
        """
        pass

    @property
    def description(self) -> str:
        """
        模式的描述.
        """
        return self.docstring.split('\n')[0]

    @property
    @abstractmethod
    def docstring(self) -> str:
        """
        模式的详细描述.
        """
        pass

    @abstractmethod
    def manifests(self) -> Manifests:
        """
        模式所管理的各种资源.
        """
        pass


class IMoss(ABC):
    """
    MOSS (model-oriented operating system shell) 的高阶抽象.

    1. 它屏蔽了 shell/interpreter 等内核模块.
    2. 它
    """

    @property
    @abstractmethod
    def manifests(self) -> Manifests:
        """
        返回当前环境下发现的 Matrix 实例.
        可以直接用于开发一个节点.
        """
        pass

    @abstractmethod
    def list_modes(self) -> dict[str, IMossMode]:
        """
        当前环境中可用的运行时模式, 用于管理不同模式下的差异化资源.
        比如 mac 模式, 机器人模式, 就可以完全隔离开.
        """
        pass

    @abstractmethod
    def matrix(self) -> Matrix:
        """
        返回当前环境下发现的 Matrix 实例.
        可以直接用于开发一个节点.
        >>> async def main(moss: IMoss):
        >>>     async with moss.matrix():
        >>>         ...
        """
        pass

    @abstractmethod
    def run(
            self,
            *,
            mode: IMossMode | str = 'default',
            session_id: str = 'default',
    ) -> IMossRuntime:
        """
        获得 moss 的运行时单例 (会校验唯一的锁, 确保 runtime 全局唯一).

        :param mode: 指定运行时的模式, 而模式控制资源. 也可以传入一个确定的 MossMode 对象.
        :param session_id: 指定一个 session id, 用来隔离上下文相关的一切资源.
        """
        pass
