import contextlib
from typing import Callable, Any, ClassVar, TYPE_CHECKING

from typing_extensions import Self
from abc import ABC, abstractmethod

from ghoshell_moss.core.concepts.channel import Channel, ChannelProvider
from ghoshell_moss.core.blueprint.cell import Cell

if TYPE_CHECKING:
    from .host import MossRuntime

__all__ = ['FractalHub', 'FractalCellProvider']


class FractalHub(ABC):
    """
    Moss 分享架构的核心.

    它可以接受其它 Moss Runtime 的接入,
    通过 Channel 形式提供给当前的 moss 运行时, 做分形能力的注册.
    未来希望能扩展到资源体系, Ghost / Agent 通讯等.

    当 MOSS 的环境中注册了 FractalHub 的实现时 (Matrix.container.bound(FractalHub) is True), 会自动集成.
    如果没有注册这个实现, 则不会启动 FractalHub.

    FractalHub 解决什么问题呢?
    MossHost 实现基于环境发现构建 HostRuntime 的能力
    """
    DEFAULT_HUB_NAME: ClassVar['str'] = 'fractal'

    @abstractmethod
    def self_explain(self) -> str:
        """
        通讯协议和其它讯息的自解释.

        预计通过 repl 等手段做显式交互, 或提供给 moss 的 meta ghost 去解释.
        """
        # 为何用自然语言解释, 而不是用强类型呢?
        # 因为未来的开发希望直接面向模型。
        # 仍然可以拥有强类型描述：
        #    1. 想要拥有强类型, 可以通过具体实现扩展函数.
        #    2. matrix.contracts 可以发现绑定
        pass

    @abstractmethod
    def get_connected(self) -> list[Cell]:
        """
        返回已经联通到当前 Hub 的节点.
        """
        pass

    @abstractmethod
    def is_running(self) -> bool:
        """是否已经运行中."""
        pass

    @abstractmethod
    def status(self) -> str:
        """
        默认的运行时描述逻辑.
        """
        pass

    @abstractmethod
    def as_channel(
            self,
            description: str = '',
            allow_all: bool = False,
            auto_start: bool = False,
    ) -> Channel:
        """
        将 Hub 定义为一个 Channel.
        可以注册到 MossRuntime.shell
        """
        pass

    def accept(self, cell_name: str):
        """
        允许 cell 进入 hub. cell 状态 alive 为 true.
        raise KeyError: 如果 cell 不存在.
        """
        pass

    def ignore(self, cell_name: str):
        """
        忽略某个连接的 cell. cell 状态 alive 为 false
        raise KeyError: 如果 cell 不存在.
        """
        pass

    @abstractmethod
    async def __aenter__(self) -> Self:
        """需要提供生命周期治理"""
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """需要提供生命周期治理"""
        pass


class FractalCellProvider(ABC):
    """
    可以将当前 MossRuntime 提供给其它 MossRuntime 的 provider.

    是一种特殊的协议.
    """

    # 为何叫 Provider 而不是 Client 或 Server?
    # 因为双工通讯的协议可能同时有 正向/反向/桥接 等各种. 单一方向不能涵盖语义.

    @abstractmethod
    def channel_provider(
            self,
            as_cell_name: str = '',
    ) -> ChannelProvider | None:
        """
        最基础的 channel provider 实现.
        """
        pass

    @abstractmethod
    def self_explain(self) -> str:
        """
        解释自身的通讯协议. 通常包含:
        1. 什么协议.
        2. 当前连接了什么路径.
        3. 从什么配置文件获得的讯息.
        4. 默认将自己用什么名字提供.
        """
        pass

    def __repr__(self):
        return f"<FractalCellProvider>\n{self.self_explain()}\n</FractalCellProvider>"

    @abstractmethod
    async def __aenter__(self) -> Self:
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    @abstractmethod
    def is_running(self) -> bool:
        pass

    async def provide(
            self,
            moss: 'MossRuntime',
            *,
            as_cell_name: str | None = None,
            on_channel_event: Callable[[Any], None] | None = None,
    ) -> None:
        """
        将当前的 moss provide 给另一个扮演 hub 角色的 moss.

        :param moss: 持有 moss runtime.
        :param as_cell_name: 可以改写自己提供给父节点时, 自身的名称. 否则使用默认名称.
        :param on_channel_event: 可以注册 channel 双工协议的事件回调. 类型与协议有关.
        """
        # 基于 code as prompt 原则将抽象使用的最小实现直接在代码里提示.
        # provide 的设计目标包含: resources / ghosts / agents / channels 等等.
        # 不过现阶段, 仅仅有条件实现 channels.
        # 为何叫做 Provider?
        # 1. 通信协议无关, 所以可能有 client -> server, server -> client, cell -> broker <- hub 等多种机制.
        # 2. 不一定有连接顺序. 比如 hub  和  cell 可能互相先后启动, 通过服务发现机制探测到.
        stack = contextlib.AsyncExitStack()
        async with stack:
            # 防蠢.
            if not self.is_running():
                await stack.enter_async_context(self)
            channel_provider = self.channel_provider(
                # 默认的名称.
                as_cell_name or f'{moss.name}_{moss.mode.name}',
            )
            if channel_provider:
                if on_channel_event:
                    channel_provider.on_proxy_event(on_channel_event)

                # 定义闭包启动.
                async def provide_moss_channels():
                    nonlocal moss, channel_provider
                    await channel_provider.arun_until_closed(
                        # main channel 的 description 由 mode 决定.
                        moss.shell.runtime if moss.shell.is_running() else moss.shell.main_channel
                    )

                # 通过 matrix 来托管 provider 的生命周期, 避免没有优雅退出.
                await moss.matrix.create_task(provide_moss_channels())
