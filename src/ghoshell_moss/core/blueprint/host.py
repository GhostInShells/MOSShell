import contextlib
from typing import Protocol

from ghoshell_container import IoCContainer
from typing_extensions import Self
from abc import ABC, abstractmethod

from ghoshell_moss.core.blueprint.manifests import Manifests
from ghoshell_moss.core.concepts.channel import Channel, ChannelProvider
from .app import AppStore
from ghoshell_moss.core.blueprint.matrix import Matrix, Cell, Mode
from ghoshell_moss.core.blueprint.session import Session
from ghoshell_moss.core.concepts.shell import MOSShell
from ghoshell_moss.message import Message

__all__ = [
    'MossRuntime', 'MossHost', 'Mode', 'FractalHub', 'FractalNodeProvider',
]


class MossLifecycleContract(Protocol):
    """
    一种约定, 在抽象设计中继承了这个协议的类,
    表示在 Moss 启动时会检查它是否存在, 如果存在会和 moss runtime 一起启动.
    同时也可以通过 container.get 去检查它是否存在.
    """

    async def __aenter__(self) -> Self:
        pass

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class MossRuntime(ABC):
    """
    通过 MossHost 环境发现构建的 MOSS Runtime.

    仍然提供 Shell 的整体使用, 同时基于环境注册运行.
    底层实现可以做参考, 用来拆分最小化引用.
    需要目标框架自行兼容输出协议.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """返回整个环境自定义的名字"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """
        返回整个环境默认的自解释描述

        覆盖逻辑是创建时传参定义 > Host 环境定义描述.
        """
        pass

    @property
    def mode(self) -> Mode:
        """当前 runtime 所处的模式. """
        return self.matrix.mode

    @abstractmethod
    def moss_instruction(self, with_static: bool = True) -> str:
        """
        返回所有的 instruction, 信息, 可以加入到 agent 的 instruction.
        :param with_static: 包含 moss static messages.
        """
        pass

    @abstractmethod
    async def moss_dynamic_messages(self, refresh: bool = True, max_wait: float = 2.0) -> list[Message]:
        """
        返回 moss 运行时的动态信息,
        仅包含组件的 interface, context messages 等等.
        """
        pass

    @abstractmethod
    def moss_static_messages(self) -> str:
        """
        返回 moss 运行时的静态信息.
        """
        pass

    @abstractmethod
    async def moss_exec(
            self,
            logos: str,
            call_soon: bool = True,
            wait_done: bool = True,
    ) -> list[Message]:
        """
        向 MOSS 的运行时添加新的指令. 通常是 CTML.
        :param logos: 基于 ctml 语法提供的 command 字符串.
        :param call_soon: 如果为 True, 会立刻中断任何运行中的命令, 否则只是追加新的指令.
        :param wait_done: 为 True 的话, 阻塞到运行结束后, 拿到观察的结果.
        """
        pass

    @abstractmethod
    async def moss_observe(
            self,
            timeout: float | None = None,
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
        """
        pass

    @abstractmethod
    async def moss_interrupt(self) -> list[Message]:
        """
        立刻中断所有运行中的命令. 并且返回中断的情况.
        """
        pass

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
    def container(self) -> IoCContainer:
        return self.matrix.container

    @property
    def session(self) -> Session:
        return self.matrix.session

    def get_fractal_hub(self) -> 'FractalHub | None':
        # default convention to get fractal hub
        return self.matrix.container.get(FractalHub)

    @property
    @abstractmethod
    def matrix(self) -> Matrix:
        """
        环境通讯的总线.
        """
        pass

    def wait_close_sync(self, timeout: float | None = None) -> bool:
        """阻塞等待关闭讯号. """
        pass

    async def wait_close(self) -> None:
        """异步阻塞等带关闭讯号."""
        pass

    def close(self) -> None:
        """发送关闭信号."""
        pass

    @abstractmethod
    async def __aenter__(self) -> Self:
        """正式启动"""
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """运行结束"""
        pass


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

    @property
    def name(self) -> str:
        """
        hub 自身的命名, 也是 as_channel 返回 channel 的默认前缀.

        需要符合 ChannelNamePattern
        对于拥有 FractalHub 的 Ghost 而言, 能通过 Channel 体系看到:
        `fractal_hub.{子节点名称}.[子节点 channels | 子节点 apps | 子节点的 fractal_hub]`
        """
        return "fractal_hub"

    @abstractmethod
    def usage(self) -> str:
        """
        通讯协议和其它讯息的自解释.

        预计通过 repl 等手段做显式交互, 或提供给 moss 的 meta ghost 去解释.
        为何用自然语言解释, 而不是用强类型呢?
        1. 强类型, 可以通过具体实现扩展函数.
        2. matrix.contracts 可以发现绑定.
        """
        pass

    @abstractmethod
    def get_connected(self) -> list[Cell]:
        """
        返回已经联通到当前 Hub 的节点.
        """
        pass

    @abstractmethod
    def is_running(self) -> bool:
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
            name: str = '',
            description: str = '',
    ) -> Channel:
        """
        将 Hub 定义为一个 Channel.
        可以注册到 MossRuntime.shell
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


class FractalNodeProvider(ABC):
    """
    可以将当前 MossRuntime 提供给其它 MossRuntime 的 provider.

    是一种特殊的协议.
    """

    # 为何叫 Provider 而不是 Client 或 Server?
    # 因为双工通讯的协议可能同时有 正向/反向/桥接 等各种. 单一方向不能涵盖语义.

    @abstractmethod
    def channel_provider(self, name: str) -> ChannelProvider | None:
        """最基础的 channel provider 实现. """
        pass

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
            moss: MossRuntime,
            name: str = '',
    ) -> None:
        """将当前的 moss provide 给另一个 moss. """
        # 基于 code as prompt 原则将抽象使用的最小实现直接在代码里提示.
        # provide 的设计目标包含: resources / ghosts / agents / channels 等等.
        # 不过现阶段, 仅仅有条件实现 channels.
        stack = contextlib.AsyncExitStack()
        async with stack:
            # 防蠢.
            if not self.is_running():
                await stack.enter_async_context(self)
            channel_provider = self.channel_provider(name or moss.name)
            if channel_provider:
                if moss.shell.is_running():
                    raise RuntimeError(f"Moss Shell is already running and occupied channels.")

                # 定义闭包启动.
                async def provide_moss_channels():
                    nonlocal moss, channel_provider
                    await channel_provider.arun_until_closed(
                        # main channel 的 description 由 mode 决定.
                        moss.shell.main_channel
                    )

                # 通过 matrix 来托管 provider 的生命周期, 避免没有优雅退出.
                await moss.matrix.create_task(provide_moss_channels())


class MossHost(ABC):
    """
    MOSS (model-oriented operating system shell) 基于环境发现的高阶抽象.

    如果不需要环境发现, 可以直接使用 ghoshell_moss.core.ctml.new_ctml_shell 来实例化 MOSShell.
    Host 用来管理和发现环境, 从环境中创建 Moss 的一切.

    1. 它屏蔽了 shell/interpreter 等内核模块.
    2. 它管理 Shell 的环境发现与运行. 核心目标包含
      - 基于约定发现能力
      - 屏蔽生命周期注册逻辑
      - 屏蔽底层抽象, 只暴露实体.
      - 将 Shell 的高阶封装作为 MossRuntime 提供.
    3. 它通过 Matrix 解决并行思考网络内的通讯体系.
    4. 支持 MossHost 的分形架构.
    5. 它缝合 Ghost 和 Shell. 作为一个独立的认知实体架构.
      - 支持 Ghost in MosShell 的实现, 不与 Shell 直接耦合.
      - 通过分形嵌套, 支持 Ghost In Shells 理念.

    架构拓扑的设计, 延续自 2019~2020 年的实现.
    https://github.com/thirdgerb/chatbot/blob/dba62e1337559c327d27ec4300366cd890a18ebc/src/Host/IHost.php#L4
    """

    @abstractmethod
    def name(self) -> str:
        """返回整个环境自定义的名字"""
        pass

    @abstractmethod
    def description(self) -> str:
        """返回整个环境默认的自解释"""
        pass

    @classmethod
    def discover(cls) -> Self:
        """
        环境发现的标准实现.
        """
        from ghoshell_moss.host import Host
        return Host.discover()

    @property
    @abstractmethod
    def manifests(self) -> Manifests:
        """
        返回当前环境下发现的 Matrix 实例.
        可以直接用于开发一个节点.
        """
        pass

    @property
    @abstractmethod
    def mode(self) -> Mode:
        """
        current mode.
        """
        pass

    @abstractmethod
    def all_modes(self) -> dict[str, Mode]:
        """
        当前环境中可用的运行时模式, 用于管理不同模式下的差异化资源.
        比如 mac 模式, 机器人模式, 就可以完全隔离开.
        """
        pass

    @abstractmethod
    def apps(self) -> AppStore:
        """
        环境发现的 App 中心
        用来管理环境中的 App (模式相关)
        """
        pass

    @abstractmethod
    def matrix(self) -> Matrix:
        """
        返回当前环境下发现的 Matrix 实例.
        可以直接用于开发一个节点.
        >>> async def main(moss: MossHost):
        >>>     async with moss.matrix():
        >>>         ...
        """
        pass

    @abstractmethod
    def run(
            self,
            *,
            run_shell: bool = True,
            with_primitives: bool = True,
            name: str | None = None,
            description: str | None = None,
    ) -> MossRuntime:
        """
        bootstrap moss runtime.
        flags:
        :param run_shell: if true, start shell when runtime aenter.
        :param with_primitives: if true, register manifests primitives into the shell
        :param name: set the moss name else use meta config
        :param description: set the moss description else use meta config
        """
        pass

    async def provide_moss_as_fractal(
            self,
            provider: FractalNodeProvider | None,
            name: str | None = None,
            description: str | None = None,
    ) -> None:
        """
        将当前的 moss runtime 作为一个分形节点提供给远程 moss.
        """
        # 协议需要业务逻辑自行判断.
        # 如果分形逻辑开箱自带的话, 两个以上的节点互联必然造成身份混乱.
        # 获取当前 runtime.
        runtime = self.run(
            # shell 本身就是主节点 channel runtime 的自解释封装.
            # 所以仍然可以运行 shell, 它不提供 logos 控制根 channel 即可.
            run_shell=True,
            # 原语只有主轨执行有意义, 所以这个 flag 设置为 false.
            with_primitives=False,
            name=name,
            description=description,
        )
        # 这里的实现作为 code as prompt 的一部分, 解释自身的使用逻辑.
        # 复杂的实现应该在理解运行逻辑基础上, 自行定义.
        async with runtime:
            # 启动 runtime, 但不注册原语.
            # 这样环境发现的能力都启动了, 但是屏蔽到原语级别功能.
            if provider is None:
                provider = runtime.matrix.container.get(FractalNodeProvider)
                if provider is None:
                    raise NotImplementedError(f"Fractal Provider is not implemented in this mode")
            async with provider:
                await provider.provide(
                    runtime,
                )
