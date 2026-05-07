from typing import Literal, Callable, Awaitable, Any, Coroutine, Iterable

from typing_extensions import Self
from abc import ABC, abstractmethod
from ghoshell_moss.core.concepts.channel import Channel, ChannelProxy
from ghoshell_moss.core.blueprint.session import Session
from ghoshell_moss.contracts import LoggerItf, ConfigStore, Workspace, SystemPrompter
from ghoshell_container import IoCContainer
import asyncio

__all__ = ['Matrix', 'Cell', 'SystemPrompter', 'ScopesKey']

from ghoshell_moss.core.blueprint.manifests import Manifests


class Cell(ABC):
    """
    在 matrix 中可以并行独立运行的单元, 比如并行思考模块, channel provider 等等.
    不需要实现它, Matrix 的实现会包含 Cell 的定义.
    """
    name: str  # 节点的名称.
    description: str  # 节点的描述.
    type: Literal['app', 'main'] | str  # 节点的类型. main 表示 moss 的 runtime, 而 app 表示是一个环境中可加载的应用.
    where: str  # 这个节点自身的工作目录.

    @property
    @abstractmethod
    def address(self) -> str:
        """节点的地址. 通常作为节点的各种通讯机制的前缀或关键环节."""
        pass

    @property
    def log_name(self) -> str:
        return '.'.join(['moss', self.type, self.name.replace('/', '.')])

    @abstractmethod
    def is_alive(self) -> bool:
        """
        节点是否在运行中.
        """
        pass

    def to_dict(self) -> dict[str, Any]:
        return {
            "address": self.address,
            "name": self.name,
            "description": self.description,
            "type": self.type,
            "where": self.where,
            "log_name": self.log_name,
            "is_alive": self.is_alive(),
        }


CELL_ADDRESS = str

ScopesKey = Literal[
    'moss_mode',  # 对环境中所有资源的隔离形式, 通过不同的 mode 隔离不同的资源组合. 使得资源如 provider, config 等可以复用.
    'session_scope',  # 运行时隔离的基本维度, 使用不同的 scope 启动, 可以用来隔离通讯/存储等. 前提是对应组件使用了这个隔离级别.
    'session_id',  # 运行时的唯一 Id. 如果一些资源或状态希望在系统关闭时就丢弃, 可以基于 session_id 构建隔离级别来通讯或存储.
    'cell_address',  # Matrix 实例作为通讯架构, 运行在每个不同的 Cell 内. 同时可以有很多个 cell 并行运行组网.
]


class Matrix(ABC):
    """
    MOSS 架构下多节点组网后形成的通讯矩阵的客户端.
    持有矩阵的抽象可以通过矩阵通讯.
    本身应该是进程级别单例.

    Matrix 是用于构建可跨进程通讯的基本抽象.
    """

    @classmethod
    def discover(cls) -> Self:
        """
        约定的环境发现逻辑.
        这里使用了反范式, discover 包含了默认实现.
        所以基于 Matrix 默认实现创建应用, 只需要调用 Matrix.discover() 根据抽象提供的能力即可.
        """
        # moss 架构的默认实现.
        from ghoshell_moss.host import Host
        return Host.discover().matrix()

    @abstractmethod
    def cell_env(self) -> dict[str, str]:
        """
        Cell 自身相关的环境变量.
        """
        pass

    @property
    @abstractmethod
    def this(self) -> Cell:
        """
        返回当前节点自身的讯息. 节点之间通讯仅仅通过 topics / parameter / action 等.
        """
        pass

    @abstractmethod
    def moss_system_prompter(self) -> SystemPrompter:
        """
        moss 全局的 system prompter.
        matrix 必须完成全局 prompter 的定义, 并注册到 IoC 容器中.
        """
        pass

    @property
    @abstractmethod
    def moss_mode(self) -> str:
        """
        返回当前 MOSS 运行的模式.
        Matrix 运行时会
        """
        pass

    def scopes(self) -> dict[ScopesKey, str]:
        """返回 Matrix 运行时的维度座标. 用来构建不同的隔离级别. """
        return {
            'session_id': self.session.session_id,
            'session_scope': self.session.session_scope,
            'moss_mode': self.moss_mode,
            'cell_address': self.this.address,
        }

    @abstractmethod
    def ctml_version(self) -> str:
        """
        当前环境定义的 ctml version.
        """
        pass

    @abstractmethod
    def get_ctml_prompt(self, version: str | None = None) -> str:
        """
        返回环境中定义的系统提示词.
        """
        pass

    @abstractmethod
    def list_cells(self) -> dict[CELL_ADDRESS, Cell]:
        """
        返回环境里的所有节点, 以及这些节点是否在运行.
        """
        pass

    @property
    @abstractmethod
    def session(self) -> Session:
        """
        共享的 Session Store.
        """
        pass

    @property
    @abstractmethod
    def container(self) -> IoCContainer:
        """
        环境中共享的 IoC 容器. 只包含进程级别的服务.
        主要是 manifests 里提供的服务.
        """
        pass

    @property
    @abstractmethod
    def manifests(self) -> Manifests:
        """
        运行环境中各种能力的声明.
        """
        pass

    @property
    @abstractmethod
    def configs(self) -> ConfigStore:
        """
        基于环境发现的配置中心.
        """
        pass

    def show_configs(self) -> Iterable[dict[str, str]]:
        """
        不返回配置值的情况下, 返回配置的介绍.
        """
        store = self.configs
        for config_info in self.manifests.configs().values():
            info = {
                "name": config_info.name,
                "description": config_info.description,
                "file": config_info.file(store),
                "type": config_info.model_path,
            }
            yield info

    @abstractmethod
    def provide_channel(self, channel: Channel) -> asyncio.Future[None]:
        """
        将 Channel 通过当前节点提供到整个 Matrix 网络中,
        可以作为 Cell 的可操控单元, 被主进程的 Shell 调用.
        一个进程只能调用一个 provide channel, 可以提供树形的 channel.
        """
        pass

    @abstractmethod
    def channel_proxy(
            self,
            address: str,
            name: str,
            description: str = '',
            id: str | None = None,
            only_allowed_in_main_cell: bool = True,
    ) -> ChannelProxy:
        """
        搭建一个 proxy 获取另一个节点里通过 provider channel 提供的 channel. 进行跨网络同构.
        通常只允许 Matrix 里的 main cell 使用 proxy 连接 channel. 因为 channel 是 matrix 内唯一的.
        多个 proxy 连接会导致 channel 频繁地重启.
        仍然允许用这个方式进行测试.

        :param address: cell address which providing a channel tree
        :param name: channel name which rewrite the providing channel.
        :param description: channel description which rewrite the providing channel.
        :param id: channel uid if given, otherwise will generate a uuid for the proxy.
        :param only_allowed_in_main_cell: only allow main cell to use channel proxy.
        :raise RuntimeError: if the current cell is not the main cell of the matrix runtime.
        """
        pass

    @property
    @abstractmethod
    def logger(self) -> LoggerItf:
        """
        日志模块. 从属于当前节点.
        """
        pass

    @property
    @abstractmethod
    def workspace(self) -> Workspace:
        """
        workspace 管理.
        """
        pass

    @abstractmethod
    def is_running(self) -> bool:
        """
        matrix 自身是否在运行.
        """
        pass

    @abstractmethod
    def is_moss_running(self) -> bool:
        """
        判断 moss 是否在运行中.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        关闭自身, 用于优雅退出.
        """
        pass

    @abstractmethod
    async def wait_closed(self) -> None:
        """
        阻塞等待自身运行退出.
        所有的功能都会关闭.
        """
        pass

    @abstractmethod
    def wait_closed_sync(self, timeout: float | None = None) -> bool:
        """
        阻塞等待自身退出.
        该方法仅限同步上下文调用
        """
        pass

    @abstractmethod
    def create_task(self, cor: Coroutine) -> asyncio.Task:
        """
        创建包含在 Matrix 生命周期内的 Task
        """
        pass

    async def arun(self, main_coro: Callable[[Self], Awaitable[Any]]) -> Any:
        """
        Matrix 运行的基本逻辑.
        可参考或直接基于这个函数运行基于 Matrix 的应用.
        如果将它包裹成 Asyncio.Task, 也可以和主协程并行运行.
        """
        if self.is_running():
            raise RuntimeError(f'Matrix already running.')

        async with self:
            loop = asyncio.get_running_loop()

            # 1. 先执行获取 Awaitable 对象
            result_or_coro = main_coro(self)

            # 2. 判断是否是协程（需要被包装成 Task 才能并发）
            if asyncio.iscoroutine(result_or_coro):
                task = loop.create_task(result_or_coro)
                exit_signal = loop.create_task(self.wait_closed())

                try:
                    done, pending = await asyncio.wait(
                        [task, exit_signal],
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    if task in done:
                        return await task
                    raise asyncio.CancelledError("Matrix identity is closing")
                finally:
                    # 3. 这里的清理逻辑必须覆盖到位
                    for t in [task, exit_signal]:
                        if not t.done():
                            t.cancel()
                    await asyncio.gather(task, exit_signal, return_exceptions=True)
            else:
                # 如果用户传的是普通 Awaitable 或已完成的结果
                return await result_or_coro

    def run(self, main_coro: Callable[[Self], Awaitable[Any]]) -> Any:
        """
        同步阻塞入口。内部自动拉起事件循环并治理生命周期。
        兼容 Python 3.10 的顶层入口。
        """
        try:
            import uvloop
        except ImportError:
            # 如果不能支持.
            uvloop = None

        try:
            if uvloop is not None:
                asyncio.set_event_loop(uvloop.new_event_loop())
            return asyncio.run(self.arun(main_coro))
        except KeyboardInterrupt:
            pass  # 底层 arun 已经处理了清理

    @abstractmethod
    async def __aenter__(self) -> Self:
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
