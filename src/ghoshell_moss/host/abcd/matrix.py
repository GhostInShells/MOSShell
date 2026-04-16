from typing import Literal, Callable, Awaitable, Any, Coroutine, Iterable
from typing_extensions import Self
from abc import ABC, abstractmethod
from ghoshell_moss.core.concepts.topic import TopicService
from ghoshell_moss.core.concepts.channel import Channel
from ghoshell_moss.contracts import LoggerItf, ConfigStore, Workspace
from ghoshell_container import IoCContainer
from ghoshell_moss.core.concepts.session import Session
from .manifests import Manifest
import asyncio

__all__ = ['Matrix', 'Cell']


class Cell(ABC):
    """
    在 matrix 中可以并行独立运行的单元, 比如并行思考模块, channel provider 等等.
    """
    name: str  # 节点的名称.
    description: str  # 节点的描述.
    docstring: str  # 节点的详细描述.
    type: Literal['app', 'main'] | str  # 节点的类型. main 表示 moss 的 runtime, 而 app 表示是一个环境中可加载的应用.
    where: str  # 这个节点自身的工作目录.

    @property
    @abstractmethod
    def address(self) -> str:
        """节点的地址. 通常作为节点的各种通讯机制的前缀或关键环节."""
        pass

    @property
    def log_name(self) -> str:
        return '.'.join(['moss', self.type, self.name])

    @abstractmethod
    def is_alive(self) -> bool:
        """
        节点是否在运行中.
        """
        pass


CELL_ADDRESS = str


class Matrix(ABC):
    """
    MOSS 架构下多节点组网后形成的通讯矩阵的客户端.
    持有矩阵的抽象可以通过矩阵通讯.
    本身应该是进程级别单例.
    """

    @classmethod
    def discover(cls) -> Self:
        """
        约定的环境发现逻辑.
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

    @property
    @abstractmethod
    def mode(self) -> str:
        """
        返回当前运行的模式.
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
    def manifests(self) -> Manifest:
        """
        返回持有的环境发现资源.
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

    def show_configs(self) -> Iterable[dict[str, str]]:
        """
        不返回配置值的情况下, 返回配置的介绍.
        """
        from ghoshell_moss.contracts import ConfigStore
        store = self.container.force_fetch(ConfigStore)
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
    def configs(self) -> ConfigStore:
        """
        本地配置中心读取.
        """
        pass

    @property
    @abstractmethod
    def workspace(self) -> Workspace:
        """
        workspace 管理.
        """
        pass

    @property
    @abstractmethod
    def topics(self) -> TopicService:
        """
        通信服务.
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
            uvloop.install()
        except ImportError:
            # 如果不能支持.
            uvloop = None

        try:
            return asyncio.run(self.arun(main_coro))
        except KeyboardInterrupt:
            pass  # 底层 arun 已经处理了清理

    @abstractmethod
    async def __aenter__(self) -> Self:
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
