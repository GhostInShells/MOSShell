from typing import Protocol, Literal
from typing_extensions import Self
from abc import ABC, abstractmethod
from ghoshell_moss.core.concepts.topic import TopicService
from ghoshell_moss.core.concepts.channel import ChannelProvider
from ghoshell_moss.contracts import LoggerItf, ConfigStore, Workspace
from ghoshell_container import IoCContainer
from .session import Session
from .manifests import Manifests


class Cell(Protocol):
    """
    在 matrix 中可以并行独立运行的单元, 比如并行思考模块, channel provider 等等.
    """
    name: str  # 节点的名称.
    description: str  # 节点的描述.
    docstring: str  # 节点的详细描述.
    address: str  # 节点的地址. 通常作为节点的各种通讯机制的前缀或关键环节.
    type: Literal['app'] | str  # 节点的类型. main 表示 moss 的 runtime, 而 app 表示是一个环境中可加载的应用.
    work_directory: str # 这个节点自身的工作目录.

    @abstractmethod
    def is_alive(self) -> bool:
        """
        节点是否在运行中.
        """
        pass


class Matrix(ABC):
    """
    MOSS 架构下多节点组网后形成的通讯矩阵的客户端.
    持有矩阵的抽象可以通过矩阵通讯.
    本身应该是进程级别单例.
    """

    @property
    @abstractmethod
    def this(self) -> Cell:
        """
        返回当前节点自身的讯息. 节点之间通讯仅仅通过 topics / parameter / action 等.
        """
        pass

    @abstractmethod
    def list_cells(self) -> list[Cell]:
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
    def manifests(self) -> Manifests:
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

    @property
    @abstractmethod
    def channel_provider(self) -> ChannelProvider:
        """
        matrix 所拥有的单独 channel provider,
        用来和主进程通讯.
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
        """
        pass

    @abstractmethod
    async def __aenter__(self) -> Self:
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
