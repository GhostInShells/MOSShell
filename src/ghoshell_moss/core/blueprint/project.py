from abc import ABC, abstractmethod
from typing import Iterable, Any, Generic, TypeVar, ClassVar, Iterator, Callable
from typing_extensions import Self
from pathlib import Path
from ghoshell_container import IoCContainer, Provider
from ghoshell_moss.core.blueprint.environment import (
    DEFAULT_NETWORK_SCOPE, DEFAULT_NETWORK_NAME, Environment,
    DEFAULT_NODES_DIR, MOSS_NAME_PATTERN,
    ENV_WORKSPACE_DIR_KEY,
)
from ghoshell_moss.core.blueprint.cell import NodeManager, CellRuntimeInfo
from ghoshell_moss.core.blueprint.ghost import GhostMeta
from ghoshell_moss.core.blueprint.states_channel import PrimeChannel
from ghoshell_moss.core.blueprint.mindflow import SignalSchema, NucleusMeta
from ghoshell_moss.core.blueprint.parameter import ParameterSchema
from ghoshell_moss.core.concepts.topic import TopicSchema
from ghoshell_moss.contracts import Workspace, ConfigType
from ghoshell_moss.contracts.resource import ResourceStorageMeta
from ghoshell_moss.contracts.logger import config_logger_from_yaml
from ghoshell_moss.message import unique_id
from pydantic import BaseModel, Field
from ghoshell_moss.core.ctml.versions import (
    search_version_file_in_dir, default_moss_ctml_meta_instruction_directory,
    get_version_from_filename, CTML_VERSION,
)
import sys
import logging

__all__ = [
    'HostModeMeta',
    'HostMode',
    'Manifest', 'HostModeManifests', 'ProjectManifest',
    'NetworkConfig', 'NetworkMetadata',
    'Project',
    'HOST_MODE_MANIFESTS_PACKAGE',
    'HOST_MODE_FILE',
    'MODE_MATRIX_MANIFESTS_PACKAGE',
]

T = TypeVar('T')

HOST_MODE_MANIFESTS_PACKAGE = 'HOST'
HOST_MODE_FILE = 'HOST.md'
MODE_MATRIX_MANIFESTS_PACKAGE = 'MATRIX.manifests'


class HostModeMeta(BaseModel):
    """
    Host 模式的元信息配置.
    通过 modes/{name}/HOST.md 读取. name 始终由发现路径的目录名覆盖.
    """
    name: str = Field(
        default='default',
        description="为启动后的 moss 实例 (躯体) 命名. 用于做本地不同模式的区分."
                    "比如 Desktop, Robot 等.",
        pattern=MOSS_NAME_PATTERN,
    )
    description: str = Field(
        default="default moss mode discovered in moss workspace",
        description="描述当前 moss 环境, 这样当这个 moss 环境提供给远程 moss 环境时, 对方可以通过命名识别自己. ",
    )
    ctml_version: str = Field(
        default=CTML_VERSION,
        description="当前 MOSS 默认使用的提示词版本."
    )

    manifest_package: str = Field(
        default=HOST_MODE_MANIFESTS_PACKAGE,
        description="发现 manifest 执行的路径. ",
    )

    system_prompt: str = Field(
        default="",
        description="补充到 CTML meta instruction 后面的内容. version 为空, 这里应该包含完整的 meta instruction"
    )

    node_paths: list[str] = Field(
        default_factory=lambda: [
            DEFAULT_NODES_DIR,
            f"${ENV_WORKSPACE_DIR_KEY}/{DEFAULT_NODES_DIR}",
        ],
        description="以 project 为出发点, 发现 nodes 的路径. ",
    )
    exclude_node_paths: list[str] = Field(
        default_factory=lambda: [],
        description="排除掉 nodes 的路径, 匹配规则从 project 出发的相对路径, 以 fnmatch 语法为准过滤. "
    )
    bringup_nodes: list[str] = Field(
        default_factory=list,
        description="mode 启动时自动拉起的 node 列表. 每项是相对 project.root 的路径, "
                    "支持目录 (找 NODE.md)、NODE.md 文件或脚本. 单个失败记日志, 不阻断 mode 启动.",
    )

    # --- 运行时的动态生成字段 --- #
    file: str = Field(
        default="",
        description="配置自身发现的位置."
    )

    @classmethod
    def read_from_directory(cls, directory: Path) -> 'HostModeMeta | None':
        file = directory / HOST_MODE_FILE
        if file.exists():
            return cls.from_file(file)
        return None

    def node_dirs(self, env: Environment) -> list[Path]:
        """基于 node_paths 解析为绝对路径."""
        result = []
        project_dir = env.project_path
        for relative_path in self.node_paths:
            relative_path = relative_path.replace(
                f'${ENV_WORKSPACE_DIR_KEY}', str(env.workspace_path),
            )
            cell_dir = project_dir / relative_path
            if cell_dir.exists():
                result.append(cell_dir.absolute())
        return result

    @classmethod
    def from_file(cls, file: Path, name: str = '') -> Self:
        """
        从文件中读取 meta instruction.
        """
        import frontmatter
        post = frontmatter.load(str(file.absolute()))
        data = post.metadata
        data['file'] = str(file.absolute())
        data['system_prompt'] = post.content
        if name:
            # 重写 name.
            data['name'] = name
        return cls(**data)

    def write_to_directory(self, directory: Path) -> None:
        file = directory / HOST_MODE_FILE
        self.write_to_file(file)

    def write_to_file(self, file: Path) -> None:
        import frontmatter
        metadata = self.model_dump(
            exclude_none=True,
            exclude={'system_prompt', 'file'},
        )
        post = frontmatter.Post(self.system_prompt, **metadata)
        file.parent.mkdir(parents=True, exist_ok=True)
        frontmatter.dump(post, file)


class NetworkMetadata(BaseModel):
    """
    通讯可以使用的 Scope 的配置元信息.
    系统基于 Project 可以查询到可用的 session scope,
    基于 session scope 可以创建 Session 通讯.
    """
    name: str = Field(
        default=DEFAULT_NETWORK_NAME,
        description="network name",
    )
    description: str = Field(
        default="",
        description="network 的介绍. 用于环境发现时展示. "
    )
    driver: str = Field(
        default="zenoh",
        description="通讯使用的模块. 需要 Matrix 支持对应的 factory 可以解析这个协议."
    )
    scope: str = Field(
        default=DEFAULT_NETWORK_SCOPE,
        description="联通到目标网络后, 通信子网所用的名称. 必须存在. ",
    )
    config: dict[str, Any] = Field(
        default_factory=dict,
        description="为 transport 提供的具体配置项, 需要和 transport 对应的模块一致. ",
    )

    @classmethod
    def read_from_file(cls, path: Path, *, throw: bool = False) -> 'NetworkMetadata | None':
        try:
            if path.is_file() and path.exists():
                data = path.read_bytes()
                return cls.model_validate_json(data)
            return None
        except Exception as e:
            if throw:
                raise e
            return None

    def write_to_file(self, scope_dir: Path) -> None:
        filename = f'scope-{unique_id()}.json'
        file = scope_dir.joinpath(filename)
        file.write_text(self.model_dump_json(indent=2, ensure_ascii=False, exclude_defaults=False, exclude_none=True))

    @classmethod
    def read_from_dir(cls, scope_dir: Path) -> Iterable['NetworkMetadata']:
        if not scope_dir.exists() or not scope_dir.is_dir():
            yield from []
            return
        for file in scope_dir.glob('*.json'):
            data = cls.read_from_file(file, throw=False)
            if data:
                yield data


class NetworkConfig(BaseModel, ABC):

    @classmethod
    @abstractmethod
    def driver_name(cls) -> str:
        pass

    def to_metadata(self, name: str, description: str = '') -> NetworkMetadata:
        return NetworkMetadata(
            name=name,
            description=description,
            driver=self.driver_name(),
            config=self.model_dump(),
        )

    @classmethod
    def from_metadata(cls, metadata: NetworkMetadata) -> 'NetworkConfig | None':
        if metadata.driver != cls.driver_name():
            return None
        return cls.model_validate(metadata.config)


_RelativePath = str
"""从项目根目录出发的相对路径."""


class Manifest(Generic[T], ABC):
    """环境中的声明对象, 可以用于运行时."""

    @abstractmethod
    def found_at(self) -> Path:
        """声明发现的路径."""
        ...

    @abstractmethod
    def import_path(self) -> str | None:
        """如果可以通过 python import, 这里提供 import 路径. """
        ...

    @abstractmethod
    def name(self) -> str:
        """发现物的名称."""
        ...

    @abstractmethod
    def description(self) -> str:
        """发现物的介绍, 比如 docstring """
        ...

    @abstractmethod
    def source(self) -> str:
        """发现相关的 source code."""
        ...

    @abstractmethod
    def value(self) -> T:
        """找到的值. 仅在 is_error() 为 False 时有效. """
        ...

    @abstractmethod
    def detail(self) -> str:
        """对值的自然语言描述. """
        ...

    def is_error(self) -> bool:
        """扫描过程中是否发生异常. True 时 error() 返回异常详情."""
        return False

    def error(self) -> Exception | None:
        """扫描异常详情. is_error() 为 False 时返回 None."""
        return None

    def update_container(self, container: IoCContainer) -> None:
        """
        所有的发现, 最终目标都是对 IoC 容器里的对象或容器本身产生影响.
        具体分为 set / register / bootstrap / shutdown 四个生命周期.
        - set 立刻生效;
        - register, 懒加载使用;
        - bootstrapper, 当 container 首次启动时生效, 可以用来对 Container 持有的某个对象做加工.
        - shutdown, container 结束时生效.
        如果需要在特定的生命周期绑定, 则不需要实现这个函数.
        """
        ...


class ProjectManifest(ABC):
    """全局基线声明 — 扫描一个 Python 包的子包，发现所有能力声明."""

    @abstractmethod
    def root_package(self) -> str:
        """此 manifest 扫描的 Python 包路径. 如 MOSS.manifests."""
        ...

    def explain(self) -> str:
        """自描述 — 此层声明的内容、来源与定位."""
        pkg = self.root_package()
        types = ["providers", "configs", "topics", "signals", "parameters",
                 "resources", "nuclei"]
        type_list = ", ".join(f"`{t}`" for t in types)
        return (
            f"# Matrix Manifest — 全局基线声明\n\n"
            f"扫描包: `{pkg}`\n"
            f"声明类型: {type_list}\n\n"
            f"这是 workspace 级别的全局声明。所有 mode 可以通过 Python import "
            f"继承这些声明，再追加或覆盖 mode 专属内容。\n"
        )

    @abstractmethod
    def providers(self) -> Iterable[Manifest[Provider]]:
        """找到的需要注册到 ioc 中的服务. """
        ...

    @abstractmethod
    def configs(self) -> Iterable[Manifest[ConfigType]]:
        """找到的配置项. """
        ...

    @abstractmethod
    def topics(self) -> Iterable[Manifest[TopicSchema]]:
        """找到的 topic 协议. """
        ...

    @abstractmethod
    def signals(self) -> Iterable[Manifest[SignalSchema]]:
        """找到的 signals 协议"""
        ...

    @abstractmethod
    def parameters(self) -> Manifest[ParameterSchema]:
        """找到的 parameter 协议. """
        ...

    @abstractmethod
    def resources(self) -> Iterable[Manifest[ResourceStorageMeta]]:
        """找到的资源存储. """
        ...

    def nuclei(self) -> Iterable[Manifest[NucleusMeta]]:
        """找到的 nucleus meta. 默认空, mode 层覆盖."""
        yield from []


class HostModeManifests(ABC):
    """Host Mode 专属声明 — 继承 Matrix 全局基线，追加 mode 特有内容."""

    @abstractmethod
    def root_package(self) -> str:
        """此 manifest 扫描的 Python 包路径. 如 HOST."""
        ...

    def explain(self) -> str:
        """自描述 — 此层声明的内容、来源与定位."""
        pkg = self.root_package()
        types = ["providers", "configs", "topics", "signals", "parameters", "resources",
                 "channel", "nuclei"]
        type_list = ", ".join(f"`{t}`" for t in types)
        return (
            f"# Mode Manifest — 当前模式的有效视图\n\n"
            f"扫描包: `{pkg}`\n"
            f"声明类型: {type_list}\n\n"
            f"Mode 通过 Python import 继承 Matrix 全局声明，再追加或覆盖。\n"
            f"`channel` 和 `nuclei` 是 mode 专属类型，不在 Matrix 层。\n"
        )

    @abstractmethod
    def channel(self) -> Manifest[PrimeChannel]:
        """找到的 main channel. """
        ...

    @abstractmethod
    def providers(self) -> Iterable[Manifest[Provider]]:
        """找到的需要注册到 ioc 中的服务. """
        ...

    @abstractmethod
    def configs(self) -> Iterable[Manifest[ConfigType]]:
        """找到的配置项. """
        ...

    @abstractmethod
    def topics(self) -> Iterable[Manifest[TopicSchema]]:
        """找到的 topic 协议. """
        ...

    @abstractmethod
    def signals(self) -> Iterable[Manifest[SignalSchema]]:
        """找到的 signals 协议"""
        ...

    @abstractmethod
    def parameters(self) -> Manifest[ParameterSchema]:
        """找到的 parameter 协议. """
        ...

    @abstractmethod
    def resources(self) -> Iterable[Manifest[ResourceStorageMeta]]:
        """找到的资源存储. """
        ...

    @abstractmethod
    def nuclei(self) -> Iterable[Manifest[NucleusMeta]]:
        """找到的 nucleus meta, 应该在 ioc 启动后, 如果有 mindflow 在场时注册. """
        ...


class HostMode(ABC):
    """
    Moss 的 Host 节点启动时的环境发现结果.
    """

    @property
    def name(self) -> str:
        """模式的名称. 可能是发现路径优先. """
        return self.meta.name

    @property
    @abstractmethod
    def env(self) -> Environment:
        pass

    def bootstrap(self) -> None:
        """
        当某个模式实际投入使用时, 也需要初始化环境信息, 主要是 .env 与 src.
        """
        if hasattr(self, '_bootstrapped') and getattr(self, '_bootstrapped', False):
            return
        setattr(self, '_bootstrapped', True)
        env_file = Environment.env_file(self.workspace_dir)
        if env_file.exists():
            import dotenv
            # !!! mode 的环境变量, 会做覆盖
            dotenv.load_dotenv(env_file, override=True)

        # 将目标模式下的环境发现代码加载.
        # 模式之间的代码彻底隔离.
        source_path = self.workspace.source().abspath()
        if source_path.exists():
            source = str(source_path)
            if source not in sys.path:
                # 让 mode 内的代码也可以加载.
                sys.path.append(source)

    @property
    @abstractmethod
    def meta(self) -> HostModeMeta:
        """
        元信息, 复用 MOSS Meta 的结构.
        """
        ...

    @property
    @abstractmethod
    def workspace_dir(self) -> Path:
        """
        mode 自己的 workspace 所在路径.
        """
        ...

    @property
    @abstractmethod
    def workspace(self) -> Workspace:
        """
        对应的 workspace.
        """
        ...

    @abstractmethod
    def manifests(self) -> HostModeManifests:
        """
        模式自己的资源声明.
        """
        ...

    @abstractmethod
    def matrix_manifests(self) -> ProjectManifest:
        """
        mode 级环境能力声明 — MATRIX.manifests.

        扫描 mode 包下的 MATRIX.manifests, 承载跨 cell 共享的环境能力
        (如音频 provider). 初始全空, mode 按需追加.
        """
        ...

    def nodes_discover_paths(self) -> list[Path]:
        # 原 HostMode.cells 已删 (UU-10/TT-7: cells 三处露头收敛为 project.cells
        # 一条链路). mode 只贡献发现路径配置, 不再持有第二个 registry 实例.
        return self.meta.node_dirs(self.env)


_project: 'Project | None' = None


class Project(ABC):
    """
    被治理领地的句柄 (governance-domain handle).

    Project 的一句话承诺: 指认一片领地, 并提供领地内治理真相的入口.
    成员几乎都指向 workspace, 因为 workspace 正是治理真相的存放地 —
    project 目录本身的内容对 MOSS 是动态可变的 (ghost 自管理的领地),
    目录 category 永不进内核 API (TT-7).
    """

    @property
    def id(self) -> str:
        """
        所有的 project 都会生成一个 uuid 作为自己唯一的身份.
        启动时如果没有, 需要生成一个.
        id 通常放置在项目 {workspace}/id 文件中.
        """
        return self.env.project_id

    @property
    @abstractmethod
    def env(self) -> Environment:
        """
        环境发现的信息对于 MOSS Project 是第一性的.
        """
        ...

    @property
    def logger(self) -> logging.Logger:
        """project 级别的日志位置. """
        return logging.getLogger('moss')

    _LOG_HANDLER_NAME = 'moss_file_handler'

    def _ensure_log_file_handler(self) -> None:
        """为 'moss' logger 添加运行时文件 handler.

        logging.yml 只配格式和等级, 文件路径是运行时确定的.
        此方法在 bootstrap() 中调用, 幂等.
        """
        from logging.handlers import TimedRotatingFileHandler
        from ghoshell_moss.contracts.logger import default_logger_formatter

        moss_logger = logging.getLogger('moss')
        for h in moss_logger.handlers:
            if h.get_name() == self._LOG_HANDLER_NAME:
                return

        log_file = self.log_file
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handler = TimedRotatingFileHandler(
            filename=str(log_file),
            when='d',
            interval=1,
            backupCount=5,
        )
        handler.set_name(self._LOG_HANDLER_NAME)
        handler.setLevel(logging.INFO)
        handler.setFormatter(default_logger_formatter())
        moss_logger.addHandler(handler)

    @classmethod
    def discover(
            cls,
            env: Environment | None = None,
    ) -> 'Project':
        """从环境中发现 Project.

        env 传入契约: 已 seal (调用方职责). None 时走 Environment.discover(),
        返回的 singleton 也已 seal — 本方法不重复 seal (§UU-1 一次性事实).
        """
        global _project
        if _project is not None:
            return _project
        from ghoshell_moss.factory import create_project
        env = env or Environment.discover()
        project = create_project(env)
        return project

    def bootstrap(self):
        """ project 产生全局副作用. """
        global _project
        # 确保自己是全局单例.
        _project = self
        # 后续副作用执行一次.
        if hasattr(self, '_bootstrapped') and getattr(self, '_bootstrapped', False):
            return
        setattr(self, '_bootstrapped', True)
        env_file = Environment.env_file(self.workspace_dir)
        if env_file.exists():
            import dotenv
            # !! 全局环境变量只做补充, 不做覆盖.
            dotenv.load_dotenv(env_file, override=False)

        # 添加 workspace 内部 共享的源码
        # 确保源码已经加载到系统.
        source_path = str(self.workspace_source_dir.absolute())
        if source_path not in sys.path:
            sys.path.append(source_path)
        # 更新日志模块 — 先加载格式/等级, 再加运行时文件 handler
        log_config_file = self.log_config_file
        if log_config_file.exists():
            config_logger_from_yaml(str(log_config_file.absolute()))
        self._ensure_log_file_handler()
        container = self.container
        container.bootstrap()

    @property
    @abstractmethod
    def container(self) -> IoCContainer:
        """project level ioc container"""
        ...

    # --- manifests --- #

    def network_metas(self) -> dict[str, NetworkMetadata]:
        """基于环境发现的, 已经配置好的通讯 scope. """
        result = {}
        for data in NetworkMetadata.read_from_dir(self.network_configs_dir):
            result[data.name] = data
        return result

    @abstractmethod
    def ghosts(self) -> Iterable[tuple[Path, GhostMeta | Exception]]:
        """
        发现不同的 Ghost 定义.
        默认通过 [workspace]/src/GHOSTS/ 发现.
        返回的 Path 是发现 GhostMeta 的路径.
        """
        ...

    def current_mode(self) -> HostMode | None:
        try:
            if self.env.no_mode:
                return None
            return self.get_mode(self.env.mode_name)
        except Exception as e:
            self.logger.error("current mode %s is not available: %s", self.env.mode_name, e)
            raise e

    def current_ghost(self) -> GhostMeta | None:
        try:
            if self.env.no_ghost:
                return None
            return self.get_ghost(self.env.ghost_name)
        except Exception as e:
            self.logger.error("current ghost %s is not available: %s", self.env.ghost_name, e)
            raise e

    @abstractmethod
    def get_ghost(self, name: str) -> GhostMeta:
        """
        获取指定的 GhostMeta
        :raise LookupError: if not found
        :raise Exception: if errors occur
        """
        ...

    @abstractmethod
    def list_modes(self) -> Iterable[tuple[Path, Manifest[HostModeMeta]]]:
        """
        返回 workspace 里定义的所有模式.
        每个模式对应的路径有自己的命名.
        用 Meta 来代替 HostMode.
        返回的 Path 是发现 ModeMeta 的路径.
        """
        ...

    @abstractmethod
    def get_mode(self, mode_name: str) -> HostMode:
        """
        使用 mode name 获取 MossMeta
        :raise LookupError: if not found
        :raise Exception: if errors occur
        """
        ...

    @abstractmethod
    def project_manifests(self) -> ProjectManifest:
        """
        workspace 级基线声明 (MOSS.manifests 包扫描产物).

        project 层承接 ProjectManifest 而非 HostModeManifests, 落实依赖分发轴:
        `pip install ghoshell_moss[matrix]` = cell 最小依赖, 不含 mode 层重依赖
        (mindflow / nuclei / audio / ml). MossRuntime 才承接 mode.manifests()
        叠加 mode 专属.

        意图: cell = package 无关的治理域快捷方式 (§WW-4) 在 pip 分发面的落实.
        """
        ...

    @property
    @abstractmethod
    def nodes(self) -> NodeManager:
        """
        治理域里可以拉起的 node 声明发现器 (基于 NODE.md 扫描).

        与 cell_runtimes() 的分工:
        - nodes: 静态声明 (领地里"可以拉起什么", 只回答 node — host 不走声明).
        - cell_runtimes: 运行时事实 (领地里"已经拉起了什么", host + node 都在).
        """
        ...

    @abstractmethod
    def kill_cell(self, address: str) -> bool:
        """
        孤儿清理: 尝试终止本 project 治理域内已死或残留的 cell 进程.

        典型场景是父进程崩溃后 ledger 残留, cell_runtimes() 迭代时按需清理.
        active cell 的正常停止不走这里, 走 owner 的 Subprocesses.

        :return: True = address 属于本 project 且 kill 尝试完成 (进程已死或成功 killpg);
                 False = address 不在本地 ledger, 无治理权限, 无操作.
        """
        ...

    @abstractmethod
    def cell_runtimes(self) -> Iterator[CellRuntimeInfo]:
        """
        遍历本 project 治理域内已拉起的所有 cell (host + node) 的 runtime 信息.

        直接读文件系统 (env.cell_runtimes_dir 下的 json), 不做活性核对;
        活性判断由调用方自负 (info.is_alive()).
        """
        ...

    # --- directory --- #

    @property
    @abstractmethod
    def workspace(self) -> Workspace:
        """
        moss 自身运行和配置所在的目录.
        将它和主 Project 区分开. 可以认为 Project root 下的内容对于 MOSS 是动态可变的.
        而 moss workspace 下保存的是 moss 自身的静态配置和运行时数据.
        """
        ...

    @property
    def workspace_dir(self) -> Path:
        """
        workspace 的起点路径.
        """
        return self.workspace.root().abspath()

    @property
    def root(self) -> Path:
        """
        project 的根目录.
        通常在根目录下包含了 moss 自身运行的 workspace.
        """
        return self.env.project_path

    @property
    def ghosts_home(self) -> Path:
        """所有 ghost 共享的路径. """
        return self.workspace_dir.joinpath('ghosts').absolute()

    def get_ghost_home(self, ghost_name: str) -> Path:
        return self.ghosts_home.joinpath(ghost_name).absolute()

    @property
    def modes_home(self) -> Path:
        return self.workspace_dir.joinpath('modes')

    def get_mode_home(self, mode_name: str) -> Path:
        """返回每个模式自己独立的 path. """
        return self.modes_home.joinpath(mode_name).absolute()

    def ctml_dir(self) -> Path:
        """
        环境中约定的 ctml versions 配置.
        """
        return self.workspace.root().abspath().joinpath("ctml").absolute()

    def ctml_versions(self) -> dict[str, Path]:
        """
        当前环境中配置的 ctml versions.
        """
        versions = search_version_file_in_dir(default_moss_ctml_meta_instruction_directory())
        version_name_to_files = {}
        for version_file in versions:
            version_name = get_version_from_filename(version_file.name)
            version_name_to_files[version_name] = version_file
        for version_file in search_version_file_in_dir(self.ctml_dir()):
            version_name = get_version_from_filename(version_file.name)
            version_name_to_files[version_name] = version_file
        return version_name_to_files

    @property
    def runtime_dir(self) -> Path:
        """放置运行时数据的路径. """
        return self.workspace.runtime().abspath()

    @property
    def sessions_dir(self) -> Path:
        """存放 sessions 的路径"""
        return self.workspace.runtime().sub_storage('sessions').abspath()

    @property
    def log_dir(self) -> Path:
        """放置日志数据的路径. """
        return self.workspace.logs().abspath()

    @property
    def log_config_file(self) -> Path:
        """约定的日志配置存在的路径. """
        return self.configs_dir.joinpath('logging.yml').absolute()

    @property
    def log_file(self) -> Path:
        # 系统约定的日志文件名.
        # 运行时所有的日志都会记录到这个文件中.
        return self.log_dir.joinpath('moss.log').absolute()

    @property
    def network_configs_dir(self) -> Path:
        return self.workspace_dir.joinpath('networks').absolute()

    @property
    def configs_dir(self) -> Path:
        return self.workspace_dir.joinpath('configs').absolute()

    _configs_store: 'ConfigStore | None' = None

    @property
    def configs(self) -> 'ConfigStore':
        """mode 专属 ConfigStore — workspace configs/ 目录的唯一构造出口, 懒加载.

        Project 级只构造一次, CLI / matrix 的 ConfigStore provider 共享同一实例,
        避免多路构造漂移. 默认取 env 推导的 mode, 无预注册.
        """
        if self._configs_store is None:
            self._configs_store = self._configs()
        return self._configs_store

    def _configs(
            self,
            *,
            on_save: Callable[[str], None] | None = None,
            mode_name: str | None = None,
            configs: Iterable[ConfigType] = (),
    ) -> 'ConfigStore':
        """ConfigStore 构造逻辑 — 唯一出口.

        - mode_name=None → 从 env 推导: '' if env.no_mode else env.mode_name.
        - mode_name 显式传入 (含 '') 原样透传给 YamlConfigStore.
        - configs 按 get_or_create 预注册 (WorkspaceYamlConfigStoreProvider 语义).
        """
        from ghoshell_moss.contracts.configs import YamlConfigStore

        if mode_name is None:
            mode_name = '' if self.env.no_mode else self.env.mode_name
        store = YamlConfigStore(
            self.workspace.configs(),
            on_save=on_save,
            mode_name=mode_name,
        )
        for config in configs:
            store.get_or_create(config)
        return store

    @property
    def workspace_source_dir(self) -> Path:
        """Project 自动加载的源码路径. """
        return self.workspace.source().abspath()

    @property
    def tmp(self) -> Path:
        """
        存储临时文件的位置. 约定在 runtime/tmp 下.
        """
        return self.workspace.runtime().sub_storage('tmp').abspath()

    def __enter__(self) -> 'Project':
        self.bootstrap()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if exc_val is not None:
                self.logger.exception(exc_val)
        finally:
            self.container.shutdown()
