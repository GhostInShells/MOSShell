from abc import ABC, abstractmethod
from typing import Iterable, Any, Generic, TypeVar
from typing_extensions import Self
from pathlib import Path
from ghoshell_container import IoCContainer, Provider
from ghoshell_moss.core.blueprint.environment import (
    DEFAULT_NETWORK_SCOPE, DEFAULT_NETWORK_NAME, Environment,
    DEFAULT_CELLS_DIR,
    ENV_WORKSPACE_DIR_KEY,
)
from ghoshell_moss.core.blueprint.cell import CellRegistry
from ghoshell_moss.core.blueprint.ghost import GhostMeta
from ghoshell_moss.core.blueprint.states_channel import PrimeChannel
from ghoshell_moss.core.blueprint.mindflow import SignalSchema, NucleusMeta
from ghoshell_moss.core.blueprint.parameter import ParameterSchema
from ghoshell_moss.core.concepts.topic import TopicSchema
from ghoshell_moss.contracts import Workspace, ConfigSchema
from ghoshell_moss.contracts.logger import config_logger_from_yaml
from ghoshell_moss.message import unique_id
from pydantic import BaseModel, Field
from ghoshell_moss.core.ctml.versions import (
    search_version_file_in_dir, default_moss_ctml_meta_instruction_directory,
    get_version_from_filename, CTML_VERSION,
)
import dotenv
import sys
import logging

__all__ = [
    'ModeMeta',
    'HostMode',
    'Manifest', 'ModeManifests', 'MatrixManifest',
    'Project',
    'MATRIX_MANIFESTS_PACKAGE', 'MODE_MANIFESTS_PACKAGE',
]

T = TypeVar('T')

MATRIX_MANIFESTS_PACKAGE = 'MOSS.manifests'
GHOST_MANIFESTS_PACKAGE = 'MOSS.ghosts'
MODE_MANIFESTS_PACKAGE = 'Host.manifests'


class ModeMeta(BaseModel):
    """
    MOSS 的元信息配置.
    通过 workspace 的 MOSS.md 读取.
    """
    name: str = Field(
        default='',
        description="为启动后的 moss 实例 (躯体) 命名. 用于做本地不同模式的区分."
                    "比如 Desktop, Robot 等.",
    )
    description: str = Field(
        default="default moss mode discovered in moss workspace",
        description="描述当前 moss 环境, 这样当这个 moss 环境提供给远程 moss 环境时, 对方可以通过命名识别自己. ",
    )
    ctml_version: str = Field(
        default=CTML_VERSION,
        description="当前 MOSS 默认使用的提示词版本."
    )

    default_network_scope: str = Field(
        default=DEFAULT_NETWORK_SCOPE,
    )

    system_prompt: str = Field(
        default="",
        description="补充到 CTML meta instruction 后面的内容. version 为空, 这里应该包含完整的 meta instruction"
    )

    cell_paths: list[str] = Field(
        default_factory=lambda: [
            DEFAULT_CELLS_DIR,
            f"${ENV_WORKSPACE_DIR_KEY}/{DEFAULT_CELLS_DIR}",
        ],
        description="以 project 为出发点, 发现 cells 的路径. ",
    )
    exclude_cell_paths: list[str] = Field(
        default_factory=lambda: [],
        description="排除掉 cells 的路径, 匹配规则从 project 出发的相对路径, 以 fnmatch 语法为准过滤. "
    )

    # --- 运行时的动态生成字段 --- #
    file: str = Field(
        default="",
        description="配置自身发现的位置."
    )

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

    def write_to_file(self, file: Path) -> None:
        import frontmatter
        metadata = self.model_dump(
            exclude_none=True,
            exclude={'system_prompt', 'project', 'file', 'where'},
        )
        content = self.system_prompt
        post = frontmatter.Post(content, **metadata)
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(post.content)

    def cell_dirs(self, env: Environment) -> list[Path]:
        """return absolute paths to find cells"""
        result = []
        project_dir = env.project_path
        for project_relative_path in self.cell_paths:
            special_key = f"${ENV_WORKSPACE_DIR_KEY}"
            project_relative_path = project_relative_path.replace(special_key, str(env.workspace_path))
            cell_dir = project_dir / project_relative_path
            if cell_dir.exists():
                result.append(cell_dir.absolute())
        return result


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
        for file in scope_dir.glob('scope-*.json'):
            data = cls.read_from_file(file, throw=False)
            if data:
                yield data


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
        """发现的配置项源码, 如果有的话."""
        ...

    @abstractmethod
    def value(self) -> T | Exception:
        """找到的值. 如果在发现中存在异常, 则应该返回异常. """
        ...

    @abstractmethod
    def detail(self) -> str:
        """对值的自然语言描述. """
        ...

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


class MatrixManifest(ABC):

    @abstractmethod
    def providers(self) -> Iterable[Manifest[Provider]]:
        """找到的需要注册到 ioc 中的服务. """
        ...

    @abstractmethod
    def configs(self) -> Iterable[Manifest[ConfigSchema]]:
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


class ModeManifests(ABC):

    @abstractmethod
    def channel(self) -> Manifest[PrimeChannel]:
        """找到的 main channel. """
        ...

    @abstractmethod
    def providers(self) -> Iterable[Manifest[Provider]]:
        """找到的需要注册到 ioc 中的服务. """
        ...

    @abstractmethod
    def configs(self) -> Iterable[Manifest[ConfigSchema]]:
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
    def nuclei(self) -> Manifest[NucleusMeta]:
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
    def meta(self) -> ModeMeta:
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
    def manifests(self) -> ModeManifests:
        """
        模式自己的资源声明.
        """
        ...

    @abstractmethod
    def cells(self) -> CellRegistry:
        """根据模式配置生成的 cell 注册表. """
        ...

    def cells_discover_paths(self) -> list[Path]:
        result = []
        for relative_path in self.meta.cell_paths:
            cell_path = self.env.project_path / relative_path
            if cell_path.exists():
                result.append(cell_path.absolute())

        for relative_path in self.meta.workspace_cell_paths:
            cell_path = self.env.workspace_path / relative_path
            if cell_path.exists():
                result.append(cell_path.absolute())
        return result


_project: 'Project | None' = None


class Project(ABC):
    """
    MOSS 所在的 Project 感知.
    环境配置的 moss workspace 所在目录为 Project.
    在 Project 中需要反应出 MOSS 可以操作的存储环境, 可以启动的模式, 能力声明等.
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
        return logging.getLogger('moss')

    @classmethod
    def discover(
            cls,
            env: Environment | None = None,
    ) -> 'Project':
        """从环境中发现 Project. """
        global _project
        if _project is not None:
            return _project
        from ghoshell_moss.factory import create_project
        env = env or Environment.discover()
        # 确认 env 启动.
        env.bootstrap()
        project = create_project(env)
        # project 设置成单例.
        project.bootstrap()
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
            # !! 全局环境变量只做补充, 不做覆盖.
            dotenv.load_dotenv(env_file, override=False)

        # 添加 workspace 内部 共享的源码
        # 确保源码已经加载到系统.
        source_path = str(self.workspace_source_dir.absolute())
        if source_path not in sys.path:
            sys.path.append(source_path)
        # 更新日志模块.
        log_config_file = self.log_config_file
        if log_config_file.exists():
            config_logger_from_yaml(str(log_config_file.absolute()))

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
    def list_modes(self) -> Iterable[tuple[Path, Manifest[ModeMeta]]]:
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

    @property
    @abstractmethod
    def cells(self) -> CellRegistry:
        """
        默认的 cells 发现机制.
        根据 env 的 project & workspace cells 约定获取.
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
        return self.root.joinpath('modes')

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
    def network_configs_dir(self) -> Path:
        return self.root.joinpath('networks').absolute()

    @property
    def configs_dir(self) -> Path:
        return self.root.joinpath('configs').absolute()

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
