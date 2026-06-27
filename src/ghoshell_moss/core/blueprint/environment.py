"""
MOSS 环境发现的关键常量.
只保留几个最核心的常量.
"""

from typing import Literal, TypeAlias, Dict
from typing_extensions import Self
from pathlib import Path
from importlib import resources

from ghoshell_moss.message import unique_id
from pydantic import BaseModel, Field
import contextlib
import frontmatter
import os
import stat

__all__ = [
    'Environment',
    # workspace
    'DEFAULT_WORKSPACE_DIR_NAME',
    'WORKSPACE_ENV_FILENAME',
    'WORKSPACE_ENV_EXAMPLE_FILENAME',
    'DEFAULT_NETWORK_SCOPE',
    'DEFAULT_NETWORK_NAME',
    # env keys
    'ENV_WORKSPACE_DIR_KEY',
    'ENV_PROJECT_DIR_KEY',
    'ENV_PROJECT_ID_KEY',
    'ENV_NETWORK_KEY',
    'ENV_NETWORK_SCOPE_KEY',
    'ENV_SESSION_ID_KEY',
    'ENV_GHOST_NAME_KEY',
    'ENV_MOSS_MODE_KEY',
    'ENV_CELL_ADDRESS_KEY',
    'ENV_PARENT_CELL_ADDRESS_KEY',

    'MOSSEnvConfigKey',
    'MOSSRuntimeScopeEnvKey',

    # internal
    'WORKSPACE_PROJECT_ID_FILE',

    'NONE_MOSS_MODE',
    'NONE_GHOST_NAME',

    # stubs
    'MODE_STUB_PACKAGE',
    'WORKSPACE_STUB_PACKAGE',

    # dir path
    'META_CONFIG_FILENAME',
    'WORKSPACE_ENV_FILENAME',
    'WORKSPACE_ENV_EXAMPLE_FILENAME',

    'WORKSPACE_CELL_RUNTIME_DIR',
    'DEFAULT_CELLS_DIR',

    'MATRIX_MANIFESTS_PACKAGE',
    'GHOST_MANIFESTS_PACKAGE',

    'MOSS_NAME_PATTERN',
    'MossMeta',
    'MOSS_META_FILE',
]

# --- moss 的 workspace 发现机制 --- #

# moss 默认的 workspace 文件夹名.
# workspace 的绝对路径优先从环境变量寻找, 找不到时按目录发现机制寻找.
# 路径发现的逻辑是: os getcwd 下, 递归搜索父级目录下, home 目录下.
DEFAULT_WORKSPACE_DIR_NAME = '.moss'
META_CONFIG_FILENAME = 'MOSS.md'

# env 文件名. workspace 启动时会从其目录下读取环境变量文件 (by loadenv)
WORKSPACE_ENV_FILENAME = '.env'
WORKSPACE_ENV_EXAMPLE_FILENAME = '.env.example'
WORKSPACE_CELL_RUNTIME_DIR = 'runtime/cells'
DEFAULT_CELLS_DIR = 'cells'

# --- stubs --- #
# workspace 的原始文件所处的 package 路径.
WORKSPACE_STUB_PACKAGE = 'ghoshell_moss.stubs.workspace'
MODE_STUB_PACKAGE = 'ghoshell_moss.stubs.workspace.modes.default'

# --- 主要的环境变量名 --- #
# 这些环境变量不在 .env 中定义, 而是启动时 发现/生成, 或者通过父子进程传递的.

# 从环境变量中获取 moss workspace 路径的环境变量名.
ENV_WORKSPACE_DIR_KEY = 'MOSS_WORKSPACE'
ENV_PROJECT_DIR_KEY = 'MOSS_PROJECT_DIR'

ENV_PROJECT_ID_KEY = 'MOSS_PROJECT_ID'
WORKSPACE_PROJECT_ID_FILE = 'project_id'

ENV_NETWORK_KEY = 'MOSS_NETWORK'
DEFAULT_NETWORK_NAME = 'local'

# 指定 network 下的通讯子空间.
ENV_NETWORK_SCOPE_KEY = 'MOSS_NETWORK_SCOPE'
DEFAULT_NETWORK_SCOPE = 'default'

# 环境变量中获取 MOSS 运行时的 SESSION ID.
ENV_SESSION_ID_KEY = 'MOSS_SESSION_ID'

ENV_MOSS_MODE_KEY = 'MOSS_MODE_NAME'
NONE_MOSS_MODE = "none"

# 如果当前 MOSS 实例启动时, 启用了 Ghost, 则 GHOST_NAME 不应该为空.
ENV_GHOST_NAME_KEY = 'MOSS_GHOST_NAME'
# none 表示没有 Ghost 在运行.
NONE_GHOST_NAME = "none"

ENV_CELL_ADDRESS_KEY = 'MOSS_CELL_ADDRESS'
ENV_PARENT_CELL_ADDRESS_KEY = 'MOSS_PARENT_CELL_ADDRESS'

MATRIX_MANIFESTS_PACKAGE = 'MOSS.manifests'
GHOST_MANIFESTS_PACKAGE = 'MOSS.ghosts'

# 与运行配置项有关的 Env Key
MOSSEnvConfigKey: TypeAlias = Literal[
    "MOSS_WORKSPACE",
]

# 与运行时状态有关的 Env Key
MOSSRuntimeScopeEnvKey: TypeAlias = Literal[
    "MOSS_WORKSPACE",
    "MOSS_PROJECT_DIR",
    "MOSS_MODE_NAME",
    "MOSS_GHOST_NAME",
    "MOSS_NETWORK",
    "MOSS_NETWORK_SCOPE",
    "MOSS_CELL_ADDRESS",
    "MOSS_PARENT_CELL_ADDRESS",
    "MOSS_PROJECT_ID",
]

MOSS_NAME_PATTERN = r'^[a-zA-Z_][a-zA-Z0-9_]*$'

MOSS_META_FILE = 'MOSS.md'


class MossMeta(BaseModel):
    """
    项目级元信息配置.
    通过 workspace/MOSS.md 读取. 所有字段均有默认值, 无文件时可自动创建.
    """
    name: str = Field(
        default='moss',
        description='name of the moss project',
        pattern=MOSS_NAME_PATTERN,
    )
    description: str = Field(
        default='',
        description='description of the moss project',
    )
    default_network: str = Field(
        default=DEFAULT_NETWORK_NAME,
        description='default moss network',
        pattern=MOSS_NAME_PATTERN,
    )
    default_network_scope: str = Field(
        default=DEFAULT_NETWORK_SCOPE,
        pattern=MOSS_NAME_PATTERN,
    )
    default_mode: str = Field(
        default=NONE_MOSS_MODE,
        description='default moss mode',
        pattern=MOSS_NAME_PATTERN,
    )
    default_ghost: str = Field(
        default=NONE_GHOST_NAME,
        description='default moss ghost',
        pattern=MOSS_NAME_PATTERN,
    )
    matrix_manifest_package: str = Field(
        default=MATRIX_MANIFESTS_PACKAGE,
    )
    project_id: str = Field(
        default='',
        description='moss project id',
    )
    system_project: str = Field(
        default='',
        description='moss system prompt',
    )
    cell_paths: list[str] = Field(
        default_factory=lambda: [
            DEFAULT_CELLS_DIR,
            f"${ENV_WORKSPACE_DIR_KEY}/{DEFAULT_CELLS_DIR}",
        ],
        description="以 project 为出发点, 发现 cells 的路径.",
    )

    file: str = Field(
        default='',
        description='moss file path',
    )

    @classmethod
    def read_from_directory(cls, directory: Path) -> 'MossMeta | None':
        file = directory / MOSS_META_FILE
        return cls.read_from_file(file)

    @classmethod
    def read_from_file(cls, file: Path) -> 'MossMeta | None':
        if not file.is_file():
            return None
        post = frontmatter.load(str(file.absolute()))
        data = post.metadata
        data['file'] = str(file.absolute())
        data['system_project'] = post.content
        return cls(**data)

    def cell_dirs(self, env: 'Environment') -> list[Path]:
        """基于 cell_paths 解析为绝对路径."""
        result = []
        project_dir = env.project_path
        for relative_path in self.cell_paths:
            relative_path = relative_path.replace(
                f'${ENV_WORKSPACE_DIR_KEY}', str(env.workspace_path),
            )
            cell_dir = project_dir / relative_path
            if cell_dir.exists():
                result.append(cell_dir.absolute())
        return result

    def write_to_directory(self, directory: Path) -> None:
        file = directory / MOSS_META_FILE
        self.write_to_file(file)

    def write_to_file(self, file: Path | None = None) -> None:
        file = file or Path(self.file)
        if not file.name:
            raise ValueError('MossMeta.write_to_file: file path required')
        if self.project_id == '':
            self.project_id = unique_id()
        metadata = self.model_dump(exclude={'system_project', 'file'})
        post = frontmatter.Post(self.system_project, **metadata)
        file.parent.mkdir(parents=True, exist_ok=True)
        frontmatter.dump(post, file)


class Environment:
    """
    MOSS 的环境配置和发现体系.
    根据文件目录约定和环境变量发现, 完成初始化讯息.
    """

    def __init__(
            self,
            # moss meta 是发现的基础.
            workspace: Path | None = None,
            project: Path | None = None,
            *,
            # --- 可以显式传入的参数 --- #
            mode: str | None = None,
            scope: str | None = None,
            ghost: str | None = None,
            network: str | None = None,
            cell_address: str | None = None,
            parent_cell_address: str | None = None,
    ):
        """
        初始化 MOSS 的进程级别环境发现.
        """
        # 通过 workspace 实现初始化.
        if not workspace:
            workspace = self.find_workspace_path()
        if not workspace or not workspace.exists():
            raise EnvironmentError(f"Expected workspace `{workspace}` not exists")
        self._workspace_path = workspace
        if project:
            project_dir = project
        else:
            env_project_where = os.environ.get(ENV_PROJECT_DIR_KEY)
            if env_project_where:
                project_dir = Path(env_project_where)
            else:
                project_dir = workspace.parent.absolute()

        if not project_dir.exists():
            raise EnvironmentError(f"Project `{project_dir}` not exists")

        # MOSS.md 是 workspace 级配置, 只在 workspace 内查找
        self._meta = self.find_moss_meta(workspace) or self.create_meta_file(workspace)

        self._project_dir = project_dir
        # 当前进程 id.
        self._self_pid: int = os.getpid()
        self._bootstrapped = False

        self._mode_name = mode or os.environ.get(ENV_MOSS_MODE_KEY, self._meta.default_mode)
        # 为当前启动的实例赋予一个 uid. 通常也可以设置在 cell 上.
        self._session_id = os.environ.get(ENV_SESSION_ID_KEY) or unique_id()
        self._ghost_name = ghost or os.environ.get(ENV_GHOST_NAME_KEY, self._meta.default_ghost)
        self._cell_address = cell_address or os.environ.get(ENV_CELL_ADDRESS_KEY, '')
        self._parent_cell_address = parent_cell_address or os.environ.get(ENV_PARENT_CELL_ADDRESS_KEY, '')
        self._network = network or os.environ.get(ENV_NETWORK_KEY, self._meta.default_network)
        self._network_scope = scope or os.environ.get(ENV_NETWORK_SCOPE_KEY, self._meta.default_network_scope)
        project_id = os.environ.get(ENV_PROJECT_ID_KEY) or self._meta.project_id
        if not project_id:
            project_id = unique_id()
            self._meta.project_id = project_id
            self._meta.write_to_file()
        self._project_id = project_id

    @classmethod
    def find_moss_meta(cls, directory: Path) -> MossMeta | None:
        return MossMeta.read_from_directory(directory)

    @classmethod
    def create_meta_file(cls, directory: Path) -> MossMeta:
        moss_meta = MossMeta()
        moss_meta.write_to_directory(directory)
        return moss_meta

    def bootstrap(self) -> None:
        """
        根据实例化的 Env, 完善进程的运行状态, 让当前进程和子进程可以分享.
        """
        if self._bootstrapped:
            return
        if not self.workspace_path.exists():
            raise EnvironmentError(f"Workspace `{self.workspace_path}` does not exist")

        # 先同步当前状态到 os.environ, 再标记 bootstrapped.
        # 标记后属性走 os.environ, 确保外部运行时修改生效.
        env_data = self.dump_runtime_scope()
        for key, value in env_data.items():
            if value:
                os.environ[key] = str(value)

        global _environment
        _environment = self
        self._bootstrapped = True

    # --- 环境发现逻辑 --- #

    @classmethod
    def discover(cls) -> Self:
        """
        从环境发现中获取进程级单例. 可以在各个模块中共享.
        """
        # Env 对象本质上是进程级别单例.
        global _environment
        # 返回进程级别单例.
        # 或者根据路径发现创建单例.
        if _environment is not None:
            # 在 workspace 中发现 runtime scope.
            return _environment
        env = cls()
        env.bootstrap()
        return env

    @classmethod
    def reset(cls) -> None:
        global _environment
        _environment = None

    @staticmethod
    def set_instance(env: 'Environment'):
        global _environment
        _environment = env

    # --- 暴露属性. --- #
    # bootstrap 前读存储属性, bootstrap 后读 os.environ — 运行时修改 env 立即生效.

    @property
    def mode_name(self) -> str:
        if self._bootstrapped:
            return os.environ.get(ENV_MOSS_MODE_KEY, NONE_MOSS_MODE)
        return self._mode_name

    @property
    def ghost_name(self) -> str:
        if self._bootstrapped:
            return os.environ.get(ENV_GHOST_NAME_KEY, NONE_GHOST_NAME)
        return self._ghost_name

    @property
    def no_ghost(self) -> bool:
        return self.ghost_name == NONE_GHOST_NAME or not self.ghost_name

    @property
    def no_mode(self) -> bool:
        return self.mode_name == NONE_MOSS_MODE or not self.mode_name

    @property
    def network_scope(self) -> str:
        if self._bootstrapped:
            return os.environ.get(ENV_NETWORK_SCOPE_KEY, DEFAULT_NETWORK_SCOPE)
        return self._network_scope

    @property
    def session_id(self) -> str:
        if self._bootstrapped:
            return os.environ.get(ENV_SESSION_ID_KEY, '') or self._session_id
        return self._session_id

    @property
    def project_id(self) -> str:
        if self._bootstrapped:
            return os.environ.get(ENV_PROJECT_ID_KEY, '') or self._project_id
        return self._project_id

    @property
    def network(self) -> str:
        if self._bootstrapped:
            return os.environ.get(ENV_NETWORK_KEY, DEFAULT_NETWORK_NAME)
        return self._network

    @property
    def this_cell_address(self) -> str:
        if self._bootstrapped:
            return os.environ.get(ENV_CELL_ADDRESS_KEY, '')
        return self._cell_address

    @property
    def parent_cell_address(self) -> str:
        if self._bootstrapped:
            return os.environ.get(ENV_PARENT_CELL_ADDRESS_KEY, '')
        return self._parent_cell_address

    @property
    def pid(self) -> int:
        """自身所处的进程 id. """
        return self._self_pid

    # --- 环境变量治理 --- #

    def dump_runtime_scope(self) -> Dict[MOSSRuntimeScopeEnvKey, str]:
        return {
            "MOSS_WORKSPACE": str(self.workspace_path),
            "MOSS_PROJECT_DIR": str(self.project_path),
            "MOSS_MODE_NAME": self.mode_name,
            "MOSS_GHOST_NAME": self.ghost_name,
            "MOSS_CELL_ADDRESS": self.this_cell_address,
            "MOSS_PARENT_CELL_ADDRESS": self.parent_cell_address,
            "MOSS_NETWORK": self.network,
            "MOSS_NETWORK_SCOPE": self.network_scope,
            "MOSS_PROJECT_ID": self.project_id,
        }

    @classmethod
    def runtime_scope_keys(cls) -> list[MOSSRuntimeScopeEnvKey]:
        return [
            "MOSS_WORKSPACE",
            "MOSS_PROJECT_DIR",
            "MOSS_PROJECT_ID",
            "MOSS_MODE_NAME",
            "MOSS_GHOST_NAME",
            "MOSS_CELL_ADDRESS",
            "MOSS_PARENT_CELL_ADDRESS",
            "MOSS_NETWORK",
            "MOSS_NETWORK_SCOPE",
        ]

    @classmethod
    def find_workspace_path(cls) -> Path:
        """
        发现 workspace 的基本方法.
        """
        # 先从环境变量中查找.
        expect_dir = os.environ.get(ENV_WORKSPACE_DIR_KEY, None)
        if expect_dir is not None:
            expect = Path(expect_dir).resolve()
            if not expect.exists():
                # 快速失败, 不要让运行出现约定幻觉.
                raise EnvironmentError(f"Workspace `{expect_dir}` from env `{ENV_WORKSPACE_DIR_KEY}` does not exist")
            return expect.absolute()

        # 从当前目录中查找.
        cwd = Path(os.getcwd())
        expect = cwd.joinpath(DEFAULT_WORKSPACE_DIR_NAME)
        if expect.exists():
            return expect.absolute()

        user_home = Path.home()
        # 从父级目录中查找.
        search_dir = cwd
        while search_dir:
            prev = search_dir
            search_dir = search_dir.parent
            if search_dir == prev:  # 已到根目录, 无法继续向上
                break
            expect = search_dir.joinpath(DEFAULT_WORKSPACE_DIR_NAME)
            if expect.exists():
                return expect.absolute()
            if search_dir == user_home:
                break

        # 最终希望在 cwd 里是项目的根目录.
        return cls.expect_cwd_workspace_path()

    @staticmethod
    def expect_home_workspace_path() -> Path:
        """如果 workspace 在 Home 目录下的位置. """
        return Path.home().joinpath(DEFAULT_WORKSPACE_DIR_NAME)

    @staticmethod
    def expect_cwd_workspace_path() -> Path:
        """如果 workspace 在 cwd 下的预计位置"""
        return Path.cwd().joinpath(DEFAULT_WORKSPACE_DIR_NAME)

    # --- workspace path conventions -- #

    @property
    def workspace_path(self) -> Path:
        """
        返回 workspace path.
        """
        return self._workspace_path

    @property
    def project_path(self) -> Path:
        """
        按约定, project 地址永远是 workspace 的父目录.
        """
        return self._project_dir

    @property
    def moss_meta(self) -> MossMeta:
        return self._meta

    def cell_dirs(self) -> list[Path]:
        """返回 MossMeta 中配置的、实际存在的 cell 发现路径."""
        return self._meta.cell_dirs(self)

    @property
    def default_project_cells_dir(self) -> Path:
        return self.project_path.joinpath(DEFAULT_CELLS_DIR)

    @property
    def default_workspace_cells_dir(self) -> Path:
        return self.workspace_path.joinpath(DEFAULT_CELLS_DIR)

    @property
    def cell_runtimes_dir(self) -> Path:
        return self.workspace_path.joinpath(WORKSPACE_CELL_RUNTIME_DIR)

    @classmethod
    def env_example_file(cls, workspace: Path) -> Path:
        """
        返回环境中的 env example file 预期地址.
        """
        return workspace.joinpath(WORKSPACE_ENV_EXAMPLE_FILENAME)

    @classmethod
    def env_file(cls, workspace: Path) -> Path:
        return workspace.joinpath(WORKSPACE_ENV_FILENAME)

    @property
    def log_config_file(self) -> Path:
        return self._workspace_path / 'configs' / 'logging.yml'

    # --- env attributes -- #

    @classmethod
    def set_mode(cls, mode: str) -> None:
        os.environ[ENV_MOSS_MODE_KEY] = mode

    @classmethod
    def set_network_scope(cls, scope: str) -> None:
        os.environ[ENV_NETWORK_SCOPE_KEY] = scope

    @classmethod
    def set_session_id(cls, session_id: str) -> None:
        os.environ[ENV_SESSION_ID_KEY] = session_id

    @classmethod
    def set_ghost_name(cls, ghost_name: str) -> None:
        os.environ[ENV_GHOST_NAME_KEY] = ghost_name

    def dump_cell_env(
            self,
            *,
            cell_address: str = "",
            parent_cell_address: str = "",
            with_os_env: bool = True,
    ) -> dict[str, str]:
        """
        生成 MOSS 自身环境相关的 env 字典, 用于展示或别的特殊需要.
        """

        data = {
            ENV_CELL_ADDRESS_KEY: cell_address,
            ENV_PARENT_CELL_ADDRESS_KEY: parent_cell_address or self.this_cell_address,
        }
        for key, value in self.dump_runtime_scope().items():
            if key not in data:
                data[key] = value
        #  去掉空值.
        data = {key: val for key, val in data.items() if val}

        if not with_os_env:
            return data
        env_data = os.environ.copy()
        env_data.update(data)
        return env_data

    @staticmethod
    @contextlib.contextmanager
    def fixture():
        global _environment
        # 保留原始值.
        origin = _environment
        origin_env = {}
        for key in Environment.runtime_scope_keys():
            if key in os.environ:
                origin_env[key] = os.environ[key]

        yield
        _environment = origin
        for key in Environment.runtime_scope_keys():
            if key in os.environ:
                os.environ.pop(key)
        os.environ.update(origin_env)

    @staticmethod
    def init_workspace(workspace_dir: Path, force: bool = False) -> None:
        """
        从 Stub Package 初始化工作空间，并设置组共享权限 (Group Writable & Setgid)。

        Args:
            workspace_dir: 目标目录。
            force: 若为 True，覆盖已存在的文件（用于 stub 升级后更新已有 workspace）。
        """
        # 1. 定义权限位
        # 目录权限：rwxrws--- (0o2770) -> 允许组成员读写，且开启 setgid 保证新建文件继承组
        DIR_MODE = stat.S_IRWXU | stat.S_IRWXG | stat.S_ISGID
        # 文件权限：rw-rw---- (0o660)
        FILE_MODE = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP

        # 确保根目录存在并设置权限
        if not workspace_dir.exists():
            workspace_dir.mkdir(parents=True, exist_ok=True)

        # 强制更新根目录权限（确保即便目录已存在，权限也是正确的）
        os.chmod(workspace_dir, DIR_MODE)

        stub_resources = resources.files(WORKSPACE_STUB_PACKAGE)

        def copy_recursive(source_node, target_dir: Path):
            for item in source_node.iterdir():
                if source_node == stub_resources and item.name == "__init__.py": continue
                target_item = target_dir / item.name

                if item.is_dir():
                    if not target_item.exists():
                        target_item.mkdir(exist_ok=True)
                    # 为子目录设置权限
                    os.chmod(target_item, DIR_MODE)
                    copy_recursive(item, target_item)
                else:
                    if force or not target_item.exists():
                        target_item.write_bytes(item.read_bytes())
                        os.chmod(target_item, FILE_MODE)

        copy_recursive(stub_resources, workspace_dir)


_environment: Environment | None = None
