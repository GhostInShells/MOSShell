"""
MOSS 环境发现的关键常量.
只保留几个最核心的常量.
"""

from typing import Literal, TypeAlias, Dict
from typing_extensions import Self
from pathlib import Path
from ghoshell_common.contracts import config_logger_from_yaml
from importlib import resources
import logging
from pydantic import BaseModel, Field
from ghoshell_moss.contracts.workspace import Workspace, LocalWorkspace, Storage
from ghoshell_moss.core.ctml.versions import (
    CTML_VERSION, search_version_file_in_dir, default_moss_ctml_meta_instruction_directory,
    get_version_from_filename,
)
from ghoshell_moss.message import unique_id
import os
import dotenv
import sys
import stat

__all__ = [
    'Environment',
    # workspace
    'DEFAULT_WORKSPACE_DIR_NAME',
    'WORKSPACE_ENV_FILENAME',
    'WORKSPACE_ENV_EXAMPLE_FILENAME',
    'DEFAULT_SESSION_SCOPE',
    # env keys
    'ENV_WORKSPACE_DIR_KEY',
    'ENV_SESSION_SCOPE_KEY',
    'ENV_SESSION_ID_KEY',
    'ENV_GHOST_NAME_KEY',
    'ENV_MOSS_MODE_KEY',
    'ENV_MOSS_HOST_PID_KEY',
    'ENV_CELL_ADDRESS_KEY',
    'ENV_PARENT_CELL_ADDRESS_KEY',

    'MOSSEnvConfigKey',
    'MOSSRuntimeScopeEnvKey',
    'MOSSSpawnCellEnvKey',

    "MossMeta",
    "RuntimeScope",

    # stubs
    'MODE_STUB_PACKAGE',
    'APP_STUB_PACKAGE',
    'WORKSPACE_STUB_PACKAGE',

    # dir path
    'META_CONFIG_FILENAME',
    'WORKSPACE_ENV_FILENAME',
    'WORKSPACE_ENV_EXAMPLE_FILENAME',
]

# --- moss 的 workspace 发现机制 --- #

# moss 默认的 workspace 文件夹名.
# workspace 的绝对路径优先从环境变量寻找, 找不到时按目录发现机制寻找.
# 路径发现的逻辑是: os getcwd 下, 递归搜索父级目录下, home 目录下.
DEFAULT_WORKSPACE_DIR_NAME = '.moss_ws'
META_CONFIG_FILENAME = 'MOSS.md'

# env 文件名. workspace 启动时会从其目录下读取环境变量文件 (by loadenv)
WORKSPACE_ENV_FILENAME = '.env'
WORKSPACE_ENV_EXAMPLE_FILENAME = '.env.example'

# --- stubs --- #
# workspace 的原始文件所处的 package 路径.
WORKSPACE_STUB_PACKAGE = 'ghoshell_moss.host.stubs.workspace'
APP_STUB_PACKAGE = 'ghoshell_moss.host.stubs.app'
MODE_STUB_PACKAGE = 'ghoshell_moss.host.stubs.mode'

# --- 主要的环境变量名 --- #
# 这些环境变量不在 .env 中定义, 而是启动时 发现/生成, 或者通过父子进程传递的.

# 从环境变量中获取 moss workspace 路径的环境变量名.
ENV_WORKSPACE_DIR_KEY = 'MOSS_WORKSPACE'

# moss 环境配置文件所在的路径.
# 影响 MOSS
ENV_SOURCE_DIR_KEY = 'MOSS_SOURCE_DIR'

# moss session scope 的环境变量 key. session scope 用于所有通讯协议的隔离.
ENV_SESSION_SCOPE_KEY = 'MOSS_SESSION_SCOPE'
DEFAULT_SESSION_SCOPE = 'default'

# 环境变量中获取 MOSS 运行时的 SESSION ID.
ENV_SESSION_ID_KEY = 'MOSS_SESSION_ID'

ENV_MOSS_MODE_KEY = 'MOSS_MODE_NAME'
DEFAULT_MOSS_MODE = "default"

# 如果当前 MOSS 实例启动时, 启用了 Ghost, 则 GHOST_NAME 不应该为空.
ENV_GHOST_NAME_KEY = 'MOSS_GHOST_NAME'
# none 表示没有 Ghost 在运行.
DEFAULT_GHOST_NAME = "none"

ENV_MOSS_HOST_PID_KEY = 'MOSS_HOST_PID'

ENV_CELL_ADDRESS_KEY = 'MOSS_CELL_ADDRESS'
ENV_PARENT_CELL_ADDRESS_KEY = 'MOSS_PARENT_ADDRESS'

# 与运行配置项有关的 Env Key
MOSSEnvConfigKey: TypeAlias = Literal[
    "MOSS_WORKSPACE",
    "MOSS_SOURCE_DIR",
]

# 与运行时状态有关的 Env Key
MOSSRuntimeScopeEnvKey: TypeAlias = Literal[
    "MOSS_MODE_NAME",
    "MOSS_GHOST_NAME",
    "MOSS_SESSION_SCOPE",
    "MOSS_SESSION_ID",
    "MOSS_HOST_PID",
]

# Spawn 一个子进程 Cell 使用的 Key.
MOSSSpawnCellEnvKey: TypeAlias = Literal[
    "MOSS_MODE_NAME",
    "MOSS_GHOST_NAME",
    "MOSS_SESSION_SCOPE",
    "MOSS_SESSION_ID",
    "MOSS_HOST_PID",
    'MOSS_PARENT_CELL_ADDRESS',
    'MOSS_CELL_ADDRESS',
]


class RuntimeScope(BaseModel):
    """
    MOSS 的运行时状态.
    在 MOSS 架构中, 所有的节点 (CELL) 都基于 RuntimeScope 构建自身, 包括:
    1. 通讯网络 (session scope)
    2. 数据的不同隔离级别.
    3. 进程生命周期的治理 (host pid).
    4. Workspace 内资源和依赖声明的隔离 (Mode)
    5. Ghost 的运行状态.

    Host 节点应该要创建 RuntimeScope, Worker 节点应该从文件中读取它作为唯一信源; 读取不到则从环境变量中获取.

    """
    source: Literal['workspace', 'env', ''] = Field(
        default='',
        description="标记 scope 如何被创建. env 是从环境变量读取, workspace 是从 workspace 读取. 默认是手动创建. ",
    )
    session_scope: str = Field(
        default=DEFAULT_SESSION_SCOPE,
        description="通讯隔离 scope. 所有 Session 通讯协议都会在同一个 Scope 下.",
    )
    session_id: str = Field(
        default_factory=unique_id,
        description="Session 为每一次重新运行独立准备的隔离级别.",
    )
    mode_name: str = Field(
        default=DEFAULT_MOSS_MODE,
        description="当前 mode 名称, 用于管理不同的 mode 资源. ",
    )
    ghost_name: str = Field(
        default=DEFAULT_GHOST_NAME,
        description="当前运行的 ghost 名称. ",
    )
    host_pid: int = Field(
        default=0,
        description="host 进程 PID，用于存活验证与运维诊断",
    )

    @classmethod
    def new(
            cls,
            *,
            mode_name: str = '',
            ghost_name: str = '',
            host_pid: int = 0,
            session_scope: str = '',
            session_id: str = '',
    ) -> 'RuntimeScope':
        """通过入参的方式构建 RuntimeScope. """
        mode_name = mode_name or os.environ.get(ENV_MOSS_MODE_KEY, DEFAULT_MOSS_MODE)
        ghost_name = ghost_name or os.environ.get(ENV_GHOST_NAME_KEY, DEFAULT_GHOST_NAME)
        session_scope = session_scope or os.environ.get(ENV_SESSION_SCOPE_KEY, DEFAULT_SESSION_SCOPE)
        session_id = session_id or os.environ.get(ENV_SESSION_ID_KEY) or unique_id()
        host_pid = host_pid
        if host_pid == 0:
            if val := os.environ.get(ENV_MOSS_HOST_PID_KEY):
                host_pid = int(val)
        return RuntimeScope(
            source='',
            mode_name=mode_name,
            ghost_name=ghost_name,
            host_pid=host_pid,
            session_scope=session_scope,
            session_id=session_id,
        )

    def write_to_directory(self, directory: Path) -> None:
        content = self.model_dump_json(indent=2, exclude_none=True, ensure_ascii=False)
        file = directory / 'runtime_scope.json'
        file.write_text(content)

    @classmethod
    def read_from_directory(cls, directory: Path) -> 'RuntimeScope | None':
        file = directory / 'runtime_scope.json'
        if not file.exists():
            return None
        try:
            content = file.read_text()
            return cls.model_validate_json(content)
        except Exception:
            return None

    @classmethod
    def create_from_env(
            cls,
            env_data: Dict[str, str] | None = None,
    ) -> 'RuntimeScope':
        env_data = env_data or os.environ.copy()
        data = {}
        if val := env_data.get(ENV_MOSS_MODE_KEY):
            data['mode_name'] = val
        if val := env_data.get(ENV_GHOST_NAME_KEY):
            data['ghost_name'] = val
        if val := env_data.get(ENV_MOSS_HOST_PID_KEY):
            data['host_pid'] = int(val) if val else 0
        if val := env_data.get(ENV_SESSION_SCOPE_KEY):
            data['session_scope'] = val
        if val := env_data.get(ENV_SESSION_ID_KEY):
            data['session_id'] = val
        data['source'] = 'env'
        return cls(**data)

    def dump_env_data(self) -> Dict[MOSSRuntimeScopeEnvKey, str]:
        return {
            "MOSS_MODE_NAME": self.mode_name,
            "MOSS_GHOST_NAME": self.ghost_name,
            "MOSS_SESSION_SCOPE": self.session_scope,
            "MOSS_SESSION_ID": self.session_id,
            "MOSS_HOST_PID": str(self.host_pid),
        }


class MossMeta(BaseModel):
    """
    MOSS 的元信息配置.
    通过 workspace 的 MOSS.md 读取.
    """

    name: str = Field(
        default='moss',
        description="为当前 moss 环境命名. 建议给环境特殊的名字, 因为可以通过分形组网, 让多个 host 互相联通.",
    )
    description: str = Field(
        default="default moss discovered in host workspace",
        description="描述当前 moss 环境, 这样当这个 moss 环境提供给远程 moss 环境时, 对方可以通过命名识别自己. ",
    )
    ctml_version: str = Field(
        default=CTML_VERSION,
        description="当前 MOSS 默认使用的提示词版本."
    )
    default_mode: str = Field(
        default=DEFAULT_MOSS_MODE,
        description="启动时默认的模式",
    )
    default_session_scope: str = Field(
        default='',
    )
    system_prompt: str = Field(
        default="",
        description="补充到 CTML meta instruction 后面的内容. version 为空, 这里应该包含完整的 meta instruction"
    )

    @classmethod
    def from_file(cls, file: Path) -> Self:
        """
        从文件中读取 meta instruction.
        """
        import frontmatter
        post = frontmatter.load(str(file.absolute()))
        data = post.metadata
        data['system_prompt'] = post.content
        return cls(**data)


class Environment:
    """
    MOSS 的环境配置和发现体系.
    根据文件目录约定和环境变量发现, 完成初始化讯息.
    """

    def __init__(
            self,
            # workspace 是唯一必要的参数.
            workspace: Workspace,
            runtime_scope: RuntimeScope | None = None,
            *,
            # --- 可以显式传入的参数 --- #
            env_file: Path | None = None,
            source_dir: Path | None = None,
    ):
        """
        初始化 MOSS 的进程级别环境发现.
        """
        # 当前进程 id.
        self._self_pid: int = os.getpid()
        self._workspace_path = workspace.root_path()
        self._workspace = workspace
        self._runtime_registry_dir = self.get_runtime_registry_dir(workspace)
        self._bootstrapped = False

        self._configured_source_dir: Path | None = source_dir

        # 默认是 {workspace}/MOSS.md
        self._meta_config_path = self._workspace_path.joinpath(META_CONFIG_FILENAME)
        if self._meta_config_path.is_file() and self._meta_config_path.exists():
            self._moss_meta_config = MossMeta.from_file(self._meta_config_path)
        else:
            self._moss_meta_config = MossMeta()

        self._modes_storage = self._workspace.root().sub_storage('modes')
        self._ghosts_storage = self._workspace.root().sub_storage('ghosts')
        self._workspace_cell_registry = self._workspace.root().sub_storage('cells')

        # 筹备 runtime scope — 必须在 env_file 选择之前,
        # 因为 env_file 依赖 moss_mode_name.
        if runtime_scope is None:
            runtime_scope = RuntimeScope.read_from_directory(self._runtime_registry_dir)
        if runtime_scope is None:
            runtime_scope = RuntimeScope.create_from_env()
        self._runtime_scope: RuntimeScope = runtime_scope

        # 初始化环境变量文件 — 依赖 runtime_scope.mode_name.
        if env_file is None:
            env_mode_filename = f'.env.{self.moss_mode_name}'
            env_mode_file = self.workspace_path.joinpath(env_mode_filename)
            if env_mode_file.exists():
                env_file = env_mode_file
            else:
                env_file = self.workspace_path.joinpath('.env')
        self._env_file = env_file


    def bootstrap(self) -> None:
        """
        根据实例化的 Env, 完善进程的运行状态, 让当前进程和子进程可以分享.
        """
        if self._bootstrapped:
            return
        self._bootstrapped = True
        if not self.workspace_path.exists():
            raise EnvironmentError(f"Workspace `{self.workspace_path}` does not exist")

        # 如果环境变量文件存在, 加载它.
        # workspace 的环境变量
        env_file = self.env_file
        if env_file is not None and env_file.exists():
            dotenv.load_dotenv(env_file)

        # 按约定加载 logging 配置: workspace/configs/logging.yml
        logging_config = self.log_config_file
        if logging_config.exists():
            config_logger_from_yaml(str(logging_config))

        # 初始化 src 路径.
        source_dir = self._configured_source_dir
        if source_dir is None:
            source_dir = os.environ.get(ENV_SOURCE_DIR_KEY) or self._workspace.source().abspath()
        # 加载 source 里的数据.
        if source_dir.exists() and source_dir.is_dir():
            abs_source_path = str(source_dir.absolute())
            if abs_source_path not in sys.path:
                sys.path.append(abs_source_path)
            self._configured_source_dir = source_dir

        # 更新当前运行状态的环境变量.
        env_data = self._runtime_scope.dump_env_data()
        os.environ[ENV_WORKSPACE_DIR_KEY] = str(self._workspace.root_path())
        os.environ[ENV_SOURCE_DIR_KEY] = str(source_dir)

        for key, value in env_data.items():
            if value:
                os.environ[key] = str(value)

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
        if _environment is None:
            # 通过 workspace 实现初始化.
            workspace_path = cls.find_workspace_path()
            if not workspace_path.exists():
                raise EnvironmentError(f"Expected workspace `{workspace_path}` not exists")
            workspace = LocalWorkspace(workspace_path)
            # 在 workspace 中发现 runtime scope.
            _environment = cls(workspace)
        return _environment

    @classmethod
    def get_runtime_registry_dir(cls, workspace: Workspace) -> Path:
        return workspace.root().sub_storage('scopes').abspath()

    @staticmethod
    def find_workspace_path() -> Path:
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
        while search_dir != user_home:
            if search_dir.joinpath(META_CONFIG_FILENAME).exists():
                # 返回找得到 MOSS.md 文件的目录作为 workspace 根目录.
                # 对于将 workspace 作为 project 使用的场景, 这样比较方便.
                return search_dir.absolute()
            search_dir = search_dir.parent
            expect = search_dir.joinpath(DEFAULT_WORKSPACE_DIR_NAME)
            if expect.exists():
                return expect.absolute()

        # 从 USER HOME 中按约定返回, 默认路径在 USER HOME.
        expect = user_home.joinpath(DEFAULT_WORKSPACE_DIR_NAME)
        return expect.absolute()

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
    def workspace(self) -> Workspace:
        return self._workspace

    @property
    def env_file(self) -> Path:
        """
        返回 workspace 中的 env 文件.
        """
        return self._env_file.absolute()

    @property
    def env_example_file(self) -> Path:
        """
        返回环境中的 env example file 预期地址.
        """
        return self._workspace_path.joinpath(WORKSPACE_ENV_EXAMPLE_FILENAME)

    @property
    def moss_meta_file(self) -> Path:
        return self._meta_config_path.absolute()

    @property
    def log_config_file(self) -> Path:
        return self._workspace_path / 'configs' / 'logging.yml'

    @property
    def modes_storage(self) -> Storage:
        """所有模式的默认路径."""
        return self._modes_storage

    @property
    def ghosts_storage(self) -> Storage:
        """所有 ghosts 持久存储的默认路径. """
        return self._ghosts_storage

    @property
    def workspace_cell_registry(self) -> Storage:
        """所有对于模型可见的 Cell 存储路径. """
        return self._workspace_cell_registry

    @property
    def runtime_registry_dir(self) -> Path:
        """运行时用来管理所有运行时状态文件, 比如 cell 文件地址. """
        return self._runtime_registry_dir

    # --- env attributes -- #

    @property
    def pid(self) -> int:
        """自身所处的进程 id. """
        return self._self_pid

    @property
    def parent_cell_address(self) -> str:
        return os.environ.get(ENV_PARENT_CELL_ADDRESS_KEY, '')

    @property
    def this_cell_address(self) -> str:
        return os.environ.get(ENV_CELL_ADDRESS_KEY, '')

    @property
    def host_pid(self) -> int:
        return self._runtime_scope.host_pid

    @property
    def moss_mode_name(self) -> str:
        return self._runtime_scope.mode_name

    @property
    def moss_meta(self) -> MossMeta:
        return self._moss_meta_config

    @property
    def session_scope(self) -> str:
        """
        返回当前的通讯隔离状态.
        """
        return self._runtime_scope.session_scope

    @property
    def session_id(self) -> str:
        return self._runtime_scope.session_id

    @property
    def ghost_name(self) -> str:
        return self._runtime_scope.ghost_name

    @property
    def runtime_scope(self) -> RuntimeScope:
        return self._runtime_scope

    @classmethod
    def set_mode(cls, mode: str) -> None:
        os.environ[ENV_MOSS_MODE_KEY] = mode

    @classmethod
    def set_session_scope(cls, session_scope: str) -> None:
        os.environ[ENV_SESSION_SCOPE_KEY] = session_scope

    @classmethod
    def set_session_id(cls, session_id: str) -> None:
        os.environ[ENV_SESSION_ID_KEY] = session_id

    @classmethod
    def set_ghost_name(cls, ghost_name: str) -> None:
        os.environ[ENV_GHOST_NAME_KEY] = ghost_name

    @property
    def logger(self) -> logging.Logger:
        self.bootstrap()
        return logging.getLogger('moss')

    def ctml_prompts_dir(self) -> Path:
        """
        环境中约定的 ctml versions 配置.
        """
        return self.workspace_path.joinpath("ctml_versions")

    def ctml_versions(self) -> dict[str, Path]:
        """
        当前环境中配置的 ctml versions.
        """
        versions = search_version_file_in_dir(default_moss_ctml_meta_instruction_directory())
        version_name_to_files = {}
        for version_file in versions:
            version_name = get_version_from_filename(version_file.name)
            version_name_to_files[version_name] = version_file
        for version_file in search_version_file_in_dir(self.ctml_prompts_dir()):
            version_name = get_version_from_filename(version_file.name)
            version_name_to_files[version_name] = version_file
        return version_name_to_files

    def dump_moss_env(
            self,
            *,
            cell_address: str = "",
            parent_cell_address: str = "",
            with_os_env: bool = True,
    ) -> dict[str, str]:
        """
        生成 MOSS 自身环境相关的 env 字典, 用于展示或别的特殊需要.
        """
        data: dict[str, str] = self._runtime_scope.dump_env_data()
        data[ENV_WORKSPACE_DIR_KEY] = str(self.workspace_path)

        if self._configured_source_dir:
            data[ENV_SOURCE_DIR_KEY] = str(self._configured_source_dir)

        if cell_address:
            data[ENV_CELL_ADDRESS_KEY] = cell_address
        if parent_cell_address:
            data[ENV_PARENT_CELL_ADDRESS_KEY] = parent_cell_address

        if not with_os_env:
            return data
        env_data = os.environ.copy()
        env_data.update(data)
        return env_data

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
