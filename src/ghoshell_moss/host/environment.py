"""
MOSS 环境发现的关键常量.
只保留几个最核心的常量.
"""

from typing import Literal
from typing_extensions import Self
from pathlib import Path
from ghoshell_common.helpers import uuid
from importlib import resources
from pydantic import BaseModel, Field
from ghoshell_moss.core.ctml.meta import CTML_VERSION
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
    # env keys
    'ENV_WORKSPACE_DIR_KEY',
    'ENV_SESSION_ID_KEY',
    'ENV_PARENT_PID_KEY',
    'ENV_GHOST_NAME_KEY',
    'ENV_CELL_NAME_KEY',
    'ENV_MOSS_MODE_KEY',
    'MOSSEnvKey',

    # stubs
    'MODE_STUB_PACKAGE',
    'APP_STUB_PACKAGE',
    'WORKSPACE_STUB_PACKAGE',

    # dir path
    'WORKSPACE_SOURCE_DIR',
    'META_INSTRUCTION_FILENAME',
    'WORKSPACE_ENV_FILENAME',
    'WORKSPACE_ENV_EXAMPLE_FILENAME',
]

from ghoshell_moss import TopicModel
from ghoshell_moss.contracts.configs import ConfigType

# --- moss 的 workspace 发现机制 --- #

# moss 默认的 workspace 文件夹名.
# workspace 的绝对路径优先从环境变量寻找, 找不到时按目录发现机制寻找.
# 路径发现的逻辑是: os getcwd 下, 递归搜索父级目录下, home 目录下.
DEFAULT_WORKSPACE_DIR_NAME = '.moss_ws'
META_INSTRUCTION_FILENAME = 'MOSS.md'

# env 文件名. workspace 启动时会从其目录下读取环境变量文件 (by loadenv)
WORKSPACE_ENV_FILENAME = '.env'
WORKSPACE_ENV_EXAMPLE_FILENAME = '.env.example'

# 源码预期所在的目录.
WORKSPACE_SOURCE_DIR = 'src'

# --- stubs --- #
# workspace 的原始文件所处的 package 路径.
WORKSPACE_STUB_PACKAGE = 'ghoshell_moss.host.stubs.workspace'
APP_STUB_PACKAGE = 'ghoshell_moss.host.stubs.app'
MODE_STUB_PACKAGE = 'ghoshell_moss.host.stubs.mode'

# --- 主要的环境变量名 --- #
# 这些环境变量不在 .env 中定义, 而是启动时 发现/生成, 或者通过父子进程传递的.

# 从环境变量中获取 moss workspace 路径的环境变量名.
ENV_WORKSPACE_DIR_KEY = 'MOSS_WORKSPACE'

# 环境变量中获取 MOSS 运行时的 SESSION ID.
# 所有通过 MOSS 架构共享本地通讯的 channel 或 topic, 都需要归属到相同的 session id 上.
ENV_SESSION_ID_KEY = 'MOSS_SESSION_ID'

ENV_MOSS_MODE_KEY = 'MOSS_MODE_NAME'
DEFAULT_MOSS_MODE = "default"

# 如果当前 MOSS 实例启动时, 启用了 Ghost, 则 GHOST_NAME 不应该为空.
ENV_GHOST_NAME_KEY = 'MOSS_GHOST_NAME'

ENV_PARENT_PID_KEY = 'MOSS_PARENT_PID'

ENV_CELL_NAME_KEY = 'MOSS_CELL_NAME'

MOSSEnvKey = Literal[
    "MOSS_WORKSPACE", "MOSS_SESSION_ID", "MOSS_MODE_NAME",
    "MOSS_GHOST_NAME", "MOSS_PARENT_PID", "MOSS_CELL_NAME"
]


class MetaInstruction(BaseModel):
    ctml_version: str = Field(
        default=CTML_VERSION,
        description="当前 MOSS 使用的提示词版本. 如果为空的话, 会忽略提示词."
    )
    content: str = Field(
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
        data['content'] = post.content
        return cls(**data)

    def get_meta_instruction(self) -> str:
        """
        获取 moss 的元提示词.
        """
        from ghoshell_moss.core.ctml.meta import get_moss_ctml_meta_instruction
        meta_instruction = ""
        if self.ctml_version:
            meta_instruction = get_moss_ctml_meta_instruction(self.ctml_version)
        return "\n\n".join([meta_instruction, self.content])

    def __str__(self):
        return self.get_meta_instruction()


class Environment:
    """
    MOSS Process Level Environment discover
    """

    def __init__(
            self,
            workspace_path: Path,
            ghost_name: str | None = None,
            moss_import_path: str | None = None,
            session_id: str | None = None,
            mode: str | None = None,
    ):
        """
        初始化 MOSS 的进程级别环境发现.
        """
        self._workspace_path = workspace_path
        self._env_file = self._workspace_path.joinpath(WORKSPACE_ENV_FILENAME)
        self._source_path = self._workspace_path.joinpath(WORKSPACE_SOURCE_DIR)
        self._meta_instruction_path = self._workspace_path.joinpath(META_INSTRUCTION_FILENAME)
        if self._meta_instruction_path.is_file() and self._meta_instruction_path.exists():
            self._meta_instruction = MetaInstruction.from_file(self._meta_instruction_path)
        else:
            self._meta_instruction = MetaInstruction()

        if mode is None:
            mode = os.environ.get(ENV_MOSS_MODE_KEY, DEFAULT_MOSS_MODE)
        self._moss_mode = mode

        # 永远要有正确的 session id.
        session_id = session_id or os.environ.get(ENV_SESSION_ID_KEY, None)
        if session_id is None:
            session_id = uuid()
        self._session_id = session_id

        self._node_name: str = os.environ.get(ENV_CELL_NAME_KEY, "")

        # 为空表示运行时不启用 ghost.
        self._ghost_name: str = ghost_name or os.environ.get(ENV_GHOST_NAME_KEY, '')

        self._self_pid: int = os.getpid()
        self._parent_pid: int = int(os.environ.get(ENV_PARENT_PID_KEY, 0))
        self._bootstrapped = False

    @classmethod
    def discover(cls) -> Self:
        """
        从环境发现中获取进程级单例. 可以在各个模块中共享.
        """
        global _environment
        # 返回进程级别单例.
        # 或者根据路径发现创建单例.
        if _environment is None:
            workspace_path = cls.find_workspace_path()
            _environment = cls(workspace_path)
        return _environment

    def dump_moss_env(self, *, cell_name: str = "", for_child_process: bool = False) -> dict[MOSSEnvKey, str]:
        """
        生成 MOSS 自身环境相关的 env 字典, 通常用于子进程做发现.
        """
        data: dict[MOSSEnvKey, str] = {
            "MOSS_WORKSPACE": str(self._workspace_path) if self._workspace_path.exists() else "",
            "MOSS_SESSION_ID": self._session_id,
            "MOSS_GHOST_NAME": self._ghost_name,
            "MOSS_MODE_NAME": self._moss_mode,
        }
        if cell_name:
            data["MOSS_CELL_NAME"] = cell_name
        if for_child_process:
            data["MOSS_PARENT_PID"] = str(self._self_pid)
        return data

    @classmethod
    def set_singleton(cls, instance: Self) -> None:
        """
        重置进程级单例.
        """
        global _environment
        _environment = instance

    def bootstrap(self) -> None:
        """
        初始化启动.
        """
        if self._bootstrapped:
            return
        self._bootstrapped = True
        if not self.workspace_path.exists():
            # 初始化 workspace.
            # 如果 workspace 不存在的话.
            # 启动脚本应该提示用户
            raise EnvironmentError(f"Workspace `{self.workspace_path}` does not exist")

        env_file = self.env_file
        # 确认加载一次环境变量.
        if env_file is not None:
            dotenv.load_dotenv(env_file)

        # 确认路径被正确加载.
        source_path = self.source_dir
        if source_path is not None:
            abs_source_path = str(source_path.absolute())
            # 加载路径.
            if abs_source_path not in sys.path:
                sys.path.append(abs_source_path)

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
            if search_dir.joinpath(META_INSTRUCTION_FILENAME).exists():
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
    def init_workspace(workspace_dir: Path) -> None:
        """
        从 Stub Package 初始化工作空间，并设置组共享权限 (Group Writable & Setgid)。
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
                    if not target_item.exists():
                        target_item.write_bytes(item.read_bytes())
                        # 为新写入的文件设置权限
                        os.chmod(target_item, FILE_MODE)

        copy_recursive(stub_resources, workspace_dir)

    @property
    def workspace_path(self) -> Path:
        """
        返回 workspace path.
        """
        return self._workspace_path

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
    def pid(self) -> int:
        return self._self_pid

    @property
    def parent_pid(self) -> int:
        return self._parent_pid

    @property
    def moss_mode(self) -> str:
        return self._moss_mode

    @property
    def meta_instruction_file(self) -> Path:
        return self._meta_instruction_path.absolute()

    @property
    def meta_instruction(self) -> MetaInstruction:
        return self._meta_instruction

    @staticmethod
    def expect_home_workspace_path() -> Path:
        return Path.home().joinpath(DEFAULT_WORKSPACE_DIR_NAME)

    @staticmethod
    def expect_cwd_workspace_path() -> Path:
        return Path.cwd().joinpath(DEFAULT_WORKSPACE_DIR_NAME)

    @property
    def session_id(self) -> str:
        """
        返回当前这次请求的 session id.
        """
        return self._session_id

    @property
    def source_dir(self) -> Path | None:
        """
        返回 workspace 中的 source 所在目录. 方便添加到 sys.paths.
        """
        if self._source_path.exists():
            return self._source_path.absolute()
        return None


_environment: Environment | None = None
