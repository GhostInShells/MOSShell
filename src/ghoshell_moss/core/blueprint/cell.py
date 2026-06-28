"""
MOSS 通讯网络中的单元节点

MOSS 作为面向模型的操作系统, 通过 Cell 管理其中的节点.
Cell 的定位类似 Node 之于 ROS, 技术核心差异是 Cell 是面向模型的, 需要能够被模型感知其存在.
不叫做 Node, 是因为不同类型的 Cell 节点拥有不同的网络角色定位. 比如主运行时单点叫 Host, 分形的 Host 组网时, 远程 Host 成为 Fractal.
这些节点在功能上并不是平级的, 所以不叫做 Node. 以及, 一个 Cell 可能通过某种机制管理其子节点, 是树上的分支.
每种子节点都有不同的发现和声明机制, 需要用统一的方式被运行时的智能模型感知其存在. Gemini3 对这种机制取名 'Cell'
"""
import time
from typing import Literal, Iterable, ClassVar, Callable
from typing_extensions import Self
from abc import ABC, abstractmethod
from enum import Enum
from pydantic import Field, BaseModel
from ghoshell_moss.core.concepts.channel import ChannelProxy, ChannelProvider, Channel
from ghoshell_moss.core.blueprint.environment import Environment
from ghoshell_moss.message import unique_id
import asyncio
import os
import sys
from pathlib import Path
import fnmatch
import frontmatter
import psutil
import shlex

__all__ = [
    'CellType',
    'CellMetadata',
    'CellLauncher',
    'CellManifest',
    'CellStatus',
    'Cell',
    'CellRegistry',
    'CellNetwork',
    'make_address',
    'CellAddress',
    'MatchPattern',
    'RelativePath',
    'normalize',
    'DuplicatedError',
]

CellAddress = str


class CellType(str, Enum):
    """
    MOSS 系统对 Cell 的默认约定.
    每种不同类型 (type) 的 cell 都可能对应不同的角色, 发现 & 声明 & 运行时机制.
    host 是 Matrix 网络中的主节点, worker 是普通功能节点.
    """
    host = 'host'
    # 普通的功能型节点.
    worker = 'worker'


# -- 类型别名 -- #

RelativePath = str
MatchPattern = str
"""通配符模式: group/name, group/*, *, */*, */name"""


class CellMetadata(BaseModel):
    """
    一个 Cell 节点的元信息, 通常用于环境发现和运行时声明.
    """
    type: CellType | str = Field(
        default=CellType.worker.value,
        description="节点的类型. 保留可扩展空间."
    )
    name: str = Field(
        description="节点的名字, 在类型下应该是唯一的. "
    )
    singleton: bool = Field(
        default=True,
        description="是否是运行时单例, 决定 network 会如何处理它的存在. "
    )
    description: str = Field(
        default='',
        description="节点的描述信息. "
    )
    channel: bool = Field(
        default=False,
        description="声明此 Cell 提供 Channel. 启动后可由 Host 自动创建 proxy. "
    )

    @classmethod
    def from_proc(cls, name: str = '', *, description: str = '') -> 'CellMetadata':
        """生成一个 worker 节点的元信息."""
        from importlib import import_module
        main = import_module('__main__')
        script_file = Path(main.__file__)
        if not name:
            name = script_file.name.split('.')[0]
        if not description:
            docstring = main.__doc__ or ''
            description = docstring.splitlines()[0] if docstring else ''
        return cls(
            type=CellType.worker,
            name=name,
            description=description,
            singleton=False,
        )


def make_address(*parts: str) -> str:
    """约定定义一个节点的 address. 使用 / 作为层级分隔符."""
    return '/'.join(parts)


def normalize(name_or_address: str) -> str:
    return (name_or_address.replace('/', '_').replace('\\', '_').
            replace('.', '_').replace('-', '_'))


class CellLauncher(BaseModel):
    """
    节点启动参数. 描述如何启动一个 cell 进程.
    """

    cwd: str = Field(
        default='',
        description="节点启动时所在的工作路径. "
                    "默认从发现 CELL.md 的路径作为 cwd, 否则从运行时 cwd / launcher.cwd"
    )
    interpreter: Literal['python', ''] | str = Field(
        default='',
        description="使用约定的运行时"
                    "为 `python` 表示使用 sys.executable(父进程) or 环境发现的解释器;"
                    "非空时请使用 cwd 相对路径, 或绝对路径指定解释器. "
    )
    cmd: str = Field(
        default='',
        description="启动节点对应的脚本. 具体路径是 cwd / script"
    )
    arguments: str = Field(
        default='',
        description="启动节点时传入的参数",
    )

    extra_env: dict[str, str] = Field(
        default_factory=dict,
        description="额外写入环境变量的讯息.",
    )

    @classmethod
    def new_empty(cls) -> 'CellLauncher':
        return cls(cwd='', interpreter='', cmd='', arguments='')

    @property
    def cwd_path(self) -> Path:
        if self.cwd == '':
            return Path.cwd()
        return Path(self.cwd).resolve()

    @classmethod
    def from_proc(cls) -> 'CellLauncher':
        """从当前进程中还原启动参数."""
        from importlib import import_module
        main = import_module('__main__')
        script_file = main.__file__
        cwd = Path.cwd()
        script_relative = Path(script_file).absolute()
        _args = str(shlex.join(sys.argv[1:]))
        return cls(
            cwd=str(cwd),
            interpreter=sys.executable,
            cmd=str(script_relative),
            arguments=_args,
        )


class CellManifest(BaseModel):
    """Cell 声明文件 (CELL.md) 的内容载体."""

    MANIFEST_FILENAME: ClassVar[str] = 'CELL.md'
    INSTALL_FILENAME: ClassVar[str] = 'INSTALL.md'
    INSTALLED_FILE: ClassVar[str] = '.installed'

    type: CellType | str = Field(
        default=CellType.worker.value,
        description="节点的类型. 保留可扩展空间."
    )
    name: str = Field(
        description="节点的名字, 在类型下应该是唯一的. "
    )
    singleton: bool = Field(
        default=True,
        description="是否是运行时单例, 可以约束 project 内部的启动个数. "
                    "对于不可重复启动的资源, 比如 robot 控制, singleton 设置为 True."
    )
    description: str = Field(
        default='',
        description="节点的描述信息. "
    )

    launcher: CellLauncher = Field(description="节点启动参数")
    instruction: str = Field(default='', description="节点的详细使用说明. 启动节点后理论要返回. ")
    installed: bool = Field(
        default=True,
        description="Cell 是否已完成安装. 无 INSTALL.md 时默认已安装; "
                    "有 INSTALL.md 时从 .installed 文件推导. "
                    "不在 CELL.md frontmatter 中.",
    )

    @classmethod
    def new(
            cls,
            *,
            meta: CellMetadata,
            launcher: CellLauncher,
            instruction: str = '',
            installed: bool = True,
    ) -> 'CellManifest':
        return cls(
            type=meta.type,
            name=meta.name,
            singleton=meta.singleton,
            description=meta.description,
            launcher=launcher,
            instruction=instruction,
            installed=installed,
        )

    def meta(self) -> CellMetadata:
        data = self.model_dump(
            exclude_defaults=True, exclude_none=True,
            include={'type', 'name', 'singleton', 'description'},
        )
        return CellMetadata(**data)

    def write_file(self, directory: Path, filename: str = '') -> None:
        """将 Manifest 写入 CELL.md 文件 (平铺 frontmatter, installed 不写入)."""
        filename = filename or self.MANIFEST_FILENAME
        flat = self.model_dump(
            exclude_defaults=True, exclude_none=True,
            exclude={'instruction', 'installed'},
        )
        instruction = self.instruction
        post = frontmatter.Post(content=instruction, **flat)
        frontmatter.dump(post, directory.joinpath(filename).resolve())

    @classmethod
    def from_proc(cls) -> 'CellManifest':
        """
        从进程中还原 Manifest.
        """
        from importlib import import_module
        main = import_module('__main__')
        script_file = Path(main.__file__)
        if script_file.exists():
            search_dir = script_file.parent
            for i in range(3):
                if not search_dir.exists():
                    break
                manifest = cls.read_from_directory(search_dir)
                if manifest:
                    launcher = CellLauncher.from_proc()
                    manifest.launcher = launcher
                    return manifest
                search_dir = search_dir.parent
        meta = CellMetadata.from_proc()
        launcher = CellLauncher.from_proc()
        data = meta.model_dump()
        data['launcher'] = launcher
        return CellManifest(**data)

    @classmethod
    def read_from_directory(cls, directory: Path) -> 'CellManifest | None':
        file = directory.joinpath(cls.MANIFEST_FILENAME)
        if file.is_file():
            return cls.read_from_file(file)
        return None

    @classmethod
    def read_from_file(cls, file: Path) -> 'CellManifest':
        """从 CELL.md 文件读取 Manifest.

        installed 从文件系统推导: 目录下存在 INSTALL.md 时,
        检查 .installed 空文件是否存在.
        """
        found_dir = file.parent.absolute()
        content = file.read_text(encoding='utf-8')
        post = frontmatter.loads(content)
        flat = post.metadata
        instruction = ''
        if stripped := post.content.strip():
            instruction = stripped
        flat['instruction'] = instruction

        directory = file.parent
        if directory.joinpath(cls.INSTALL_FILENAME).exists():
            flat['installed'] = directory.joinpath(cls.INSTALLED_FILE).exists()
        else:
            flat['installed'] = True  # 无 INSTALL.md 表示无需安装, 直接视为已安装
        manifest = cls(**flat)
        # cwd 必须更新成绝对地址. 空值默认取 CELL.md 所在目录, 与 launcher.cwd 字段 docstring 一致.
        if manifest.launcher.cwd:
            cwd = found_dir / manifest.launcher.cwd
            manifest.launcher.cwd = str(cwd.absolute())
        else:
            manifest.launcher.cwd = str(found_dir)
        return manifest


class DuplicatedError(RuntimeError):
    """cell 重复启动异常."""


class CellStatus(BaseModel):
    """
    Cell 运行时快照. 描述进程级状态.
    """
    uid: str = Field(
        default_factory=unique_id,
        description="每个 runtime status 启动时分配的唯一id. ",
    )
    project_id: str = Field(
        default='',
        description="启动后所属的 project id ",
    )
    state: Literal['starting', 'alive', 'stopped'] = Field(
        default='stopped',
        description="节点当前的状态.",
    )
    version: int = Field(
        default=0,
        description="更新和广播时的自增计数. 用于防止低版本覆盖高版本. ",
    )
    updated: float = Field(
        default_factory=time.time,
        description="最后更新的时间戳"
    )
    pid: int = Field(
        default=0,
        description="节点的进程 Id. 为 0 表示未运行."
    )
    failure: str = Field(
        default='',
        description="节点的致命故障讯息."
    )
    stdout_log: str = Field(
        default='',
        description='是否有 stdout log, 有的话给予绝对路径. '
    )
    stderr_log: str = Field(
        default='',
        description="是否有 stderr log, 有的话给予绝对路径."
    )

    @classmethod
    def from_proc(cls) -> 'CellStatus':
        return cls(
            pid=os.getpid(),
            state='starting',
            failure='',
        )


class Cell(BaseModel):
    """
    运行时发现的节点完整讯息.
    """

    meta: CellMetadata
    launcher: CellLauncher
    status: CellStatus = Field(default_factory=CellStatus)

    @property
    def name(self) -> str:
        """节点的名称. 在 project 内部是无重复的. """
        if self.meta.singleton:
            return self.normalized_name
        return self.unique_name

    @property
    def normalized_name(self) -> str:
        return normalize(self.meta.name).lower()

    @property
    def unique_name(self) -> str:
        """运行时确保唯一的名称. 用 uid 前 8 位防碰撞. """
        name = normalize(self.meta.name)
        return '_'.join([name, self.status.uid[:8]])

    @property
    def type(self) -> str:
        return self.meta.type.value if isinstance(self.meta.type, CellType) else str(self.meta.type)

    @property
    def cell_locker_name(self) -> str:
        """
        project 级别的 cell 单一锁, 用来放置 cell 重复启动.
        """
        if self.meta.singleton:
            return '-'.join([self.meta.type, self.normalized_name])
        # 争取名字可以自解释.
        return '-'.join([self.meta.type, self.normalized_name, self.status.uid])

    @property
    def address(self) -> CellAddress:
        """在整个体系内, 通讯时的唯一地址 (包含了唯一 id). """
        return '/'.join([self.meta.type, self.normalized_name, self.status.uid])

    @property
    def logger_name(self) -> str:
        """cell 自身的日志名. 通常决定 matrix 的日志. """
        return '.'.join(['moss', self.type, self.normalized_name, self.status.uid])

    @classmethod
    def new(
            cls,
            meta: CellMetadata,
            launcher: CellLauncher | None = None,
            status: CellStatus | None = None,
    ) -> 'Cell':
        """创建一个尚未运行的 cell."""
        launcher = launcher or CellLauncher()
        return cls(
            meta=meta,
            launcher=launcher,
            status=status or CellStatus(),
        )

    @property
    def is_host(self) -> bool:
        """是否是一个 host 节点. """
        return CellType.host.value == self.meta.type

    @classmethod
    def from_proc(cls) -> 'Cell':
        """从当前进程中还原 cell 信息. 通常只在启动 Cell 时调用."""
        return cls(
            meta=CellMetadata.from_proc(),
            launcher=CellLauncher.from_proc(),
            status=CellStatus.from_proc(),
        )

    @classmethod
    def from_manifest(cls, manifest: CellManifest, *, status_from_proc: bool = True) -> 'Cell':
        """从 Manifest 构造 Cell."""
        return cls(
            meta=manifest.meta(),
            launcher=manifest.launcher,
            status=CellStatus.from_proc() if status_from_proc else CellStatus(),
        )

    def as_manifest(self, instruction: str = '') -> CellManifest:
        """提取 Manifest 部分."""
        data = self.meta.model_dump()
        data['launcher'] = self.launcher
        data['instruction'] = instruction
        return CellManifest(**data)

    def is_alive(self) -> bool:
        if self.status.state != 'alive' or self.status.pid == 0:
            return False
        try:
            p = psutil.Process(self.status.pid)
            # 严格校验：确保这个 PID 不是僵尸进程.
            return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False

    def update(self):
        self.status.version += 1
        self.status.updated = time.time()

    def set_alive(self, pid: int | None = None) -> None:
        """标记 cell 为 alive 状态.

        pid 为 None 时使用当前进程 PID.
        调用方 (Matrix bootstrap / Host) 应在 cell 成功启动后调用此方法.
        """
        self.status.pid = pid or os.getpid()
        self.status.state = 'alive'
        self.status.failure = ''
        self.update()

    def set_failed(self, reason: str) -> None:
        """标记 cell 为 stopped 状态并记录失败原因."""
        self.status.state = 'stopped'
        self.status.failure = reason
        self.update()

    def to_json(self) -> str:
        return self.model_dump_json(
            ensure_ascii=True, indent=0,
            exclude_defaults=True, serialize_as_any=True,
        )

    @classmethod
    def discover(
            cls,
            env: Environment | None = None,
            *,
            cell_runtime_dir: Path | None = None,
    ) -> 'Cell':
        """从环境中发现当前的 Cell.

        优先从运行时文件恢复; 无文件时从当前进程构造.
        不做进程生命期治理 — 冲突检测 / 旧进程清理由 Matrix 层负责.
        """
        env = env or Environment.discover()
        address = env.this_cell_address
        if address:
            runtime_dir = cell_runtime_dir or env.cell_runtimes_dir
            file = runtime_dir / cls.make_runtime_filename(address)
            if file.exists():
                try:
                    return cls.read_from_runtime_file(file)
                except Exception:
                    pass
        return cls.from_proc()

    def write_runtime_file(self, cell_runtime_dir: Path | None = None) -> None:
        """写入运行时注册文件. 在一个项目中的注册文件可以用来判断哪些 cell 启动和运行状态. """
        cell_runtime_dir = cell_runtime_dir or Environment.discover().cell_runtimes_dir
        file_path = self.runtime_filepath(cell_runtime_dir)
        data = self.to_json()
        file_path.write_text(data)

    def runtime_filepath(self, scope_dir: Path) -> Path:
        filename = self.make_runtime_filename(self.address)
        return scope_dir.joinpath(filename)

    @staticmethod
    def make_runtime_filename(cell_address: str) -> str:
        address = normalize(cell_address)
        return f"cell-{address}.json"

    @classmethod
    def find_runtime_cells(cls, directory: Path | None = None, *, throw: bool = False) -> Iterable['Cell']:
        directory = directory or Environment.discover().cell_runtimes_dir
        for file in directory.glob('cell-*.json'):
            try:
                yield cls.read_from_runtime_file(file)
            except Exception as e:
                if throw:
                    raise e

    @classmethod
    def read_from_runtime_file(cls, file: Path) -> 'Cell':
        """
        父子进程启动 Cell 的方式:
        父进程写入子进程节点的 runtime file, pid 为 0.
        传入 address 环境变量启动子进程.
        子进程读取 runtime file, 获得 Cell,  更新 pid 并且启动.
        """
        cell = cls.model_validate_json(file.read_text(encoding='utf-8'))
        return cell

    def launch_cwd(self, cwd: Path | None = None) -> Path:
        cwd = cwd or Path.cwd()
        if self.launcher.cwd:
            cwd = cwd / self.launcher.cwd
            return cwd.resolve()
        return cwd

    def launch_program(self) -> str:
        if self.launcher.interpreter == 'python':
            return sys.executable
        elif self.launcher.interpreter:
            # cwd 相对或绝对路径.
            return self.launcher.interpreter
        else:
            return self.launcher.cmd

    def launch_args(self) -> list[str]:
        args = []
        if self.launcher.interpreter and self.launcher.cmd:
            args.append(self.launcher.cmd)

        if self.launcher.arguments:
            args.extend(shlex.split(self.launcher.arguments))
        return args


class CellRegistry(ABC):
    """
    Project 级别的 Cell 发现和注册中心.

    负责静态发现 (CELL.md 文件扫描)、本地运行时注册、
    以及本地进程 spawn
    """

    @abstractmethod
    def list_cell_manifests(
            self,
            refresh: bool = True,
            *,
            installed: bool = True,
            include: list[MatchPattern] | None = None,
            exclude: list[MatchPattern] | None = None,
    ) -> dict[RelativePath, CellManifest]:
        """
        列出所有环境中静态发现的 Cell 声明.
        :param refresh: 重新扫描文件系统.
        :param installed: 是否只返回已安装的.
        :param include: 匹配模式筛选.
        :param exclude: 排除模式筛选.
        """
        pass

    @abstractmethod
    def get_cell_manifest(self, relative_path: str | Path) -> CellManifest | None:
        """获取指定 目录 路径的 Cell 声明. 目录路径用 '/' 做分割 """
        pass

    @property
    @abstractmethod
    def cell_runtimes_dir(self) -> Path:
        """管理 cells 的运行时文件. 通常就是 env 约定的路径. 否则会有问题. """
        pass

    def local_runtime_cells(self) -> list[Cell]:
        """返回本地运行时注册的 cells."""
        return list(Cell.find_runtime_cells(self.cell_runtimes_dir, throw=False))

    def kill_all_runtime_cells(self) -> None:
        """删除并消灭 runtime 中所有的 cells. """
        cell_runtimes_dir = self.cell_runtimes_dir
        for cell in Cell.find_runtime_cells(cell_runtimes_dir):
            file = cell.runtime_filepath(cell_runtimes_dir)
            if cell.status.pid:
                self.recursively_kill_process(cell.status.pid)
            file.unlink()

    @staticmethod
    def recursively_kill_process(pid: int) -> None:
        try:
            parent = psutil.Process(pid)
            # recursive=True 找出的子进程列表，默认就是从外围孙子到核心儿子的拓扑序
            for child in parent.children(recursive=True):
                try:
                    child.kill()  # 遇到已经死掉的子进程自动忽略
                except psutil.NoSuchProcess:
                    pass
            parent.kill()  # 最后强杀父进程本身
        except psutil.NoSuchProcess:
            pass  # 父进程本身就不存在

    @staticmethod
    def match_cells(
            cells: dict[RelativePath, CellManifest],
            include: list[MatchPattern] | None = None,
            *,
            exclude: list[MatchPattern] | None = None,
    ) -> Iterable[tuple[RelativePath, CellManifest]]:
        """基于 fnmatch 通配符筛选 Cell.

        include 为空时返回全部 (仅受 exclude 约束).
        """
        include_patterns = set(include) if include else set()
        exclude_patterns = set(exclude or [])

        for relative_path, cell in cells.items():
            if include_patterns:
                if not any(fnmatch.fnmatch(relative_path, p) for p in include_patterns):
                    continue
            if exclude_patterns:
                if any(fnmatch.fnmatch(relative_path, p) for p in exclude_patterns):
                    continue
            yield relative_path, cell

    @abstractmethod
    def dump_spawn_env(self, address: str) -> dict[str, str]:
        """返回一个给子节点准备的环境变量. """
        pass

    async def spawn_cell(
            self,
            manifest: CellManifest,
            *,
            stdout: int | None = asyncio.subprocess.DEVNULL,
            stderr: int | None = asyncio.subprocess.DEVNULL,
            start_new_session: bool = True,
            kill_exists: bool = True,
            env: dict[str, str] | None = None,
    ) -> asyncio.subprocess.Process:
        """
        基于 Manifest 在本地启动一个 cell 子进程.
        不包含生命周期治理 —— 生命周期由调用方管理.
        """
        # 1. 创建 Cell 对象, pid 设置为 0. 更新 launcher 内的路径为绝对路径.
        # 2. 写入 runtime scopes 目录.
        # 3. 使用 manifest.launcher 创建子进程. 治理好 stdout 与 stderr.
        # pid 应该是空.
        cell = Cell.from_manifest(manifest, status_from_proc=False)
        # 判断旧 cell 文件如何处理.
        runtime_dir = self.cell_runtimes_dir
        file = cell.runtime_filepath(runtime_dir)
        if file.exists():
            exists = None
            try:
                exists = Cell.read_from_runtime_file(file)
            except Exception:
                pass
            if exists:
                if exists.is_alive() and not kill_exists:
                    raise RuntimeError(f"Cell {cell.address} already running")
                self.recursively_kill_process(cell.status.pid)
            file.unlink()
        # 先添加到环境. 写入运行环境.
        # 这样节点从环境中发现时, 会拿到 pid 为 None 的完整配置.
        # 准备好子进程的 env, 与父进程共享, 同时传入 address 身份.
        env_data = os.environ.copy()
        update = self.dump_spawn_env(cell.address)
        env_data.update(update)
        if env:
            env_data.update(env)

        cell.write_runtime_file(runtime_dir)
        try:
            proc = await asyncio.subprocess.create_subprocess_exec(
                cell.launch_program(),
                *cell.launch_args(),
                cwd=cell.launch_cwd(),
                start_new_session=start_new_session,
                env=env_data,
                stdout=stdout,
                stderr=stderr,
            )
        except Exception:
            if file.exists():
                file.unlink()
            raise
        return proc


class CellNetwork(ABC):
    """
    Cell 网络通讯层.
    负责运行时发现 (网络查询存活 cell)、provider/proxy 连线、
    通过网络只获取通讯链路上存在的 cells.

    Cell 在网络间通讯的基本原理:
    1. liveness: 声明自己存活和下线.
    2. queryable: 通过地址可以读取.
    3. pub: 广播改动内容.

    方便构建动态更新和缓存机制:
    1. query 全量节点, 构建启动缓存.
    2. 监听 liveness, 上线配合 query 构建缓存; 下线删除缓存.
    3. 通过 pub 仅在有改动时监听改动.
    """

    # -- 存活发现 -- #

    @property
    @abstractmethod
    def name(self) -> str:
        """通常用于 debug. """
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @property
    @abstractmethod
    def scope(self) -> str:
        """
        通讯子空间标识. 同一 scope 下的 cell 可以互相发现, scope 之间隔离.
        zenoh key 空间: MOSS/matrix/scopes/{scope}/cells/...
        Host 节点同时发布到 scope 空间和 hosts 空间.
        """
        pass

    @abstractmethod
    async def get_host(self) -> Cell | None:
        """
        尝试获取当前 Network 内的 Host 节点.
        """
        pass

    @abstractmethod
    async def all_hosts(self) -> list[Cell]:
        """
        返回所处网络中所有的 host 节点.
        """
        pass

    @abstractmethod
    async def get_live_cells(self, *, type: str | None = None, refresh: bool = False) -> dict[CellAddress, Cell]:
        """
        主动获取当前网络中在线的 cell.
        所有的 Cell 入网后, 应该用类似 Queryable 的逻辑支持被主动调用获取数据.
        :param type: 指定按特定的 type 类型查询.
        :param refresh: 是否全量查询.
        """
        pass

    @abstractmethod
    def live_cells(self) -> dict[CellAddress, Cell]:
        """
        直接通过缓存获取 cells.
        """
        ...

    @abstractmethod
    def on_change(self, callback: Callable[[Cell, bool], None]) -> None:
        """
        注册 callback, 监听 tuple[Cell, alive] 的改动.
        """
        ...

    # --- channel -- #

    @abstractmethod
    def create_provider(
            self,
            address: CellAddress,
            channel: Channel,
    ) -> ChannelProvider:
        """
        基于当前 Network 的通讯协议创建一个 ChannelProvider.
        address 必须是自己 update 过的 cell.
        """
        pass

    @abstractmethod
    def create_proxy(
            self,
            address: CellAddress,
            *,
            name: str = '',
            description: str = '',
    ) -> ChannelProxy:
        """
        基于 address 创建一个 CellNetwork 内部的 Proxy.
        实际上 CellNetwork 只负责创建, 不负责生命周期治理.
        """
        pass

    # --- announce -- #

    @abstractmethod
    async def update_cell(self, cell: Cell) -> None:
        """
        声明一个 Cell 的存在, 或更新 cell 的数据.
        首次更新会先检查 liveness 的唯一性, 然后抛出 liveness.
        同时支持 query. 非首次更新会 pub. Network 关闭时会自动下线.

        如果是 Host 节点, 同时会被广播到特殊的 Host 命名空间下.
        :raise DuplicatedError: 如果目标节点已经被别的地方声明过, 则会抛出异常.
        """
        pass

    @abstractmethod
    async def revoke_cell(self, cell: Cell) -> None:
        """
        取消环中中对一个 cell 的声明.
        :raise LookupError: 如果并不是自己 update 过的 cell, 会抛出异常.
        """
        pass

    # -- 生命周期 -- #

    @abstractmethod
    async def __aenter__(self) -> Self:
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        pass
