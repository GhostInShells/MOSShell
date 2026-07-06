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
    'HOST_TYPE',
    'WORKER_TYPE',
    'CellMetadata',
    'CellLauncher',
    'CellManifest',
    'CellStatus',
    'Cell',
    'CellLog',
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


# -- 框架保留的 cell type 字面量. 用户可自由扩展 type 字符串 (sensors / bodies / tools / ...).
# 框架行为只对 host 与 worker 有特殊语义; 其它 type 一视同仁.
HOST_TYPE = 'host'
WORKER_TYPE = 'worker'


# -- 类型别名 -- #

RelativePath = str
MatchPattern = str
"""通配符模式: group/name, group/*, *, */*, */name"""


class CellMetadata(BaseModel):
    """
    一个 Cell 节点的元信息, 通常用于环境发现和运行时声明.
    type 是开放命名空间: 'host' / 'worker' 是框架保留字, 其它字符串由用户自由命名
    (sensors / bodies / tools / ...). 框架行为只对 host 区分对待, 其余 type 一视同仁.
    """
    type: str = Field(
        default=WORKER_TYPE,
        description="节点的类型. 开放命名空间, 'host' / 'worker' 是框架保留字."
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
            type=WORKER_TYPE,
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

    type: str = Field(
        default=WORKER_TYPE,
        description="节点的类型. 开放命名空间, 'host' / 'worker' 是框架保留字."
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
        return self.meta.type

    @property
    def identity(self) -> str:
        """type/name. project 内稳定 id, 跨 cell 重启不变. 模型心智锚."""
        return '/'.join([self.meta.type, self.meta.name])

    @property
    def cell_locker_name(self) -> str:
        """
        project 级别的 cell 单一锁, 用来防止 cell 重复启动.
        """
        if self.meta.singleton:
            return '-'.join([self.meta.type, self.normalized_name])
        # 争取名字可以自解释.
        return '-'.join([self.meta.type, self.normalized_name, self.status.uid])

    @property
    def address(self) -> CellAddress:
        """type/name/uid. network 唯一地址, uid 防 stale 与碰撞.
        注意: 使用原始 meta.name (非 normalized) — name 含 / 会破坏路径假设, 由上层保证字面合法.
        """
        return '/'.join([self.meta.type, self.meta.name, self.status.uid])

    @property
    def channel_name(self) -> str:
        """proxy channel 名 — 由 identity 派生, 模型读到就知道怎么调.
        - singleton: normalize(identity).lower()
        - non-singleton: + '_' + uid[:8] 防碰撞
        """
        base = normalize(self.identity).lower()
        if not self.meta.singleton:
            return f"{base}_{self.status.uid[:8]}"
        return base

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
        return self.meta.type == HOST_TYPE

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


class DuplicatedError(RuntimeError):
    """cell 重复启动异常."""


class CellLog(BaseModel):
    """CellNetwork 上轻量事件 — 不广播 cell snapshot, 只广播事件信号 (§SS-2).

    用途: status 变更 (failure/degraded 等中间态), cell 死亡通知, channel ready/gone.
    接收方语义:
    - terminal=False → append log + 触发 follow-up query 刷新 cache.
    - terminal=True  → append log + pop cache entry, 不 query (cell 已没了).
    """

    address: CellAddress = Field(
        description="事件来源 cell 的 address (含 uid, 网络精确)."
    )
    content: str = Field(
        default='',
        description="自由文本事件内容. 第一期不约定 enum — 隐藏 type 留在心里, 不暴露给模型."
    )
    timestamp: float = Field(
        default_factory=time.time,
        description="事件产生时刻."
    )
    terminal: bool = Field(
        default=False,
        description="True 表示 cell 已没了, 接收方别 query, 直接删 cache."
    )


class CellNetwork(ABC):
    """
    Cell 网络通讯层 — cell 发现 + provider/proxy 单一真相源.

    屏蔽底层 hub (zenoh / 未来 IPC 等) 的差异性. 调用方拿到 CellNetwork 就拿到了:
    - 网络上谁在线 (get_live_cells / live_cells)
    - 状态变更轻量事件 (recent_logs, broadcast_log)
    - 跨 cell 通讯通道 (provide / proxies / get_proxy / wait_connected)
    - 上下文唤醒回调 (on_change / on_provider_online / on_provider_offline)

    二元真相承诺 (§NN):
      live_cells() 返回的是延迟视图. 只承诺 online/offline 准确, status 字段可能滞后到
      下次 reconcile. 需要实时 status 的调用方必须 get_live_cells(refresh=True).
      cell 想让网络知道 degraded, 应主动 revoke_cell + update_cell — liveness DELETE+PUT
      会传出去. 这是 cell 有意识的传播, 不是框架默认行为.

    底层原语 (zenoh 实现):
    1. liveness: 声明自己存活和下线. PUT=上线 DELETE=下线 subscriber 实时感知.
    2. queryable: 通过地址按需拉取 Cell snapshot.
    3. logs publisher/subscriber: 轻量事件 (CellLog) 广播, 不广播 snapshot.

    缓存机制:
    1. seed: 全量 liveness get + queryable, 启动缓存.
    2. liveness subscriber: 上线 PUT → query + 写 cache; 下线 DELETE → pop cache.
    3. log subscriber: terminal=False → re-query 刷 cache; terminal=True → 提前 pop cache.
    4. reconcile loop: 低频全量对账, 兜底极端丢事件.
    """

    # ── 标识 ──────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str:
        """通常用于 debug. """
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        ...

    @property
    @abstractmethod
    def scope(self) -> str:
        """通讯子空间标识. 同一 scope 下的 cell 可以互相发现, scope 之间隔离.
        zenoh key 空间: MOSS/matrix/scopes/{scope}/cells/...
        Host 节点同时发布到 scope 空间和 hosts 空间.
        """
        ...

    # ── 发现 ──────────────────────────────────────────────────────────

    @abstractmethod
    async def get_host(self) -> Cell | None:
        """尝试获取当前 Network 内的 Host 节点."""
        ...

    @abstractmethod
    async def all_hosts(self) -> list[Cell]:
        """返回所处网络中所有的 host 节点 (跨 scope)."""
        ...

    @abstractmethod
    async def get_live_cells(
            self,
            *,
            type: str | None = None,
            local: bool | None = None,
            refresh: bool = False,
    ) -> dict[CellAddress, Cell]:
        """
        主动获取当前网络中在线的 cell.

        :param type: 仅返回指定 type 的 cell (开放命名空间, 'host' / 'worker' / 用户自定义).
        :param local: None=不过滤; True=仅本 project; False=仅其它 project.
                     "local" 判定: cell.status.project_id == self_project_id.
                     如果 CellNetwork 不知道 self_project_id, local 过滤一律忽略.
        :param refresh: True=全量 liveness get + queryable refresh, False=读 cache.
        """
        ...

    @abstractmethod
    def live_cells(self) -> dict[CellAddress, Cell]:
        """直接通过 cache 获取 cells (零延时, 延迟视图)."""
        ...

    @abstractmethod
    def on_change(self, callback: Callable[[Cell, bool], None]) -> None:
        """注册 callback, 监听 (Cell, online) 的改动.

        回调可能在 zenoh subscriber 线程或 reconcile loop 任务中触发,
        caller 负责线程安全.
        """
        ...

    # ── 唯一性 ────────────────────────────────────────────────────────

    @abstractmethod
    async def check_unique(self, cell: Cell) -> None:
        """检查 cell 在网络上是否唯一可宣告.

        - singleton 节点: 同 identity (type/name) 在 scope 内不允许多个.
          实现层面: 查 identity wildcard, 命中即冲突.
        - non-singleton 节点: 同 address (含 uid) 不允许重复.
          实现层面: 查 address exact 即可.

        :raise DuplicatedError: 已被别处声明.
        """
        ...

    # ── 轻量事件 (CellLog) ─────────────────────────────────────────────

    @abstractmethod
    async def broadcast_log(
            self,
            address: CellAddress,
            content: str,
            *,
            terminal: bool = False,
    ) -> None:
        """向 network 广播一个 CellLog 事件 (§SS-2).

        不广播 cell snapshot — 接收方按需 query cache.

        terminal=True 触发场景:
        - 自身 revoke_cell 时
        - 观察到 liveness DELETE → 自动 broadcast cell-gone (terminal=True)
        - CellsManager 观察到 ProcessManager on_exit → broadcast exited (terminal=True)
        """
        ...

    @abstractmethod
    def recent_logs(
            self,
            *,
            limit: int = 20,
            local: bool | None = None,
    ) -> list[CellLog]:
        """最近 CellLog 时间窗口 (ring buffer).

        :param limit: 返回条数上限 (FIFO, 最新优先).
        :param local: None=不过滤; True=仅本 project 的 cell; False=仅其它 project.
        """
        ...

    # ── 宣告 ──────────────────────────────────────────────────────────

    @abstractmethod
    async def update_cell(self, cell: Cell, *, log: str = '') -> None:
        """声明一个 Cell 的存在, 或更新 cell 的数据.

        首次更新前检查 liveness 唯一性, 然后宣告 liveness + queryable.
        如果是 Host 节点, 同时被广播到特殊的 hosts 命名空间下.

        :param log: 若非空, 同时 broadcast_log(address, log) (terminal=False).
        :raise DuplicatedError: 已被别处声明.
        """
        ...

    @abstractmethod
    async def revoke_cell(self, cell: Cell, *, log: str = '') -> None:
        """取消网络中对一个 cell 的声明.

        :param log: 若非空, 同时 broadcast_log(address, log, terminal=True).
        :raise LookupError: 并不是自己 update 过的 cell.
        """
        ...

    # ── channel (provider/proxy) ─────────────────────────────────────

    @abstractmethod
    async def provide(
            self,
            channel: Channel,
            *,
            address: CellAddress | None = None,
    ) -> ChannelProvider:
        """将 Channel 通过本 network 暴露出去, 让远端 cell 能 proxy 到.

        三层调用: matrix.provide → network.provide → hub.provider.

        :param address: 必须是已 update 过的 cell address (自己 own 的).
                       None 时由实现决定 (Matrix 通常会传 self.this.address).
        :raise ValueError: address 为 None 且实现无默认.
        """
        ...

    @abstractmethod
    def proxies(self) -> dict[CellAddress, ChannelProxy]:
        """当前自动构建的全部 proxy. 每个 key 对应一个网络上观察到的 provider.

        触发链: zenoh hub liveness PUT (channels_ns/{address}) → CellNetwork
        自动 build proxy → 放入 proxies dict → broadcast_log "channel-ready".

        第一期约束 (§SS-5): 只有 host 视角的 CellNetwork 才会构建 proxy.
        worker 视角的 network proxies() 永远空.
        """
        ...

    @abstractmethod
    def get_proxy(self, address: CellAddress) -> ChannelProxy | None:
        """按 address 取 proxy. 不存在返回 None."""
        ...

    @abstractmethod
    async def wait_connected(
            self,
            address: CellAddress,
            *,
            timeout: float = 30,
    ) -> bool:
        """等待 address 对应的 proxy 上线并可用.

        实现: 注册 on_provider_online callback + 检查 proxies() 当前是否已含 address.
        :return: True=超时前 ready, False=超时.
        """
        ...

    # ── channel 上下线回调 ────────────────────────────────────────────

    @abstractmethod
    def on_provider_online(
            self,
            callback: Callable[[CellAddress], None],
    ) -> Callable[[], None]:
        """注册 provider 上线回调. 返回 unsubscribe 函数.

        回调可能在 zenoh 后台线程 fire, 调用方负责线程安全.
        """
        ...

    @abstractmethod
    def on_provider_offline(
            self,
            callback: Callable[[CellAddress], None],
    ) -> Callable[[], None]:
        """注册 provider 下线回调. 返回 unsubscribe 函数."""
        ...

    # ── 生命周期 ──────────────────────────────────────────────────────

    @abstractmethod
    async def __aenter__(self) -> Self:
        ...

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        ...
