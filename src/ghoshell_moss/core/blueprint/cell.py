"""
MOSS 通讯网络中的单元节点

MOSS 作为面向模型的操作系统, 通过 Cell 管理其中的节点.
Cell 的定位类似 Node 之于 ROS, 技术核心差异是 Cell 是面向模型的, 需要能够被模型感知其存在.
不叫做 Node, 是因为不同类型的 Cell 节点拥有不同的网络角色定位. 比如主运行时单点叫 Host, 分形的 Host 组网时, 远程 Host 成为 Fractal.
这些节点在功能上并不是平级的, 所以不叫做 Node. 以及, 一个 Cell 可能通过某种机制管理其子节点, 是树上的分支.
每种子节点都有不同的发现和声明机制, 需要用统一的方式被运行时的智能模型感知其存在. Gemini3 对这种机制取名 'Cell'
"""
import signal
from typing import Literal, Iterable, ClassVar, Callable
from typing_extensions import Self
from abc import ABC, abstractmethod
from enum import Enum
from pydantic import Field, BaseModel
from ghoshell_moss.core.concepts.channel import ChannelProxy, ChannelProvider, Channel
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
    'split_address',
    'make_default_logger_name',
    'make_bridge_address',
    'split_bridge_address',
    'CellAddress',
    'CellBridgeAddress',
    'MatchPattern',
    'RelativePath',
    'normalize',
]


class CellType(str, Enum):
    """
    MOSS 系统对 Cell 的默认约定.
    每种不同类型 (type) 的 cell 都可能对应不同的角色, 发现 & 声明 & 运行时机制.

    host / worker / fractal 为 framework 保留 type。
    worker 无 owner channel，用于无语义兜底 —— 入网不被拒，semantic 不被承诺。
    """
    host = 'host'
    fractal = 'fractal'
    worker = 'worker'


# -- 类型别名 -- #

RelativePath = str
MatchPattern = str
"""通配符模式: group/name, group/*, *, */*, */name"""
CellAddress = str
CellBridgeAddress = str


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


def make_address(cell_type: str | CellType, name: str) -> str:
    """约定定义一个节点的 address. 使用 / 作为层级分隔符."""
    cell_type = (cell_type.value if isinstance(cell_type, CellType) else cell_type).lower().strip()
    name = name.strip()
    if not cell_type or not name:
        raise ValueError(f"Invalid address parts: type={cell_type!r}, name={name!r}")
    if '/' in cell_type:
        raise ValueError(f"Cell type must not contain '/': {cell_type!r}")
    return '/'.join([cell_type, name])


def normalize(name_or_address: str) -> str:
    return (name_or_address.replace('/', '_').replace('\\', '_').
            replace('.', '_').replace('-', '_'))


def make_bridge_address(prefix: str, uid: str) -> CellBridgeAddress:
    return '/'.join([prefix, uid])


def split_address(address: str) -> tuple[str, str]:
    got = address.split('/', 1)
    if len(got) == 2:
        return got[0], got[1]
    raise ValueError(f"Invalid cell address: {address}")


def split_bridge_address(address: CellBridgeAddress) -> tuple[str, str]:
    got = address.split('/')
    if len(got) >= 2:
        return '/'.join(got[0: -1]), got[-1]
    raise ValueError(f"Invalid cell bridge address: {address}")


def make_default_logger_name(address: str) -> str:
    return address.replace('/', '.').replace('\\', '.')


class CellLauncher(BaseModel):
    """
    节点启动参数. 描述如何启动一个 cell 进程.
    """

    cwd: str = Field(
        default='./',
        description="节点启动时所在的工作路径. 为空表示不是本地运行的进程. "
                    "默认从发现 CELL.md 的路径作为 cwd, 否则从运行时 cwd / launcher.cwd"
    )

    interpreter: Literal['python', ''] | str = Field(
        default='python',
        description="使用约定的运行时"
                    "为 `python` 表示使用 sys.executable(父进程) or 环境发现的解释器;"
                    "非空时请使用 cwd 相对路径, 或绝对路径指定解释器. "
    )
    cmd: str = Field(
        default='main.py',
        description="启动节点对应的脚本. 具体路径是 cwd / script"
    )
    args: str = Field(
        default='',
        description="启动节点时传入的参数",
    )

    extra_env: dict[str, str] = Field(
        default_factory=dict,
        description="额外写入环境变量的讯息.",
    )

    @classmethod
    def new_empty(cls) -> 'CellLauncher':
        return cls(cwd='', interpreter='', cmd='', args='')

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
            args=_args,
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
        description="是否是运行时单例, 决定 network 会如何处理它的存在. "
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
        # cwd 必须更新成绝对地址.
        cwd = found_dir / manifest.launcher.cwd
        if manifest.launcher.cwd:
            manifest.launcher.cwd = str(cwd.absolute())
        return manifest


class CellStatus(BaseModel):
    """
    Cell 运行时快照. 描述进程级状态.
    """
    uid: str = Field(
        default_factory=unique_id,
        description="每个 runtime status 启动时分配的唯一id. ",
    )
    state: Literal['starting', 'alive', 'stopped'] = Field(
        default='stopped',
        description="节点当前的状态.",
    )
    pid: int | None = Field(
        default=None,
        description="节点的进程 Id. 为空表示未运行或不是本地进程."
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
    def address(self) -> CellAddress:
        """运行时唯一的地址."""
        if self.meta.singleton:
            return make_address(self.meta.type, self.meta.name)
        else:
            return make_address(self.meta.type, self.status.uid)

    @property
    def name(self) -> str:
        return self.meta.name

    @property
    def type(self) -> str:
        return self.meta.type.value if isinstance(self.meta.type, CellType) else str(self.meta.type)

    @property
    def bridge_address(self) -> CellBridgeAddress:
        return make_bridge_address(self.type, self.status.uid)

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

    @classmethod
    def from_proc(cls) -> 'Cell':
        """从当前进程中还原 cell 信息."""
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
        if self.status.state != 'alive' or self.status.pid is None:
            return False
        try:
            p = psutil.Process(self.status.pid)
            # 严格校验：确保这个 PID 不是僵尸进程.
            return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False

    def set_alive(self, pid: int | None = None) -> None:
        """标记 cell 为 alive 状态.

        pid 为 None 时使用当前进程 PID.
        调用方 (Matrix bootstrap / Host) 应在 cell 成功启动后调用此方法.
        """
        self.status.pid = pid or os.getpid()
        self.status.state = 'alive'
        self.status.failure = ''

    def set_failed(self, reason: str) -> None:
        """标记 cell 为 stopped 状态并记录失败原因."""
        self.status.state = 'stopped'
        self.status.failure = reason

    def to_json(self) -> str:
        return self.model_dump_json(
            ensure_ascii=True, indent=0,
            exclude_defaults=True, serialize_as_any=True,
        )

    def write_runtime_file(self, directory: Path) -> None:
        """写入运行时注册文件."""
        file_path = self.runtime_filepath(directory)
        data = self.to_json()
        file_path.write_text(data)

    def runtime_filepath(self, scope_dir: Path) -> Path:
        filename = self.make_runtime_filename(self.address)
        return scope_dir.joinpath(filename)

    @staticmethod
    def make_runtime_filename(address: str) -> str:
        address = normalize(address)
        return f"cell-{address}.json"

    def normalized_address(self) -> str:
        return normalize(self.address)

    def normalized_name(self) -> str:
        return normalize(self.meta.name).lower()

    @classmethod
    def find_runtime_cells(cls, directory: Path, *, throw: bool = False) -> Iterable['Cell']:
        for file in directory.glob('cell-*.json'):
            try:
                yield cls.read_from_runtime_file(file)
            except Exception:
                if throw:
                    raise

    @classmethod
    def read_from_runtime_file(cls, file: Path, *, pid: int | None = None) -> 'Cell':
        """
        父子进程启动 Cell 的方式:
        父进程写入子进程节点的 runtime file, pid 为 0.
        传入 address 环境变量启动子进程.
        子进程读取 runtime file, 获得 Cell,  更新 pid 并且启动.
        """
        cell = cls.model_validate_json(file.read_text(encoding='utf-8'))
        if pid:
            cell.status.pid = pid
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
            return self.launcher.interpreter
        else:
            return self.launcher.cmd

    def launch_args(self) -> list[str]:
        args = []
        if self.launcher.interpreter and self.launcher.cmd:
            args.append(self.launcher.cmd)
        if self.launcher.args:
            args.extend(shlex.split(self.launcher.args))
        return args


class CellRegistry(ABC):
    """
    Cell 注册中心.

    负责静态发现 (CELL.md 文件扫描)、本地运行时注册 (runtime/cells/ 读写)、
    以及本地进程 spawn。在 Host 层暴露，Matrix 上不可见。
    """

    @abstractmethod
    def root(self) -> Path:
        """Cell 注册文件的根目录."""
        pass

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
    def get_cell_manifest(self, relative_path: str) -> CellManifest | None:
        """获取指定 目录 路径的 Cell 声明. 目录路径用 '/' 做分割 """
        pass

    @abstractmethod
    def discover_current_cell(self) -> Cell:
        """找到当前进程拿到的 Cell 身份. """
        pass

    @abstractmethod
    def local_runtime_cells(self) -> list[Cell]:
        """返回本地运行时注册的 cells."""
        pass

    def kill_all_runtime_cells(self) -> None:
        """删除并消灭 runtime 中所有的 cells. """
        import os
        for cell in self.local_runtime_cells():
            if cell.status.pid is not None:
                os.kill(cell.status.pid, signal.SIGKILL)
            self.remove_cell_runtime(cell.address)

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
    def add_cell_runtime(self, cell: Cell) -> None:
        """注册一个启动的 Cell 到本地运行时."""
        pass

    @abstractmethod
    def remove_cell_runtime(self, address: str) -> bool:
        """删除本地运行时中的 Cell 注册信息."""
        pass

    @abstractmethod
    def get_cell_runtime(self, address: str) -> Cell | None:
        pass

    @abstractmethod
    def cell_runtime_exists(self, address: str) -> bool:
        """判断一个 address 对应的 cell, 其讯息是否存在. """
        pass

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
        exists = self.get_cell_runtime(cell.address)
        if exists:
            if not exists.is_alive():
                # 永远删除.
                self.remove_cell_runtime(cell.address)
            else:
                if not kill_exists:
                    raise RuntimeError(f"Cell {cell.address} already running")
                os.kill(exists.status.pid, signal.SIGKILL)
                self.remove_cell_runtime(cell.address)
        # 先添加到环境. 写入运行环境.
        # 这样节点从环境中发现时, 会拿到 pid 为 None 的完整配置.
        # 准备好子进程的 env, 与父进程共享, 同时传入 address 身份.
        env = os.environ.copy()
        update = self.dump_spawn_env(cell.address)
        env.update(update)

        self.add_cell_runtime(cell)
        try:
            proc = await asyncio.subprocess.create_subprocess_exec(
                cell.launch_program(),
                *cell.launch_args(),
                cwd=cell.launch_cwd(),
                start_new_session=start_new_session,
                env=env,
                stdout=stdout,
                stderr=stderr,
            )
        except Exception:
            self.remove_cell_runtime(cell.address)
            raise
        return proc


class CellNetwork(ABC):
    """
    Cell 网络通讯层.
    负责运行时发现 (网络查询存活 cell)、provider/proxy 连线、
    通过网络只获取通讯链路上存在的 cells.
    """

    # -- 存活发现 -- #

    @abstractmethod
    async def get_live_cells(self, refresh: bool = False) -> dict[CellAddress, Cell]:
        """
        主动获取当前网络中在线的 cell.
        启动监听前是同步获取 (queryable), 启动监听后走缓存获取.
        """
        pass

    # -- provider / proxy -- #

    @abstractmethod
    def provide(
            self,
            address: CellAddress,
            channel: Channel,
    ) -> ChannelProvider:
        """
        创建 ChannelProvider，将 Channel 通过当前节点提供到整个 Matrix 网络.
        :param address: 提供方的 cell bridge address.
        :param channel: 要提供的 Channel 树根节点.
        """
        pass

    @abstractmethod
    def create_proxy(
            self,
            cell: Cell,
    ) -> ChannelProxy:
        """
        为目标 address 创建 ChannelProxy.
        :raise LookupError: 目标 cell 未在网络中发现.
        """
        pass

    @abstractmethod
    def proxies(self) -> dict[str, ChannelProxy]:
        """返回所有已创建的 proxies."""
        pass

    @abstractmethod
    def list_providers(self) -> list[CellAddress]:
        """通过网络通讯主动查询当前提供 channel 的 cell addresses."""
        # 首次启动时会等待, 后续通过动态缓存获取.
        pass

    # --- announce -- #

    @abstractmethod
    def announce_cell(self, cell: Cell) -> None:
        """声明一个 Cell 的存在, 或更新 cell 的数据. """
        pass

    @abstractmethod
    def revoke_cell(self, cell: Cell) -> None:
        """取消环中中对一个 cell 的声明. """
        pass

    # -- 生命周期 -- #

    @abstractmethod
    async def __aenter__(self) -> Self:
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        pass
