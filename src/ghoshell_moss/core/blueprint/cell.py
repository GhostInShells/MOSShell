"""
MOSS 通讯网络中的单元节点 (Cell).

Cell 是在 Moss 的网络 (Matrix) 中运行的进程单元. 可以将之想象为一个数字城市中的一个房间, 用来提供不同的功能.
Cell 可以用来控制机器人, 创建图形界面, 运行独立的思考或 Agent. 可以把它理解为手机里的 App.
Cell 运行时, 通过映射进入 Matrix 网络, Cell 之间的通讯可以通过文件或者 Matrix.session 协议.

每个 Cell 都有自己的地址 - address, 类似于电话; 关于 cell 的通讯基本都从 address 出发.
当 Cell 通过 Matrix 提供 Channel 等能力时, 将能够让 Moss 在运行时控制它. 一个通过 moss 启动的 Ghost 可以用 Channel 控制它看到的 Cell.

各种各样的 Cell 运行时组织成 Matrix 的网络, 通过 Host 节点提供给 Ghost 一个可用的操作系统.
"""
import contextlib
import sys
import time
from pathlib import Path
from typing import Callable, ClassVar, Iterable, Literal
from typing_extensions import Self
from abc import ABC, abstractmethod

import fnmatch
import frontmatter
import shlex
from pydantic import BaseModel, Field, AwareDatetime

from ghoshell_moss.core.concepts.channel import Channel, ChannelProvider, ChannelProxy
from ghoshell_moss.message import unique_id
from .environment import Environment
import datetime
import dateutil
import dataclasses
import psutil
import asyncio

__all__ = [
    'CellAddress',
    'CellRole',
    'CellProtocol',
    'HOST_ROLE',
    'NODE_ROLE',
    'normalize',
    'make_address',
    'parse_address',
    'build_node_from_manifest',
    'build_host_cell',
    'discover_this_node',
    'ExecSpec',
    'NodeManifest',
    'CellRuntimeInfo',
    'Cell',
    'CellEvent',
    'CellPresence',
    'CellNetwork',
    'NodeManager',
    'DuplicatedError',

    'CellName',
    'CellNamePattern',
    'AbsolutePath',
    'ProjectRelativePath',
    'MatchPattern',
    'NodeLauncher',
    'enter_cell_lifecycle',
]

CellRole = Literal['host', 'node']
"""
描述 Cell 在 Matrix 网络拓扑中的位置角色.
- 'host': 网络的中心节点, 将所有能力组织在一起供躯体或智能体驱动.
- 'node': 功能性节点. 可能控制躯体, 提供 gui, 有独立的应用能力等等.
"""
HOST_ROLE: CellRole = 'host'
NODE_ROLE: CellRole = 'node'

ROLES = frozenset({HOST_ROLE, NODE_ROLE})

CellProtocol = Literal['channel']
"""
Cell 提供的 MOSS 双工通讯协议名. 封闭集: 每个值对应 duplex/ 里一整套 Provider/Proxy
角色对 + 事件词汇 (见 duplex/protocol.py). 未来加协议 = 加一套 Provider/Proxy 实现,
且在此 type alias 追加值 — 不是自由字符串, 不能靠约定加名字.
"""

CellAddress = str
"""
cell 在网络上的唯一地址: address = CellRole / unique_name / uid
"""

ProjectRelativePath = str
AbsolutePath = str
MatchPattern = str
"""通配符模式: group/name, group/*, *, */*, */name"""

CellName = str
CellNamePattern = r"^[a-zA-Z0-9_]+$"
"""只允许用这种方式定义 cell name, 与 python命名风格兼容. """


class Cell(BaseModel):
    """
    Cell 是 Matrix 网络中 **运行中** 节点的声明讯息.
    可以把 Matrix 想象成一个大楼, 里的一个房间就是 Cell.
    通常是一个独立的进程.
    """
    role: CellRole = Field(
        description="cell role"
    )
    name: str = Field(
        description="cell name",
        pattern=CellNamePattern,
    )
    uid: str = Field(
        default_factory=unique_id,
        description="cell uid",
    )
    singleton: bool = Field(
        default=False,
        description="声明本 cell 在治理域内是否保持唯一实例. "
                    "True: 同名 cell 已在运行时, 重复拉起会被拒绝 (DuplicatedError). "
                    "适合有硬件独占 (麦克风/摄像头/机器人) 或状态独占 (数据库连接) 的场景. "
                    "False (默认): 可多实例并行, 各自有独立 uid.",
    )
    category: Literal['ghost', 'shell', 'script'] | str = Field(
        default='',
        pattern=r"^[a-zA-Z0-9_]*$",
        description='cell 的分类.'
    )
    description: str = Field(
        default='',
        description="cell description",
    )
    project_id: str = Field(
        default='',
        description="cell 所属治理域 (project) 标识. 可以用于判别是否本地的 cell. ",
    )
    project_name: str = Field(
        default='',
        description='cell 所在项目的名字.'
    )
    providing: list[CellProtocol] = Field(
        default_factory=list,
        description="本 cell 当前运行状态提供的 MOSS 双工通讯协议. "
                    "只标记协议名, 内容靠各自 Provider/Proxy 桥拉取. "
                    "值域受 CellProtocol 封闭 — 加协议要先在 duplex/ 落地对应角色.",
    )
    updated: AwareDatetime = Field(
        default_factory=lambda: datetime.datetime.now(dateutil.tz.gettz()),
        description="本 cell 最后更新的时间戳.",
    )
    home: str = Field(
        description="进程工作目录, 绝对路径.",
    )
    parent_address: str = Field(
        default='',
        description="运行当前 Cell 的父节点讯息.",
    )

    def update(self) -> None:
        self.updated = datetime.datetime.now(dateutil.tz.gettz())

    @property
    def address(self) -> CellAddress:
        return make_address(self.role, self.name, self.uid)

    @property
    def fullname(self) -> str:
        if self.category:
            return '_'.join([self.category, normalize(self.name)])
        return normalize(self.name)

    @property
    def is_host(self) -> bool:
        """本 cell 是否是网络的 host — 从 address[0] 推断 (§ZZ-10)."""
        return self.role == HOST_ROLE

    def is_local(self, env: Environment) -> bool:
        return self.project_id == env.project_id

    @property
    def unique_name(self) -> str:
        return '_'.join([self.fullname, self.uid[:8]])


class ExecSpec(BaseModel):
    """
    运行一个进程 (主要是 Node) 的声明.
    """
    command: Literal['python'] | str = Field(
        default='python',
        description="不为空时, 作为启动命令 argv[0]. "
                    "应当是 cwd 的相对路径. ",
    )
    args: str = Field(
        default='main.py',
        description="启动命令的参数列表.",
    )
    env: dict[str, str] = Field(
        default_factory=dict,
        description="额外注入的环境变量. 也可以考虑在启动脚本内部通过 dotenv 等方式自行加载. ",
    )

    @property
    def arguments(self) -> list[str]:
        return shlex.split(self.args)


NodeScriptCategory = 'script'


class NodeManifest(BaseModel):
    """
    Node 类型的 Cell 节点的声明信息.
    通过声明文件完成定义, 也兼容没有声明文件的脚本启动场景.
    """
    MANIFEST_FILENAME: ClassVar[str] = 'NODE.md'
    """声明文件的约定文件, 可以理解为 windows 的快捷方式. """

    INSTALL_FILENAME: ClassVar[str] = 'INSTALL.md'
    """如何完成安装的文件, 如果存在, 则需要配套 INSTALLED_FILE 描述安装的状态. 
    node 可能拥有独立的项目依赖, 这时需要定义 INSTALL_FILENAME. 如果未完成安装则应该返回该文件地址. """

    INSTALLED_FILE: ClassVar[str] = '.installed'
    """通过文件标记一个 Node 是否完成了安装. """

    name: str = Field(
        description="Node 的名字. 治理域内的身份锚.",
        pattern=CellNamePattern,
    )
    description: str = Field(
        default='',
        description="Node 的一句话描述.",
    )
    category: str = Field(
        default='',
        description="纯分类标签 (如 sensors / bodies / scripts / tools), 自由命名, 不驱动任何机制.",
        pattern=r"^[a-zA-Z0-9_]*$",
    )
    singleton: bool = Field(
        default=True,
        description="声明本 cell 在治理域内是否保持唯一实例. "
                    "True (默认): 同名 cell 已在运行时, 重复拉起会被拒绝. "
                    "适合有硬件独占 (麦克风/摄像头/机器人) 或状态独占 (数据库连接) 的场景. "
                    "False: 可多实例并行, 各自有独立 uid.",
    )
    exec: 'ExecSpec' = Field(
        default_factory=ExecSpec,
        description="默认启动入口 (frontmatter `run:` 声明). "
                    "无声明的 cell 只能以显式脚本路径拉起.",
    )
    instruction: str = Field(
        default='',
        description="节点详细的使用说明",
    )
    installed: bool = Field(
        default=True,
        description="是否已完成安装. 未安装的 cell 可被发现但拒绝拉起, "
                    "错误信息会给出 INSTALL.md 路径. 由文件系统推导, 不在 frontmatter 中.",
    )

    file: AbsolutePath = Field(
        default='',
        description="当前 NodeManifest 生成时的文件绝对路径地址.",
    )

    @property
    def cwd(self) -> Path:
        if self.file:
            return Path(self.file).parent.resolve()
        return Path.cwd()

    @classmethod
    def read_from_file(cls, file: Path) -> 'NodeManifest':
        """从 CELL.md 文件读取声明. 正文即 instruction, frontmatter 即字段."""
        content = file.read_text(encoding='utf-8')
        post = frontmatter.loads(content)
        data = dict(post.metadata)
        data['instruction'] = post.content.strip()
        data['file'] = str(file.absolute())
        directory = file.parent
        # installed 由文件系统推导:
        #   有 INSTALL.md → 需要额外安装步骤, 靠 .installed 标记是否完成
        #   无 INSTALL.md → 无额外依赖, 天然视为已安装
        if directory.joinpath(cls.INSTALL_FILENAME).exists():
            data['installed'] = directory.joinpath(cls.INSTALLED_FILE).exists()
        else:
            data['installed'] = True
        return cls(**data)

    @classmethod
    def read_from_directory(cls, directory: Path) -> 'NodeManifest | None':
        """从目录中获取 manifest"""
        file = directory.joinpath(cls.MANIFEST_FILENAME)
        if file.is_file():
            return cls.read_from_file(file)
        return None

    def save(self) -> None:
        file = Path(self.file).absolute()
        self.write_file(file)

    def write_file(self, directory: Path, filename: str = '') -> None:
        """将声明写入 CELL.md"""
        filename = filename or self.MANIFEST_FILENAME
        data = self.model_dump(
            exclude_defaults=True, exclude_none=True,
            exclude={'instruction', 'installed', 'file'},
        )
        post = frontmatter.Post(content=self.instruction, **data)
        frontmatter.dump(post, directory.joinpath(filename).resolve())

    @classmethod
    def find_upward(cls, start: Path) -> 'NodeManifest | None':
        """从 start 出发向上查找最近的 CELL.md (找到第一个即停)."""
        directory = start if start.is_dir() else start.parent
        home = Path.home()
        for candidate in [directory, *directory.parents]:
            manifest = cls.read_from_directory(candidate)
            if manifest is not None:
                return manifest
            if candidate == home:
                break
        return None

    @classmethod
    def from_script(cls, script: Path, *, exec_spec: ExecSpec | None = None) -> 'NodeManifest':
        """
        以脚本为入口构造 Manifest: 向上认亲最近的 CELL.md;
        找不到时降级为临时身份, 不拒绝运行.
        """
        script = script.resolve()
        found = cls.find_upward(script)
        exec_spec = exec_spec or ExecSpec(command=sys.executable, args=str(script))
        if found is None:
            manifest = cls(
                name=script.stem,
                category=NodeScriptCategory,
                singleton=True,
                description=f'ad-hoc node from {script}',
                file=str(script.absolute()),
            )
        else:
            manifest = found
        manifest.exec = exec_spec
        return manifest

    @classmethod
    def from_proc(cls) -> 'NodeManifest':
        """从当前进程自述身份: 以 __main__ 脚本向上认亲, 找不到则降级临时身份."""
        from importlib import import_module
        import inspect
        main = import_module('__main__')
        script_file = Path(inspect.getfile(main))
        exec_spec = ExecSpec(command=sys.executable, args=' '.join(sys.argv))
        return cls.from_script(script_file, exec_spec=exec_spec)


class CellRuntimeInfo(BaseModel):
    """
    MOSS Project 管理 Cell 进程时的数据.
    包含运维面信息: 只有能对该进程直接行动的一侧 (owner / 本机 CLI) 才应消费这些字段.
    """

    # -- 运行时文件命名约定 -- #
    RUNTIME_SUBDIR: ClassVar[str] = 'runtime'
    SUFFIX_JSON: ClassVar[str] = '.json'
    SUFFIX_STDOUT: ClassVar[str] = '.stdout.log'
    SUFFIX_STDERR: ClassVar[str] = '.stderr.log'

    address: CellAddress = Field(
        description="cell 的网络地址.",
    )
    pid: int = Field(
        default=0,
        description="进程 id. 为 0 的话表示还没启动过. ",
    )
    pgid: int = Field(
        default=0,
        description="进程组 id (start_new_session 后即进程自身的组). killpg 的作用对象.",
    )
    start_time: float = Field(
        default_factory=time.time,
        description="进程启动时间戳. 与 pid 一起构成防 pid 复用的核对依据.",
    )
    cell: Cell = Field(
        description="cell 运行时用于重建和广播的数据.",
    )

    @classmethod
    def filename(cls, address: CellAddress, *, suffix: str = SUFFIX_JSON) -> str:
        return normalize(address) + suffix

    @classmethod
    def default_stdout_log(cls, cell_home: Path, address: CellAddress) -> Path:
        filename = cls.filename(address, suffix=cls.SUFFIX_STDOUT)
        return cell_home.joinpath(filename).resolve()

    @classmethod
    def default_stderr_log(cls, cell_home: Path, address: CellAddress) -> Path:
        filename = cls.filename(address, suffix=cls.SUFFIX_STDERR)
        return cell_home.joinpath(filename).resolve()

    @classmethod
    def get_normalized_address_from_file(cls, file: Path) -> str:
        if not file.is_file():
            raise FileNotFoundError(f'{file} is not a valid file')
        return file.stem

    @classmethod
    def filepath(cls, runtime_dir: Path, address: CellAddress) -> Path:
        filename = cls.filename(address)
        return runtime_dir.joinpath(filename).resolve()

    def write_to_runtime_dir(self, runtime_dir: Path) -> None:
        content = self.model_dump_json(indent=0, exclude_defaults=True, exclude_none=True, ensure_ascii=True)
        self.filepath(runtime_dir, self.address).write_text(content)

    def delete_invalid(self, runtime_dir: Path) -> None:
        file = self.filepath(runtime_dir, self.address)
        if file.exists():
            file.unlink()

    @classmethod
    def read_from_runtime_dir(
            cls,
            runtime_dir: Path, address: CellAddress, *, delete_invalid: bool = True,
    ) -> 'CellRuntimeInfo | None':
        filepath = cls.filepath(runtime_dir, address)
        return cls.read_from_file(filepath, delete_invalid=delete_invalid)

    @classmethod
    def read_from_file(
            cls,
            filepath: Path,
            *,
            delete_invalid: bool = True,
    ) -> 'CellRuntimeInfo | None':
        if filepath.exists():
            data = filepath.read_text()
            try:
                info = cls.model_validate_json(data)
                return info
            except Exception:
                if delete_invalid:
                    filepath.unlink()
        return None

    @classmethod
    def iter_runtime_info(cls, runtime_dir: Path) -> 'Iterable[CellRuntimeInfo]':
        for file in runtime_dir.glob(f'*{cls.SUFFIX_JSON}'):
            found = cls.read_from_file(file, delete_invalid=False)
            if found is not None:
                yield found

    @classmethod
    def clear_dead_runtimes(cls, runtime_dir: Path) -> int:
        """扫 runtime_dir 里所有 ledger, 进程已死的连日志文件一起清. 返回清理数."""
        cleaned = 0
        for info in cls.iter_runtime_info(runtime_dir):
            if info.is_alive():
                continue
            info.delete_invalid(runtime_dir)
            for suffix in (cls.SUFFIX_STDOUT, cls.SUFFIX_STDERR):
                f = runtime_dir / cls.filename(info.address, suffix=suffix)
                if f.exists():
                    f.unlink()
            cleaned += 1
        return cleaned

    def is_alive(self) -> bool:
        return psutil.pid_exists(self.pid)

    def locker_name(self) -> str:
        """
        本 cell 的锁名 — singleton 排他机制的唯一权威出口.

        锁真相载体 = env.workspace.lock(locker_name()) 文件锁.
        名字用 fullname (category_name 或 name), 治理域内 fullname 唯一即锁唯一.
        uid 不进锁名: 不同 uid 的同名 cell 才需要互斥, 加 uid 就退化成"永远拿得到锁".
        """
        return normalize(self.cell.fullname)


class CellEvent(BaseModel):
    """
    网络上的 on-change 通知: 一个 cell 广播的变更 hint (推拉结合的推侧).

    事件本身是廉价的推送信号 ("我变了"), 具体内容永远由消费侧按需拉:
      refetch=True → 消费方 refetch Cell 更新缓存 (推拉的拉)
      refetch=False → 消费方仅记事件, 不动缓存 (纯 signal/debug 信号)

    双重消费面 (Watcher 上的两个订阅点分别服务):
      结构变化 → Watcher.on_change (Cell 快照消费者: cache 视图 / CLI 展示)
      注意力候选 → Watcher.on_event (nucleus 消费者: 转 Signal 送 mindflow)
    """
    address: CellAddress = Field(
        description="事件来源 cell 的 address.",
    )
    content: str = Field(
        default='',
        description="事件的自由文本 hint. 可空. 消费方作参考, 不作调度依据.",
    )
    created: AwareDatetime = Field(
        default_factory=lambda: datetime.datetime.now(dateutil.tz.gettz()),
        description="时间签发时间.",
    )
    refetch: bool = Field(
        default=True,
        description="True → 消费方应 refetch Cell 更新缓存 "
                    "(cell 状态/膜类型可能变了); "
                    "False → 仅追加事件缓冲, 缓存不动 (纯 signal/debug).",
    )


@dataclasses.dataclass
class NodeLauncher:
    """
    一个 Node cell 的启动参数打包.

    只描述"怎么起一个进程", 不描述"谁负责起". 生产环境由 spawner
    (通常是 Subprocesses.execute) 消费本 dataclass, 用 start_new_session=True
    起进程, 起完后回填 runtime.pid / runtime.pgid, 再 write_to_runtime_dir.
    """
    cwd: Path
    env: dict[str, str]
    run: list[str]

    runtime: CellRuntimeInfo

    @classmethod
    def from_manifest(
            cls,
            env: Environment,
            manifest: NodeManifest,
    ) -> 'NodeLauncher':
        """筹备运行一个 Cell 节点. """
        cell = build_node_from_manifest(env, manifest)
        cwd = manifest.cwd
        # pid/pgid 留 0, 由 spawner 起进程后回填.
        runtime_info = CellRuntimeInfo(address=cell.address, cell=cell)
        env_data = env.dump_cell_env(cell_address=cell.address, parent_cell_address=env.this_cell_address)
        run = []
        if manifest.exec.command:
            command = manifest.exec.command
            if command == 'python':
                command = sys.executable
            run.append(command)
        run.extend(manifest.exec.arguments)
        return cls(
            cwd=cwd,
            env=env_data,
            run=run,
            runtime=runtime_info,
        )


def make_address(role: CellRole, name: CellName, uid: str) -> str:
    """
    构造 address (§ZZ-10 三段结构).

    :param role: address[0] 保留字, 必须是 CellRole 值域之一.
    :param name: address[1] 治理域路径
    :param uid: address[-1] 唯一性来源, 短随机字符串.
    """
    if not name:
        raise ValueError(
            f'address must have at least one middle segment (kind={role!r}, uid={uid!r})'
        )
    elif not uid:
        raise ValueError(f'address uid must be non-empty (kind={role!r})')
    elif role not in ROLES:
        raise ValueError(f'address role must be in ROLES {ROLES}')
    for seg in (role, name, uid):
        if '/' in seg:
            raise ValueError(f'address segment must not contain "/": {seg!r}')
    return '/'.join([role, name, uid])


def parse_address(address: CellAddress) -> tuple[CellRole, CellName, str]:
    """
    拆解 address 到三 slice: (kind, middle_path, uid).

    :raise ValueError: address 段数 < 3 或 kind 不在 CellRole 值域.
    """
    parts = address.split('/')
    if len(parts) != 3:
        raise ValueError(
            f'address must have at least 3 segments (kind/middle+/uid), got {address!r}'
        )
    role, name, uid = parts
    if role not in ROLES:
        raise ValueError(
            f'address[0] must be in CellRole {ROLES}, got {role!r}'
        )
    elif not name or not uid:
        raise ValueError(f'address {address} parts should not be empty')
    return role, name, uid  # type: ignore[return-value]


def normalize(name_or_address: str) -> str:
    """将名称或 address 归一化为可作文件名 / python 标识符的形式."""
    return (name_or_address.replace('/', '_').replace('\\', '_').
            replace('.', '_').replace('-', '_'))


def build_node_from_manifest(
        env: Environment,
        manifest: 'NodeManifest',
        *,
        name: str = '',
) -> 'Cell':
    """
    基于 Node 的声明来构造一个 Cell 实例.
    :param env: 环境载体
    :param manifest: 本 cell 的 NodeManifest.
    :param name: 给 node 赋予的别名.
    """
    # node uid 每次 spawn 独立生成, 保证 address 全局唯一.
    # 不用 env.session_id: 同一父进程连续 spawn 多个 node 时 session_id 相同会撞.
    uid = unique_id()
    cell_name = name or manifest.name
    if manifest.file:
        # 以发现 node 声明文件的位置作为 cell 的 workspace.
        home = Path(manifest.file).parent.resolve()
    else:
        # 在 workspace 内部为 cell 创建一个临时的 workspace.
        # fullname 表达与 Cell.fullname property 同源 — 二者变则同变.
        if manifest.category:
            fullname = '_'.join([manifest.category, normalize(cell_name)])
        else:
            fullname = normalize(cell_name)
        home = env.cell_runtimes_dir.joinpath(fullname).resolve()
    return Cell(
        role=NODE_ROLE,
        name=cell_name,
        category=manifest.category,
        uid=uid,
        singleton=manifest.singleton,
        project_id=env.project_id,
        project_name=env.project_name,
        home=str(home.absolute()),
    )


def build_host_cell(
        env: Environment,
) -> 'Cell':
    """
    构建一个 host 类型的 cell 节点.

    host address = host / {moss_name} / {project_id}
    - moss_name 来自 MOSS.md.name (workspace 静态声明).
    - project_id 作 uid, 一个 project 内唯一.
    - singleton=True: 同一 project 只能起一个 host.
    """
    return Cell(
        role=HOST_ROLE,
        name=env.moss_meta.name,
        uid=env.project_id,
        category='',
        singleton=True,
        project_id=env.project_id,
        project_name=env.project_name,
        home=str(env.workspace_path.absolute())
    )


def discover_this_node(
        env: Environment,
) -> CellRuntimeInfo:
    """从当前运行时中发现正在运行的 cell runtime info.

    路径:
    1. env.this_cell_address 有值 → 从 runtime dir 读父进程写的文件 (spawn 路径).
    2. runtime file 缺失或损坏 → 降级为从当前进程自述 (from_proc + build).
    3. env.this_cell_address 为空 → 直接走 from_proc 分支 (裸脚本运行).

    最后统一用当前进程 pid 覆盖 runtime_info.pid.
    """
    address = env.this_cell_address
    cell_runtime_info: CellRuntimeInfo | None = None
    if address:
        cell_runtime_info = CellRuntimeInfo.read_from_runtime_dir(env.cell_runtimes_dir, address)
    if cell_runtime_info is None:
        manifest = NodeManifest.from_proc()
        cell = build_node_from_manifest(env, manifest)
        cell_runtime_info = CellRuntimeInfo(
            address=address or cell.address,
            pid=env.pid,
            cell=cell,
        )
    cell_runtime_info.pid = env.pid
    return cell_runtime_info


def clear_cell_runtimes(
        env: Environment,
        kill: Callable[[CellRuntimeInfo], None],
        *,
        throw: bool = False
):
    for found in CellRuntimeInfo.iter_runtime_info(env.cell_runtimes_dir):
        try:
            if found.is_alive():
                kill(found)
            found.delete_invalid(env.cell_runtimes_dir)
        except Exception as e:
            if throw:
                raise e


def enter_cell_lifecycle(
        stack: contextlib.ExitStack,
        env: Environment,
        runtime_info: CellRuntimeInfo,
        kill: Callable[[CellRuntimeInfo], None],
):
    if runtime_info.cell.singleton:
        # 单写者纪律 (§UU-6): singleton cell 的进程锁由 cell 自身争抢, 且必须
        # fast-fail — 父进程 (CLI / matrix.run_node) 只写 ledger, 不抢锁.
        # timeout=0: 撞车立即报, 不阻塞. FileLocker.acquire 默认就是 timeout=0
        # (契约 fast-fail), 此处显式写值表达意图.
        locker = env.workspace.lock(runtime_info.locker_name())
        if not locker.acquire(timeout=0):
            raise DuplicatedError(
                f"singleton cell {runtime_info.cell.fullname!r} lock "
                f"{runtime_info.locker_name()!r} held by another process; "
                f"a live instance already exists."
            )
        stack.callback(locker.release)

    @contextlib.contextmanager
    def _runtime_info_ctx():
        try:
            runtime_info.pid = env.pid
            # host shall key all.
            if runtime_info.cell.is_host:
                clear_cell_runtimes(env, kill=kill)
            runtime_info.write_to_runtime_dir(env.cell_runtimes_dir)
            yield
        finally:
            runtime_info.delete_invalid(env.cell_runtimes_dir)
            if runtime_info.cell.is_host:
                clear_cell_runtimes(env, kill=kill)

    stack.enter_context(_runtime_info_ctx())


class CellPresence(ABC):
    """
    cell 的入网侧: 让自己在网络上可被发现、可被查询、可提供 channel.

    一个 cell 只 announce 一个 presence, 生命周期 = 本对象生命周期.
    """

    @property
    @abstractmethod
    def this(self) -> Cell:
        """本 cell 当前宣告的 presence 内容."""
        ...

    @abstractmethod
    async def provide_channel(self, channel: Channel) -> ChannelProvider:
        """
        立刻将 Channel 提供到网络中. 同时广播更新.
        返回 provider 实例作为可操作句柄.
        """
        ...

    @abstractmethod
    async def publish_event(self, content: str, *, updated: bool = True) -> None:
        """向网络广播一个本 cell 的轻量事件 (CellEvent)."""
        ...

    @abstractmethod
    async def __aenter__(self) -> Self:
        # 1. 管理 This 的 lifecycle (locker 检查, runtime file 写入).
        # 2. 声明 liveness / queryable / event 之类的通讯资源.
        # 3. 广播自己上线.
        ...

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        # 撤回上线声明, 释放 lifecycle 资源. 下线动作归此处, 不单开 revoke.
        ...


class CellNetwork(ABC):
    """
    Matrix 网络的观测与连接层.
    用于发现 Cell 存在与连接 Cell 的能力.
    """

    @abstractmethod
    def view(
            self,
            *,
            project_id: str | None = None,
    ) -> dict[CellAddress, Cell]:
        """
        从缓存中获取最新的 Cell 试图.
        :param project_id: 仅返回指定治理域的 cell (local/foreign 过滤的原料).
        """
        ...

    @abstractmethod
    async def refresh(self, address: CellAddress | None = None) -> dict[CellAddress, Cell]:
        """拉取指定 cell (None=全量) 的最新 presence 并更新视图."""
        ...

    @abstractmethod
    def on_updated(
            self,
            callback: Callable[[Cell, bool], None],
    ) -> Callable[[], None]:
        """
        注册 (Cell, online) 结构变化回调, 返回 unsubscribe 函数.
        触发时机: cell 有增删或 refetch 后内容变化.
        回调可能在网络后台线程触发, 调用方负责线程安全.
        """
        ...

    @abstractmethod
    def on_event(
            self,
            callback: Callable[['CellEvent'], None],
    ) -> Callable[[], None]:
        """
        注册 CellEvent 到达回调, 返回 unsubscribe 函数.
        触发时机: 网络上任何一条 CellEvent 到达 (无论 refetch 值).
        消费者: 将回调事件纳入决策, 或记录日志.
        """
        ...

    @abstractmethod
    async def wait_present(
            self,
            address: CellAddress,
            *,
            timeout: float = 30,
    ) -> Cell | None:
        """
        等待某个 cell 的 presence 出现
        :return: cell, 或超时 None.
        """
        ...

    @abstractmethod
    def has_host(self) -> bool:
        """
        本 network 是否有 host 在运行 (view 层判断).

        host 在 network 级别唯一 — 无需按 project_id 过滤. 消费者 (通常是
        worker cell 或 CLI) 判断组网状态的 code as prompt.
        """
        ...

    # -- 网络域治理动词: accept / reject -- #
    #
    # accept/reject 表达"是否承认某 cell 的资源", 与该 cell 是否在线正交.
    # 实现层维护 accept 表 (含默认接受策略) 和 reject 表:
    #   cell present + 在 accept 表 (或匹配默认策略) → 自动组装 channel_proxy
    #   cell present + 在 reject 表 → 忽略, 不建句柄
    #   cell offline → 已建的句柄按实现决定保留或清理
    # cell update 事件到达时按当前表状态自动重装/撤销资源句柄.

    @abstractmethod
    def set_auto_accept(
            self,
            *,
            local: bool | None = None,
            foreign: bool | None = None,
    ) -> None:
        """
        切换 auto-accept 默认策略. None 表示不改动.

        触发即扫: 调用后立即按新策略扫一遍当前视图 —
          - 新纳入策略 (原不接受, 现接受) 的 cell 会补加进 accept 表 + 组装句柄
          - 移出策略 (原接受, 现不接受) 的 cell 会撤销句柄
        显式 accept/reject 表覆盖默认策略, toggle 不动它们.

        典型使用: 上层 channel (如 cells channel) 通过 command 暴露给模型,
        运行时可自主开关是否自动接纳 foreign cell 的资源.

        :param local: is_local(env) 的 cell 是否自动 accept. None=不改.
        :param foreign: 非 local 的 cell 是否自动 accept. None=不改.
        """
        ...

    @abstractmethod
    async def accept(self, address: CellAddress, *, lookup: bool = False) -> None:
        """
        承认远端 cell 的资源: 加入 accept 表, 若 cell 已 present 立即组装资源句柄.

        :param lookup: True 时若视图中无该 address 会先 refresh 一次再判断;
                      False 时依赖当前视图, 视图中无则等下次 present 时自动生效.
        :raise LookupError: lookup=True 且 refresh 后仍不在网络上.
        """
        ...

    @abstractmethod
    async def reject(self, address: CellAddress) -> None:
        """
        拒绝远端 cell 的资源: 加入 reject 表, 已存在的句柄立即撤回.
        与 accept 对称, 语义是资源承认与否, 不影响对方在线状态.
        """
        ...

    @abstractmethod
    def channel_proxies(self) -> dict[CellAddress, ChannelProxy]:
        """当前已 accept 且 present 的 cell 对应的 channel proxy."""
        ...

    @abstractmethod
    def recent_events(self, *, limit: int = 20) -> list[CellEvent]:
        """最近的网络轻量事件窗口 (ring buffer, 最新优先)."""
        ...

    @abstractmethod
    def cell_events(self, address: CellAddress, *, limit: int = 20) -> list[CellEvent]:
        """某个 cell 的最新事件. """
        ...

    @abstractmethod
    async def __aenter__(self) -> Self:
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        pass


class NodeManager(ABC):
    """
    管理所有 Node 的抽象.
    """

    @abstractmethod
    def list_nodes(
            self,
            refresh: bool = True,
            *,
            paths: list[Path] | None = None,
            installed: bool | None = None,
            include: list[MatchPattern] | None = None,
            exclude: list[MatchPattern] | None = None,
    ) -> dict[ProjectRelativePath, NodeManifest]:
        """
        列出领地内发现的全部 Cell 声明.
        :param refresh: 重新扫描文件系统.
        :param paths: 指定扫描的根目录, 否则使用默认的.
        :param installed: None=全部; True=仅已安装; False=仅未安装.
        :param include: 匹配模式筛选.
        :param exclude: 排除模式筛选.
        """
        ...

    @abstractmethod
    def get_node(self, relative_path: 'str | Path') -> 'NodeManifest | None':
        """获取指定目录路径的 Cell 声明. 目录路径用 '/' 分割."""
        ...

    @abstractmethod
    def get_node_launcher(self, relative_path: 'str | Path') -> NodeLauncher | None:
        ...

    @staticmethod
    def match_nodes(
            cells: dict[ProjectRelativePath, NodeManifest],
            include: list[MatchPattern] | None = None,
            *,
            exclude: list[MatchPattern] | None = None,
    ) -> Iterable[tuple[ProjectRelativePath, NodeManifest]]:
        """基于 fnmatch 通配符筛选 Cell. include 为空时返回全部 (仅受 exclude 约束)."""
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


class DuplicatedError(RuntimeError):
    """cell 重复启动异常. singleton 声明的执法产物, 错误信息应引用声明原文."""
