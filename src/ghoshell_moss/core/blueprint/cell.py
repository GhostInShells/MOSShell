"""
Cell — a unit node in the MOSS communication network, projected into the Matrix under an
address.

This module defines the Cell abstraction (``Cell``, ``CellAddress``, ``CellProtocol``),
the node manifest that declares a cell (``NodeManifest``), and the entry points that
build and tear down cells (``build_cell_from_node``, ``enter_cell_lifecycle``).

Cell 是在 Moss 的网络 (Matrix) 中运行的进程单元. 可以将之想象为一个数字城市中的一个房间, 用来提供不同的功能.
Cell 可以用来控制机器人, 创建图形界面, 运行独立的思考或 Agent. 可以把它理解为手机里的 App.
Cell 运行时, 通过映射进入 Matrix 网络, Cell 之间的通讯可以通过文件或者 Matrix.session 协议.

每个 Cell 都有自己的地址 - address, 类似于电话; 关于 cell 的通讯基本都从 address 出发.
当 Cell 通过 Matrix 提供 Channel 等能力时, 将能够让 Moss 在运行时控制它. 一个通过 moss 启动的 Ghost 可以用 Channel 控制它看到的 Cell.

各种各样的 Cell 运行时组织成 Matrix 的网络, 通过 Host 节点提供给 Ghost 一个可用的操作系统.
"""
import contextlib
import os
import sys
import time
from enum import IntEnum
from pathlib import Path
from typing import Callable, ClassVar, Iterable, Literal
from typing_extensions import Self
from abc import ABC, abstractmethod

import fnmatch
import frontmatter
import shlex
from pydantic import BaseModel, Field, AwareDatetime

from ghoshell_moss.core.concepts.channel import Channel, ChannelProvider, ChannelProxy
from ghoshell_moss.contracts.subprocesses import CaptureSpec, ManagedProcess
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
    'CellEventLevel',
    'HOST_ROLE',
    'NODE_ROLE',
    'normalize',
    'make_address',
    'parse_address',
    'build_cell_from_node',
    'build_host_cell',
    'discover_this_node',
    'ExecSpec',
    'NodeManifest',
    'CellRuntimeInfo',
    'Cell',
    'CellEvent',
    'CELL_EVENT_CHANNEL_ADDED',
    'CellPresence',
    'CellNetwork',
    'AutoAcceptPolicy',
    'NodeManager',
    'DuplicatedError',

    'CellName',
    'CellNamePattern',
    'AbsolutePath',
    'ProjectRelativePath',
    'MatchPattern',
    'NodeLauncher',
    'enter_cell_lifecycle',
    'CellAddressCodec', 'NodeProbeError',
]

CellRole = Literal['host', 'node']
"""
Role of a cell in the Matrix network topology.

- 'host': the network's central node — organizes all capabilities for a body or agent to drive.
- 'node': a functional node — may control a body, provide a GUI, run an independent app, etc.
"""
HOST_ROLE: CellRole = 'host'
NODE_ROLE: CellRole = 'node'

ROLES = frozenset({HOST_ROLE, NODE_ROLE})


class CellEventLevel(IntEnum):
    """Perception level of a cell lifecycle event — aligned to logging levels.

    Perception follows logging filter semantics:
      event_level >= INFO (threshold) -> send_signal (perceivable)
      event_level <  INFO             -> no send_signal (zero-cost); kept in event_buffer (pullable)

    Mapping: DEBUG->drop, INFO->BACKGROUND, WARNING->WARNING, ERROR->ERROR, CRITICAL->CRITICAL.
    """

    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50

    @classmethod
    def resolve(cls, level: 'CellEventLevel | None') -> 'CellEventLevel':
        """Normalize None (system default) to INFO — the single entry point for the perception check."""
        return level if level is not None else cls.INFO

    @classmethod
    def is_perceivable(cls, level: 'CellEventLevel | None') -> bool:
        """Whether this level produces a ghost signal (below the INFO threshold it does not)."""
        return cls.resolve(level) >= cls.INFO

CellProtocol = Literal['channel']
"""
A MOSS duplex protocol a cell provides. Closed set — each value maps to a Provider/Proxy
role pair + event vocabulary in duplex/. Adding a protocol means adding that implementation
and appending the value here; it is not a free string.
"""

CellAddress = str
"""
The cell's unique network address: CellRole / unique_name / uid
"""

ProjectRelativePath = str
AbsolutePath = str
MatchPattern = str
"""Wildcard pattern: group/name, group/*, *, */*, */name"""

CellName = str
# 不硬约束 -/. : from_script 从文件名 (moss-ghost 等) 导出的 name 会反复炸.
CellNamePattern = r"^[a-zA-Z0-9_.-]+$"
"""A cell name is a governance-domain path segment, allowing ``-`` ``.``.

``make_address`` normalizes ``-``/``.`` to ``_`` for identifier safety; ``Cell.name``
keeps the original value.
"""


class Cell(BaseModel):
    """
    The declaration of a **running** node in the Matrix network — a room in the building
    that is Matrix. Usually one process.
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
        description="Whether this cell keeps a single instance within the governance domain. "
                    "True: a live same-name cell rejects a duplicate launch (DuplicatedError) — "
                    "for hardware-exclusive (mic/camera/robot) or state-exclusive (database) cases. "
                    "False (default): multiple instances run in parallel, each with its own uid.",
    )
    category: Literal['ghost', 'shell', 'script'] | str = Field(
        default='',
        pattern=r"^[a-zA-Z0-9_]*$",
        description="The cell's category.",
    )
    event_level: CellEventLevel | None = Field(
        default=None,
        description="Perception level of this cell's lifecycle events for listeners. "
                    "None = system default (resident node -> INFO perceivable, one-shot -> DEBUG silent). "
                    "Below the INFO threshold no ghost signal is produced — the event stays "
                    "in event_buffer (pullable) but does not enter attention.",
    )
    description: str = Field(
        default='',
        description="cell description",
    )
    project_id: str = Field(
        default='',
        description="Governance-domain (project) id of this cell; used to tell local from foreign cells.",
    )
    project_name: str = Field(
        default='',
        description="Name of the project this cell belongs to.",
    )
    providing: list[CellProtocol] = Field(
        default_factory=list,
        description="The MOSS duplex protocols this cell currently provides. "
                    "Marks protocol names only — content is pulled through each Provider/Proxy bridge. "
                    "Value domain is closed by CellProtocol; adding one requires landing the role pair in duplex/.",
    )
    updated: AwareDatetime = Field(
        default_factory=lambda: datetime.datetime.now(dateutil.tz.gettz()),
        description="Timestamp of this cell's last update.",
    )
    home: str = Field(
        description="Process working directory, absolute path.",
    )
    persist: bool = Field(
        default=False,
    )
    parent_address: str = Field(
        default='',
        description="The parent node running this cell.",
    )

    def update(self) -> None:
        self.updated = datetime.datetime.now(dateutil.tz.gettz())

    @property
    def address(self) -> CellAddress:
        return make_address(self.role, self.name, self.uid)

    @property
    def address_codec(self) -> 'CellAddressCodec':
        return CellAddressCodec(self.address)

    @property
    def fullname(self) -> str:
        if self.category:
            return '_'.join([self.category, normalize(self.name)])
        return normalize(self.name)

    @property
    def is_host(self) -> bool:
        """Whether this cell is the network's host."""
        return self.role == HOST_ROLE

    def is_local(self, env: Environment) -> bool:
        return self.project_id == env.project_id

    @property
    def unique_name(self) -> str:
        return self.address_codec.short


class ExecSpec(BaseModel):
    """
    Declaration of a process to run (usually a Node).
    """
    command: Literal['python'] | str = Field(
        default='python',
        description="When non-empty, argv[0] of the launch command. "
                    "Should be a path relative to cwd.",
    )
    args: str = Field(
        default='main.py',
        description="Argument list of the launch command.",
    )
    env: dict[str, str] = Field(
        default_factory=dict,
        description="Extra environment variables to inject. The launch script may also load its own via dotenv.",
    )
    timeout: float | None = Field(
        default=None,
        description="Process timeout in seconds. Currently consumed only by the NodeManifest.check "
                    "probe: on timeout it is judged broken and its process group is terminated. "
                    "None = unlimited. (Not consumed by the main exec path yet.)",
    )

    @property
    def arguments(self) -> list[str]:
        return shlex.split(self.args)


NodeScriptCategory = 'script'


class NodeManifest(BaseModel):
    """
    Declaration of a Node-type cell — defined through a declaration file, and also
    supporting script-launch scenarios without one.
    """
    MANIFEST_FILENAME: ClassVar[str] = 'NODE.md'
    """The conventional declaration file — think of it as a Windows shortcut. """

    INSTALL_FILENAME: ClassVar[str] = 'INSTALL.md'
    """The file describing how to install. When present, it pairs with INSTALLED_FILE to
    mark install state — a node with its own project dependencies declares its steps here."""

    INSTALLED_FILE: ClassVar[str] = '.installed'
    """A file marker recording whether a node has completed installation."""

    name: str = Field(
        description="The node's name — its identity anchor within the governance domain.",
        pattern=CellNamePattern,
    )
    description: str = Field(
        default='',
        description="One-line description of the node.",
    )
    category: str = Field(
        default='',
        description="A free-form category label (e.g. sensors / bodies / scripts / tools) that drives no mechanism.",
        pattern=r"^[a-zA-Z0-9_]*$",
    )
    singleton: bool = Field(
        default=True,
        description="Whether this cell keeps a single instance within the governance domain. "
                    "True (default): a live same-name cell rejects a duplicate launch — "
                    "for hardware-exclusive (mic/camera/robot) or state-exclusive (database) cases. "
                    "False: multiple instances run in parallel, each with its own uid.",
    )
    persist: bool = Field(
        default=True,
        description="Whether this node is resident. "
                    "True (default): resident node cell — provides a channel long-running, "
                    "lifecycle events are perceivable (event_level INFO by convention). "
                    "False: one-shot run-to-completion — events silent (event_level=DEBUG), "
                    "no channel provided, results read via a blocking nodes:run for stdout/stderr/exitcode.",
    )
    exec: 'ExecSpec' = Field(
        default_factory=ExecSpec,
        description="Default launch entry (frontmatter `run:` declaration). "
                    "A cell without one can only be launched by an explicit script path.",
    )
    check: 'ExecSpec | None' = Field(
        default=None,
        description="Pre-launch probe (frontmatter `check:` declaration), an independent process "
                    "requiring zero cooperation from the target. "
                    "exit 0 -> pass, launch the main script; nonzero + stderr -> broken reason, do not launch. "
                    "Verifies whether the environment can run now (real dependency import / smoke call). "
                    "Omitted -> probe skipped.",
    )
    instruction: str = Field(
        default='',
        description="Detailed usage instructions for the node.",
    )
    installed: bool = Field(
        default=True,
        description="Whether installation is complete. An uninstalled cell is discoverable but "
                    "refuses to launch; the error points at the INSTALL.md path. Derived from the "
                    "filesystem, not from frontmatter.",
    )

    file: AbsolutePath = Field(
        default='',
        description="Absolute path of the file this NodeManifest was generated from.",
    )

    @property
    def cwd(self) -> Path:
        if self.file:
            return Path(self.file).parent.resolve()
        return Path.cwd()

    @classmethod
    def read_from_file(cls, file: Path) -> 'NodeManifest':
        """Read the declaration from a NODE.md file. The body is the instruction; the
        frontmatter is the fields."""
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
        """Read the manifest from a directory."""
        file = directory.joinpath(cls.MANIFEST_FILENAME)
        if file.is_file():
            return cls.read_from_file(file)
        return None

    def save(self) -> None:
        file = Path(self.file).absolute()
        self.write_file(file)

    def write_file(self, directory: Path, filename: str = '') -> None:
        """Write the declaration to NODE.md."""
        filename = filename or self.MANIFEST_FILENAME
        data = self.model_dump(
            exclude_none=True,
            exclude={'instruction', 'installed', 'file'},
        )
        post = frontmatter.Post(content=self.instruction, **data)
        frontmatter.dump(post, directory.joinpath(filename).resolve())

    @classmethod
    def find_upward(cls, start: Path) -> 'NodeManifest | None':
        """Walk up from ``start`` to the nearest NODE.md (stops at the first hit)."""
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
        Build a Manifest from a script entry point: adopt the nearest NODE.md found by
        walking upward. When none is found, degrade to an ad-hoc identity rather than
        refusing to run.
        """
        script = script.resolve()
        found = cls.find_upward(script)
        exec_spec = exec_spec or ExecSpec(command=sys.executable, args=str(script))
        if found is None:
            manifest = cls(
                name=script.stem,
                category=NodeScriptCategory,
                singleton=True,
                persist=False,
                description=f'ad-hoc node from {script}',
                file=str(script.absolute()),
            )
        else:
            manifest = found
        manifest.exec = exec_spec
        return manifest

    @classmethod
    def from_proc(cls) -> 'NodeManifest':
        """Build identity from the current process: walk up from the __main__ script; degrade
        to a temporary identity when none is found."""
        from importlib import import_module
        import inspect
        main = import_module('__main__')
        script_file = Path(inspect.getfile(main))
        exec_spec = ExecSpec(command=sys.executable, args=' '.join(sys.argv))
        return cls.from_script(script_file, exec_spec=exec_spec)

    @classmethod
    def new(
            cls,
            name: str,
            *,
            description: str = '',
            category: str = '',
    ) -> 'NodeManifest':
        """Create a node manifest in the current process."""
        manifest = cls.from_proc()
        manifest.name = name
        manifest.description = description
        manifest.category = category
        return manifest


class CellRuntimeInfo(BaseModel):
    """
    Runtime data MOSS Project keeps for a cell process — the operational surface: only the
    side that can act directly on the process (owner / local CLI) should consume these fields.
    """

    # -- 运行时文件命名约定 -- #
    RUNTIME_SUBDIR: ClassVar[str] = 'runtime'
    SUFFIX_JSON: ClassVar[str] = '.json'
    SUFFIX_STDOUT: ClassVar[str] = '.stdout.log'
    SUFFIX_STDERR: ClassVar[str] = '.stderr.log'

    address: CellAddress = Field(
        description="The cell's network address.",
    )
    pid: int = Field(
        default=0,
        description="Process id; 0 means not yet started.",
    )
    pgid: int = Field(
        default=0,
        description="Process group id (the process's own group after start_new_session); the target of killpg.",
    )
    start_time: float = Field(
        default_factory=time.time,
        description="Process start timestamp; combined with pid as a guard against pid reuse.",
    )
    cell: Cell = Field(
        description="The cell's runtime data, used to rebuild and broadcast identity.",
    )

    @classmethod
    def from_cell(cls, cell: Cell) -> 'CellRuntimeInfo':
        return cls(address=cell.address, cell=cell)

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
        """Scan all ledgers in runtime_dir; clear dead processes together with their log files. Returns the cleared count."""
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
        This cell's lock name — the single authority for the singleton exclusion mechanism.

        The lock's carrier is the file lock env.workspace.lock(locker_name()). The name is
        the fullname (category_name or name); within a governance domain fullname uniqueness
        is lock uniqueness. The uid is deliberately not in the lock name — same-name cells
        across different uids are the ones that must exclude each other.
        """
        return normalize(self.cell.fullname)


CELL_EVENT_CHANNEL_ADDED = 'channel added'
"""CellEvent content a cell publishes when it provides a channel.

The channel dimension's truth lives on the observer side (mesh mount), not the
producer's self-report — the consumer filters this out of the signal path.
"""


class CellEvent(BaseModel):
    """
    An on-change notification on the network: the change hint a cell broadcasts (the push
    side of push-pull).

    The event itself is a cheap push ("I changed"); the concrete content is always pulled
    on demand by the consumer:
      refetch=True  -> consumer refetches the Cell to update its cache (the pull)
      refetch=False -> consumer only records the event, cache untouched (pure signal/debug)

    Two consumption surfaces (two subscription points on the Watcher):
      structural change -> Watcher.on_change (Cell snapshot consumers: cache view / CLI display)
      attention candidate -> Watcher.on_event (nucleus consumers: turn into Signal for mindflow)
    """
    address: CellAddress = Field(
        description="Address of the cell the event came from.",
    )
    content: str = Field(
        default='',
        description="Free-text hint of the event; may be empty. A reference for the consumer, not a scheduling criterion.",
    )
    created: AwareDatetime = Field(
        default_factory=lambda: datetime.datetime.now(dateutil.tz.gettz()),
        description="Time the event was issued.",
    )
    refetch: bool = Field(
        default=True,
        description="True -> the consumer should refetch the Cell to update its cache "
                    "(cell state / membrane type may have changed); "
                    "False -> only append the event buffer, cache untouched (pure signal/debug).",
    )
    event_level: CellEventLevel | None = Field(
        default=None,
        description="Perception level (CellEventLevel) of the source cell. "
                    "The listener decides whether to produce a ghost signal: "
                    "below the INFO threshold no send_signal, kept pullable.",
    )

    @property
    def address_codec(self) -> 'CellAddressCodec':
        return CellAddressCodec(self.address)


@dataclasses.dataclass
class NodeLauncher:
    """
    The launch parameters of a node cell, packaged.

    Describes only "how to start a process", not "who starts it". The spawner (usually
    Subprocesses.execute) consumes this dataclass, starts the process with
    start_new_session=True, then backfills runtime.pid / runtime.pgid and writes to
    the runtime dir.
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
        """Prepare the launch of a node cell."""
        cell = build_cell_from_node(env, manifest)
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
    Build an address from three segments.

    :param role: address[0] — a reserved word, one of the CellRole values.
    :param name: address[1] — the governance-domain path.
    :param uid: address[-1] — the uniqueness source, a short random string.
    """
    # name 是治理域路径, -/. 归一化为 _ 保持 address 标识符安全;
    # 原始值留在 Cell.name, 此处是 address 生成的唯一落点.
    name = name.replace('-', '_').replace('.', '_')
    return CellAddressCodec.make(role, name, uid).address


def parse_address(address: CellAddress) -> tuple[CellRole, CellName, str]:
    """
    Split an address into three slices: (role, name, uid).

    :raise ValueError: fewer than 3 segments, or role not in CellRole.
    """
    return CellAddressCodec.parse(address)


def normalize(name_or_address: str) -> str:
    """Normalize a name or address into a filename / python-identifier-safe form."""
    return CellAddressCodec.normalize(name_or_address)


class CellAddressCodec:
    """Form conversion and validation for CellAddress (str).

    The address stays a str representation (type alias); this class is the single entry
    point for converting, displaying and matching it. Types that hold an address
    (Cell / CellEvent) expose an instance via ``.address_codec`` instead of parsing the
    address string themselves.
    """

    SHORT_UID_LEN = 6

    def __init__(self, address: CellAddress, *, validate: bool = True) -> None:
        self.address: CellAddress = address
        self._parts: tuple[CellRole, CellName, str] | None = None
        if validate:
            self._parts = self.parse(address)

    @property
    def parts(self) -> tuple[CellRole, CellName, str]:
        if self._parts is None:
            self._parts = self.parse(self.address)
        return self._parts

    @property
    def role(self) -> CellRole:
        return self.parts[0]

    @property
    def name(self) -> CellName:
        return self.parts[1]

    @property
    def uid(self) -> str:
        return self.parts[2]

    # -- 别名 -------------------------------------------------

    @property
    def short(self) -> str:
        """short 形态: ``name_uid[-6:]``, 全链统一的地址短标."""
        # 取尾部随机段而非头部: uid 是 ULID, 头部 10 字符是毫秒时间戳,
        # 同 name 多实例在 ~4.4 分钟内 `uid[:6]` 相同 → 短标撞车.
        # 尾部落在 80 位随机段, 每个 spawn 唯一.
        return f'{self.name}_{self.uid[-CellAddressCodec.SHORT_UID_LEN:]}'

    @property
    def dot_address(self) -> str:
        """Dot-separated form: ``role.name.uid``."""
        return self.address.replace('/', '.')

    @classmethod
    def from_dot_address(cls, dot_address: str) -> 'CellAddressCodec':
        """Rebuild from the dot-separated form (best effort)."""
        return cls(dot_address.replace('.', '/'), validate=True)

    @property
    def normalized(self) -> str:
        """Filesystem-safe form: ``/`` ``.`` ``-`` replaced with ``__``."""
        return self.normalize(self.address)

    @classmethod
    def from_normalized(cls, normalized: str) -> 'CellAddressCodec':
        """Rebuild from normalize output (best effort); a validate failure raises ValueError."""
        return cls(normalized.replace('__', '/'), validate=True)

    # -- from / to ---------------------------------------------

    @classmethod
    def make(cls, role: CellRole, name: CellName, uid: str) -> 'CellAddressCodec':
        """Build an address from three segments (role/name/uid)."""
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
        return cls('/'.join([role, name, uid]))

    @classmethod
    def parse(cls, address: CellAddress) -> tuple[CellRole, CellName, str]:
        """Parse the three segments (role, name, uid)."""
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

    @classmethod
    def normalize(cls, name_or_address: str) -> str:
        """Normalize into a filesystem-safe name: ``/`` ``.`` ``-`` -> ``__``."""
        return (name_or_address.replace('/', '__').replace('\\', '__').
                replace('.', '__').replace('-', '__'))

    def __str__(self) -> str:
        return self.address

    def __repr__(self) -> str:
        return f'CellAddressCodec({self.address})'

    # -- 匹配 -------------------------------------------------

    def match(self, query: str) -> bool:
        """Whether ``query`` matches this address.

        Five paths, by priority: exact full address -> exact short -> exact name segment
        -> uid prefix (>= 3 chars) -> address prefix (>= 3 chars). An empty / one-two
        char query never matches — a semantic threshold against false matches.
        """
        if not query:
            return False
        if query == self.address:
            return True
        if query == self.short:
            return True
        if query == self.name:
            return True
        if len(query) >= 3:
            if self.uid.startswith(query):
                return True
            if self.address.startswith(query):
                return True
        return False

    @classmethod
    def suggest(
            cls,
            query: str,
            candidates: Iterable[CellAddress],
            *,
            limit: int = 3,
    ) -> list[CellAddress]:
        """did you want? — collect approximate hits from candidates (name prefix/substring,
        uid prefix).

        A fallback hint for a parse failure or ambiguity, turning fuzzy input into a
        correctable dialog. Returns the full addresses of the candidates.
        """
        if not query:
            return []
        q = query.lower()
        scored: list[tuple[int, CellAddress]] = []
        for addr in candidates:
            try:
                _, name, uid = parse_address(addr)
            except ValueError:
                continue
            nl, ul = name.lower(), uid.lower()
            score = 0
            if ul.startswith(q):
                score += 3
            if nl == q:
                score += 4
            elif nl.startswith(q):
                score += 2
            elif q in nl:
                score += 1
            if score:
                scored.append((score, addr))
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [addr for _, addr in scored[:limit]]


def build_cell_from_node(
        env: Environment,
        manifest: 'NodeManifest',
        *,
        name: str = '',
) -> 'Cell':
    """
    Build a Cell instance from a Node declaration.
    :param env: the environment carrier.
    :param manifest: this cell's NodeManifest.
    :param name: an alias to give the node.
    """
    # node uid 每次 spawn 独立生成, 保证 address 全局唯一.
    # 不用 env.run_id: 同一父进程连续 spawn 多个 node 时 run_id 相同会撞.
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
    # NodeManifest 暂未暴露 event_level; 显式值优先, persist 仅作未声明时的兜底.
    event_level = getattr(manifest, 'event_level', None)
    if event_level is None:
        event_level = None if manifest.persist else CellEventLevel.DEBUG
    return Cell(
        role=NODE_ROLE,
        name=cell_name,
        category=manifest.category,
        persist=manifest.persist,
        uid=uid,
        singleton=manifest.singleton,
        event_level=event_level,
        project_id=env.project_id,
        project_name=env.project_name,
        home=str(home.absolute()),
    )


def build_host_cell(
        env: Environment,
) -> 'Cell':
    """
    Build a host-type cell.

    host address = host / {moss_name} / {project_id}
    - moss_name comes from MOSS.md.name (the workspace static declaration).
    - project_id acts as the uid, unique within one project.
    - singleton=True: a project can run at most one host.
    """
    return Cell(
        role=HOST_ROLE,
        name=env.moss_meta.name,
        uid=env.project_id,
        category='',
        persist=True,
        singleton=True,
        project_id=env.project_id,
        project_name=env.project_name,
        home=str(env.workspace_path.absolute())
    )


def discover_this_node(
        env: Environment,
) -> CellRuntimeInfo:
    """Discover the running cell runtime info from the current runtime.

    Paths:
    1. env.this_cell_address set -> read the file the parent wrote into the runtime dir (spawn path).
    2. runtime file missing or corrupt -> degrade to self-description from the current process (from_proc + build).
    3. env.this_cell_address empty -> go straight to the from_proc branch (bare script run).

    Finally, override runtime_info.pid with the current process pid.
    """
    address = env.this_cell_address
    cell_runtime_info: CellRuntimeInfo | None = None
    if address:
        cell_runtime_info = CellRuntimeInfo.read_from_runtime_dir(env.cell_runtimes_dir, address)
    if cell_runtime_info is None:
        manifest = NodeManifest.from_proc()
        cell = build_cell_from_node(env, manifest)
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
            pgid = _current_pgid(env.pid)
            if pgid is not None:
                runtime_info.pgid = pgid
            runtime_info.write_to_runtime_dir(env.cell_runtimes_dir)
            yield
        finally:
            runtime_info.delete_invalid(env.cell_runtimes_dir)
            if runtime_info.cell.is_host:
                clear_cell_runtimes(env, kill=kill)

    stack.enter_context(_runtime_info_ctx())


def _current_pgid(pid: int) -> int | None:
    """The current process group id — returned when the system supports it (POSIX getpgid), else None (Windows fallback)."""
    if not hasattr(os, 'getpgid'):
        return None
    try:
        return os.getpgid(pid)
    except OSError:
        # 进程已退出或组不可得
        return None


class CellPresence(ABC):
    """
    A cell's network-facing side: making itself discoverable, queryable and channel-providing
    on the network.

    A cell announces exactly one presence; its lifecycle equals this object's lifecycle.
    """

    @property
    @abstractmethod
    def this(self) -> Cell:
        """The presence content this cell currently announces."""
        ...

    @abstractmethod
    async def provide_channel(self, channel: Channel) -> ChannelProvider:
        """
        Provide the Channel to the network immediately, broadcasting an update.
        Returns the provider instance as an operable handle.
        """
        ...

    @abstractmethod
    async def publish_event(
            self,
            content: str,
            *,
            updated: bool = True,
            event_level: CellEventLevel | None = None,
    ) -> None:
        """Broadcast a lightweight event (CellEvent) of this cell to the network.

        event_level: override this cell's default perception level; None = use cell.event_level.
        """
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


@dataclasses.dataclass(frozen=True)
class AutoAcceptPolicy:
    """The two switches of the auto-accept default policy (the read side, symmetric to
    ``CellNetwork.set_auto_accept``).

    The explicit accept / reject tables override this policy — it only decides how cells
    that have not been explicitly stated are treated by default. When both switches are
    on, the network accepts all resources by default and explicit accept / reject become
    meaningless.
    """
    local: bool
    foreign: bool


class CellNetwork(ABC):
    """
    The observation and connection layer of the Matrix network — discovers cell presence
    and connects cell capabilities.
    """

    @abstractmethod
    def view(
            self,
            *,
            project_id: str | None = None,
    ) -> dict[CellAddress, Cell]:
        """
        Get the latest Cell view from the cache.
        :param project_id: only return cells of the given governance domain (the raw material for local/foreign filtering).
        """
        ...

    @abstractmethod
    async def refresh(self, address: CellAddress | None = None) -> dict[CellAddress, Cell]:
        """Fetch the latest presence of the given cell (None = all) and update the view."""
        ...

    @abstractmethod
    def on_updated(
            self,
            callback: Callable[[Cell, bool], None],
    ) -> Callable[[], None]:
        """
        Register a (Cell, online) structural-change callback; returns the unsubscribe function.
        Fires when: a cell is added/removed, or its content changes after a refetch.
        The callback may fire on a network background thread; the caller owns thread safety.
        """
        ...

    @abstractmethod
    def on_event(
            self,
            callback: Callable[['CellEvent'], None],
    ) -> Callable[[], None]:
        """
        Register a CellEvent-arrival callback; returns the unsubscribe function.
        Fires when: any CellEvent arrives on the network (regardless of its refetch value).
        Consumers: fold the event into decisions, or log it.
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
        Wait for a cell's presence to appear.
        :return: the cell, or None on timeout.
        """
        ...

    @abstractmethod
    def has_host(self) -> bool:
        """
        Whether this network has a host running (decided at the view layer).

        A host is unique at the network level — no project_id filtering needed. Consumers
        (usually a worker cell or the CLI) use this as code-as-prompt to judge group state.
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
        Toggle the auto-accept default policy. None means no change.

        Fires a scan: immediately re-scans the current view under the new policy —
          - cells newly brought into policy (was not accepted, now is) get added to the accept table + handle assembled
          - cells moved out of policy (was accepted, now not) get their handle revoked
        The explicit accept/reject tables override the default policy; toggling does not touch them.

        Typical use: an upper channel (like the mesh channel) exposes this to the model via a
        command, toggling at runtime whether foreign cells' resources are auto-accepted.

        :param local: whether is_local(env) cells are auto-accepted. None = no change.
        :param foreign: whether non-local cells are auto-accepted. None = no change.
        """
        ...

    @abstractmethod
    def auto_accept(self) -> AutoAcceptPolicy:
        """
        The current auto-accept default policy — the read side of set_auto_accept.

        Without a read side, consumers can only hardcode their own "can't read" constant (the
        policy state becomes unknowable) and policy visibility is lost. Must be implemented
        in pair with set_auto_accept.

        Reads the policy only; whether a cell has been accepted lives in the accept/reject
        tables, see channel_proxies.
        """
        ...

    @abstractmethod
    async def accept(self, address: CellAddress, *, lookup: bool = False) -> None:
        """
        Acknowledge a remote cell's resources: add to the accept table, and assemble the
        resource handle immediately if the cell is already present.

        :param lookup: if True, refresh once first when the address is not in the view;
                      if False, rely on the current view, taking effect on the next presence.
        :raise LookupError: lookup=True and still not on the network after refresh.
        """
        ...

    @abstractmethod
    async def reject(self, address: CellAddress) -> None:
        """
        Reject a remote cell's resources: add to the reject table and revoke any existing
        handle immediately. Symmetric to accept — it is about resource acknowledgement, not
        the other side's online status.
        """
        ...

    @abstractmethod
    def channel_proxies(self) -> dict[CellAddress, ChannelProxy]:
        """The channel proxies of the cells that are currently accepted and present."""
        ...

    @abstractmethod
    def recent_events(self, *, limit: int = 20) -> list[CellEvent]:
        """The recent network lightweight-event window (ring buffer, newest first)."""
        ...

    @abstractmethod
    def cell_events(self, address: CellAddress, *, limit: int = 20) -> list[CellEvent]:
        """The latest events of a specific cell."""
        ...

    @abstractmethod
    async def __aenter__(self) -> Self:
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        pass


class NodeManager(ABC):
    """
    Abstraction managing all nodes.
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
        List all cell declarations discovered in the territory.
        :param refresh: re-scan the filesystem.
        :param paths: roots to scan; default roots when omitted.
        :param installed: None = all; True = installed only; False = uninstalled only.
        :param include: include-match filter.
        :param exclude: exclude-match filter.
        """
        ...

    @abstractmethod
    def get_node(self, relative_path: 'str | Path') -> 'NodeManifest | None':
        """Get the cell declaration at a given directory path (segments separated by '/')."""
        ...

    @abstractmethod
    def get_node_launcher(self, relative_path: 'str | Path') -> NodeLauncher | None:
        ...

    @abstractmethod
    def resolve_node(self, target: 'str | Path') -> 'NodeManifest':
        """Resolve target -> NodeManifest.

        A relative path resolves against project root and is made absolute; a NODE.md path is
        read directly; a directory looks for NODE.md inside; a script uses
        NodeManifest.from_script to walk up and claim its parent.
        :raise FileNotFoundError: target does not exist.
        :raise LookupError: no NODE.md in the directory.
        """
        ...

    @staticmethod
    def match_nodes(
            cells: dict[ProjectRelativePath, NodeManifest],
            include: list[MatchPattern] | None = None,
            *,
            exclude: list[MatchPattern] | None = None,
    ) -> Iterable[tuple[ProjectRelativePath, NodeManifest]]:
        """Filter cells by fnmatch wildcards. Empty include returns all (only constrained by exclude)."""
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
    async def spawn_node(
            self,
            manifest: NodeManifest,
            *,
            extra_env: dict[str, str] | None = None,
            extra_args: list[str] | None = None,
            capture: Callable[[CellRuntimeInfo], CaptureSpec] | None = None,
    ) -> tuple[CellRuntimeInfo, ManagedProcess]:
        """
        Launch a node cell — the single spawn choke point.

        Only does: installed check -> NodeLauncher packing -> probe gate (manifest.check) ->
        singleton pre-check (read-only is_locked; raises DuplicatedError on collision) -> write
        the first ledger entry (identity uid, pid/pgid placeholder 0) -> Subprocesses.execute.
        Does not do: holding the singleton lock / ledger cleanup / pid·pgid backfill — those
        belong to the child enter_cell_lifecycle.

        extra_args: extra argv tokens appended after the declared ``exec.args`` —
        for per-instance identity/binding (device index, stream address, ...).
        Append-only; ``exec.args`` is never replaced. The pre-launch probe does
        not receive them.

        capture: an optional factory taking the packed CellRuntimeInfo and returning a
        CaptureSpec (the on-disk path can use runtime.address). None = no capture (inherit terminal).

        A failed probe (manifest.check) raises NodeProbeError; a singleton lock collision raises
        DuplicatedError; a failed installed check raises RuntimeError. Returns (runtime, managed)
        — runtime lets the caller assemble a CellHandle / track.
        """
        ...

    @abstractmethod
    def list_runtimes(self) -> list[CellRuntimeInfo]:
        """Read the ledger and return all launched cell runtimes in this governance domain (host + node)."""
        ...

    @abstractmethod
    def get_runtime(self, address: CellAddress) -> CellRuntimeInfo | None:
        """Read a single runtime by address; None when not in the ledger."""
        ...

    @abstractmethod
    def kill_cell(self, address: CellAddress, *, force: bool = False) -> bool:
        """Terminate a cell process (SIGTERM -> grace -> SIGKILL) and clear its ledger.

        :return: True = address is in this domain's ledger, termination + cleanup attempted;
                 False = not in the ledger, no-op.
        """
        ...

    @abstractmethod
    def prune(self, *, keep_alive: bool = False, force: bool = False) -> tuple[int, int, int]:
        """Clear orphan runtime ledgers. Returns (removed, killed, skipped).

        By default kills live orphans (they hold singleton locks); keep_alive=True removes only
        dead ledgers.
        """
        ...


class DuplicatedError(RuntimeError):
    """Duplicate cell-launch error, produced by singleton enforcement; the message should quote the declaration."""


class NodeProbeError(RuntimeError):
    """Pre-launch probe (check:) failure — the broken reason is carried in the message; the gate does not launch."""
