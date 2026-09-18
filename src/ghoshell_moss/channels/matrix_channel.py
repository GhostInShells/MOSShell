"""Matrix 治理集成: nodes/mesh/matrix 三 channel 单文件 | 系统管理 | alpha

三 channel 分工 (matrix-channel.md §5):
- nodes: 本地治理 (list/read/run/stop/status/read_output). 数据源 =
  matrix.project.nodes() + matrix.handled_cells() + matrix.dead_cells().
- mesh: 网络投影 (accept/reject/set_auto_accept + events). virtual_children
  镜像 mesh.channel_proxies(). CellEvent -> Signal 生产侧归本 channel.
- matrix: 集成点. 静态挂 nodes/mesh. 本轮无 own commands.

表面分层 (cold=instruction / warm=notice / hot=context, 见 channel_builder):
本文件没有 perception 级数据, 不占热面. 所有运行时状态都是**状态级**变更, 走 notice:
nodes 的 running/dead, mesh 的事件尾部. 内核按文本差分投递 notice, 变了才重发,
所以两件事必须守住:
- 尾部必须有上界 (show_running / show_dead / show_events), 否则 transcript 单调膨胀;
- 文本必须**稳定**: 行内不得出现 uptime / "N ago" 这类每次渲染都变的字段,
  那会永久击穿差分, 退化成每轮全文重发. 实时量 (精确 uptime / 更长事件历史)
  交 status() / events() 主动拉取.

OS 工具 (bash / file_editor) 已迁至 desktop channel, 与 matrix 平级.

Example:
    # matrix 环境自动装配 (推荐, IoC 走 container)
    from ghoshell_moss import new_shell_main_channel
    from ghoshell_moss.channels.matrix_channel import build_matrix_channel
    main = new_shell_main_channel()
    main.import_channels(build_matrix_channel())

    # 只挂 nodes / mesh 之一 (罕见, 用于精细组装)
    from ghoshell_moss.channels.matrix_channel import build_nodes_channel
    main.import_channels(build_nodes_channel())
"""

from __future__ import annotations

import re
import shlex
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from ghoshell_container import IoCContainer

from ghoshell_moss.core.blueprint.channel_builder import (
    ChannelFactory,
    CommandUtil,
    new_channel,
)
from ghoshell_moss.core.blueprint.cell import (
    AutoAcceptPolicy,
    CELL_EVENT_CHANNEL_ADDED,
    CellEvent,
    CellAddress,
    CellAddressCodec,
    CellEventLevel,
    DuplicatedError, NodeManifest, NodeProbeError,
)
from ghoshell_moss.core.blueprint.matrix import CellHandle, Matrix
from ghoshell_moss.core.blueprint.mindflow import Priority
from ghoshell_moss.core.blueprint.states_channel import PrimeChannel
from ghoshell_moss.core.concepts.channel import Channel, ChannelNamePattern
from ghoshell_moss.signals import CellEventSignalMeta, CellTransition

__all__ = [
    "build_matrix_channel",
    "build_nodes_channel",
    "build_mesh_channel",
    "new_matrix_channel",
    "new_nodes_channel",
    "new_mesh_channel",
    "CellAliasRegistry",
]

# ---- constants ----
_DEFAULT_SHOW_RUNNING = 8
_DEFAULT_SHOW_DEAD = 3
_DEFAULT_SHOW_EVENTS = 8
_EVENT_BUFFER = 128
_STDERR_TAIL_LINES = 5
_ONE_SHOT_OUTPUT_TAIL = 200


# ==== cell alias registry ========================================


class CellAliasRegistry:
    """Process-local alias bookkeeping shared by the nodes and mesh channels.

    Naming is a channel-tree concern, not a Matrix identity concern — it lives
    here rather than on the facade. Two pieces:

    - counter: base name → next suffix. Monotonic (never reused), so a name in
      the transcript refers to exactly one spawn forever.
    - pending: address → final name. Reserved at ``run``, consumed (popped) at
      mount, pruned when the process dies without providing a channel.
    """

    def __init__(self) -> None:
        self._counter: dict[str, int] = {}
        self._pending: dict[CellAddress, str] = {}

    def reserve(self, address: CellAddress, base: str) -> str:
        """Mint the real name for ``base`` and record it as pending for ``address``."""
        n = self._counter.get(base, 0)
        self._counter[base] = n + 1
        name = base if n == 0 else f'{base}_{n + 1}'
        self._pending[address] = name
        return name

    def consume(self, address: CellAddress) -> str | None:
        """Pop and return the pending name at mount time; None if never reserved."""
        return self._pending.pop(address, None)

    def prune(self, live_addresses: set[CellAddress]) -> None:
        """Drop pending entries whose process is gone (never provided a channel)."""
        for address in list(self._pending):
            if address not in live_addresses:
                del self._pending[address]


# ==== helpers ====================================================

# cell event level → signal priority 映射 (一处). None (系统约定) 视同 INFO.
_EVENT_LEVEL_PRIORITY = {
    CellEventLevel.INFO: Priority.BACKGROUND,
    CellEventLevel.WARNING: Priority.WARNING,
    CellEventLevel.ERROR: Priority.ERROR,
    CellEventLevel.CRITICAL: Priority.CRITICAL,
}


def _signal_priority_for(event_level: CellEventLevel | None) -> Priority:
    # 调用方保证 event_level 已感知 (>= INFO); DEBUG 由 _dispatch_event 提前 return,
    # 不经过本映射. fallback 到 BACKGROUND 只是 fail-safe (未知档按最低感知处理).
    level = CellEventLevel.resolve(event_level)
    return _EVENT_LEVEL_PRIORITY.get(level, Priority.BACKGROUND)


def _now_ts() -> float:
    return datetime.now(timezone.utc).timestamp()



def _fmt_uptime(seconds: float) -> str:
    s = int(seconds)
    if s < 60:
        return f'{s}s'
    if s < 3600:
        return f'{s // 60}m{s % 60}s'
    return f'{s // 3600}h{(s % 3600) // 60}m'


def _resolve_handled_address(
        target: str, handled: dict[CellAddress, CellHandle],
) -> CellHandle | None:
    """target 由 CellAddressCodec.match 唯一命中才返回 (short/名段/uid前缀/全名)."""
    if not target:
        return None
    if target in handled:
        return handled[target]
    matches: list[CellHandle] = []
    for addr, handle in handled.items():
        if CellAddressCodec(addr).match(target):
            matches.append(handle)
    if len(matches) == 1:
        return matches[0]
    return None


def _find_handle_in_all(
        target: str,
        handled: dict[CellAddress, CellHandle],
        dead: list[CellHandle],
) -> CellHandle | None:
    """先查活的, 再查死的. 供 read_output / status 用."""
    handle = _resolve_handled_address(target, handled)
    if handle is not None:
        return handle
    if not target:
        return None
    for h in dead:
        if CellAddressCodec(h.address).match(target):
            return h
    return None


# 行渲染分两套, 消费者不同:
#   _fmt_status_* — 命令主动拉取, 可以带 uptime / "N ago" 这类易变字段.
#   _fmt_notice_* — 进 notice 参与文本差分, 必须逐字稳定, 否则每轮都被判为"已变更".
def _fmt_status_running(handle: CellHandle) -> str:
    meta = handle.process.meta
    short = CellAddressCodec(handle.address).short
    uptime = _fmt_uptime(_now_ts() - meta.created)
    return f'  {short}  uptime={uptime} pid={meta.pid}'


def _fmt_status_dead(handle: CellHandle) -> str:
    meta = handle.process.meta
    short = CellAddressCodec(handle.address).short
    code = meta.exit_code
    when = _fmt_uptime(_now_ts() - meta.updated)
    tail = ''
    if code not in (0, None):
        tail = f' — read_output({short}) for stderr'
    return f'  {short}  exit={code} ({when} ago){tail}'


def _fmt_notice_running(handle: CellHandle) -> str:
    short = CellAddressCodec(handle.address).short
    return f'  {short}  pid={handle.process.meta.pid}'


def _fmt_notice_dead(handle: CellHandle) -> str:
    meta = handle.process.meta
    short = CellAddressCodec(handle.address).short
    code = meta.exit_code
    tail = ''
    if code not in (0, None):
        tail = f' — read_output({short}) for stderr'
    return f'  {short}  exit={code}{tail}'


def _render_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """Column-align a small table; only the last column is left unpadded."""
    if not rows:
        return []
    widths = [
        max(len(headers[i]), *(len(r[i]) for r in rows))
        for i in range(len(headers))
    ]

    def line(cells: list[str]) -> str:
        padded = [
            c if i == len(widths) - 1 else c.ljust(widths[i])
            for i, c in enumerate(cells)
        ]
        return ('  ' + '  '.join(padded)).rstrip()

    return [line(headers)] + [line(r) for r in rows]


def _node_ident(rel_path: str, manifest: NodeManifest) -> str:
    """rel_path is the run/read target; the manifest name adds identity only when
    it differs from the directory name (e.g. sensors/listener → voice)."""
    base = Path(rel_path).name
    if manifest.name and manifest.name != base:
        return f'{rel_path} ({manifest.name})'
    return rel_path


# ==== nodes channel ==============================================


def new_nodes_channel(
        matrix: Matrix,
        *,
        name: str = 'nodes',
        description: str | None = None,
        show_running: int = _DEFAULT_SHOW_RUNNING,
        show_dead: int = _DEFAULT_SHOW_DEAD,
        aliases: CellAliasRegistry | None = None,
) -> Channel:
    """本地 node 治理 channel. 五动词全 nonblocking, 数据源来自 matrix."""

    aliases = aliases or CellAliasRegistry()

    default_desc = (
        'Local node governance — list/read/run/stop/status/read_output.'
    )
    chan = new_channel(name=name, description=description or default_desc)

    # -- list ---------------------------------------------------------

    @chan.build.command(name='list', blocking=False, always_observe=True)
    async def list_nodes(
            path: str = '',
            category: str = '',
            installed: bool | None = None,
            refresh: bool = False,
    ) -> str:
        """List discoverable node declarations. path='' scans default roots."""
        nodes_mgr = matrix.project.nodes
        scan_paths: list[Path] | None = None
        if path:
            p = Path(path)
            if not p.is_absolute():
                p = matrix.project.root.abspath_of(p)
            scan_paths = [p]
        found = nodes_mgr.list_nodes(
            refresh=refresh, paths=scan_paths, installed=installed,
        )

        # 数运行中的 cell — 按 (category, name) 聚合, 与 Cell.fullname 同源.
        running_count: dict[tuple[str, str], int] = {}
        for h in matrix.handled_cells().values():
            cell = h.runtime.cell
            key = (cell.category, cell.name)
            running_count[key] = running_count.get(key, 0) + 1

        rows: list[list[str]] = []
        for rel_path, manifest in sorted(found.items()):
            if category and manifest.category != category:
                continue
            running = running_count.get((manifest.category, manifest.name), 0)
            rows.append([
                _node_ident(rel_path, manifest),
                manifest.category or '-',
                'yes' if manifest.installed else 'no',
                str(running) if running else '-',
                manifest.description or '',
            ])
        if not rows:
            return '[nodes] (empty)'
        scope = f' category={category}' if category else ''
        head = f'[nodes] discovered ({len(rows)}{scope}):'
        table = _render_table(
            ['path', 'category', 'installed', 'running', 'description'], rows,
        )
        return '\n'.join([head] + table)

    # -- read ---------------------------------------------------------

    @chan.build.command(name='read', blocking=False, always_observe=True)
    async def read_node(target: str) -> str:
        """Read a node manifest — frontmatter + instruction body."""
        if not target:
            CommandUtil.raise_observe(
                "target required. Use list() to discover paths."
            )
        manifest = matrix.project.nodes.get_node(target)
        if manifest is None:
            CommandUtil.raise_observe(
                f"node {target!r} not found. list() shows available paths."
            )
        lines = [
            f'[nodes:read {target}]',
            f'name={manifest.name}',
            f'description={manifest.description}',
            f'category={manifest.category or "(none)"}',
            f'singleton={manifest.singleton}',
            f'persist={manifest.persist}',
            f'installed={manifest.installed}',
            f'exec={manifest.exec.command} {manifest.exec.args}',
            f'file={manifest.file}',
        ]
        if not manifest.installed:
            install_path = Path(manifest.file).parent / manifest.INSTALL_FILENAME
            lines.append(f'install_hint: see {install_path} for setup steps')
        if manifest.instruction:
            lines.append('')
            lines.append('---- instruction ----')
            lines.append(manifest.instruction)
        return '\n'.join(lines)

    # -- run ----------------------------------------------------------

    @chan.build.command(name='run', blocking=False, always_observe=True)
    async def run_node(target: str, name: str, extra_args: str | None = None) -> str:
        """Spawn a node cell under a model-chosen name. Nonblocking for persist
        nodes; blocking for one-shot.

        name: how this cell is addressed on the network — the mount key under
        matrix.mesh. Duplicate names are auto-suffixed (_2, _3...); the receipt
        reports the final name. A one-shot node never mounts, so its name is inert.

        One-shot (persist=false) cells run to completion — this command blocks
        until exit and returns stdout/stderr tail + exit code (standard bash call).

        extra_args: shell-like extra argv appended after the node's declared entry
        args, shlex-split here. Use it for per-instance binding, e.g.
        run('nodes/visions/stream', 'vision', extra_args='--address rtmp://127.0.0.1/live').
        """
        if not target:
            CommandUtil.raise_observe(
                "target required. list() to discover paths."
            )
        if not re.fullmatch(ChannelNamePattern, name):
            CommandUtil.raise_observe(
                f"name {name!r} is not a valid channel name "
                f"(pattern {ChannelNamePattern})."
            )
        argv = shlex.split(extra_args) if extra_args else None
        try:
            handle = await matrix.run_node(Path(target), extra_args=argv)
        except DuplicatedError as e:
            CommandUtil.raise_observe(
                f'Singleton conflict: {e}. status() to inspect; '
                f'stop(<address>) to release.'
            )
        except NodeProbeError as e:
            CommandUtil.raise_observe(
                f'Node probe (check: in NODE.md) failed — refusing to launch. '
                f'{e}'
            )
        except FileNotFoundError as e:
            CommandUtil.raise_observe(f'target not found: {e}')
        except RuntimeError as e:
            CommandUtil.raise_observe(str(e))
        short = CellAddressCodec(handle.address).short

        # 一次性 node (persist=false → event_level 低于 INFO): 阻塞等退出拿结果.
        if not CellEventLevel.is_perceivable(handle.runtime.cell.event_level):
            meta = await handle.wait()
            output = handle.process.output
            code = meta.exit_code
            lines = [f'[{short}] exited code={code}']
            if output is not None:
                stdout = output.stdout(limit=_ONE_SHOT_OUTPUT_TAIL)
                stderr = output.stderr(limit=_ONE_SHOT_OUTPUT_TAIL)
                if stdout:
                    lines.append(f'--- stdout (tail {_ONE_SHOT_OUTPUT_TAIL}) ---\n{stdout.rstrip()}')
                if stderr:
                    lines.append(f'--- stderr (tail {_ONE_SHOT_OUTPUT_TAIL}) ---\n{stderr.rstrip()}')
                # 完整输出落盘 — 提示文件路径, 供模型按需读全量.
                full_files = [str(f) for f in (output.stdout_file, output.stderr_file) if f]
                if full_files:
                    lines.append('--- full output ---\n' + '\n'.join(full_files))
            return '\n'.join(lines)

        alias = aliases.reserve(handle.address, name)
        return (
            f'[{short}] alias={alias} pid={handle.process.meta.pid} — '
            f'if it announces a channel: matrix.mesh.{alias}'
        )

    # -- stop ---------------------------------------------------------

    @chan.build.command(name='stop', blocking=False, always_observe=False)
    async def stop_node(address: str, timeout: float = 5.0) -> str:
        """Stop a running node (SIGTERM -> grace -> killpg)."""
        handled = matrix.handled_cells()
        handle = _resolve_handled_address(address, handled)
        if handle is None:
            CommandUtil.raise_observe(
                f'{address!r} does not uniquely match any running cell. '
                f'status() shows current cells.'
            )
        await handle.stop(timeout=timeout)
        code = handle.process.meta.exit_code
        short = CellAddressCodec(handle.address).short
        return f'[{short}] stopped, exit={code}'

    # -- status -------------------------------------------------------

    @chan.build.command(name='status', blocking=False, always_observe=True)
    async def status_node(address: str = '') -> str:
        """Inspect running + recently exited cells. address='' = all."""
        handled = matrix.handled_cells()
        dead = list(matrix.dead_cells())

        if address:
            handle = _find_handle_in_all(address, handled, dead)
            if handle is None:
                CommandUtil.raise_observe(
                    f'{address!r} not found in handled or dead cells.'
                )
            return _fmt_single_brief(handle, address in handled)

        lines: list[str] = []
        if handled:
            lines.append(f'running ({len(handled)}):')
            lines.extend(_fmt_status_running(h) for h in handled.values())
        if dead:
            lines.append(f'recently exited ({len(dead)}):')
            lines.extend(_fmt_status_dead(h) for h in dead)
        if not lines:
            return '[nodes:status] no cells running or recently exited.'
        return '[nodes:status]\n' + '\n'.join(lines)

    # -- read_output --------------------------------------------------

    @chan.build.command(name='read_output', blocking=False, always_observe=True)
    async def read_output(
            address: str, stream: str = 'stderr', limit: int = 50,
    ) -> str:
        """Read stdout/stderr tail from memory ring buffer."""
        handled = matrix.handled_cells()
        dead = list(matrix.dead_cells())
        handle = _find_handle_in_all(address, handled, dead)
        if handle is None:
            CommandUtil.raise_observe(
                f'{address!r} not found in handled or dead cells.'
            )
        output = handle.process.output
        short = CellAddressCodec(handle.address).short
        if output is None:
            return f'[{short}] no capture buffer (spawn without CaptureSpec).'
        if stream == 'stdout':
            body = output.stdout(limit=limit)
        else:
            body = output.stderr(limit=limit)
        if not body:
            return f'[{short}] {stream} empty.'
        return f'[{short}] {stream} tail:\n{body.rstrip()}'

    # -- notice -------------------------------------------------------

    @chan.build.notice
    def nodes_notice() -> str:
        """Warm state: what is running now, plus the installed node catalog.

        Everything here is state-level, so it rides notice (diffed by text) instead
        of the hot context band. Rows are rendered without uptime / "N ago" — a
        field that moves every second would defeat the diff and force a full
        re-emission on every refresh. Live numbers come from status().
        """
        lines: list[str] = []
        handled = matrix.handled_cells()
        dead = list(matrix.dead_cells())
        if handled:
            lines.append(f'running ({len(handled)}):')
            for h in list(handled.values())[:show_running]:
                lines.append(_fmt_notice_running(h))
            if len(handled) > show_running:
                extra = len(handled) - show_running
                lines.append(f'  ...+{extra} more, status() for full list')
        if dead:
            recent = dead[-show_dead:]
            lines.append(f'recently exited ({len(recent)}):')
            for h in recent:
                lines.append(_fmt_notice_dead(h))
        # refresh=False: notice re-renders on every meta refresh; never rescan
        # the filesystem here (cache fills on first call, list(refresh=True) refreshes).
        found = matrix.project.nodes.list_nodes(refresh=False, installed=True)
        if found:
            lines.append(f'installed nodes ({len(found)}):')
            for rel_path, manifest in sorted(found.items()):
                label = _node_ident(rel_path, manifest)
                desc = manifest.description or ''
                lines.append(f'  {label} — {desc}' if desc else f'  {label}')
        return '\n'.join(lines)

    # -- instruction --------------------------------------------------

    @chan.build.instruction
    def nodes_instruction() -> str:
        return (
            'Local node cell governance. All verbs are nonblocking. run() '
            'returns immediately — the spawned organ surfaces on the network '
            'once it announces. read(target) before running: the declaration '
            'carries how the node wants to be used.'
        )

    return chan


def build_nodes_channel(
        *,
        name: str = 'nodes',
        description: str | None = None,
        show_running: int = _DEFAULT_SHOW_RUNNING,
        show_dead: int = _DEFAULT_SHOW_DEAD,
) -> ChannelFactory:
    """High-order factory: config → ChannelFactory. Resolves Matrix from container."""

    def factory(container: IoCContainer) -> Channel:
        matrix = container.force_fetch(Matrix)
        return new_nodes_channel(
            matrix,
            name=name, description=description,
            show_running=show_running, show_dead=show_dead,
        )

    return factory


# ==== helpers for status brief ====================================


def _fmt_single_brief(handle: CellHandle, alive: bool) -> str:
    cell = handle.runtime.cell
    meta = handle.process.meta
    code = meta.exit_code
    short = CellAddressCodec(handle.address).short
    lines = [
        f'[{short}] {"running" if alive else "dead"}',
        f'  fullname={cell.fullname}',
        f'  category={cell.category or "(none)"}',
        f'  pid={meta.pid}',
        f'  providing={cell.providing}',
        f'  spawn_cwd={meta.cwd}  (stderr.log/stdout.log persist here)',
    ]
    if alive:
        lines.append(f'  uptime={_fmt_uptime(_now_ts() - meta.created)}')
    else:
        lines.append(f'  exit_code={code}')
        output = handle.process.output
        if output is not None:
            tail = output.stderr(limit=_STDERR_TAIL_LINES)
            if tail:
                lines.append('  stderr tail:')
                for tail_line in tail.rstrip().splitlines():
                    lines.append(f'    {tail_line}')
    return '\n'.join(lines)


# ==== mesh channel ===============================================


def new_mesh_channel(
        matrix: Matrix,
        *,
        name: str = 'mesh',
        description: str | None = None,
        show_events: int = _DEFAULT_SHOW_EVENTS,
        aliases: CellAliasRegistry | None = None,
) -> Channel:
    """网络投影 channel. virtual_children 镜像 mesh.channel_proxies(),
    CellEvent 生产侧订阅 mesh.on_event 双扇出 (事件 ring + Signal)."""

    aliases = aliases or CellAliasRegistry()

    default_desc = (
        'Network projection — accepted cells surface as matrix.mesh.<name>.'
    )
    chan: PrimeChannel = new_channel(name=name, description=description or default_desc)

    # 事件 ring (上界 _EVENT_BUFFER), 喂 notice 尾部. 只做有上界的"最近 N 条",
    # 不做逐条去重: 尾部是状态视图, 事件到达即更新, 由 notice 差分决定是否重发.
    event_buffer: deque[CellEvent] = deque(maxlen=_EVENT_BUFFER)

    # unsub 句柄, on_close 时释放
    unsub_holder: list[Callable[[], None] | None] = [None]

    # virtual children 缓存: address → alias
    proxy_aliases: dict[CellAddress, str] = {}

    # auto_accept 策略缓存. matrix.network() 是 async, 而 available 谓词是 sync,
    # 所以策略在 refresh_meta / set_auto_accept 时更新一次 (nonlocal), 谓词与
    # notice 只读它, 不每帧 re-await 网络.
    policy: AutoAcceptPolicy | None = None

    # -- lifecycle: subscribe mesh.on_event 双扇出 --------------------

    def _dispatch_event(event: CellEvent) -> None:
        # 1) 入事件 ring (notice 尾部数据源). 更长历史另有 pull 路径:
        #    events() 读网络侧 ring (mesh.recent_events).
        event_buffer.append(event)
        # 2) 感知判决: 低于阈值 INFO (DEBUG) 不产生 signal (零值/不调用)
        if not CellEventLevel.is_perceivable(event.event_level):
            return
        # 2.5) channel 维度的真相在观察者侧 (mesh 挂载), 生产者的自述时机不准
        #      (未 accept / 未 connected) 且无名字 — 不进 signal, 留在 ring.
        if event.content == CELL_EVENT_CHANNEL_ADDED:
            return
        # 3) 转 Signal 送 CellEventNucleus (M7.5)
        try:
            meta = CellEventSignalMeta(
                address=event.address,
                # 本轮 CellEvent 无 transition 字段, 统一 READY (§5.9 简化).
                # 未来扩展 CellEvent 或 on_exit 补 exited/crashed 时再分档.
                transition=CellTransition.READY,
            )
            short = CellAddressCodec(event.address).short
            content = event.content or f'cell {short} updated'
            signal = meta.to_signal(
                content,
                description=f'cell_event {short}',
                priority=_signal_priority_for(event.event_level),
            )
            CommandUtil.send_signal(signal)
        except Exception:
            # signal 送不出去不该阻塞 mesh 事件消费
            logger = CommandUtil.logger()
            if logger is not None:
                logger.exception('mesh channel: failed to dispatch cell_event')

    @chan.build.startup
    async def _startup() -> None:
        mesh = await matrix.network()
        unsub_holder[0] = mesh.on_event(_dispatch_event)

    @chan.build.close
    async def _close() -> None:
        unsub = unsub_holder[0]
        unsub_holder[0] = None
        if unsub is not None:
            try:
                unsub()
            except Exception:
                pass

    # -- refresh_meta: 同步 virtual_children + auto_accept 策略 --------

    @chan.build.refresh_meta
    async def _refresh() -> None:
        nonlocal policy
        mesh = await matrix.network()
        policy = mesh.auto_accept()
        proxies = mesh.channel_proxies()
        # 计算增删差异
        current = set(proxy_aliases.keys())
        target = set(proxies.keys())
        # 剪枝: 进程已死且从未 provide channel 的 pending 名字 (一次性信箱清垃圾).
        aliases.prune(set(matrix.handled_cells().keys()))
        # remove: 掉线的 accepted cells
        for gone_addr in current - target:
            alias = proxy_aliases.pop(gone_addr, None)
            if alias is not None:
                try:
                    chan.remove_virtual_channel(alias)
                except Exception:
                    pass
        # add: 新 accept 的 cells — run 时 reserve 的名字在此消费, 否则回落 short.
        for new_addr in target - current:
            proxy = proxies[new_addr]
            alias = aliases.consume(new_addr) or CellAddressCodec(new_addr).short
            try:
                chan.add_virtual_channel(proxy, alias=alias)
                proxy_aliases[new_addr] = alias
            except Exception:
                logger = CommandUtil.logger()
                if logger is not None:
                    logger.exception(
                        'mesh channel: failed to add virtual %s', new_addr,
                    )

    # -- auto_accept 策略: accept/reject 的可见性由它决定 --------------

    def _auto_accept_covers_all() -> bool:
        """策略全开 (local 与 foreign 都自动承认) 时, accept/reject 失去意义."""
        return policy is not None and policy.local and policy.foreign

    def _accept_available() -> bool:
        return not _auto_accept_covers_all()

    # -- commands: accept / reject / set_auto_accept ----------------

    @chan.build.command(
        name='accept', blocking=False, always_observe=False,
        available=_accept_available,
    )
    async def accept_cell(address: str, lookup: bool = False) -> str:
        """Trust a network cell — build channel proxy immediately."""
        mesh = await matrix.network()
        try:
            await mesh.accept(address, lookup=lookup)
        except LookupError as e:
            CommandUtil.raise_observe(str(e))
        return f'[mesh:accept {address}] resource acknowledged.'

    @chan.build.command(
        name='reject', blocking=False, always_observe=False,
        available=_accept_available,
    )
    async def reject_cell(address: str) -> str:
        """Refuse a network cell — tear down active proxy."""
        mesh = await matrix.network()
        await mesh.reject(address)
        return f'[mesh:reject {address}] resource withdrawn.'

    @chan.build.command(
        name='set_auto_accept', blocking=False, always_observe=False,
    )
    async def set_auto_accept(
            local: bool | None = None, foreign: bool | None = None,
    ) -> str:
        """Toggle auto-accept policy. None = keep current."""
        nonlocal policy
        mesh = await matrix.network()
        mesh.set_auto_accept(local=local, foreign=foreign)
        # 报结果状态, 不是回显请求参数 (None 只表示"未改动该开关").
        policy = mesh.auto_accept()
        return (
            f'[mesh:set_auto_accept] local={policy.local} '
            f'foreign={policy.foreign}'
        )

    @chan.build.command(name='events', blocking=False, always_observe=True)
    async def events_cmd(address: str = '', limit: int = 20) -> str:
        """Read recent cell events from network."""
        mesh = await matrix.network()
        if address:
            events = mesh.cell_events(address, limit=limit)
        else:
            events = mesh.recent_events(limit=limit)
        if not events:
            return '[mesh:events] (empty)'
        lines = [f'[mesh:events {"@" + address if address else "all"}]']
        for ev in events:
            when = ev.created.strftime('%H:%M:%S')
            short = ev.address_codec.short
            content = ev.content or '(no content)'
            lines.append(f'  {when}  {short}  {content}')
        return '\n'.join(lines)

    # -- notice: 事件尾部 (温数据) -----------------------------------

    @chan.build.notice
    def mesh_notice() -> str:
        """Current trust policy, then a bounded tail of recent cell events.

        Warm, so it rides notice: the kernel re-sends it only when the text changes.
        The tail window is capped at ``show_events``; the rest of the history is
        pulled on demand by events() rather than replayed here every refresh.
        """
        lines: list[str] = []
        if policy is not None:
            lines.append(f'auto_accept: local={policy.local}, foreign={policy.foreign}')
        if event_buffer:
            shown = list(event_buffer)[-show_events:]
            lines.append(f'recent events ({len(shown)}):')
            for ev in shown:
                when = ev.created.strftime('%H:%M:%S')
                short = ev.address_codec.short
                content = ev.content or 'updated'
                lines.append(f'  {when}  {short}  {content}')
            if len(event_buffer) > len(shown):
                lines.append(f'  ...+{len(event_buffer) - len(shown)} more, events() for the tail')
        return '\n'.join(lines)

    # -- instruction --------------------------------------------------

    @chan.build.instruction
    def mesh_instruction() -> str:
        return (
            'Network cell mesh: accepted cells appear here as sub-channels '
            '(matrix.mesh.<short>). accept/reject govern resource trust; '
            'set_auto_accept toggles the default policy. A bounded tail of '
            'recent cell events rides along with this channel\'s notice; '
            'events() pulls the fuller history on demand.'
        )

    return chan


def build_mesh_channel(
        *,
        name: str = 'mesh',
        description: str | None = None,
        show_events: int = _DEFAULT_SHOW_EVENTS,
) -> ChannelFactory:
    """High-order factory: config → ChannelFactory. Resolves Matrix from container."""

    def factory(container: IoCContainer) -> Channel:
        matrix = container.force_fetch(Matrix)
        return new_mesh_channel(
            matrix,
            name=name, description=description, show_events=show_events,
        )

    return factory


# ==== matrix channel (integration point) ========================


def new_matrix_channel(
        matrix: Matrix,
        *,
        name: str = 'matrix',
        description: str | None = None,
        extra_children: tuple[Channel | ChannelFactory, ...] = (),
) -> Channel:
    """集成点. import_channels 挂 nodes + mesh. 本轮 matrix 自身无 own commands."""

    default_desc = (
        'Matrix integration point: network projection and local organ '
        'governance. Children: nodes (local cell declarations + spawn), '
        'mesh (accepted network cells surface here). '
        'OS tools live under desktop, not here.'
    )
    chan = new_channel(name=name, description=description or default_desc)

    # 静态挂 nodes + mesh (composed inline, share matrix reference + alias registry)
    aliases = CellAliasRegistry()
    nodes = new_nodes_channel(matrix, aliases=aliases)
    mesh = new_mesh_channel(matrix, aliases=aliases)
    chan.import_channels(nodes, mesh, *extra_children)

    @chan.build.instruction
    def matrix_instruction() -> str:
        return (
            'Cell governance integration point. '
            'nodes: what organs can be declared / running locally. '
            'mesh: what organs (yours and others\') are on the network.'
        )

    return chan


def build_matrix_channel(
        *,
        name: str = 'matrix',
        description: str | None = None,
        extra_children: tuple[Channel | ChannelFactory, ...] = (),
) -> ChannelFactory:
    """High-order factory: config → ChannelFactory.

    Resolves Matrix from container, composes nodes + mesh as static children.
    OS tools (bash / file_editor) live under desktop channel, not here.

    :param name: matrix channel tag (default 'matrix')
    :param description: override default description
    :param extra_children: additional Channel or ChannelFactory to import
    """

    def factory(container: IoCContainer) -> Channel:
        matrix = container.force_fetch(Matrix)
        return new_matrix_channel(
            matrix, name=name, description=description,
            extra_children=extra_children,
        )

    return factory
