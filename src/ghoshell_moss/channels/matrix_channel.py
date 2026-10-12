"""Matrix governance — nodes / mesh / matrix channels in one module | system-management | alpha

三 channel 分工 (matrix-channel.md §5):
- nodes: 本地治理 (list/read/run/stop/status/read_output). 数据源 =
  matrix.project.nodes() + matrix.handled_cells() + matrix.dead_cells().
- mesh: 网络投影 (accept/reject/set_auto_accept + events). virtual_children
  镜像 mesh.channel_proxies(). CellEvent -> Signal 生产侧归本 channel.
- matrix: 集成点. 静态挂 nodes/mesh. 本轮无 own commands.

表面分层 (cold=instruction / warm=notice / hot=context, 见 channel_builder):
本文件没有 perception 级数据, 不占热面. 所有运行时状态都是**状态级**变更, 走 warm 面:
- nodes: 有生命周期的表面拆成 named_notices 片段 (running / exited / installed),
  各自独立差分, 一个片段变动不拖其余. running/exited 尾部有上界
  (show_running / show_dead); installed 只报计数, 目录交 list(). 实时量
  (精确 uptime / 更长事件历史) 交 status() / read_output() 主动拉取.
- mesh: auto_accept 策略走无名 notice (常驻), 事件尾部有上界 (show_events),
  历史交 events() 主动拉取.

两条纪律: 尾部必须有上界, 否则 transcript 单调膨胀; 行内文本必须**稳定**, 不得出现
uptime / "N ago" 这类每次渲染都变的字段, 否则击穿差分退化成每轮全文重发.

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

import asyncio
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
    CellEvent,
    CellAddress,
    CellAddressCodec,
    CellEventLevel,
    CellNetwork,
    DuplicatedError, NodeManifest, NodeProbeError,
)
from ghoshell_moss.core.blueprint.matrix import CellHandle, Matrix
from ghoshell_moss.core.blueprint.mindflow import Priority
from ghoshell_moss.core.blueprint.states_channel import PrimeChannel, new_prime_channel
from ghoshell_moss.core.concepts.channel import Channel, ChannelProxy
from ghoshell_moss.signals import CellEventSignalMeta, CellTransition

__all__ = [
    "build_matrix_channel",
    "build_nodes_channel",
    "build_mesh_channel",
    "new_matrix_channel",
    "new_nodes_channel",
    "new_mesh_channel",
]

# ---- constants ----
_DEFAULT_SHOW_RUNNING = 8
_DEFAULT_SHOW_DEAD = 3
_DEFAULT_SHOW_EVENTS = 8
_DEFAULT_SHOW_BRANCHES = 8
_EVENT_BUFFER = 128
_STDERR_TAIL_LINES = 5
_ONE_SHOT_OUTPUT_TAIL = 200


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


def _live_aliases(matrix: Matrix) -> dict[CellAddress, str]:
    """address → alias for cells THIS process spawned — the naming authority's map.

    ``NodeManager`` mints every alias at spawn and drops the entry the moment the
    process exits: **an alias names a live cell and nothing else.** The address is
    the identity (read-only truth); the ledger's ``runtime.alias`` is the spawn-time
    trace it was written as, and no surface reads it back.
    """
    return matrix.project.nodes.spawned_nodes()


def _alias_index(aliases: dict[CellAddress, str]) -> dict[str, list[CellAddress]]:
    """alias → addresses, derived from the naming authority — never from the ledger.

    An alias is a naming promise, an address is an identity: they are two
    namespaces, so an alias is never looked up *in* an address-keyed ledger.
    """
    index: dict[str, list[CellAddress]] = {}
    for address, alias in aliases.items():
        if alias:
            index.setdefault(alias, []).append(address)
    return index


def _match_pid(handle: CellHandle, target: str) -> bool:
    """pid is the debug handle: a plain integer, meaningful inside this process only."""
    if not target.isdigit():
        return False
    pid = handle.process.meta.pid
    return pid is not None and str(pid) == target


def _resolve_live(
        target: str,
        handled: dict[CellAddress, CellHandle],
        aliases: dict[CellAddress, str],
) -> CellHandle | None:
    """Resolve a target to a live handle — three forms, and only the ones a row prints.

    The address, the pid, the alias. A shorthand cut from an address (a name segment,
    a uid prefix) is **not** a form here: what a row shows is what a call takes, which
    is what keeps the facade free of "which shape does this command want" inferences.

    A target that is all digits is a **pid** and nothing else — an alias can never be
    numeric (``_validate_cell_alias`` demands a leading letter), so the two cannot
    collide. An alias resolves through the naming authority, and only when exactly one
    live cell owns it. Nothing resolves when the target is ambiguous.
    """
    if not target:
        return None
    if not isinstance(target, str):
        target = str(target)  # defensive: pid hint 可以把 int 直接喂进来
    if target in handled:
        return handled[target]
    if target.isdigit():
        owned = [h for h in handled.values() if _match_pid(h, target)]
        return owned[0] if len(owned) == 1 else None
    owned_alias = _alias_index(aliases).get(target)
    if owned_alias is not None:
        return handled.get(owned_alias[0]) if len(owned_alias) == 1 else None
    return None


def _resolve_cell(
        target: str,
        handled: dict[CellAddress, CellHandle],
        dead: list[CellHandle],
        aliases: dict[CellAddress, str],
) -> tuple[CellHandle, bool] | None:
    """Resolve a target to ``(handle, running)``.

    Live first (see ``_resolve_live``). An exited cell answers to the **pid or the
    whole address** it was printed with — never to a name: an alias names a live cell,
    and it does not outlive the process that minted it. Liveness is decided by
    **which ledger the handle came from**, never by whether the target happens to be
    a key of the live ledger.
    """
    if not target:
        return None
    if not isinstance(target, str):
        target = str(target)  # defensive: pid hint 可以把 int 直接喂进来
    handle = _resolve_live(target, handled, aliases)
    if handle is not None:
        return handle, True
    for h in reversed(dead):  # newest first: a reused pid must name its latest owner
        if _match_pid(h, target) or h.address == target:
            return h, False
    return None


# 行渲染分两套, 消费者不同:
#   _fmt_status_* — 命令主动拉取, 可以带 uptime / "N ago" 这类易变字段.
#   _fmt_notice_* — 进 notice 参与文本差分, 必须逐字稳定, 否则每轮都被判为"已变更".
# 每行统一三件套, 让模型拿到句柄时不看文档就能用对:
#   名字 (alias) — 只在活着时有, 取自命名权威; 身份 (address) — 永远有效; 手柄 (pid) — 本地 debug.
def _live_label(handle: CellHandle, alias: str | None) -> str:
    """A live cell's row: ``alias (address) pid=N``."""
    head = f'{alias} ({handle.address})' if alias else handle.address
    return f'{head} pid={handle.process.meta.pid}'


def _dead_label(handle: CellHandle) -> str:
    """An exited cell's row: ``address pid=N`` — the alias died with the process."""
    return f'{handle.address} pid={handle.process.meta.pid}'


def _stderr_hint(handle: CellHandle) -> str:
    """The debug pointer for an exited cell: its pid (or its address) — a name does
    not outlive the process that minted it."""
    return f' — read_output(target={handle.process.meta.pid}) for stderr'


def _fmt_status_running(handle: CellHandle, alias: str | None = None) -> str:
    uptime = _fmt_uptime(_now_ts() - handle.process.meta.created)
    return f'  {_live_label(handle, alias)}  uptime={uptime}'


def _fmt_status_dead(handle: CellHandle) -> str:
    meta = handle.process.meta
    when = _fmt_uptime(_now_ts() - meta.updated)
    tail = _stderr_hint(handle) if meta.exit_code not in (0, None) else ''
    return f'  {_dead_label(handle)}  exit={meta.exit_code} ({when} ago){tail}'


def _fmt_notice_running(handle: CellHandle, alias: str | None = None) -> str:
    return f'  {_live_label(handle, alias)}'


def _fmt_notice_dead(handle: CellHandle) -> str:
    meta = handle.process.meta
    tail = _stderr_hint(handle) if meta.exit_code not in (0, None) else ''
    return f'  {_dead_label(handle)}  exit={meta.exit_code}{tail}'


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


def _branch_name(address: CellAddress, spawned: dict[CellAddress, str]) -> str:
    """A mesh branch's name — simultaneously its CTML path segment.

    A locally spawned cell is named by the alias it promised to the model; anything
    else by its normalized address. Because this string *is* the lookup key (the tree
    joins it into ``matrix.mesh.<name>``), it must stay a single token: never decorate
    it with the address for display — surfacing the address is the notice's job.
    """
    return spawned.get(address) or CellAddressCodec(address).normalized


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
) -> Channel:
    """本地 node 治理 channel. 五动词全 nonblocking, 数据源来自 matrix."""

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
            p = Path(path).expanduser()
            if not p.is_absolute():
                p = (matrix.project.root / p).resolve()
            scan_paths = [p]
        found = await asyncio.to_thread(
            nodes_mgr.list_nodes,
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
        manifest = await asyncio.to_thread(matrix.project.nodes.get_node, target)
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
    async def run_node(target: str, alias: str = '', extra_args: str | None = None) -> str:
        """Spawn a node cell. Nonblocking for persist nodes; blocking for one-shot.

        alias: the branch name this process promises under matrix.mesh.<alias>.
        Empty → the node name is used; duplicates get _2/_3. A one-shot node never
        mounts, so its alias is inert.

        One-shot (persist=false) cells run to completion — this command blocks
        until exit and returns stdout/stderr tail + exit code (standard bash call).

        extra_args: shell-like extra argv appended after the node's declared entry
        args, shlex-split here. Use it for per-instance binding, e.g.
        run('nodes/visions/stream', extra_args='--address rtmp://127.0.0.1/live').
        """
        if not target:
            CommandUtil.raise_observe(
                "target required. list() to discover paths."
            )
        argv = shlex.split(extra_args) if extra_args else None
        try:
            handle = await matrix.run_node(
                Path(target), extra_args=argv, alias=alias or None,
            )
        except DuplicatedError as e:
            CommandUtil.raise_observe(
                f'Singleton conflict: {e}. status() to inspect; '
                f'stop(target) to release.'
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
        # 三件套回执: 名字 / 身份 / debug 手柄 —— 拿到任一个都能马上用.
        aliases = _live_aliases(matrix)
        label = _live_label(handle, aliases.get(handle.address))

        # 一次性 node (persist=false → event_level 低于 INFO): 阻塞等退出拿结果.
        if not CellEventLevel.is_perceivable(handle.runtime.cell.event_level):
            meta = await handle.wait()
            output = handle.process.output
            code = meta.exit_code
            lines = [f'[{label}] exited code={code}']
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

        return (
            f'[{label}] — if it announces a channel: '
            f'matrix.mesh.{_branch_name(handle.address, aliases)}'
        )

    # -- stop ---------------------------------------------------------

    @chan.build.command(name='stop', blocking=False, always_observe=False)
    async def stop_node(target: str, timeout: float = 5.0) -> str:
        """Stop a running node (SIGTERM -> grace -> killpg).

        ``target``: a pid, the address, or the alias of a live cell.
        """
        handled = matrix.handled_cells()
        aliases = _live_aliases(matrix)
        handle = _resolve_live(target, handled, aliases)
        if handle is None:
            CommandUtil.raise_observe(
                f'{target!r} does not uniquely match any running cell. '
                f'status() shows current cells.'
            )
        await handle.stop(timeout=timeout)
        code = handle.process.meta.exit_code
        return f'[{_live_label(handle, aliases.get(handle.address))}] stopped, exit={code}'

    # -- status -------------------------------------------------------

    @chan.build.command(name='status', blocking=False, always_observe=True)
    async def status_node(target: str = '') -> str:
        """Inspect running + recently exited cells. target='' = all.

        ``target``: a pid, the address, or the alias of a live cell.
        """
        handled = matrix.handled_cells()
        dead = list(matrix.dead_cells())
        aliases = _live_aliases(matrix)

        if target:
            found = _resolve_cell(target, handled, dead, aliases)
            if found is None:
                CommandUtil.raise_observe(
                    f'{target!r} not found in handled or dead cells.'
                )
            handle, running = found
            return _fmt_single_brief(handle, running, aliases.get(handle.address))

        lines: list[str] = []
        if handled:
            lines.append(f'running ({len(handled)}):')
            lines.extend(
                _fmt_status_running(h, aliases.get(h.address)) for h in handled.values()
            )
        if dead:
            lines.append(f'recently exited ({len(dead)}):')
            lines.extend(_fmt_status_dead(h) for h in dead)
        if not lines:
            return '[nodes:status] no cells running or recently exited.'
        return '[nodes:status]\n' + '\n'.join(lines)

    # -- read_output --------------------------------------------------

    @chan.build.command(name='read_output', blocking=False, always_observe=True)
    async def read_output(
            target: str, stream: str = 'stderr', limit: int = 50,
    ) -> str:
        """Read stdout/stderr tail from memory ring buffer — the debug read.

        ``target``: a pid, the address, or the alias of a live cell.
        """
        handled = matrix.handled_cells()
        dead = list(matrix.dead_cells())
        aliases = _live_aliases(matrix)
        found = _resolve_cell(target, handled, dead, aliases)
        if found is None:
            CommandUtil.raise_observe(
                f'{target!r} not found in handled or dead cells.'
            )
        handle, running = found
        output = handle.process.output
        label = (
            _live_label(handle, aliases.get(handle.address)) if running
            else _dead_label(handle)
        )
        if output is None:
            return f'[{label}] no capture buffer (spawn without CaptureSpec).'
        if stream == 'stdout':
            body = output.stdout(limit=limit)
        else:
            body = output.stderr(limit=limit)
        if not body:
            return f'[{label}] {stream} empty.'
        return f'[{label}] {stream} tail:\n{body.rstrip()}'

    # -- startup: prime the catalog off the loop ----------------------

    @chan.build.startup
    async def _warm_node_catalog() -> None:
        # 同步 notice 回调不得做 IO (见 channel.py on_refresh_meta 契约): nodes_notices
        # 走 list_nodes(refresh=False), 而缓存未填时那一次全目录扫描会落在 event loop
        # 线程上. 在这里先扫掉 — 之后 refresh=False 只命中缓存拷贝 (NodeManager._cache
        # 一旦填上不会再被写回 None).
        try:
            await asyncio.to_thread(
                matrix.project.nodes.list_nodes, refresh=False, installed=True,
            )
        except Exception:
            # 预热是性能防御, 扫描失败不该拖垮 channel 启动.
            logger = CommandUtil.logger()
            if logger is not None:
                logger.exception('nodes channel: node catalog warm-up failed')

    # -- notice (named fragments) -------------------------------------

    @chan.build.named_notices
    def nodes_notices() -> dict[str, str | None]:
        """Warm state, split by lifecycle so one fragment moving does not re-send the rest.

        ``running`` / ``exited`` keep a bounded tail of rows (absent when nothing is in
        that state, which reads as "removed"); ``installed`` is a bare count — the catalog
        itself is a deliberate pull via list(). Rows carry pid / exit code only, never
        uptime / "N ago", so the text is byte-stable between refreshes and the facade diff
        stays quiet. Live detail comes from status() / read_output().
        """
        out: dict[str, str | None] = {}
        aliases = _live_aliases(matrix)

        handled = matrix.handled_cells()
        if handled:
            lines = [f'{len(handled)} running:']
            lines += [
                _fmt_notice_running(h, aliases.get(h.address))
                for h in list(handled.values())[:show_running]
            ]
            if len(handled) > show_running:
                lines.append(f'  ...+{len(handled) - show_running} more, status() for full list')
            out['running'] = '\n'.join(lines)

        dead = list(matrix.dead_cells())
        if dead:
            recent = dead[-show_dead:]
            out['exited'] = '\n'.join(
                [f'{len(recent)} recently exited:'] + [_fmt_notice_dead(h) for h in recent]
            )

        # refresh=False: reads the cache primed at startup; this path must never be the
        # one that scans. A bare count keeps the catalog out of the warm band —
        # descriptions and paths are what list() is for.
        found = matrix.project.nodes.list_nodes(refresh=False, installed=True)
        if found:
            out['installed'] = str(len(found))

        return out

    # -- instruction --------------------------------------------------

    @chan.build.instruction
    def nodes_instruction() -> str:
        return (
            'Local node cell governance. All verbs are nonblocking. run() '
            'returns immediately — the spawned organ surfaces on the network '
            'once it announces. read(target) before running: the declaration '
            'carries how the node wants to be used. The warm notice carries only '
            'counts and a short tail — status() / list() pull the detail.'
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


def _fmt_single_brief(handle: CellHandle, alive: bool, alias: str | None = None) -> str:
    cell = handle.runtime.cell
    meta = handle.process.meta
    code = meta.exit_code
    head = _live_label(handle, alias) if alive else _dead_label(handle)
    lines = [
        f'[{head}] {"running" if alive else "dead"}',
        f'  fullname={cell.fullname}',
        f'  category={cell.category or "(none)"}',
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
        show_branches: int = _DEFAULT_SHOW_BRANCHES,
) -> Channel:
    """网络投影 channel. virtual_children 直接镜像 mesh.channel_proxies() (sync 直读),
    CellEvent 生产侧订阅 mesh.on_event (事件 ring + Signal), on_channel_provided 发
    connected signal."""

    default_desc = (
        'Network projection — accepted cells surface as matrix.mesh.<name>.'
    )
    chan: PrimeChannel = new_prime_channel(name=name, description=description or default_desc)

    # 事件 ring (上界 _EVENT_BUFFER), 喂 notice 尾部. 只做有上界的"最近 N 条".
    event_buffer: deque[CellEvent] = deque(maxlen=_EVENT_BUFFER)

    # mesh 引用在 startup 缓存 (matrix.network() 是 async, virtual_children 是 sync).
    mesh: CellNetwork | None = None
    # unsub 句柄, on_close 时释放.
    unsubs: list[Callable[[], None]] = []

    # auto_accept 策略缓存. available 谓词是 sync, 读缓存不每帧 re-await.
    policy: AutoAcceptPolicy | None = None

    # -- branch name: 命名权威归 spawn 侧 (module-level `_branch_name`) ----

    def _resolve_mesh_address(query: str) -> CellAddress | None:
        # _branch_name 的反向: accept/reject 的入参接受**展示形态** (raw / normalized
        # address / alias) —— 也就是 mesh 的 notice 与事件里印出来的那几种, 而不是任何
        # 打字简写. 归一化完就只剩地址.
        if mesh is None:
            return None
        proxies = mesh.channel_proxies()
        view = mesh.view()
        if query in proxies or query in view:
            return query
        try:
            codec = CellAddressCodec.from_normalized(query)
            if codec.address in proxies or codec.address in view:
                return codec.address
        except ValueError:
            pass
        for addr, alias in matrix.project.nodes.spawned_nodes().items():
            if alias == query:
                return addr
        return None

    # -- lifecycle: 订阅 mesh.on_event + mesh.on_channel_provided ---------

    def _dispatch_event(event: CellEvent) -> None:
        # 空 content = 纯 refetch 提示, 不进 ring 也不进 signal.
        if not event.content:
            return
        event_buffer.append(event)
        # 感知判决: 低于阈值 INFO (DEBUG) 不产生 signal (零值/不调用).
        if not CellEventLevel.is_perceivable(event.event_level):
            return
        try:
            meta = CellEventSignalMeta(address=event.address, transition=CellTransition.READY)
            name = _branch_name(event.address, matrix.project.nodes.spawned_nodes())
            signal = meta.to_signal(
                event.content,
                description=f'cell {name}',
                priority=_signal_priority_for(event.event_level),
            )
            CommandUtil.send_signal(signal)
        except Exception:
            # signal 送不出去不该阻塞 mesh 事件消费.
            logger = CommandUtil.logger()
            if logger is not None:
                logger.exception('mesh channel: failed to dispatch cell_event')

    def _on_channel_provided(address: CellAddress, proxy: ChannelProxy) -> None:
        # channel 上线 → 发 connected signal. 契约: 在 network event loop 上调用.
        name = _branch_name(address, matrix.project.nodes.spawned_nodes())
        try:
            meta = CellEventSignalMeta(address=address, transition=CellTransition.READY)
            signal = meta.to_signal(
                f'cell {name} connected',
                description=f'cell {name}',
                priority=Priority.BACKGROUND,
            )
            matrix.send_signal_to_ghost(signal)
        except Exception:
            logger = CommandUtil.logger()
            if logger is not None:
                logger.exception('mesh channel: failed to signal channel provided %s', address)

    @chan.build.startup
    async def _startup() -> None:
        nonlocal mesh, policy
        mesh = await matrix.network()
        unsubs.append(mesh.on_event(_dispatch_event))
        unsubs.append(mesh.on_channel_provided(_on_channel_provided))
        policy = mesh.auto_accept()

    @chan.build.close
    async def _close() -> None:
        for unsub in unsubs:
            try:
                unsub()
            except Exception:
                pass
        unsubs.clear()

    # -- virtual children: sync 直读 (树的 refresh 自己 diff 增删) --------

    @chan.build.virtual_children
    def _virtual_children() -> dict[str, Channel]:
        if mesh is None:
            return {}
        spawned = matrix.project.nodes.spawned_nodes()
        return {
            _branch_name(address, spawned): proxy
            for address, proxy in mesh.channel_proxies().items()
        }

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
    async def accept_cell(target: str, lookup: bool = False) -> str:
        """Trust a network cell — build channel proxy immediately.

        ``target``: any form the notice/events() prints — the raw address, its
        normalized form, or a local alias.
        """
        resolved = _resolve_mesh_address(target)
        if resolved is None:
            CommandUtil.raise_observe(
                f'{target!r} does not match any network cell. events() to inspect.'
            )
        try:
            await mesh.accept(resolved, lookup=lookup)
        except LookupError as e:
            CommandUtil.raise_observe(str(e))
        return f'[mesh:accept {resolved}] resource acknowledged.'

    @chan.build.command(
        name='reject', blocking=False, always_observe=False,
        available=_accept_available,
    )
    async def reject_cell(target: str) -> str:
        """Refuse a network cell — tear down active proxy.

        ``target``: the same forms accept() takes — the raw address, its normalized
        form, or a local alias.
        """
        resolved = _resolve_mesh_address(target)
        if resolved is None:
            CommandUtil.raise_observe(
                f'{target!r} does not match any network cell. events() to inspect.'
            )
        await mesh.reject(resolved)
        return f'[mesh:reject {resolved}] resource withdrawn.'

    @chan.build.command(
        name='set_auto_accept', blocking=False, always_observe=False,
    )
    async def set_auto_accept(
            local: bool | None = None, foreign: bool | None = None,
    ) -> str:
        """Toggle auto-accept policy. None = keep current."""
        nonlocal policy
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
        if address:
            events = mesh.cell_events(address, limit=limit)
        else:
            events = mesh.recent_events(limit=limit)
        if not events:
            return '[mesh:events] (empty)'
        lines = [f'[mesh:events {"@" + address if address else "all"}]']
        spawned = matrix.project.nodes.spawned_nodes()
        for ev in events:
            when = ev.created.strftime('%H:%M:%S')
            name = _branch_name(ev.address, spawned)
            content = ev.content or '(no content)'
            lines.append(f'  {when}  {name}  {content}')
        return '\n'.join(lines)

    # -- notice: 事件尾部 (温数据) -----------------------------------

    @chan.build.notice
    def mesh_notice() -> str:
        """Current trust policy, then the branch names, then a bounded event tail.

        Warm, so it rides notice: the kernel re-sends it only when the text changes.
        Both tails are capped — the branch table at ``show_branches`` (the rest is a
        pull via status()), the event window at ``show_events`` (the rest via
        events()) — rather than replayed here every refresh.
        """
        lines: list[str] = []
        if policy is not None:
            lines.append(f'auto_accept: local={policy.local}, foreign={policy.foreign}')
        lines.extend(_alias_lines())
        if event_buffer:
            shown = list(event_buffer)[-show_events:]
            lines.append(f'recent events ({len(shown)}):')
            spawned = matrix.project.nodes.spawned_nodes()
            for ev in shown:
                when = ev.created.strftime('%H:%M:%S')
                name = _branch_name(ev.address, spawned)
                content = ev.content or 'updated'
                lines.append(f'  {when}  {name}  {content}')
            if len(event_buffer) > len(shown):
                lines.append(f'  ...+{len(event_buffer) - len(shown)} more, events() for the tail')
        return '\n'.join(lines)

    def _alias_lines() -> list[str]:
        """``alias = address`` for the branches a local spawn named — bounded.

        Mesh is the one surface where the alias is load-bearing: ``matrix.mesh.<name>``
        *is* the alias while the cell lives. So the branch table spells both namespaces
        out side by side. Foreign cells need no line; their branch name already *is*
        their address. Rows are capped like every other warm tail — the rest is a pull
        (status()).
        """
        spawned = matrix.project.nodes.spawned_nodes()
        proxied = mesh.channel_proxies() if mesh is not None else {}
        rows = [
            [spawned[address], address]
            for address in sorted(proxied)
            if address in spawned
        ]
        if not rows:
            return []
        shown = rows[:show_branches]
        lines = [f'branches ({len(rows)}):']
        lines += _render_table(['alias', 'address'], shown)
        if len(rows) > len(shown):
            lines.append(f'  ...+{len(rows) - len(shown)} more, status() for full list')
        return lines

    # -- instruction --------------------------------------------------

    @chan.build.instruction
    def mesh_instruction() -> str:
        return (
            'Network cell mesh: accepted cells appear here as sub-channels '
            '(matrix.mesh.<name>). accept/reject govern resource trust; '
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
        show_branches: int = _DEFAULT_SHOW_BRANCHES,
) -> ChannelFactory:
    """High-order factory: config → ChannelFactory. Resolves Matrix from container."""

    def factory(container: IoCContainer) -> Channel:
        matrix = container.force_fetch(Matrix)
        return new_mesh_channel(
            matrix,
            name=name, description=description, show_events=show_events,
            show_branches=show_branches,
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

    # 静态挂 nodes + mesh (composed inline, share matrix reference).
    nodes = new_nodes_channel(matrix)
    mesh = new_mesh_channel(matrix)
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
