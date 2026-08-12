"""Matrix 治理集成: nodes/mesh/matrix 三 channel 单文件 | 系统管理 | alpha

三 channel 分工 (matrix-channel.md §5):
- nodes: 本地治理 (list/read/run/stop/status/read_output). 数据源 =
  matrix.project.nodes() + matrix.handled_cells() + matrix.dead_cells().
- mesh: 网络投影 (accept/reject/set_auto_accept + events). virtual_children
  镜像 mesh.channel_proxies(). CellEvent -> Signal 生产侧归本 channel.
- matrix: 集成点. 静态挂 nodes/mesh. 本轮无 own commands.

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
    CellEvent,
    CellAddress,
    CellAddressCodec,
    CellEventLevel,
    DuplicatedError,
)
from ghoshell_moss.core.blueprint.matrix import CellHandle, Matrix
from ghoshell_moss.core.blueprint.mindflow import Priority
from ghoshell_moss.core.blueprint.states_channel import PrimeChannel
from ghoshell_moss.core.concepts.channel import Channel
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


def _fmt_running_row(handle: CellHandle) -> str:
    meta = handle.process.meta
    short = CellAddressCodec(handle.address).short
    uptime = _fmt_uptime(_now_ts() - meta.created)
    return f'  {short}  uptime={uptime} pid={meta.pid}'


def _fmt_dead_row(handle: CellHandle) -> str:
    meta = handle.process.meta
    short = CellAddressCodec(handle.address).short
    code = meta.exit_code
    when = _fmt_uptime(_now_ts() - meta.updated)
    tail = ''
    if code not in (0, None):
        tail = f' — nodes:read_output({short}) for stderr'
    return f'  {short}  exit={code} ({when} ago){tail}'


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
            p = Path(path)
            if not p.is_absolute():
                p = matrix.project.root.abspath_of(p)
            scan_paths = [p]
        found = nodes_mgr.list_nodes(
            refresh=refresh, paths=scan_paths, installed=installed,
        )
        if not found:
            return '[nodes] (empty)'

        # 数运行中的 cell (按 fullname 聚合)
        handled = matrix.handled_cells()
        running_count: dict[str, int] = {}
        for h in handled.values():
            fn = h.runtime.cell.fullname
            running_count[fn] = running_count.get(fn, 0) + 1

        lines = [f'[nodes] discovered ({len(found)}):']
        for rel_path, manifest in found.items():
            if category and manifest.category != category:
                continue
            marker_installed = 'installed' if manifest.installed else 'NOT installed'
            running = running_count.get(manifest.name, 0)
            running_hint = f' running={running}' if running else ''
            cat_hint = f' [{manifest.category}]' if manifest.category else ''
            desc = manifest.description or ''
            if desc:
                desc = f' — {desc}'
            lines.append(
                f'  {rel_path:<32}{cat_hint} {marker_installed}'
                f'{running_hint}{desc}'
            )
        return '\n'.join(lines)

    # -- read ---------------------------------------------------------

    @chan.build.command(name='read', blocking=False, always_observe=True)
    async def read_node(target: str) -> str:
        """Read a node manifest — frontmatter + instruction body."""
        if not target:
            CommandUtil.raise_observe(
                "target required. Use nodes:list() to discover paths."
            )
        manifest = matrix.project.nodes.get_node(target)
        if manifest is None:
            CommandUtil.raise_observe(
                f"node {target!r} not found. nodes:list() shows available paths."
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
    async def run_node(target: str) -> str:
        """Spawn a node cell. Nonblocking for persist nodes; blocking for one-shot.

        One-shot (persist=false) cells run to completion — this command blocks
        until exit and returns stdout/stderr tail + exit code (standard bash call).
        """
        if not target:
            CommandUtil.raise_observe(
                "target required. nodes:list() to discover paths."
            )
        try:
            handle = await matrix.run_node(Path(target))
        except DuplicatedError as e:
            CommandUtil.raise_observe(
                f'Singleton conflict: {e}. nodes:status() to inspect; '
                f'nodes:stop(<address>) to release.'
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

        return (
            f'[{short}] pid={handle.process.meta.pid} — '
            f'organ appears next frame under matrix.mesh once announced.'
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
                f'nodes:status() shows current cells.'
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
            lines.extend(_fmt_running_row(h) for h in handled.values())
        if dead:
            lines.append(f'recently exited ({len(dead)}):')
            lines.extend(_fmt_dead_row(h) for h in dead)
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

    # -- context messages --------------------------------------------

    @chan.build.context_messages
    def nodes_context() -> list[str]:
        handled = matrix.handled_cells()
        dead = list(matrix.dead_cells())
        if not handled and not dead:
            return []
        lines: list[str] = []
        if handled:
            lines.append(f'[nodes] running ({len(handled)}):')
            for h in list(handled.values())[:show_running]:
                lines.append(_fmt_running_row(h))
            if len(handled) > show_running:
                extra = len(handled) - show_running
                lines.append(
                    f'  ...+{extra} more, nodes:status() for full list'
                )
        if dead:
            recent = dead[-show_dead:]
            lines.append(f'recently exited ({len(recent)}):')
            for h in recent:
                lines.append(_fmt_dead_row(h))
        return ['\n'.join(lines)]

    # -- instruction --------------------------------------------------

    @chan.build.instruction
    def nodes_instruction() -> str:
        return (
            'Local node cell governance. list/read/run/stop/status/read_output '
            'are all nonblocking. run() returns immediately — the spawned '
            'organ appears next frame under matrix.mesh once announced. '
            'Read the NodeManifest with read(target) before running to know '
            'how it wants to be used.'
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
) -> Channel:
    """网络投影 channel. virtual_children 镜像 mesh.channel_proxies(),
    CellEvent 生产侧订阅 mesh.on_event 双扇出 (ring buffer + Signal)."""

    default_desc = (
        'Network projection — accepted cells surface as matrix.mesh.<short>.'
    )
    chan: PrimeChannel = new_channel(name=name, description=description or default_desc)

    # 自持事件 ring buffer, 喂 context + events 命令
    event_buffer: deque[CellEvent] = deque(maxlen=_EVENT_BUFFER)

    # unsub 句柄, on_close 时释放
    unsub_holder: list[Callable[[], None] | None] = [None]

    # virtual children 缓存: address → alias
    proxy_aliases: dict[CellAddress, str] = {}

    # -- lifecycle: subscribe mesh.on_event 双扇出 --------------------

    def _dispatch_event(event: CellEvent) -> None:
        # 1) 写自持 ring buffer (喂 context / events 命令) — 所有事件都可拉取
        event_buffer.append(event)
        # 2) 感知判决: 低于阈值 INFO (DEBUG) 不产生 signal (零值/不调用)
        if not CellEventLevel.is_perceivable(event.event_level):
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

    # -- refresh_meta: 同步 virtual_children 到 mesh.channel_proxies() -

    @chan.build.refresh_meta
    async def _refresh() -> None:
        mesh = await matrix.network()
        proxies = mesh.channel_proxies()
        # 计算增删差异
        current = set(proxy_aliases.keys())
        target = set(proxies.keys())
        # remove: 掉线的 accepted cells
        for gone_addr in current - target:
            alias = proxy_aliases.pop(gone_addr, None)
            if alias is not None:
                try:
                    chan.remove_virtual_channel(alias)
                except Exception:
                    pass
        # add: 新 accept 的 cells
        for new_addr in target - current:
            proxy = proxies[new_addr]
            alias = CellAddressCodec(new_addr).short
            try:
                chan.add_virtual_channel(proxy, alias=alias)
                proxy_aliases[new_addr] = alias
            except Exception:
                logger = CommandUtil.logger()
                if logger is not None:
                    logger.exception(
                        'mesh channel: failed to add virtual %s', new_addr,
                    )

    # -- auto_accept 状态查询 -----------------------------------------

    def _auto_accept_covers_all() -> bool:
        """auto_accept 全开时 accept/reject 命令不出现在 perspective."""
        # CellNetwork ABC 没暴露 auto_accept 查询接口, 用一个"守卫函数"占位.
        # 未来 mesh 加 get_auto_accept() 时替换掉 (matrix-channel.md §5.2).
        return False

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
        mesh = await matrix.network()
        mesh.set_auto_accept(local=local, foreign=foreign)
        return (
            f'[mesh:set_auto_accept] applied '
            f'(local={local}, foreign={foreign}).'
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

    # -- context messages --------------------------------------------

    @chan.build.context_messages
    def mesh_context() -> list[str]:
        # 从自持 ring buffer 取尾 (mesh.recent_events 是并行数据源, 本 channel
        # 消费自己 on_event 累积的 buffer, 保证与 signal 生产同步)
        if not event_buffer:
            return []
        events = list(event_buffer)[-show_events:]
        lines = [f'[mesh] recent events ({len(events)}):']
        for ev in events:
            when = ev.created.strftime('%H:%M:%S')
            short = ev.address_codec.short
            content = ev.content or 'updated'
            lines.append(f'  {when}  {short}  {content}')
        # 网络概要
        try:
            # 若 mesh 尚未惰性 fetch 过, mesh.view() 无 await 也应能返回缓存
            # (CellNetwork.view 是 sync). 但 mesh() 是 async factory, 需要
            # await — context_messages 是 sync, 只能读上次 refresh 的缓存.
            # 简化: mesh 概要只显示 accepted proxy 数量 (proxy_aliases 已同步)
            lines.append(
                f'network: {len(proxy_aliases)} cells accepted (proxies mounted)'
            )
        except Exception:
            pass
        return ['\n'.join(lines)]

    # -- instruction --------------------------------------------------

    @chan.build.instruction
    def mesh_instruction() -> str:
        return (
            'Network cell mesh: accepted cells appear here as sub-channels '
            '(matrix.mesh.<short>). accept/reject govern resource trust; '
            'set_auto_accept toggles the default policy. events() reads the '
            'recent event stream; live events also appear as background '
            'hints when idle.'
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

    # 静态挂 nodes + mesh (composed inline, share matrix reference)
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
