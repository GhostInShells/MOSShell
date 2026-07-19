"""Matrix 治理集成: nodes/mesh/matrix 三 channel 单文件 | 系统管理 | alpha

三 channel 分工 (matrix-channel.md §5):
- nodes: 本地治理 (list/read/run/stop/status/read_output). 数据源 =
  matrix.project.nodes() + matrix.handled_cells() + matrix.dead_cells().
- mesh: 网络投影 (accept/reject/set_auto_accept + events). virtual_children
  镜像 mesh.channel_proxies(). CellEvent → Signal 生产侧归本 channel.
- matrix: 集成点. 静态挂 nodes/mesh 及可选的 terminal/file_editor. 本轮
  无 own commands, 极简自我介绍.

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
    DuplicatedError,
)
from ghoshell_moss.core.blueprint.matrix import CellHandle, Matrix
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


# ==== helpers ====================================================


def _now_ts() -> float:
    return datetime.now(timezone.utc).timestamp()


def _uid_short(address: str) -> str:
    # address = kind/name/uid, uid 取前 8 位作短标识
    parts = address.split('/')
    if len(parts) >= 3 and parts[-1]:
        return parts[-1][:8]
    return address


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
    """target 是完整 address 或 fullname_uidprefix 短形式. 唯一匹配才返回."""
    if not target:
        return None
    if target in handled:
        return handled[target]
    # 尝试短形式匹配: fullname 或 uid 前缀
    matches: list[CellHandle] = []
    for addr, handle in handled.items():
        cell = handle.runtime.cell
        if cell.fullname == target:
            matches.append(handle)
        elif addr.endswith(target):
            matches.append(handle)
        elif cell.uid.startswith(target):
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
        cell = h.runtime.cell
        if h.address == target or cell.fullname == target or cell.uid.startswith(target):
            return h
    return None


def _fmt_running_row(handle: CellHandle) -> str:
    cell = handle.runtime.cell
    meta = handle.process.meta
    uptime = _fmt_uptime(_now_ts() - meta.created)
    return (
        f'  {cell.fullname:<24} uid={_uid_short(handle.address)} '
        f'uptime={uptime} pid={meta.pid}'
    )


def _fmt_dead_row(handle: CellHandle) -> str:
    cell = handle.runtime.cell
    meta = handle.process.meta
    code = meta.exit_code
    when = _fmt_uptime(_now_ts() - meta.updated)
    tail = ''
    if code not in (0, None):
        tail = f' — nodes:read_output({cell.fullname}) for stderr'
    return f'  {cell.fullname:<24} exit={code} ({when} ago){tail}'


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
        'Local node governance: list / read / run / stop / status / '
        'read_output declared cells in this project. '
        'Running cells surface as siblings under matrix.mesh once accepted.'
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
        """List discoverable node declarations in this project.

        Empty ``path`` scans project.nodes_discover_paths (default). Absolute
        or project-relative ``path`` scans that root — useful for exploring
        outside the governed area (cognitive fields, file editor).

        :param path: scan root; empty = default
        :param category: filter by NodeManifest.category ('' = all)
        :param installed: True = installed only; False = uninstalled only;
            None = both
        :param refresh: rescan filesystem (default = use cache)
        """
        nodes_mgr = matrix.project.nodes
        scan_paths: list[Path] | None = None
        if path:
            p = Path(path)
            if not p.is_absolute():
                p = matrix.project.root.abspath_of(p)
            scan_paths = [p]
        found = nodes_mgr().list_nodes(
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
        """Read a node's manifest — signature (frontmatter) + usage (instruction body).

        The manifest is "how to use this node before you run it": name,
        category, singleton flag, ExecSpec, and the free-form instruction.

        :param target: project-relative path to the node (from ``list`` output)
        """
        if not target:
            CommandUtil.raise_observe(
                "target required. Use nodes:list() to discover paths."
            )
        manifest = matrix.project.nodes().get_node(target)
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
        """Spawn a node cell. Nonblocking — the organ becomes available next frame.

        The receipt just confirms the spawn. Watch for the node's channel
        under ``matrix.mesh.<fullname>`` on the next perspective refresh.
        Errors (unknown target, uninstalled, singleton conflict) return
        immediately without spawn.

        :param target: project-relative path (from ``list``) or absolute path
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
        return (
            f'[{handle.address}] pid={handle.process.meta.pid} — '
            f'organ appears next frame under matrix.mesh once announced.'
        )

    # -- stop ---------------------------------------------------------

    @chan.build.command(name='stop', blocking=False, always_observe=False)
    async def stop_node(address: str, timeout: float = 5.0) -> str:
        """Stop a running node cell (SIGTERM → grace period → killpg).

        :param address: full CellAddress ``node/name/uid`` or short form
            (``fullname`` or ``uid_prefix``) if it uniquely matches
        :param timeout: grace period in seconds before force kill
        """
        handled = matrix.handled_cells()
        handle = _resolve_handled_address(address, handled)
        if handle is None:
            CommandUtil.raise_observe(
                f'{address!r} does not uniquely match any running cell. '
                f'nodes:status() shows current cells.'
            )
        await handle.stop(timeout=timeout)
        code = handle.process.meta.exit_code
        return f'[{handle.address}] stopped, exit={code}'

    # -- status -------------------------------------------------------

    @chan.build.command(name='status', blocking=False, always_observe=True)
    async def status_node(address: str = '') -> str:
        """Inspect running cells + recently exited.

        Empty ``address`` returns the full picture. Given ``address`` returns
        a single-cell brief (handle + process meta + spawn cwd path).

        :param address: full CellAddress or short form; empty = full list
        """
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
        """Read a cell's captured output tail (memory ring buffer, ~200 lines).

        Works for both running and recently dead cells. For dead cells the
        full log is also persisted at ``{spawn_cwd}/{stdout,stderr}.log``
        until the next host restart (matrix-channel.md §5.4).

        :param address: full CellAddress or short form
        :param stream: 'stdout' or 'stderr' (default 'stderr')
        :param limit: max lines to return (0 = whole window)
        """
        handled = matrix.handled_cells()
        dead = list(matrix.dead_cells())
        handle = _find_handle_in_all(address, handled, dead)
        if handle is None:
            CommandUtil.raise_observe(
                f'{address!r} not found in handled or dead cells.'
            )
        output = handle.process.output
        if output is None:
            return f'[{handle.address}] no capture buffer (spawn without CaptureSpec).'
        if stream == 'stdout':
            body = output.stdout(limit=limit)
        else:
            body = output.stderr(limit=limit)
        if not body:
            return f'[{handle.address}] {stream} empty.'
        return f'[{handle.address}] {stream} tail:\n{body.rstrip()}'

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
    lines = [
        f'[{handle.address}] {"running" if alive else "dead"}',
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
        'Network projection: accepted cells surface as sub-channels '
        '(matrix.mesh.<fullname>:<command>). accept/reject/set_auto_accept '
        'govern which network cells are trusted resources.'
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
        # 1) 写自持 ring buffer (喂 context / events 命令)
        event_buffer.append(event)
        # 2) 转 Signal 送 CellEventNucleus (M7.5)
        try:
            meta = CellEventSignalMeta(
                address=event.address,
                # 本轮 CellEvent 无 transition 字段, 统一 READY (§5.9 简化).
                # 未来扩展 CellEvent 或 on_exit 补 exited/crashed 时再分档.
                transition=CellTransition.READY,
            )
            content = event.content or f'cell {event.address} updated'
            signal = meta.to_signal(
                content,
                description=f'cell_event {event.address}',
            )
            CommandUtil.send_signal(signal)
        except Exception:
            # signal 送不出去不该阻塞 mesh 事件消费
            logger = CommandUtil.logger()
            if logger is not None:
                logger.exception('mesh channel: failed to dispatch cell_event')

    @chan.build.startup
    async def _startup() -> None:
        mesh = await matrix.mesh()
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
        mesh = await matrix.mesh()
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
            # alias 用 cell.fullname (未来场景倒逼时可加 uid 后缀去冲突)
            mesh_view = mesh.view()
            cell = mesh_view.get(new_addr)
            alias = cell.fullname if cell is not None else new_addr.replace('/', '_')
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
        # CellMesh ABC 没暴露 auto_accept 查询接口, 用一个"守卫函数"占位.
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
        """Trust a network cell's resources — build its channel proxy immediately.

        :param address: full CellAddress ``kind/name/uid``
        :param lookup: True = refresh mesh view first if address not visible;
            False = wait for its presence to arrive naturally
        """
        mesh = await matrix.mesh()
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
        """Refuse a network cell's resources — tear down any active proxy.

        :param address: full CellAddress ``kind/name/uid``
        """
        mesh = await matrix.mesh()
        await mesh.reject(address)
        return f'[mesh:reject {address}] resource withdrawn.'

    @chan.build.command(
        name='set_auto_accept', blocking=False, always_observe=False,
    )
    async def set_auto_accept(
            local: bool | None = None, foreign: bool | None = None,
    ) -> str:
        """Toggle auto-accept policy (None = leave alone).

        Immediately re-scans the view: cells newly covered by policy get
        proxies built; cells no longer covered get proxies torn down.
        Explicit accept/reject tables override policy — this does not touch them.

        :param local: auto-accept cells in this project (None = keep)
        :param foreign: auto-accept cells outside this project (None = keep)
        """
        mesh = await matrix.mesh()
        mesh.set_auto_accept(local=local, foreign=foreign)
        return (
            f'[mesh:set_auto_accept] applied '
            f'(local={local}, foreign={foreign}).'
        )

    @chan.build.command(name='events', blocking=False, always_observe=True)
    async def events_cmd(address: str = '', limit: int = 20) -> str:
        """Read recent cell events (network-level).

        :param address: filter to a single cell; empty = all
        :param limit: max events to return (default 20)
        """
        mesh = await matrix.mesh()
        if address:
            events = mesh.cell_events(address, limit=limit)
        else:
            events = mesh.recent_events(limit=limit)
        if not events:
            return '[mesh:events] (empty)'
        lines = [f'[mesh:events {"@" + address if address else "all"}]']
        for ev in events:
            when = ev.created.strftime('%H:%M:%S')
            content = ev.content or '(no content)'
            lines.append(f'  {when}  {ev.address}  {content}')
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
            content = ev.content or 'updated'
            lines.append(f'  {when}  {ev.address}  {content}')
        # 网络概要
        try:
            # 若 mesh 尚未惰性 fetch 过, mesh.view() 无 await 也应能返回缓存
            # (CellMesh.view 是 sync). 但 mesh() 是 async factory, 需要
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
            '(matrix.mesh.<fullname>). accept/reject govern resource trust; '
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
    """集成点. import_channels 挂 nodes + mesh + extras (terminal / file_editor).
    本轮 matrix 自身无 own commands, 极简自我介绍."""

    default_desc = (
        'Matrix integration point: network projection and local organ '
        'governance. Children: nodes (local cell declarations + spawn), '
        'mesh (accepted network cells surface here), plus attached tools.'
    )
    chan = new_channel(name=name, description=description or default_desc)

    # 静态挂 nodes + mesh (composed inline, share matrix reference)
    nodes = new_nodes_channel(matrix)
    mesh = new_mesh_channel(matrix)
    chan.import_channels(nodes, mesh, *extra_children)

    @chan.build.instruction
    def matrix_instruction() -> str:
        return (
            'This is your body\'s integration point. '
            'nodes: what organs can be declared / running locally. '
            'mesh: what organs (yours and others\') are on the network. '
            'Tools attached here run in this process, addressable as '
            'matrix.<tool>:<command>.'
        )

    return chan


def build_matrix_channel(
        *,
        name: str = 'matrix',
        description: str | None = None,
        with_terminal: bool = True,
        with_file_editor: bool = True,
        extra_children: tuple[Channel | ChannelFactory, ...] = (),
) -> ChannelFactory:
    """High-order factory: config → ChannelFactory.

    Resolves Matrix from container, composes nodes + mesh + optional tools
    (terminal / file_editor) as static children of matrix.

    :param name: matrix channel tag (default 'matrix')
    :param description: override default description
    :param with_terminal: attach ``terminal_channel`` as ``matrix.bash``
    :param with_file_editor: attach ``file_editor_channel`` as ``matrix.file_editor``
    :param extra_children: additional Channel or ChannelFactory to import
    """

    def factory(container: IoCContainer) -> Channel:
        matrix = container.force_fetch(Matrix)

        extras: list[Channel | ChannelFactory] = list(extra_children)
        if with_terminal:
            from ghoshell_moss.channels.terminal_channel import build_terminal_channel
            extras.append(build_terminal_channel())
        if with_file_editor:
            from ghoshell_moss.channels.file_editor_channel import (
                build_file_editor_channel,
            )
            extras.append(build_file_editor_channel())

        return new_matrix_channel(
            matrix, name=name, description=description,
            extra_children=tuple(extras),
        )

    return factory
