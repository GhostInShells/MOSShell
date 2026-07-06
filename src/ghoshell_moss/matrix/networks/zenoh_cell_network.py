"""Zenoh-based CellNetwork — 推拉结合的 cell 发现 + 事件总线 + 自动 proxy.

Cell 在网络上的存在 = 1 个 key, 2 种 zenoh 原语:
  {cells_ns}/{cell.address}  → liveness token (推: 上线/下线, subscriber 实时感知)
  {cells_ns}/{cell.address}  → queryable   (拉: Cell JSON 全量, 按需 get)

Host 额外跨域宣告:
  {hosts_ns}/{scope}/cells/liveness/{cell.address} → liveness token

CellLog 总线 (§SS-2 轻量事件, 不广播 cell snapshot):
  {cells_ns}/logs/{cell.address}  → cell-owned zenoh.Publisher pub/sub
  - owner: announce 时 declare_publisher, broadcast_log 时 publisher.put(version+=1)
  - subscriber: 订阅 {cells_ns}/logs/** , put_nowait 到 janus queue 卸载到 event loop
  - consumer: 单 asyncio task 串行消费, 按 address 维度 last-version-wins
  - terminal=True → pop cache + fire on_change(False); terminal=False → refetch cache

发现:
  subscriber 实时推 — PUT=上线 DELETE=下线, 零轮询
  reconcile loop 低频兜底 — 60s+ 全量对账, 覆盖极端丢事件

自动 build proxy (§SS-3 / §SS-5, 仅 allow_create_proxy=True 启用):
  hub liveness PUT (channels_ns/{address})
    → hub._on_provider_online → fan-out to self._auto_build_proxy
    → 查 cells_ns 拿 Cell snapshot → hub.proxy(address, name=cell.channel_name)
    → broadcast_log(address, "channel-ready")
"""

import asyncio
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import janus

from ghoshell_moss.depends import depend_zenoh

depend_zenoh()
import zenoh

from ghoshell_moss.core.blueprint.cell import (
    Cell,
    CellAddress,
    CellLog,
    CellNetwork,
    DuplicatedError,
)
from ghoshell_moss.core.concepts.channel import Channel, ChannelProxy, ChannelProvider
from ghoshell_moss.tools.zenoh_helper import MatrixNamespace
from ghoshell_moss.bridges.zenoh_bridge import ZenohChannelHub

import logging

__all__ = ['ZenohCellNetwork']

# 大数, log 是低频事件, 这里不做背压. 入队失败说明上游异常, 走 logger.error 暴露问题.
_LOG_QUEUE_MAXSIZE = 10000


@dataclass
class _Announce:
    """一次 cell announce 持有的 zenoh 资源 + log publisher."""
    liveness_token: zenoh.LivelinessToken
    queryable: zenoh.Queryable
    log_publisher: zenoh.Publisher
    cell: Cell
    host_token: zenoh.LivelinessToken | None = None

    def close(self) -> None:
        for resource_name in ("log_publisher", "queryable", "liveness_token", "host_token"):
            resource = getattr(self, resource_name, None)
            if resource is None:
                continue
            try:
                resource.undeclare()
            except RuntimeError:
                pass


class ZenohCellNetwork(CellNetwork):
    """基于 zenoh 的 CellNetwork 实现.

    线程模型:
    - zenoh subscriber 回调线程: liveness sample, log sample. 只做最薄工作 (put_nowait 入队 / 同步 cache 操作).
    - asyncio event loop 线程: 主流程, log consumer task, reconcile loop.
    - 跨线程数据卸载用 janus.Queue (避免锁竞争阻塞 zenoh 线程).

    缓存写者:
    - liveness subscriber (zenoh 线程): cache PUT/DELETE.
    - log consumer task (event loop): cache PUT (refetch) / DELETE (terminal log).
    - reconcile loop (event loop): cache diff 补救.
    三方写者共享 self._cache, 用 threading.Lock 保护 (跨线程必需).
    """

    def __init__(
            self,
            *,
            session: zenoh.Session,
            logger: logging.Logger,
            namespace: MatrixNamespace,
            scope: str,
            allow_create_proxy: bool = False,
            reconcile_interval: float = 60.0,
            self_project_id: str | None = None,
            log_buffer_size: int = 100,
    ):
        self._session = session
        self._logger = logger
        self._ns = namespace
        self._scope = scope
        self._allow_create_proxy = allow_create_proxy
        self._reconcile_interval = reconcile_interval
        self._self_project_id = self_project_id

        # -- zenoh key 约定, 全部在 init 生成 ---------------------------
        self._scopes_root = "MOSS/matrix/scopes"
        self._cells_ns_prefix = self._ns.cells_ns + '/'
        self._hosts_ns_prefix = self._ns.hosts_ns + '/'
        # host 跨域 key 中的语义分隔标记: {scope}/cells/liveness/{address}
        self._host_liveness_marker = "/cells/liveness/"

        # cells wildcards
        self._cell_liveness_wildcard = self._join(self._ns.cells_ns, "**")
        self._host_discovery_wildcard = self._join(self._ns.cells_ns, "host", "**")
        self._cross_hosts_wildcard = self._join(
            self._ns.hosts_ns, "*", "cells", "liveness", "**",
        )

        # CellLog 总线 — 挂在 cells 分组下 (§SS-2). 单 key per cell, address 段后不再追加.
        # 同 key 多次 put 在 zenoh pub/sub 模型下都会送达 subscriber.
        self._cells_logs_ns = self._join(self._ns.cells_ns, "logs")
        self._cells_logs_ns_prefix = self._cells_logs_ns + '/'
        self._cells_logs_wildcard = self._join(self._cells_logs_ns, "**")

        # -- channel 层委托 hub ------------------------------------------
        self._hub = ZenohChannelHub(
            zenoh_session=session,
            scope=scope,
            logger=logger,
            namespace=namespace,
        )

        # -- 自身宣告 ----------------------------------------------------
        self._announcements: dict[CellAddress, _Announce] = {}

        # -- 发现缓存 — 写者跨线程, 必须锁
        self._cache: dict[CellAddress, Cell] = {}
        self._cache_lock = threading.Lock()

        # change callbacks — 调用线程不固定, caller 负责线程安全
        self._change_callbacks: list[Callable[[Cell, bool], None]] = []

        # -- CellLog ---------------------------------------------------
        # ring buffer 容量上限, 单写者 (consumer task) 不需要锁
        self._log_buffer: deque[CellLog] = deque(maxlen=log_buffer_size)
        # janus 队列 — zenoh 线程 put_nowait, consumer task get
        self._log_queue: janus.Queue[CellLog] | None = None
        # consumer task, last-version-wins state 由其私有持有
        self._log_consumer_task: asyncio.Task | None = None

        # subscribers
        self._cell_subscriber: zenoh.Subscriber | None = None
        self._log_subscriber: zenoh.Subscriber | None = None

        # hub callback unsubscribe handles
        self._hub_unsubs: list[Callable[[], None]] = []

        # main event loop ref — zenoh 后台线程 schedule async 工作时用
        self._loop: asyncio.AbstractEventLoop | None = None

        # reconcile — 低频兜底
        self._reconcile_task: asyncio.Task | None = None

        self._started = False
        self._closed = False

    # ── properties ────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return f"zenoh-cell-network/{self._scope}"

    @property
    def description(self) -> str:
        return f"Zenoh CellNetwork — scope={self._scope}"

    @property
    def scope(self) -> str:
        return self._scope

    # ==================================================================
    # key 生成 — 参数化的 key 构建方法 (非参数化约定在 __init__)
    # ==================================================================

    @staticmethod
    def _join(*segments: str) -> str:
        return '/'.join(segments)

    def _cell_key(self, address: CellAddress) -> str:
        """scope 内 cell 的 liveness + queryable key: {cells_ns}/{address}"""
        return self._join(self._ns.cells_ns, address)

    def _cell_log_key(self, address: CellAddress) -> str:
        """cell 的 log publisher key: {cells_ns}/logs/{address}. 单 key, 不带后缀."""
        return self._join(self._cells_logs_ns, address)

    def _host_liveness_key(self, scope: str, address: CellAddress) -> str:
        """host 跨域 liveness key: {hosts_ns}/{scope}/cells/liveness/{address}"""
        return self._join(self._ns.hosts_ns, scope, "cells", "liveness", address)

    def _remote_cell_key(self, scope: str, address: CellAddress) -> str:
        """跨 scope 回查 cell queryable key: scopes/{scope}/cells/{address}"""
        return self._join(self._scopes_root, scope, "cells", address)

    def _key_to_address(self, key: str) -> CellAddress | None:
        """从 {cells_ns}/{address} liveness key 提取 cell address.

        用 prefix strip 而非段数索引 — address 自身含 / 不受影响.
        注意排除 log key: {cells_ns}/logs/... 也匹配 cells_ns_prefix.
        """
        if not key.startswith(self._cells_ns_prefix):
            return None
        rest = key[len(self._cells_ns_prefix):]
        # logs/... 是 log key, 不是 cell liveness key
        if rest.startswith('logs/'):
            return None
        return rest

    def _log_key_to_address(self, key: str) -> CellAddress | None:
        """从 {cells_ns}/logs/{address} log key 提取 cell address."""
        if not key.startswith(self._cells_logs_ns_prefix):
            return None
        return key[len(self._cells_logs_ns_prefix):]

    def _parse_host_liveness_key(self, key: str) -> tuple[str, CellAddress] | None:
        """从 {hosts_ns}/{scope}/cells/liveness/{address} 解析 (scope, address)."""
        if not key.startswith(self._hosts_ns_prefix):
            return None
        rest = key[len(self._hosts_ns_prefix):]
        idx = rest.find(self._host_liveness_marker)
        if idx == -1:
            return None
        scope = rest[:idx]
        address = rest[idx + len(self._host_liveness_marker):]
        if not scope or not address:
            return None
        return scope, address

    # ==================================================================
    # announce
    # ==================================================================

    async def update_cell(self, cell: Cell, *, log: str = '') -> None:
        address = cell.address

        if address in self._announcements:
            # 同 address 重复宣告: 仅更新 cell 引用, queryable handler 下次查询返回新数据
            self._announcements[address].cell = cell
            if log:
                await self.broadcast_log(address, log, terminal=False)
            return

        key = self._cell_key(address)

        # 首次宣告前检查唯一性
        await self.check_unique(cell)

        # queryable 必须先于 liveness token 注册 — 避免 subscriber 收到 PUT
        # 时 queryable 尚未就位导致 _sync_fetch_cell 空返回
        def _on_query(query: zenoh.Query):
            try:
                ann = self._announcements.get(address)
                if ann is None:
                    return
                payload = ann.cell.to_json().encode('utf-8')
                query.reply(query.key_expr, payload)
            except Exception:
                self._logger.exception("cell queryable error for %s", address)

        queryable = self._session.declare_queryable(key, _on_query)

        # log publisher — cell-owned, 每个 cell 一个 publisher 复用 routing
        log_publisher = self._session.declare_publisher(self._cell_log_key(address))

        # liveness token — 推: 上线/下线由 subscriber 实时感知
        liveness_token = self._session.liveliness().declare_token(key)

        host_token = None
        if cell.is_host:
            host_key = self._host_liveness_key(self._scope, address)
            host_token = self._session.liveliness().declare_token(host_key)

        self._announcements[address] = _Announce(
            liveness_token=liveness_token,
            queryable=queryable,
            log_publisher=log_publisher,
            cell=cell,
            host_token=host_token,
        )

        self._logger.debug("cell announced: address=%s host=%s", address, cell.is_host)

        if log:
            await self.broadcast_log(address, log, terminal=False)

    async def revoke_cell(self, cell: Cell, *, log: str = '') -> None:
        address = cell.address

        ann = self._announcements.get(address)
        if ann is None:
            raise LookupError(
                f"cell address '{address}' was not announced by this instance"
            )

        # 先 broadcast terminal log (publisher 还活着), 再 close
        if log:
            try:
                await self.broadcast_log(address, log, terminal=True)
            except Exception:
                self._logger.exception("broadcast revoke log failed for %s", address)

        self._announcements.pop(address, None)
        ann.close()
        self._logger.debug("cell revoked: address=%s", address)

    # ==================================================================
    # check_unique
    # ==================================================================

    async def check_unique(self, cell: Cell) -> None:
        """检查 cell 在网络上是否唯一可宣告.

        - host: scope 内只允许 1 个 host (无论 name) — 查 {cells_ns}/host/**.
        - 其它 singleton: 同 identity (type/name) 唯一 — 查 {cells_ns}/type/name/*.
        - non-singleton: 仅检 address exact key (含 uid).

        :raise DuplicatedError: 已被别处声明.
        """
        if cell.is_host:
            check_key = self._host_discovery_wildcard
        elif cell.meta.singleton:
            check_key = self._join(
                self._ns.cells_ns, cell.meta.type, cell.meta.name, "*",
            )
        else:
            check_key = self._cell_key(cell.address)

        replies = await asyncio.to_thread(
            self._session.liveliness().get, check_key,
        )
        for reply in replies:
            if reply.ok:
                raise DuplicatedError(
                    f"cell '{cell.identity}' is already announced on the network "
                    f"(is_host={cell.is_host}, singleton={cell.meta.singleton})"
                )

    # ==================================================================
    # discovery
    # ==================================================================

    async def get_host(self) -> Cell | None:
        replies = await asyncio.to_thread(
            self._session.liveliness().get,
            self._host_discovery_wildcard,
        )
        for reply in replies:
            if not reply.ok:
                continue
            address = self._key_to_address(str(reply.result.key_expr))
            if address is None:
                continue
            cell = await self._fetch_cell(address)
            if cell is not None:
                return cell
        return None

    async def all_hosts(self) -> list[Cell]:
        replies = await asyncio.to_thread(
            self._session.liveliness().get,
            self._cross_hosts_wildcard,
        )
        result: list[Cell] = []
        for reply in replies:
            if not reply.ok:
                continue
            parsed = self._parse_host_liveness_key(str(reply.result.key_expr))
            if parsed is None:
                continue
            remote_scope, address = parsed
            cell = await self._fetch_cell_via(
                self._remote_cell_key(remote_scope, address),
            )
            if cell is not None:
                result.append(cell)
        return result

    async def get_live_cells(
            self,
            *,
            type: str | None = None,
            local: bool | None = None,
            refresh: bool = False,
    ) -> dict[CellAddress, Cell]:
        if not refresh:
            with self._cache_lock:
                cells = dict(self._cache)
        else:
            cells = await self._refresh_cache_full()

        if type is not None:
            cells = {a: c for a, c in cells.items() if c.type == type}
        if local is not None and self._self_project_id:
            cells = {
                a: c for a, c in cells.items()
                if (c.status.project_id == self._self_project_id) == local
            }
        return cells

    async def _refresh_cache_full(self) -> dict[CellAddress, Cell]:
        """liveness get 全量 → queryable get 每个 cell → 写 cache. 返回结果副本."""
        cells: dict[CellAddress, Cell] = {}
        replies = await asyncio.to_thread(
            self._session.liveliness().get,
            self._cell_liveness_wildcard,
        )
        for reply in replies:
            if not reply.ok:
                continue
            address = self._key_to_address(str(reply.result.key_expr))
            if address is None:
                continue
            cell = await self._fetch_cell(address)
            if cell is not None:
                cells[address] = cell

        with self._cache_lock:
            self._cache.update(cells)
        self._logger.debug("live cells refreshed: %d cells", len(cells))
        return cells

    def live_cells(self) -> dict[CellAddress, Cell]:
        with self._cache_lock:
            return dict(self._cache)

    def on_change(self, callback: Callable[[Cell, bool], None]) -> None:
        self._change_callbacks.append(callback)

    # ==================================================================
    # channel (provider / proxy delegate to hub)
    # ==================================================================

    async def provide(
            self,
            channel: Channel,
            *,
            address: CellAddress | None = None,
    ) -> ChannelProvider:
        if not address:
            raise ValueError(
                "ZenohCellNetwork.provide requires explicit address — "
                "Matrix layer should pass self.this.address"
            )
        return self._hub.provider(address)

    def proxies(self) -> dict[CellAddress, ChannelProxy]:
        """delegate to hub. 仅在 allow_create_proxy=True 时有非空内容."""
        if not self._allow_create_proxy:
            return {}
        return self._hub.proxies

    def get_proxy(self, address: CellAddress) -> ChannelProxy | None:
        if not self._allow_create_proxy:
            return None
        return self._hub.get_proxy(address)

    async def wait_connected(
            self,
            address: CellAddress,
            *,
            timeout: float = 30,
    ) -> bool:
        """等 address 对应 proxy 上线. 已 ready 立即返回 True; 超时返回 False."""
        if self._hub.get_proxy(address) is not None:
            return True

        loop = asyncio.get_running_loop()
        fut: asyncio.Future[bool] = loop.create_future()

        def _on_online(addr: str) -> None:
            if addr != address or fut.done():
                return
            loop.call_soon_threadsafe(
                lambda: fut.done() or fut.set_result(True)
            )

        unsub = self._hub.on_provider_online(_on_online)
        try:
            # double-check 防 race
            if self._hub.get_proxy(address) is not None:
                return True
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            return False
        finally:
            unsub()

    def on_provider_online(
            self,
            callback: Callable[[CellAddress], None],
    ) -> Callable[[], None]:
        return self._hub.on_provider_online(callback)

    def on_provider_offline(
            self,
            callback: Callable[[CellAddress], None],
    ) -> Callable[[], None]:
        return self._hub.on_provider_offline(callback)

    # ==================================================================
    # CellLog 总线 (broadcast_log / recent_logs)
    # ==================================================================

    async def broadcast_log(
            self,
            address: CellAddress,
            content: str,
            *,
            terminal: bool = False,
    ) -> None:
        """broadcast 一条 CellLog 到 network.

        约定: 只能为本 network 已 announce 的 cell broadcast log.
        别的 cell 死亡观察靠 liveness subscriber, 不是再次 broadcast.

        :raise LookupError: address 没在本 network announce.
        """
        ann = self._announcements.get(address)
        if ann is None:
            raise LookupError(
                f"cannot broadcast log for {address}: not announced by this network"
            )
        log = CellLog(
            address=address,
            content=content,
            timestamp=time.time(),
            terminal=terminal,
        )
        payload = log.model_dump_json().encode('utf-8')
        try:
            await asyncio.to_thread(ann.log_publisher.put, payload)
        except Exception:
            self._logger.exception(
                "publisher.put failed for log: address=%s content=%s",
                address, content,
            )

    def recent_logs(
            self,
            *,
            limit: int = 20,
            local: bool | None = None,
    ) -> list[CellLog]:
        """FIFO 最近 N 条 (最新优先). local 过滤需要 self_project_id, 否则忽略 local.

        log_buffer 是 deque, 单消费者 (log consumer task) 写, 多读者 snapshot.
        CPython GIL 保证 list(deque) atomic, 不需要锁.
        """
        snapshot = list(self._log_buffer)
        snapshot.reverse()  # 最新优先
        if local is not None and self._self_project_id:
            filtered = []
            for log in snapshot:
                with self._cache_lock:
                    cell = self._cache.get(log.address)
                if cell is None:
                    # cache miss — terminal=True 的 log 留下 (cell 已没了, 无法回查)
                    if log.terminal:
                        filtered.append(log)
                    continue
                cell_local = (cell.status.project_id == self._self_project_id)
                if cell_local == local:
                    filtered.append(log)
            snapshot = filtered
        return snapshot[:limit]

    # ==================================================================
    # log subscriber + consumer (janus 卸载)
    # ==================================================================

    def _on_log_sample(self, sample: zenoh.Sample) -> None:
        """zenoh log subscriber 回调 — zenoh 后台线程. 只做最薄工作:
        parse + put_nowait 入队. 处理逻辑卸载到 consumer task.
        """
        if sample.kind != zenoh.SampleKind.PUT:
            return
        try:
            log = CellLog.model_validate_json(sample.payload.to_bytes())
        except Exception:
            self._logger.exception(
                "failed to parse CellLog from sample, key=%s", sample.key_expr,
            )
            return

        if self._log_queue is None:
            return  # 已关闭

        try:
            self._log_queue.sync_q.put_nowait(log)
        except janus.SyncQueueFull:
            # log 是低频事件, 满了说明 consumer 严重落后. 报 error 暴露问题, 丢弃此条.
            self._logger.error(
                "log queue full (maxsize=%d), dropping log: address=%s content=%s",
                _LOG_QUEUE_MAXSIZE, log.address, log.content,
            )
        except janus.SyncQueueShutDown:
            pass

    async def _log_consumer_loop(self) -> None:
        """单消费者循环 — 串行处理 log.

        单 publisher per cell + zenoh subscriber 串行回调 + 单 consumer task =
        天然按 cell 维度有序. 不需要 version 去重.
        """
        while not self._closed:
            try:
                if self._log_queue is None:
                    return
                log = await self._log_queue.async_q.get()
            except janus.AsyncQueueShutDown:
                return
            except asyncio.CancelledError:
                return

            try:
                await self._process_log(log)
            except asyncio.CancelledError:
                raise
            except Exception:
                self._logger.exception(
                    "log consumer failed on: address=%s content=%s",
                    log.address, log.content,
                )

    async def _process_log(self, log: CellLog) -> None:
        """串行处理一条 log. 单 consumer task 内, 无并发.

        on_change 语义: cell 上下线状态变化, 不是 cache snapshot 刷新.
        - terminal=True: 主动终结信号, fire on_change(False) (与 liveness DELETE 同义).
        - terminal=False: snapshot 已变, refetch 写 cache. **不** fire on_change —
          cell 仍在线, observer 需要新 snapshot 时自取 live_cells / get_live_cells.
        """
        # 单写者 (本 task), deque 无锁
        self._log_buffer.append(log)

        if log.terminal:
            with self._cache_lock:
                cell = self._cache.pop(log.address, None)
            if cell is not None:
                self._fire_on_change(cell, False)
            return

        # 非 terminal — refetch cell snapshot 更新 cache, 不 fire on_change
        try:
            cell = await self._fetch_cell(log.address)
        except Exception:
            self._logger.debug(
                "refetch_into_cache failed: address=%s", log.address,
            )
            return
        if cell is None:
            return
        with self._cache_lock:
            self._cache[log.address] = cell

    # ==================================================================
    # internal: auto build proxy chain (hub callback fan-out)
    # ==================================================================

    def _auto_build_proxy(self, address: CellAddress) -> None:
        """hub.on_provider_online 回调 (zenoh 后台线程) — 自动 build proxy.

        仅 allow_create_proxy=True 启用 (host 视角).
        """
        if not self._allow_create_proxy:
            return
        if self._hub.get_proxy(address) is not None:
            return
        try:
            cell = _sync_fetch_cell(self._session, self._cell_key(address))
        except Exception:
            self._logger.exception("auto_build_proxy: fetch %s failed", address)
            return
        if cell is None:
            # cell snapshot 尚未就位 — 跳过, 后续 hub callback 会重触发
            self._logger.debug(
                "auto_build_proxy: cell snapshot not yet ready for %s, skip",
                address,
            )
            return
        try:
            self._hub.proxy(address, name=cell.channel_name)
        except Exception:
            self._logger.exception("auto_build_proxy: hub.proxy %s failed", address)
            return
        self._logger.debug("auto built proxy: address=%s name=%s", address, cell.channel_name)
        # schedule broadcast_log to event loop — 当前在 zenoh 线程, 不能 await
        if self._loop is not None and not self._loop.is_closed():
            asyncio.run_coroutine_threadsafe(
                self._safe_broadcast_log(address, "channel-ready", terminal=False),
                self._loop,
            )

    def _auto_drop_proxy(self, address: CellAddress) -> None:
        """hub.on_provider_offline 回调 — hub 已自动 pop proxy, 这里 broadcast log."""
        if not self._allow_create_proxy:
            return
        if self._loop is not None and not self._loop.is_closed():
            asyncio.run_coroutine_threadsafe(
                self._safe_broadcast_log(address, "channel-gone", terminal=True),
                self._loop,
            )

    async def _safe_broadcast_log(
            self,
            address: CellAddress,
            content: str,
            *,
            terminal: bool,
    ) -> None:
        """broadcast_log 的容错包装 — 用于 schedule 入口, 异常吞掉记日志."""
        try:
            await self.broadcast_log(address, content, terminal=terminal)
        except LookupError:
            # 这种情况是: hub 看到的 provider 不是本 network announce 的 cell.
            # 这是合法的 — hub 跨 network 都能看, 只是本 network 不该广播.
            self._logger.debug(
                "auto broadcast skipped: %s not announced here", address,
            )
        except Exception:
            self._logger.exception(
                "auto broadcast failed: address=%s content=%s", address, content,
            )

    # ==================================================================
    # lifecycle
    # ==================================================================

    async def __aenter__(self):
        if self._started:
            return self
        self._started = True
        self._loop = asyncio.get_running_loop()
        self._log_queue = janus.Queue(maxsize=_LOG_QUEUE_MAXSIZE)

        await self._hub.__aenter__()

        # 先订阅 — 避免 declare 和 get 之间的 TOCTOU 丢事件
        self._cell_subscriber = self._session.liveliness().declare_subscriber(
            self._cell_liveness_wildcard,
            self._on_liveness_sample,
        )
        self._log_subscriber = self._session.declare_subscriber(
            self._cells_logs_wildcard,
            self._on_log_sample,
        )

        # log consumer task
        self._log_consumer_task = self._loop.create_task(self._log_consumer_loop())

        # 注册 hub 回调 — auto build/drop proxy
        self._hub_unsubs.append(self._hub.on_provider_online(self._auto_build_proxy))
        self._hub_unsubs.append(self._hub.on_provider_offline(self._auto_drop_proxy))

        # 初始全量 seed — subscriber 已就位, 重复事件无害
        await self._seed_cache()

        # 低频 reconcile — 兜底极端丢事件
        self._reconcile_task = self._loop.create_task(self._reconcile_loop())

        self._logger.debug(
            "ZenohCellNetwork started: scope=%s cells_ns=%s",
            self._scope, self._ns.cells_ns,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._closed = True

        # 取消 reconcile + log consumer
        for task in (self._reconcile_task, self._log_consumer_task):
            if task is None or task.done():
                continue
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                self._logger.exception("task cancellation surfaced")
        self._reconcile_task = None
        self._log_consumer_task = None

        # 关闭 subscribers
        for sub in (self._cell_subscriber, self._log_subscriber):
            if sub is None:
                continue
            try:
                sub.undeclare()
            except RuntimeError:
                pass
        self._cell_subscriber = None
        self._log_subscriber = None

        # 关闭 janus queue
        if self._log_queue is not None:
            self._log_queue.shutdown(immediate=True)
            self._log_queue = None

        # 注销 hub 回调
        for unsub in self._hub_unsubs:
            try:
                unsub()
            except Exception:
                self._logger.exception("hub unsub failed")
        self._hub_unsubs.clear()

        # 关闭所有自身宣告 (含 log publisher / queryable / liveness / host_token)
        for ann in list(self._announcements.values()):
            ann.close()
        self._announcements.clear()

        await self._hub.__aexit__(exc_type, exc_val, exc_tb)

        with self._cache_lock:
            self._cache.clear()

        self._started = False
        self._loop = None
        self._logger.debug("ZenohCellNetwork stopped: scope=%s", self._scope)

    # ==================================================================
    # internal: data fetch
    # ==================================================================

    async def _fetch_cell(self, address: CellAddress) -> Cell | None:
        return await self._fetch_cell_via(self._cell_key(address))

    async def _fetch_cell_via(self, info_key: str) -> Cell | None:
        try:
            replies = await asyncio.to_thread(self._session.get, info_key)
            for reply in replies:
                if reply.ok:
                    return Cell.model_validate_json(reply.ok.payload.to_bytes())
        except Exception:
            self._logger.debug("failed to fetch cell info via %s", info_key)
        return None

    # ==================================================================
    # internal: liveness subscriber
    # ==================================================================

    def _on_liveness_sample(self, sample: zenoh.Sample) -> None:
        """zenoh liveness subscriber 回调 — zenoh 后台线程.

        不持锁做 I/O. cache 锁只在写入瞬间持有.
        """
        address = self._key_to_address(str(sample.key_expr))
        if address is None:
            return

        if sample.kind == zenoh.SampleKind.PUT:
            try:
                cell = _sync_fetch_cell(self._session, self._cell_key(address))
            except Exception:
                self._logger.exception(
                    "failed to fetch cell on liveness PUT: %s", address,
                )
                return
            if cell is None:
                return
            with self._cache_lock:
                self._cache[address] = cell
            self._fire_on_change(cell, True)

        elif sample.kind == zenoh.SampleKind.DELETE:
            with self._cache_lock:
                cell = self._cache.pop(address, None)
            if cell is not None:
                self._fire_on_change(cell, False)

    # ==================================================================
    # internal: reconcile
    # ==================================================================

    async def _seed_cache(self) -> None:
        """启动时全量 liveness get, 填充初始缓存."""
        try:
            replies = await asyncio.to_thread(
                self._session.liveliness().get,
                self._cell_liveness_wildcard,
            )
        except Exception:
            self._logger.exception("initial cell liveness query failed")
            return

        for reply in replies:
            if not reply.ok:
                continue
            address = self._key_to_address(str(reply.result.key_expr))
            if address is None:
                continue
            with self._cache_lock:
                if address in self._cache:
                    continue
            cell = await self._fetch_cell(address)
            if cell is None:
                continue
            with self._cache_lock:
                self._cache.setdefault(address, cell)

        self._logger.debug("cache seeded: %d cells", len(self._cache))

    async def _reconcile_loop(self) -> None:
        """低频全量对账 — 兜底 subscriber 极端丢事件."""
        while not self._closed:
            try:
                await asyncio.sleep(self._reconcile_interval)
                if self._closed:
                    return

                replies = await asyncio.to_thread(
                    self._session.liveliness().get,
                    self._cell_liveness_wildcard,
                )
                live_addresses: set[CellAddress] = set()
                for reply in replies:
                    if not reply.ok:
                        continue
                    address = self._key_to_address(str(reply.result.key_expr))
                    if address is not None:
                        live_addresses.add(address)

                with self._cache_lock:
                    cached = set(self._cache.keys())
                    added = live_addresses - cached
                    removed = cached - live_addresses

                for address in added:
                    cell = await self._fetch_cell(address)
                    if cell is None:
                        continue
                    with self._cache_lock:
                        self._cache[address] = cell
                    self._fire_on_change(cell, True)

                for address in removed:
                    with self._cache_lock:
                        cell = self._cache.pop(address, None)
                    if cell is not None:
                        self._fire_on_change(cell, False)

                if added or removed:
                    self._logger.debug(
                        "reconcile: +%d -%d cells (total=%d)",
                        len(added), len(removed), len(live_addresses),
                    )

            except asyncio.CancelledError:
                return
            except Exception:
                self._logger.exception("reconcile loop error")

    # ==================================================================
    # internal: callbacks
    # ==================================================================

    def _fire_on_change(self, cell: Cell, online: bool) -> None:
        for cb in self._change_callbacks:
            try:
                cb(cell, online)
            except Exception:
                self._logger.exception(
                    "on_change callback error for %s online=%s",
                    cell.address, online,
                )


def _sync_fetch_cell(session: zenoh.Session, key: str) -> Cell | None:
    """同步拉取 cell queryable 数据 — 供 zenoh 回调线程使用."""
    for reply in session.get(key):
        if reply.ok:
            return Cell.model_validate_json(reply.ok.payload.to_bytes())
    return None
