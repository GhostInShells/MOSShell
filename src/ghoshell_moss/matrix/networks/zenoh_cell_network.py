"""Zenoh-based CellNetwork — 推拉结合的 cell 发现与宣告.

每个 cell 在网络上的存在 = 1 个 key, 2 种 zenoh 原语:
  {cells_ns}/{cell.address}  → liveness token (推: 上线/下线, subscriber 实时感知)
  {cells_ns}/{cell.address}  → queryable   (拉: Cell JSON 全量, 按需 get)

Host 额外跨域宣告:
  {hosts_ns}/{scope}/cells/liveness/{cell.address} → liveness token

发现:
  subscriber 实时推 — PUT=上线 DELETE=下线, 零轮询
  reconcile loop 低频兜底 — 60s+ 全量对账, 覆盖极端丢事件

cell 不变 (meta/launcher 静态, status 只有 starting→alive→stopped),
所以不需要 change pub. 上下线由 liveness 推, 数据由 queryable 拉.
"""

import asyncio
import threading
from dataclasses import dataclass
from typing import Callable

from ghoshell_moss.depends import depend_zenoh

depend_zenoh()
import zenoh

from ghoshell_moss.core.blueprint.cell import (
    Cell,
    CellAddress,
    CellNetwork,
    DuplicatedError,
)
from ghoshell_moss.core.concepts.channel import Channel, ChannelProxy, ChannelProvider
from ghoshell_moss.tools.zenoh_helper import MatrixNamespace
from ghoshell_moss.bridges.zenoh_bridge import ZenohChannelHub

import logging

__all__ = ['ZenohCellNetwork']


@dataclass
class _Announce:
    """一次 cell announce 持有的 zenoh 资源."""
    liveness_token: zenoh.LivelinessToken
    queryable: zenoh.Queryable
    cell: Cell

    def close(self) -> None:
        try:
            self.queryable.undeclare()
        except RuntimeError:
            pass
        try:
            self.liveness_token.undeclare()
        except RuntimeError:
            pass


class ZenohCellNetwork(CellNetwork):
    """基于 zenoh 的 CellNetwork 实现.

    不持有 cell 引用, 不假设 host/worker 身份.
    推拉结合: liveness token 推上下线, queryable 拉全量数据.
    subscriber 实时感知, reconcile loop 低频兜底.
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
    ):
        self._session = session
        self._logger = logger
        self._ns = namespace
        self._scope = scope
        self._allow_create_proxy = allow_create_proxy
        self._reconcile_interval = reconcile_interval

        # -- zenoh key 约定, 全部在 init 生成 ---------------------------
        # 跨 scope queryable 回查根前缀
        self._scopes_root = "MOSS/matrix/scopes"
        # prefix strip 用
        self._cells_ns_prefix = self._ns.cells_ns + '/'
        self._hosts_ns_prefix = self._ns.hosts_ns + '/'
        # host 跨域 key 中的语义分隔标记: {scope}/cells/liveness/{address}
        self._host_liveness_marker = "/cells/liveness/"
        # wildcard
        self._cell_liveness_wildcard = self._join(self._ns.cells_ns, "**")
        self._host_discovery_wildcard = self._join(self._ns.cells_ns, "host", "**")
        self._cross_hosts_wildcard = self._join(
            self._ns.hosts_ns, "*", "cells", "liveness", "**",
        )

        # -- channel 层委托 hub ------------------------------------------
        self._hub = ZenohChannelHub(
            zenoh_session=session,
            scope=scope,
            logger=logger,
            namespace=namespace,
        )

        # -- 自身宣告 ----------------------------------------------------
        self._announcements: dict[CellAddress, _Announce] = {}
        self._host_tokens: dict[CellAddress, zenoh.LivelinessToken] = {}

        # -- 发现缓存 — 写者: subscriber (zenoh 线程) + reconcile (event loop 线程)
        self._cache: dict[CellAddress, Cell] = {}
        self._lock = threading.Lock()

        # change callbacks — 调用线程不固定, caller 负责线程安全
        self._change_callbacks: list[Callable[[Cell, bool], None]] = []

        # subscriber — 推: 实时感知上下线
        self._subscriber: zenoh.Subscriber | None = None

        # reconcile — 低频兜底
        self._reconcile_task: asyncio.Task | None = None

        self._started = False
        self._closed = False

    # -- properties ----------------------------------------------------

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

    def _host_liveness_key(self, scope: str, address: CellAddress) -> str:
        """host 跨域 liveness key: {hosts_ns}/{scope}/cells/liveness/{address}"""
        return self._join(self._ns.hosts_ns, scope, "cells", "liveness", address)

    def _remote_cell_key(self, scope: str, address: CellAddress) -> str:
        """跨 scope 回查 cell queryable key: scopes/{scope}/cells/{address}"""
        return self._join(self._scopes_root, scope, "cells", address)

    def _key_to_address(self, key: str) -> CellAddress | None:
        """从 {cells_ns}/{address} liveness key 提取 cell address.

        用 prefix strip 而非段数索引 — address 自身含 / 不受影响.
        """
        if not key.startswith(self._cells_ns_prefix):
            return None
        return key[len(self._cells_ns_prefix):]

    def _parse_host_liveness_key(self, key: str) -> tuple[str, CellAddress] | None:
        """从 {hosts_ns}/{scope}/cells/liveness/{address} 解析 (scope, address).

        用 marker 匹配 — address 自身含 /, 段数不确定.
        """
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

    async def update_cell(self, cell: Cell) -> None:
        address = cell.address

        if address in self._announcements:
            # 同 address 重复宣告: 仅更新 cell 引用, queryable handler 下次查询返回新数据
            self._announcements[address].cell = cell
            return

        key = self._cell_key(address)

        # 首次宣告前检查重复 — host 一个 scope 只允许一个, worker 检查 exact key
        if cell.is_host:
            dup_check_key = self._host_discovery_wildcard
        else:
            dup_check_key = key

        existing = await asyncio.to_thread(
            self._session.liveliness().get, dup_check_key,
        )
        for reply in existing:
            if reply.ok:
                raise DuplicatedError(
                    f"cell address '{address}' is already announced on the network"
                )

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

        # liveness token — 推: 上线/下线由 subscriber 实时感知
        liveness_token = self._session.liveliness().declare_token(key)

        self._announcements[address] = _Announce(
            liveness_token=liveness_token,
            queryable=queryable,
            cell=cell,
        )

        if cell.is_host:
            host_key = self._host_liveness_key(self._scope, address)
            self._host_tokens[address] = self._session.liveliness().declare_token(host_key)

        self._logger.debug("cell announced: address=%s host=%s", address, cell.is_host)

    async def revoke_cell(self, cell: Cell) -> None:
        address = cell.address

        ann = self._announcements.pop(address, None)
        if ann is None:
            raise LookupError(
                f"cell address '{address}' was not announced by this instance"
            )
        ann.close()

        host_token = self._host_tokens.pop(address, None)
        if host_token is not None:
            try:
                host_token.undeclare()
            except RuntimeError:
                pass

        self._logger.debug("cell revoked: address=%s", address)

    # ==================================================================
    # discovery
    # ==================================================================

    async def get_host(self) -> Cell | None:
        """liveness get scope 内 host 节点, 返回第一个."""
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
        """跨 scope 发现所有 host.

        liveness get {hosts_ns}/*/cells/liveness/** → key 解析得 scope + address
        → 回查 scopes/{scope}/cells/{address} queryable 拿完整 Cell.
        """
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
            refresh: bool = False,
    ) -> dict[CellAddress, Cell]:
        if not refresh:
            with self._lock:
                cells = dict(self._cache)
            if type is not None:
                cells = {a: c for a, c in cells.items() if c.type == type}
            return cells

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
            if type is not None and not address.startswith(f"{type}/"):
                continue
            cell = await self._fetch_cell(address)
            if cell is not None:
                cells[address] = cell

        with self._lock:
            self._cache.update(cells)
        self._logger.debug("live cells refreshed: %d cells", len(cells))
        return cells

    def live_cells(self) -> dict[CellAddress, Cell]:
        with self._lock:
            return dict(self._cache)

    def on_change(self, callback: Callable[[Cell, bool], None]) -> None:
        """注册 cell 上下线回调.

        回调在 zenoh subscriber 线程 (上线) 或 reconcile loop (下线对账) 触发.
        caller 负责线程安全.
        """
        self._change_callbacks.append(callback)

    # ==================================================================
    # channel
    # ==================================================================

    def create_provider(
            self,
            address: CellAddress,
            channel: Channel,
    ) -> ChannelProvider:
        return self._hub.provider(address)

    def create_proxy(
            self,
            address: CellAddress,
            *,
            name: str = '',
            description: str = '',
    ) -> ChannelProxy:
        if not self._allow_create_proxy:
            raise RuntimeError(
                "create_proxy is disabled on this CellNetwork. "
                "Set allow_create_proxy=True for host nodes that manage channel proxies."
            )
        return self._hub.proxy(address, name=name, description=description)

    # ==================================================================
    # lifecycle
    # ==================================================================

    async def __aenter__(self):
        if self._started:
            return self
        self._started = True

        await self._hub.__aenter__()

        # 先订阅 — 避免 declare 和 get 之间的 TOCTOU 丢事件
        self._subscriber = self._session.liveliness().declare_subscriber(
            self._cell_liveness_wildcard,
            self._on_liveness_sample,
        )

        # 再初始全量 seed — subscriber 已就位, 重复事件无害 (缓存写入幂等)
        await self._seed_cache()

        # 低频 reconcile — 兜底极端丢事件, 不替代 subscriber
        loop = asyncio.get_running_loop()
        self._reconcile_task = loop.create_task(self._reconcile_loop())

        self._logger.debug(
            "ZenohCellNetwork started: scope=%s cells_ns=%s",
            self._scope, self._ns.cells_ns,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._closed = True

        if self._reconcile_task is not None:
            self._reconcile_task.cancel()
            try:
                await self._reconcile_task
            except asyncio.CancelledError:
                pass
            self._reconcile_task = None

        if self._subscriber is not None:
            try:
                self._subscriber.undeclare()
            except RuntimeError:
                pass
            self._subscriber = None

        for ann in list(self._announcements.values()):
            ann.close()
        self._announcements.clear()

        for host_token in list(self._host_tokens.values()):
            try:
                host_token.undeclare()
            except RuntimeError:
                pass
        self._host_tokens.clear()

        await self._hub.__aexit__(exc_type, exc_val, exc_tb)

        with self._lock:
            self._cache.clear()

        self._started = False
        self._logger.debug("ZenohCellNetwork stopped: scope=%s", self._scope)

    # ==================================================================
    # internal: data fetch
    # ==================================================================

    async def _fetch_cell(self, address: CellAddress) -> Cell | None:
        return await self._fetch_cell_via(self._cell_key(address))

    async def _fetch_cell_via(self, info_key: str) -> Cell | None:
        """queryable get 拉取单个 cell 的完整数据."""
        try:
            replies = await asyncio.to_thread(self._session.get, info_key)
            for reply in replies:
                if reply.ok:
                    return Cell.model_validate_json(reply.ok.payload.to_bytes())
        except Exception:
            self._logger.debug("failed to fetch cell info via %s", info_key)
        return None

    # ==================================================================
    # internal: subscriber
    # ==================================================================

    def _on_liveness_sample(self, sample: zenoh.Sample) -> None:
        """zenoh liveness subscriber 回调 — 在 zenoh 后台线程执行.

        不持锁做 I/O (_sync_fetch_cell 是同步阻塞的 session.get).
        锁仅用于缓存写入的最小临界区.
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
            with self._lock:
                self._cache[address] = cell
            self._fire_on_change(cell, True)

        elif sample.kind == zenoh.SampleKind.DELETE:
            with self._lock:
                cell = self._cache.pop(address, None)
            if cell is not None:
                self._fire_on_change(cell, False)

    # ==================================================================
    # internal: reconcile
    # ==================================================================

    async def _seed_cache(self) -> None:
        """启动时全量 liveness get, 填充初始缓存.

        subscriber 已先于本调用就位, 期间到达的事件不会丢失 —
        subscriber 写入的缓存条目会被 seed 的 update 覆盖, 幂等.
        """
        try:
            replies = await asyncio.to_thread(
                self._session.liveliness().get,
                self._cell_liveness_wildcard,
            )
        except Exception:
            self._logger.exception("initial cell liveness query failed")
            return

        with self._lock:
            for reply in replies:
                if not reply.ok:
                    continue
                address = self._key_to_address(str(reply.result.key_expr))
                if address is None or address in self._cache:
                    continue
                cell = await self._fetch_cell(address)
                if cell is not None:
                    self._cache[address] = cell

        self._logger.debug("cache seeded: %d cells", len(self._cache))

    async def _reconcile_loop(self) -> None:
        """低频全量对账 — 兜底 subscriber 极端丢事件.

        对比 liveness 全量结果与本地缓存 diff → fire on_change.
        subscriber 是主路径, 此 loop 是备用, 间隔长 (默认 60s).
        """
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

                with self._lock:
                    cached = set(self._cache.keys())
                    added = live_addresses - cached
                    removed = cached - live_addresses

                for address in added:
                    cell = await self._fetch_cell(address)
                    if cell is None:
                        continue
                    with self._lock:
                        self._cache[address] = cell
                    self._fire_on_change(cell, True)

                for address in removed:
                    with self._lock:
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
