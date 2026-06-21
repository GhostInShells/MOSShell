"""Zenoh-based CellNetwork — cell 发现、provider/proxy、detection loop.

两层分离:
  cell 层  — liveness token + queryable + detection loop  (本模块自己管)
  channel 层 — provider/proxy 工厂                          (委托 ZenohChannelHub)

announce_cell() 是 cell 宣告的入口: 创建 liveness token + queryable。
__aenter__ 走 announce_cell 宣告自身，__aexit__ 回收所有宣告。
"""

import asyncio
import threading
import time
from dataclasses import dataclass
from typing import Callable

from ghoshell_moss.depends import depend_zenoh

depend_zenoh()
import zenoh

from ghoshell_moss.core.blueprint.cell import (
    Cell,
    CellAddress,
    CellBridgeAddress,
    CellNetwork,
    split_bridge_address,
)
from ghoshell_moss.core.concepts.channel import Channel, ChannelProxy, ChannelProvider
from ghoshell_moss.contracts import LoggerItf, get_moss_logger
from ghoshell_moss.bridges.zenoh_bridge import ZenohChannelHub

__all__ = ['ZenohCellNetwork']



class CellKeyExpr:

    def __init__(self, scope: str):
        self.prefix  = f"MOSS/{scope}/network/cells"


# ── announce 记录 ───────────────────────────────────────────────────


@dataclass
class _CellAnnounce:
    """一次 cell announce 对应的 zenoh 资源."""

    liveness_token: zenoh.LivelinessToken
    queryable: zenoh.Queryable
    cell_factory: Callable[[], Cell]

    def close(self):
        try:
            self.queryable.undeclare()
        except RuntimeError:
            pass
        try:
            self.liveness_token.undeclare()
        except RuntimeError:
            pass


# ── ZenohCellNetwork ────────────────────────────────────────────────


class ZenohCellNetwork(CellNetwork):
    """基于 zenoh 的 CellNetwork 实现。

    announce_cell(factory) 宣告一个 cell 的存在 (liveness token + queryable)。
    __aenter__ 对自身调用 announce_cell，__aexit__ 回收所有宣告。
    host 节点额外运行 detection loop: 轮询 cell liveness → 查询 cell info → 更新缓存。
    """

    def __init__(
        self,
        this: Cell,
        *,
        zenoh_session: zenoh.Session,
        scope: str = 'default',
        detection_interval: float = 3.0,
        stale_timeout: float | None = None,
        logger: LoggerItf | None = None,
    ):
        self._this = this
        self._scope = scope
        self._session = zenoh_session
        self._logger = logger or get_moss_logger()
        self._detection_interval = detection_interval
        self._stale_timeout = stale_timeout or (detection_interval * 3)

        # -- channel 层委托给 hub --
        self._hub = ZenohChannelHub(
            zenoh_session=zenoh_session,
            scope=scope,
            logger=logger,
        )

        # -- detection loop state --
        self._detection_task: asyncio.Task | None = None
        self._closed = False

        # -- announces --
        self._announces: list[_CellAnnounce] = []

        # -- caches --
        self._cached_cells: dict[CellAddress, Cell] = {}
        self._cached_providers: set[CellBridgeAddress] = set()
        self._last_seen: dict[CellAddress, float] = {}
        self._lock = threading.Lock()

    # ── announce ────────────────────────────────────────────────

    def announce_cell(self, cell: Callable[[], Cell]) -> None:
        """宣告一个 cell 的存在: 创建 liveness token + cell info queryable。

        cell factory 会在 queryable 被查询时调用，返回当前 Cell 快照。
        """
        bridge = cell().bridge_address
        scope = self._scope

        liveness_key = _cell_liveness_key(scope, bridge)
        liveness_token = self._session.liveliness().declare_token(liveness_key)

        info_key = _cell_info_key(scope, bridge)

        def _on_query(query: zenoh.Query):
            try:
                c = cell()
                payload = c.model_dump_json().encode('utf-8')
                query.reply(info_key, payload)
            except Exception:
                self._logger.exception(
                    "cell info queryable error for %s", bridge,
                )

        queryable = self._session.declare_queryable(info_key, _on_query)

        self._announces.append(_CellAnnounce(
            liveness_token=liveness_token,
            queryable=queryable,
            cell_factory=cell,
        ))

        # 若 announce 时 detection loop 已在运行，立即写入缓存
        with self._lock:
            if self._detection_task is not None:
                c = cell()
                self._cached_cells[c.address] = c
                self._last_seen[c.address] = time.monotonic()

        self._logger.debug(
            "cell announced: address=%s bridge=%s", cell().address, bridge,
        )

    # ── CellNetwork ABC ─────────────────────────────────────────

    async def get_live_cells(self) -> dict[CellAddress, Cell]:
        result: dict[CellAddress, Cell] = {}
        online = await self.list_providers()
        for bridge in online:
            if bridge == self._this.bridge_address:
                result[self._this.address] = self._this
                continue
            try:
                address, _ = split_bridge_address(bridge)
            except ValueError:
                continue
            if address not in result:
                cell_data = await self._fetch_cell_info(bridge)
                if cell_data is not None:
                    result[address] = Cell.model_validate_json(cell_data)
        return result

    async def run_cell(
        self,
        cell: Cell,
        *,
        wait_alive: bool = False,
        timeout: float = None,
    ) -> asyncio.subprocess.Process:
        raise NotImplementedError(
            "start_cell is owned by Matrix.spawn, not CellNetwork"
        )

    # -- channel 层: 全部委托 hub --

    def provide(
        self,
        address: CellBridgeAddress,
        channel: Channel,
    ) -> ChannelProvider:
        return self._hub.provider(address)

    def create_proxy(
        self,
        address: CellBridgeAddress,
        name: str = '',
        description: str = '',
    ) -> ChannelProxy:
        return self._hub.proxy(address, name=name, description=description)

    def proxies(self) -> dict[CellBridgeAddress, ChannelProxy]:
        return self._hub.proxies

    # -- 发现 --

    async def list_providers(self) -> list[CellBridgeAddress]:
        """查询当前宣告 cell liveness 的 bridge addresses."""
        wildcard = f"{_cell_liveness_prefix(self._scope)}/**"
        prefix = _cell_liveness_prefix(self._scope) + '/'
        result: list[CellBridgeAddress] = []
        for sample in await asyncio.to_thread(
            self._session.liveliness().get, wildcard
        ):
            if not sample.ok:
                continue
            key = str(sample.result.key_expr)
            if key.startswith(prefix):
                result.append(key[len(prefix):])
        return result

    # -- 缓存 --

    def cached_cells(self) -> dict[CellAddress, Cell]:
        with self._lock:
            return dict(self._cached_cells)

    def cached_providers(self) -> list[CellBridgeAddress]:
        with self._lock:
            return list(self._cached_providers)

    # -- detection loop (host only) --

    async def start_detection_loop(self):
        if self._detection_task is not None:
            return
        with self._lock:
            self._cached_cells[self._this.address] = self._this
            self._last_seen[self._this.address] = time.monotonic()
        self._detection_task = asyncio.create_task(self._detection_loop())
        self._logger.debug(
            "detection loop started (interval=%.1fs, stale=%.1fs)",
            self._detection_interval, self._stale_timeout,
        )

    async def stop_detection_loop(self):
        if self._detection_task is not None:
            self._detection_task.cancel()
            try:
                await self._detection_task
            except asyncio.CancelledError:
                pass
            self._detection_task = None

    # ── lifecycle ───────────────────────────────────────────────

    async def __aenter__(self):
        self.announce_cell(lambda: self._this)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._closed = True
        await self.stop_detection_loop()

        for ann in self._announces:
            ann.close()
        self._announces.clear()

        with self._lock:
            self._cached_cells.clear()
            self._cached_providers.clear()
            self._last_seen.clear()

        self._logger.debug("ZenohCellNetwork exited: cell=%s", self._this.address)

    # ── internal ─────────────────────────────────────────────────

    async def _fetch_cell_info(self, bridge: CellBridgeAddress) -> bytes | None:
        info_key = _cell_info_key(self._scope, bridge)
        try:
            replies = await asyncio.to_thread(self._session.get, info_key)
            for reply in replies:
                if reply.ok:
                    return reply.result.payload.to_bytes()
        except Exception:
            self._logger.debug("failed to fetch cell info for %s", bridge)
        return None

    async def _detection_loop(self):
        while not self._closed:
            try:
                await asyncio.sleep(self._detection_interval)
                if self._closed:
                    return

                online = await self.list_providers()
                now = time.monotonic()
                seen: set[CellAddress] = set()

                for bridge in online:
                    if bridge == self._this.bridge_address:
                        seen.add(self._this.address)
                        continue
                    try:
                        address, _ = split_bridge_address(bridge)
                    except ValueError:
                        continue
                    seen.add(address)

                    with self._lock:
                        if address not in self._cached_cells:
                            cell_data = await self._fetch_cell_info(bridge)
                            if cell_data is not None:
                                self._cached_cells[address] = Cell.model_validate_json(
                                    cell_data
                                )
                        self._last_seen[address] = now

                # self always present
                with self._lock:
                    self._cached_cells[self._this.address] = self._this
                    self._last_seen[self._this.address] = now

                # reconcile stale
                with self._lock:
                    stale = [
                        addr
                        for addr, ts in self._last_seen.items()
                        if now - ts > self._stale_timeout
                    ]
                    for addr in stale:
                        self._cached_cells.pop(addr, None)
                        self._last_seen.pop(addr, None)

                with self._lock:
                    self._cached_providers = set(online)

            except asyncio.CancelledError:
                return
            except Exception:
                self._logger.exception("detection loop error")