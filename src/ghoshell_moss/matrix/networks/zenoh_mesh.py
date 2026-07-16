"""
ZenohCellMesh — 对网络的观察侧 + accept/reject 实现 (CellMesh ABC).

一个 Mesh 实例治理 (key 表见 _utils.py):
  liveness subscriber (cell_liveness_wildcard)  → PUT/DELETE 更新 cache 边缘
  event subscriber   (events_wildcard)           → CellEvent 到达 → refetch + on_event
  queryable get       (按需拉 Cell 快照)
  reconcile loop      (低频对账兜底)
  hub.proxy(address)  (accept: 本地 dict 查重 + hub 建 duplex proxy)

opt-in: 每 runtime 至多一个 (§UU-7 shared informer 同构). 纯 worker cell 不需要.

延迟视图承诺 (§NN):
  只有 online/offline 边缘是实时的 (liveness 推送); cell 内容可能滞后到
  下次 refresh. 要实时内容就 refresh(address). 消费者面对这一延迟做视图/信号
  的二次消费, 不当强一致源.
"""
# -- §UU-7 拆分: subscriber + cache + reconcile 归这里 (O(N) 主动).
#    "我看不见 X" → 审讯本对象.
# -- §UU-8: accept/deny 归 Mesh. accept 即建 proxy (owner = 本 Mesh 持有者),
#    reject 即调 hub.drop_proxy. auto-build-proxy 已删 (那等于自动 accept 全网络).
# -- 蝴蝶横 8 字里 Mesh 是左翼数据源:
#      liveness 边缘 + CellEvent → on_updated 服务视图 / on_event 服务 nucleus.
#    nucleus 把 event 加工成 Signal 送 mindflow 争夺 Attention.
#    Mesh 只做分发, 不 signal 化 (职责单一, TT-1 融合检验).

import asyncio
import threading
from collections import deque
from typing import Callable

import janus
import zenoh
from typing_extensions import Self

from ghoshell_moss.bridges.zenoh_bridge import ZenohChannelHub
from ghoshell_moss.core.blueprint.cell import (
    CellAddress,
    CellEvent,
    Cell,
    CellMesh,
    normalize,
)
from ghoshell_moss.core.concepts.channel import ChannelProxy
from ghoshell_moss.matrix.networks._utils import CellsKeyspace

import logging

__all__ = ['ZenohCellMesh']

# 大数, 事件是低频, 满了说明消费严重落后, 报 error 暴露问题.
_QUEUE_MAXSIZE = 10000


class ZenohCellMesh(CellMesh):
    """
    基于 zenoh 的观察侧实现.

    线程模型:
      zenoh subscriber 回调线程: liveness sample / event sample.
        只做最薄工作 (parse + janus put_nowait), 处理卸载到 event loop.
      event loop 线程: 主流程 + consumer task + reconcile loop.
      跨线程共享: self._cache 用 threading.Lock 保护.
    """

    def __init__(
            self,
            *,
            session: zenoh.Session,
            logger: logging.Logger,
            keyspace: CellsKeyspace,
            scope: str,
            hub: ZenohChannelHub,
            self_project_id: str | None = None,
            reconcile_interval: float = 60.0,
            event_buffer_size: int = 100,
    ):
        self._session = session
        self._logger = logger
        self._keyspace = keyspace
        self._scope = scope
        self._hub = hub
        self._self_project_id = self_project_id
        self._reconcile_interval = reconcile_interval

        # -- 缓存 & 回调 -------------------------------------------------
        self._cache: dict[CellAddress, Cell] = {}
        self._cache_lock = threading.Lock()
        self._change_callbacks: list[Callable[[Cell, bool], None]] = []
        self._event_callbacks: list[Callable[[CellEvent], None]] = []
        # accepted proxy 本地表 — UU-8 零网络往返查重.
        self._accepted: dict[CellAddress, ChannelProxy] = {}
        self._event_buffer: deque[CellEvent] = deque(maxlen=event_buffer_size)

        # -- 内部队列 & task ---------------------------------------------
        self._event_queue: janus.Queue | None = None
        self._liveness_queue: janus.Queue | None = None
        self._event_consumer_task: asyncio.Task | None = None
        self._liveness_consumer_task: asyncio.Task | None = None
        self._reconcile_task: asyncio.Task | None = None

        self._cell_subscriber: zenoh.Subscriber | None = None
        self._event_subscriber: zenoh.Subscriber | None = None

        self._loop: asyncio.AbstractEventLoop | None = None
        self._started = False
        self._closed = False

    # ── 视图 ──────────────────────────────────────────────────────────

    def view(
            self,
            *,
            project_id: str | None = None,
    ) -> dict[CellAddress, Cell]:
        with self._cache_lock:
            snapshot = dict(self._cache)
        if project_id is not None:
            snapshot = {a: p for a, p in snapshot.items() if p.project_id == project_id}
        return snapshot

    async def refresh(
            self,
            address: CellAddress | None = None,
    ) -> dict[CellAddress, Cell]:
        # None → 全量 liveness get + per-cell queryable → 覆盖 cache.
        # 具体地址 → 单个 queryable get → 更新 cache 对应条目.
        if address is not None:
            presence = await self._fetch_presence(address)
            if presence is None:
                # 拉不到就当离线, 从 cache 移除.
                with self._cache_lock:
                    self._cache.pop(address, None)
                return {}
            with self._cache_lock:
                self._cache[address] = presence
            return {address: presence}

        result: dict[CellAddress, Cell] = {}
        try:
            replies = await asyncio.to_thread(
                self._session.liveliness().get,
                self._keyspace.cell_liveness_wildcard,
            )
        except Exception:
            self._logger.exception("refresh full liveness query failed")
            return {}
        for reply in replies:
            if not reply.ok:
                continue
            addr = self._keyspace.address_from_cell_key(str(reply.result.key_expr))
            if addr is None:
                continue
            presence = await self._fetch_presence(addr)
            if presence is not None:
                result[addr] = presence
        with self._cache_lock:
            self._cache = result.copy()
        return result

    # ── 回调注册 (返回 unsubscribe) ────────────────────────────────────

    def on_updated(
            self,
            callback: Callable[[Cell, bool], None],
    ) -> Callable[[], None]:
        self._change_callbacks.append(callback)

        def _unsub() -> None:
            try:
                self._change_callbacks.remove(callback)
            except ValueError:
                pass

        return _unsub

    def on_event(
            self,
            callback: Callable[[CellEvent], None],
    ) -> Callable[[], None]:
        self._event_callbacks.append(callback)

        def _unsub() -> None:
            try:
                self._event_callbacks.remove(callback)
            except ValueError:
                pass

        return _unsub

    def recent_events(self, *, limit: int = 20) -> list[CellEvent]:
        # deque 单写者 (consumer task), 读快照免锁靠 CPython GIL 保护 list(deque).
        snapshot = list(self._event_buffer)
        snapshot.reverse()
        return snapshot[:limit]

    async def wait_present(
            self,
            address: CellAddress,
            *,
            timeout: float = 30,
    ) -> Cell | None:
        # 已在 cache 即返回; 否则挂 on_updated 等待.
        # 缓存里出现 = liveness token 在网络上 = "present" (§NN 边缘实时).
        with self._cache_lock:
            existing = self._cache.get(address)
        if existing is not None:
            return existing

        loop = asyncio.get_running_loop()
        fut: asyncio.Future[Cell | None] = loop.create_future()

        def _on_change(presence: Cell, online: bool) -> None:
            if presence.address != address:
                return
            if not online:
                return
            if fut.done():
                return
            loop.call_soon_threadsafe(
                lambda: fut.done() or fut.set_result(presence),
            )

        unsub = self.on_updated(_on_change)
        try:
            # double-check 防 race.
            with self._cache_lock:
                existing = self._cache.get(address)
            if existing is not None:
                return existing
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            return None
        finally:
            unsub()

    # ── accept / reject / accepted ────────────────────────────────────

    async def accept(self, address: CellAddress) -> ChannelProxy:
        # UU-8: accept = 本地 dict 查重 → hub.proxy(address, name=...) → track.
        # owner = 本 Mesh, reject 或 __aexit__ 时清理.
        existing = self._accepted.get(address)
        if existing is not None:
            return existing

        # proxy name 派生规则:
        #   优先 cell.fullname (category_name 归一, 稳定跨启动),
        #   缺则 normalize(address) 兜底. 命名冲突交给上层.
        with self._cache_lock:
            presence = self._cache.get(address)
        if presence is None:
            raise LookupError(
                f"cell {address} not in network view (refresh first if needed)"
            )
        name = normalize(presence.fullname) if presence.fullname else normalize(address)

        proxy = self._hub.proxy(address, name=name)
        self._accepted[address] = proxy
        self._logger.debug("accepted proxy: address=%s name=%s", address, name)
        return proxy

    async def reject(self, address: CellAddress) -> None:
        # 幂等. hub.drop_proxy 内部关闭 zenoh 连接.
        proxy = self._accepted.pop(address, None)
        if proxy is None:
            return
        try:
            self._hub.drop_proxy(address)
        except Exception:
            self._logger.exception("hub.drop_proxy failed for %s", address)
        self._logger.debug("released proxy: address=%s", address)

    def channel_proxies(self) -> dict[CellAddress, ChannelProxy]:
        return dict(self._accepted)

    # ── 生命周期 ──────────────────────────────────────────────────────

    async def __aenter__(self) -> Self:
        if self._started:
            return self
        self._started = True
        self._loop = asyncio.get_running_loop()
        self._event_queue = janus.Queue(maxsize=_QUEUE_MAXSIZE)
        self._liveness_queue = janus.Queue(maxsize=_QUEUE_MAXSIZE)

        # 先订阅, 再 seed, 避免 declare 和 get 之间丢事件.
        self._cell_subscriber = self._session.liveliness().declare_subscriber(
            self._keyspace.cell_liveness_wildcard,
            self._on_liveness_sample,
        )
        self._event_subscriber = self._session.declare_subscriber(
            self._keyspace.events_wildcard,
            self._on_event_sample,
        )

        self._event_consumer_task = self._loop.create_task(
            self._event_consumer_loop(),
        )
        self._liveness_consumer_task = self._loop.create_task(
            self._liveness_consumer_loop(),
        )

        # 全量 seed — subscriber 已就位, 重复事件无害.
        await self.refresh()

        self._reconcile_task = self._loop.create_task(self._reconcile_loop())

        self._logger.debug(
            "ZenohCellMesh started: scope=%s cells_ns=%s",
            self._scope, self._keyspace.cells_ns,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self._closed = True

        # 停 task 循环.
        for task in (
                self._reconcile_task,
                self._event_consumer_task,
                self._liveness_consumer_task,
        ):
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
        self._event_consumer_task = None
        self._liveness_consumer_task = None

        # 关 subscriber.
        for sub in (self._cell_subscriber, self._event_subscriber):
            if sub is None:
                continue
            try:
                sub.undeclare()
            except RuntimeError:
                pass
        self._cell_subscriber = None
        self._event_subscriber = None

        # 关 janus queues.
        for q in (self._event_queue, self._liveness_queue):
            if q is not None:
                q.shutdown(immediate=True)
        self._event_queue = None
        self._liveness_queue = None

        # 释放所有 accept 的 proxy — owner 关闭即释放 (UU-8).
        for addr in list(self._accepted.keys()):
            await self.reject(addr)

        with self._cache_lock:
            self._cache.clear()

        self._started = False
        self._loop = None
        self._logger.debug("ZenohCellMesh stopped: scope=%s", self._scope)

    # ── zenoh 后台线程回调 ────────────────────────────────────────────

    def _on_liveness_sample(self, sample: zenoh.Sample) -> None:
        # zenoh 后台线程. 只做最薄工作: 入队, 卸载到 consumer.
        address = self._keyspace.address_from_cell_key(str(sample.key_expr))
        if address is None:
            return
        if self._liveness_queue is None:
            return
        try:
            self._liveness_queue.sync_q.put_nowait((address, sample.kind))
        except janus.SyncQueueFull:
            self._logger.error(
                "liveness queue full, dropping: address=%s kind=%s",
                address, sample.kind,
            )
        except janus.SyncQueueShutDown:
            pass

    def _on_event_sample(self, sample: zenoh.Sample) -> None:
        # zenoh 后台线程. parse + 入队.
        if sample.kind != zenoh.SampleKind.PUT:
            return
        try:
            event = CellEvent.model_validate_json(sample.payload.to_bytes())
        except Exception:
            self._logger.exception(
                "failed to parse CellEvent from sample, key=%s", sample.key_expr,
            )
            return
        if self._event_queue is None:
            return
        try:
            self._event_queue.sync_q.put_nowait(event)
        except janus.SyncQueueFull:
            self._logger.error(
                "event queue full, dropping: address=%s content=%s",
                event.address, event.content,
            )
        except janus.SyncQueueShutDown:
            pass

    # ── consumer loops ────────────────────────────────────────────────

    async def _liveness_consumer_loop(self) -> None:
        while not self._closed:
            try:
                if self._liveness_queue is None:
                    return
                address, kind = await self._liveness_queue.async_q.get()
            except janus.AsyncQueueShutDown:
                return
            except asyncio.CancelledError:
                return
            try:
                await self._process_liveness(address, kind)
            except asyncio.CancelledError:
                raise
            except Exception:
                self._logger.exception(
                    "liveness consumer failed: address=%s", address,
                )

    async def _process_liveness(self, address: CellAddress, kind) -> None:
        if kind == zenoh.SampleKind.PUT:
            presence = await self._fetch_presence(address)
            if presence is None:
                # 存活但拉不到 payload — 视为"存在但状态未知", 不更新 cache 也不 fire.
                return
            with self._cache_lock:
                self._cache[address] = presence
            self._fire_on_change(presence, True)
        elif kind == zenoh.SampleKind.DELETE:
            with self._cache_lock:
                presence = self._cache.pop(address, None)
            if presence is not None:
                # 反映"下线"事实, 传给回调时 state 保留最后一次记录 (仅供参考).
                self._fire_on_change(presence, False)

    async def _event_consumer_loop(self) -> None:
        while not self._closed:
            try:
                if self._event_queue is None:
                    return
                event: CellEvent = await self._event_queue.async_q.get()
            except janus.AsyncQueueShutDown:
                return
            except asyncio.CancelledError:
                return
            try:
                await self._process_event(event)
            except asyncio.CancelledError:
                raise
            except Exception:
                self._logger.exception(
                    "event consumer failed: address=%s content=%s",
                    event.address, event.content,
                )

    async def _process_event(self, event: CellEvent) -> None:
        # deque 单写者 (本 task), 无锁.
        self._event_buffer.append(event)

        # refetch=True → 拉最新 presence 更新 cache (推拉结合的拉).
        if event.refetch:
            try:
                presence = await self._fetch_presence(event.address)
            except Exception:
                self._logger.debug(
                    "refetch on event failed: address=%s", event.address,
                )
                presence = None
            if presence is not None:
                with self._cache_lock:
                    self._cache[event.address] = presence
                # cache 内容变化了 (即使 online 状态不变), fire on_change
                # 供视图消费者刷视图.
                self._fire_on_change(presence, True)

        # 无论 refetch 与否, 事件本身都要分发给 on_event 订阅者
        # (nucleus / signal 消费者).
        self._fire_on_event(event)

    # ── reconcile ─────────────────────────────────────────────────────

    async def _reconcile_loop(self) -> None:
        while not self._closed:
            try:
                await asyncio.sleep(self._reconcile_interval)
                if self._closed:
                    return
                try:
                    replies = await asyncio.to_thread(
                        self._session.liveliness().get,
                        self._keyspace.cell_liveness_wildcard,
                    )
                except Exception:
                    self._logger.exception("reconcile liveness query failed")
                    continue

                live_addresses: set[CellAddress] = set()
                for reply in replies:
                    if not reply.ok:
                        continue
                    addr = self._keyspace.address_from_cell_key(str(reply.result.key_expr))
                    if addr is not None:
                        live_addresses.add(addr)

                with self._cache_lock:
                    cached = set(self._cache.keys())
                    added = live_addresses - cached
                    removed = cached - live_addresses

                for addr in added:
                    presence = await self._fetch_presence(addr)
                    if presence is None:
                        continue
                    with self._cache_lock:
                        self._cache[addr] = presence
                    self._fire_on_change(presence, True)

                for addr in removed:
                    with self._cache_lock:
                        presence = self._cache.pop(addr, None)
                    if presence is not None:
                        self._fire_on_change(presence, False)

                if added or removed:
                    self._logger.debug(
                        "reconcile: +%d -%d cells (total=%d)",
                        len(added), len(removed), len(live_addresses),
                    )
            except asyncio.CancelledError:
                return
            except Exception:
                self._logger.exception("reconcile loop error")

    # ── 数据拉取 ──────────────────────────────────────────────────────

    async def _fetch_presence(self, address: CellAddress) -> Cell | None:
        key = self._keyspace.cell_key(address)
        try:
            replies = await asyncio.to_thread(self._session.get, key)
            for reply in replies:
                if reply.ok:
                    return Cell.model_validate_json(
                        reply.ok.payload.to_bytes(),
                    )
        except Exception:
            self._logger.debug("fetch presence failed via %s", key)
        return None

    # ── 回调分发 ──────────────────────────────────────────────────────

    def _fire_on_change(self, presence: Cell, online: bool) -> None:
        # 在 event loop 线程内 fire — 回调者不需要跨线程防御.
        # 快照回调列表, 避免回调内 add/remove 导致遍历破坏.
        for cb in list(self._change_callbacks):
            try:
                cb(presence, online)
            except Exception:
                self._logger.exception(
                    "on_change callback error for %s online=%s",
                    presence.address, online,
                )

    def _fire_on_event(self, event: CellEvent) -> None:
        for cb in list(self._event_callbacks):
            try:
                cb(event)
            except Exception:
                self._logger.exception(
                    "on_event callback error for %s content=%s",
                    event.address, event.content,
                )
