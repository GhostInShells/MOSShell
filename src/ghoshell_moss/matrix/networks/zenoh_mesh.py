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
from ghoshell_moss.core.blueprint.environment import Environment
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
            env: Environment,
            auto_accept_local: bool = True,
            auto_accept_foreign: bool = False,
            reconcile_interval: float = 60.0,
            event_buffer_size: int = 100,
    ):
        self._session = session
        self._logger = logger
        self._keyspace = keyspace
        self._scope = scope
        self._hub = hub
        # env: 用于 cell.is_local(env) 判定 (§UU-7 local/foreign 分档来源).
        # 传 env 而非 project_id str, 与 blueprint Cell.is_local(env) 语义对齐.
        self._env = env
        self._reconcile_interval = reconcile_interval

        # -- 缓存 & 回调 -------------------------------------------------
        self._cache: dict[CellAddress, Cell] = {}
        self._cache_lock = threading.Lock()
        self._change_callbacks: list[Callable[[Cell, bool], None]] = []
        self._event_callbacks: list[Callable[[CellEvent], None]] = []
        # accept/reject 意图集 (§UU-8, 与在线状态正交).
        # 优先级 (在 _should_build_proxy 里判定): reject > accept > auto_accept.
        self._accept_set: set[CellAddress] = set()
        self._reject_set: set[CellAddress] = set()
        # auto_accept 默认策略 — 可通过 set_auto_accept toggle.
        # 默认: local True, foreign False — host 自然接纳本 project 的 cell,
        # 拒绝 foreign; foreign 场景需上层 (cells channel command) 显式开.
        self._auto_accept_local = auto_accept_local
        self._auto_accept_foreign = auto_accept_foreign
        # 已建 proxy 表 — 上下线的 owner 视角 dict (§UU-8 "从 dict 拿掉即下线").
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
                self._try_drop_proxy(address)
                return {}
            with self._cache_lock:
                self._cache[address] = presence
            # 单点 refresh 后按规则触发上下线 (auto_accept + accept 表都靠这个入口生效).
            if self._should_build_proxy(presence):
                self._try_build_proxy(presence)
            else:
                self._try_drop_proxy(address)
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
            # 差集: 之前 cache 有的 address 但这次全量拉不到 → 视为下线.
            gone = set(self._cache.keys()) - set(result.keys())
            self._cache = result.copy()
        # 全量 refresh 后走一遍上下线规则 (seed 阶段/reconcile 补漏时也走同一条路径).
        for addr in gone:
            self._try_drop_proxy(addr)
        for cell in result.values():
            if self._should_build_proxy(cell):
                self._try_build_proxy(cell)
            else:
                self._try_drop_proxy(cell.address)
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

    # ── accept / reject / set_auto_accept ─────────────────────────────

    def set_auto_accept(
            self,
            *,
            local: bool | None = None,
            foreign: bool | None = None,
    ) -> None:
        """
        切换默认 auto-accept 策略, 触发即扫.

        新策略立刻作用于 cache 中已知的每个 cell:
          - 应建 (通过 _should_build_proxy) 且未建 → 补建
          - 不应建但已建 → 撤销 (显式 accept 集里的不会被撤, reject 优先级最高)
        """
        if local is not None:
            self._auto_accept_local = local
        if foreign is not None:
            self._auto_accept_foreign = foreign
        # 触发即扫: 用 cache 快照过一遍规则.
        with self._cache_lock:
            cells = list(self._cache.values())
        for cell in cells:
            if self._should_build_proxy(cell):
                self._try_build_proxy(cell)
            else:
                self._try_drop_proxy(cell.address)

    async def accept(self, address: CellAddress, *, lookup: bool = False) -> None:
        # UU-8: accept 集 (承认表达, 与在线正交). 显式 accept 覆盖 reject.
        self._reject_set.discard(address)
        self._accept_set.add(address)

        with self._cache_lock:
            cached = self._cache.get(address)
        if cached is None and lookup:
            # lookup=True: 视图中无该 address 先 refresh 一次再判.
            await self.refresh(address)
            with self._cache_lock:
                cached = self._cache.get(address)
            if cached is None:
                raise LookupError(
                    f"cell {address} not present in network after refresh"
                )
        if cached is not None:
            self._try_build_proxy(cached)
        # lookup=False 且视图中无: 只加集, 等 present-later 触发.

    async def reject(self, address: CellAddress) -> None:
        # 幂等. reject 集覆盖 accept 集与 auto_accept 规则.
        self._accept_set.discard(address)
        self._reject_set.add(address)
        self._try_drop_proxy(address)

    def channel_proxies(self) -> dict[CellAddress, ChannelProxy]:
        return dict(self._accepted)

    def has_host(self) -> bool:
        """
        本 network 是否有 host 在 view 里. host 在 network 级唯一, 不需按
        project_id 过滤 (§YY: host = 抢到 zenoh listen 端口的那个).
        """
        with self._cache_lock:
            for cell in self._cache.values():
                if cell.is_host:
                    return True
        return False

    # ── 规则判定 & proxy 上下线原语 (event loop 线程内调用) ───────────

    def _should_build_proxy(self, cell: Cell) -> bool:
        """判断是否应为该 cell 建 proxy.

        优先级 (从高到低): reject > !has_channel > accept > auto_accept.
        """
        address = cell.address
        if address in self._reject_set:
            return False
        if 'channel' not in cell.providing:
            return False
        if address in self._accept_set:
            return True
        # auto_accept 分档: is_local(env) 决定看哪个开关.
        if cell.is_local(self._env):
            return self._auto_accept_local
        return self._auto_accept_foreign

    def _try_build_proxy(self, cell: Cell) -> None:
        """幂等: 应建且未建时走 hub.proxy 建 (name_hint=fullname)."""
        address = cell.address
        if address in self._accepted:
            return
        if not self._should_build_proxy(cell):
            return
        hint = cell.fullname or address.replace('/', '_')
        try:
            proxy = self._hub.proxy(address, name_hint=hint)
        except Exception:
            self._logger.exception(
                "hub.proxy failed: address=%s hint=%s", address, hint,
            )
            return
        self._accepted[address] = proxy
        self._logger.debug(
            "proxy built: address=%s hint=%s", address, hint,
        )

    def _try_drop_proxy(self, address: CellAddress) -> None:
        """幂等: 从 _accepted pop + hub.drop_proxy 释放 zenoh 资源."""
        proxy = self._accepted.pop(address, None)
        if proxy is None:
            return
        try:
            self._hub.drop_proxy(address)
        except Exception:
            self._logger.exception("hub.drop_proxy failed for %s", address)
        self._logger.debug("proxy dropped: address=%s", address)

    def recent_events(self, *, limit: int = 20) -> list[CellEvent]:
        # deque 单写者 (consumer task), 读快照免锁靠 CPython GIL 保护 list(deque).
        snapshot = list(self._event_buffer)
        snapshot.reverse()
        return snapshot[:limit]

    def cell_events(
            self,
            address: CellAddress,
            *,
            limit: int = 20,
    ) -> list[CellEvent]:
        """某个 cell 的最近事件 (按到达时间倒序). 从 event_buffer 按 address 过滤."""
        snapshot = list(self._event_buffer)
        filtered = [e for e in snapshot if e.address == address]
        filtered.reverse()
        return filtered[:limit]

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
        # 直接走 _try_drop_proxy (幂等 pop + hub.drop_proxy), 不用 reject
        # (reject 会往 _reject_set 加, 这里只是清资源).
        for addr in list(self._accepted.keys()):
            self._try_drop_proxy(addr)

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
            # 上线: 按规则决定是否建 proxy (幂等, 应建才建).
            self._try_build_proxy(presence)
        elif kind == zenoh.SampleKind.DELETE:
            with self._cache_lock:
                presence = self._cache.pop(address, None)
            # 下线: 从 dict 拿掉 proxy (owner 视角). hub 内部 _on_provider_offline
            # 会并行清 hub._proxies, 本处仍显式 drop 保幂等.
            self._try_drop_proxy(address)
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
                # 补触发: cell 添了 channel → 应建就建; 移了 channel → 应撤就撤.
                # 两个 helper 都幂等, 允许每次 refetch 都走一遍.
                if self._should_build_proxy(presence):
                    self._try_build_proxy(presence)
                else:
                    self._try_drop_proxy(event.address)

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
                    # reconcile 补 auto-accept (subscriber 漏包时的兜底路径).
                    self._try_build_proxy(presence)

                for addr in removed:
                    with self._cache_lock:
                        presence = self._cache.pop(addr, None)
                    # 下线 proxy 兜底.
                    self._try_drop_proxy(addr)
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
