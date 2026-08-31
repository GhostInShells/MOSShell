"""
ZenohOperator — ServiceOperator ABC 的 zenoh 实现.

每个 cell 一个实例 (通过 ``matrix.service_operator()`` 惰性获取).
管理本 cell 的 service terminal 集合 + 全网 service 发现.

发现管线:
  ZenohLivenessListener({services_ns}/**) → on_online/offline
  → parse kind + dotted_addr → on_service_start/stop callbacks
  get_services_by_kind → liveness cache filter + meta queryable per service

异步桥纪律 (2026-09-01 实测定型, 见 _bridge.py 模块 docstring):

- ``get()`` 全回调化: ``session.get(key, Callback(on_reply, on_drop))``,
  replies 在回调线程收集, drop 终结信号经 ``call_soon_threadsafe`` 完成
  loop 侧 Future — 零线程占用, 广播 N 个目标互不干扰。
- ``sub()`` 无独立线程: callback subscriber → 共享分发管线 →
  create_task-per-sample。
- ``emit()`` 的 ``session.put`` 经出站 worker 执行, loop 零阻塞。
- 广播目标解析只读 liveness cache, 不做 per-service meta round-trip。
"""

import asyncio
import json
import time
from typing import Callable, Awaitable

from ghoshell_moss.depends import depend_matrix

depend_matrix()

import zenoh
from typing_extensions import Self

from ghoshell_moss.core.blueprint.service import (
    ServiceOperator,
    ServiceDeclaration,
    ServiceProvider,
    ServiceMeta,
    Reply,
    Sample,
    Handle,
)
from ghoshell_moss.core.blueprint.cell import CellAddress
from ghoshell_moss.matrix.zenoh_helper import ZenohLivenessListener

from ._utils import ServiceKeyspace
from ._bridge import ZenohHandle, LoopDispatcher, OutboundWorker, invoke_handler
from .zenoh_service_terminal import (
    ZenohServiceTerminal,
    _encode_query_payload,
)

import logging

__all__ = ['ZenohOperator']

# Bound a single query's lifetime.  With callback-based get the timeout only
# bounds when zenoh fires the drop (completion) signal — no thread is held.
_QUERY_TIMEOUT = 5.0

# Safety margin for the loop-side wait: the zenoh drop signal should always
# arrive within _QUERY_TIMEOUT; the extra second guards against a lost drop.
_QUERY_WAIT_TIMEOUT = _QUERY_TIMEOUT + 1.0

_SUB_QUEUE_MAXSIZE = 1000
_LIVENESS_QUEUE_MAXSIZE = 1000


class ZenohOperator(ServiceOperator):
    """ServiceOperator 的 zenoh 实现 — per-cell 的 service 接线员."""

    def __init__(
            self,
            *,
            session: zenoh.Session,
            network_ns: str,
            this_address: CellAddress,
            logger: logging.Logger,
    ):
        self._session = session
        self._this_address = this_address
        self._logger = logger
        self._keyspace = ServiceKeyspace(network_ns)

        # per-kind terminals (本地 provide 的 service)
        self._terminals: dict[str, ZenohServiceTerminal] = {}

        # liveness-driven discovery
        self._liveness_listener = ZenohLivenessListener(
            liveness_prefix=self._keyspace.services_ns,
            session=session,
            logger=logger,
            on_online=self._on_service_online,
            on_offline=self._on_service_offline,
        )

        # kind → list[(callback, Handle)]
        self._start_callbacks: dict[str, list[tuple[Callable, ZenohHandle]]] = {}
        self._stop_callbacks: dict[str, list[tuple[Callable, ZenohHandle]]] = {}

        # subscriber handles for sub() — removed on close (no leak)
        self._sub_handles: list[ZenohHandle] = []

        # last known meta per live service — lets on_service_stop deliver a
        # K4-valid meta (from_meta must round-trip) after the service is gone
        self._meta_cache: dict[tuple[str, str], ServiceMeta] = {}

        # -- loop-side pipelines -------------------------------------------
        # sub(): (handler, key_expr, payload) → create_task-per-sample
        self._sub_dispatcher: LoopDispatcher[tuple[Callable, str, bytes]] = LoopDispatcher(
            "operator-sub", logger, maxsize=_SUB_QUEUE_MAXSIZE,
        )
        # liveness: (direction, dotted_addr, kind) → create_task-per-event
        self._liveness_dispatcher: LoopDispatcher[tuple[str, str, str]] = LoopDispatcher(
            "operator-liveness", logger, maxsize=_LIVENESS_QUEUE_MAXSIZE,
        )
        # emit(): sync zenoh puts off the loop
        self._outbound = OutboundWorker("operator", logger)

        self._loop: asyncio.AbstractEventLoop | None = None
        self._started = False

    def _require_started(self) -> None:
        if not self._started or self._loop is None:
            raise RuntimeError(
                "operator not running: use it between __aenter__ and __aexit__"
            )

    # -- ServiceOperator: provide ----------------------------------------

    async def provide(self, declaration: ServiceDeclaration) -> ServiceProvider:
        kind = declaration.kind()
        if kind in self._terminals:
            raise RuntimeError(
                f"service kind {kind!r} already provided by this cell"
            )
        keys = self._keyspace.per_service(self._this_address, kind)
        terminal = ZenohServiceTerminal(
            session=self._session,
            keys=keys,
            declaration=declaration,
            logger=self._logger,
        )
        try:
            await terminal.__aenter__()
        except Exception:
            # rollback: __aenter__ may have partially started (e.g.
            # liveness token declared but consumer task failed).
            # best-effort cleanup so zenoh handles don't leak.
            try:
                await terminal.__aexit__(None, None, None)
            except Exception:
                self._logger.exception(
                    "terminal rollback error: kind=%s", kind,
                )
            raise
        self._terminals[kind] = terminal
        self._logger.info(
            "service provided: address=%s kind=%s", self._this_address, kind,
        )
        return terminal

    # -- ServiceOperator: discovery --------------------------------------

    def _live_services(self) -> list[tuple[str, str]]:
        """Snapshot of live (dotted_addr, kind) pairs from the liveness cache."""
        result = []
        for identity in self._liveness_listener.live_keys:
            parsed = self._keyspace.parse_live_identity(identity)
            if parsed is not None:
                result.append(parsed)
        return result

    def _live_addresses_by_kind(self, kind: str) -> list[str]:
        """Resolve broadcast targets from the liveness cache alone —
        no meta round-trip needed for addressing."""
        return [
            dotted.replace('.', '/')
            for dotted, k in self._live_services()
            if k == kind
        ]

    async def get_services_by_kind(self, kind: str) -> list[ServiceMeta]:
        pairs = [(d, k) for d, k in self._live_services() if k == kind]
        metas = await asyncio.gather(
            *[self._fetch_meta(d, k) for d, k in pairs],
        )
        return [m for m in metas if m is not None]

    async def get_services_by_address(self, address: str) -> list[ServiceMeta]:
        dotted_target = address.replace('/', '.')
        pairs = [(d, k) for d, k in self._live_services() if d == dotted_target]
        metas = await asyncio.gather(
            *[self._fetch_meta(d, k) for d, k in pairs],
        )
        return [m for m in metas if m is not None]

    def on_service_start(
            self,
            kind: str,
            callback: Callable[[ServiceMeta], Awaitable[None] | None],
    ) -> Handle:
        handle = ZenohHandle(kind, lambda: self._remove_callback(
            self._start_callbacks, kind, callback,
        ))
        lst = self._start_callbacks.setdefault(kind, [])
        lst.append((callback, handle))
        return handle

    def on_service_stop(
            self,
            kind: str,
            callback: Callable[[ServiceMeta], Awaitable[None] | None],
    ) -> Handle:
        handle = ZenohHandle(kind, lambda: self._remove_callback(
            self._stop_callbacks, kind, callback,
        ))
        lst = self._stop_callbacks.setdefault(kind, [])
        lst.append((callback, handle))
        return handle

    @staticmethod
    def _remove_callback(
            registry: dict[str, list],
            kind: str,
            callback: Callable,
    ) -> None:
        lst = registry.get(kind, [])
        for cb, h in list(lst):
            if cb is callback:
                lst.remove((cb, h))
                return

    # -- ServiceOperator: get --------------------------------------------

    def _issue_get(self, query_key: str, payload: bytes | None) -> 'asyncio.Future[list[zenoh.Reply]]':
        """Issue a callback-based zenoh get, bridged to a loop-side Future.

        Replies are collected on the zenoh callback thread; the drop
        (completion) signal completes the Future via call_soon_threadsafe.
        Value flow is strictly one-way (zenoh thread → loop) — no thread
        ever waits on a loop-side result.
        """
        loop = self._loop
        fut: asyncio.Future = loop.create_future()
        box: list[zenoh.Reply] = []

        def _on_reply(reply: zenoh.Reply) -> None:
            box.append(reply)

        def _complete() -> None:
            if not fut.done():  # guard: caller may have cancelled / timed out
                fut.set_result(box)

        def _on_drop() -> None:
            try:
                loop.call_soon_threadsafe(_complete)
            except RuntimeError:
                pass  # loop closed during shutdown

        self._session.get(
            query_key,
            zenoh.handlers.Callback(_on_reply, _on_drop),
            payload=payload,
            timeout=_QUERY_TIMEOUT,
        )
        return fut

    async def get(
            self,
            kind: str,
            key: str,
            params: bytes | None,
            *services: ServiceMeta,
    ) -> list[Reply]:
        self._require_started()
        if services:
            targets = [meta['address'] for meta in services]
        else:
            targets = self._live_addresses_by_kind(kind)
        if not targets:
            return []

        payload = _encode_query_payload(self._this_address, params)

        futures: list[asyncio.Future] = []
        addresses: list[str] = []
        for address in targets:
            query_key = self._keyspace.per_service(address, kind).query_key(key)
            try:
                fut = self._issue_get(query_key, payload)
            except Exception:
                self._logger.exception(
                    "get dispatch failed: kind=%s key=%s address=%s",
                    kind, key, address,
                )
                continue
            futures.append(fut)
            addresses.append(address)

        gathered = await asyncio.gather(
            *[asyncio.wait_for(f, timeout=_QUERY_WAIT_TIMEOUT) for f in futures],
            return_exceptions=True,
        )

        results: list[Reply] = []
        for address, outcome in zip(addresses, gathered):
            if isinstance(outcome, BaseException):
                # safety net — normally the zenoh timeout fires the drop first
                self._logger.warning(
                    "get did not complete: kind=%s key=%s address=%s error=%r",
                    kind, key, address, outcome,
                )
                continue
            for reply in outcome:
                if reply.ok:
                    results.append(Reply(
                        address=address,
                        key=key,
                        payload=reply.ok.payload.to_bytes(),
                        timestamp=time.time(),
                    ))
                    break
                else:
                    self._logger.warning(
                        "get query rejected: kind=%s key=%s address=%s error=%s",
                        kind, key, address,
                        reply.err.payload.to_string() if reply.err.payload else 'unknown',
                    )
        return results

    # -- ServiceOperator: sub --------------------------------------------

    def sub(
            self,
            kind: str,
            key: str,
            handler: Callable[[Sample], Awaitable[None] | None],
            *services: ServiceMeta,
    ) -> Handle:
        """Subscribe to a service's pub stream.

        Callback subscriber → shared dispatch pipeline → one task per
        sample.  No per-subscription thread.
        """
        self._require_started()
        targets = list(services) if services else []
        if not targets:
            wildcard = self._keyspace.kind_pub_wildcard(kind, key)
            return self._start_sub(wildcard, handler, f"{kind}/{key}")
        handles = [
            self._start_sub(
                self._keyspace.per_service(meta['address'], kind).pub_key(key),
                handler,
                f"{meta['address']}/{kind}/{key}",
            )
            for meta in targets
        ]

        def _close_all() -> None:
            for h in handles:
                h.close()

        # aggregate handle is NOT tracked in _sub_handles — its children
        # already are; tracking both would double-close on exit.
        return ZenohHandle(f"{kind}/{key}", _close_all)

    def _start_sub(
            self,
            key_expr: str,
            handler: Callable[[Sample], Awaitable[None] | None],
            handle_key: str,
    ) -> ZenohHandle:
        def _on_sample(sample: zenoh.Sample) -> None:
            # zenoh callback thread: extract + enqueue, never block
            if sample.kind != zenoh.SampleKind.PUT:
                return
            item = (handler, str(sample.key_expr), sample.payload.to_bytes())
            if not self._sub_dispatcher.push_from_thread(item):
                self._logger.warning(
                    "sub queue full, dropping sample: %s", handle_key,
                )

        zenoh_sub = self._session.declare_subscriber(key_expr, _on_sample)

        def _close() -> None:
            try:
                zenoh_sub.undeclare()
            except Exception:
                self._logger.info(
                    "subscriber already undeclared (session likely closed): %s",
                    handle_key,
                )
            try:
                self._sub_handles.remove(h)
            except ValueError:
                pass

        h = ZenohHandle(handle_key, _close)
        self._sub_handles.append(h)
        return h

    async def _dispatch_sub(self, item: tuple[Callable, str, bytes]) -> None:
        """loop task (one per sample): parse + run handler."""
        handler, key_expr, payload = item
        sample = self._parse_pub_sample(key_expr, payload)
        try:
            await invoke_handler(handler, sample)
        except asyncio.CancelledError:
            raise
        except Exception:
            self._logger.exception("sub handler error: key=%s", sample['key'])

    # -- ServiceOperator: emit -------------------------------------------

    async def emit(
            self,
            kind: str,
            key: str,
            payload: bytes,
            *services: ServiceMeta,
    ) -> None:
        self._require_started()
        if services:
            targets = [meta['address'] for meta in services]
        else:
            targets = self._live_addresses_by_kind(kind)
        for address in targets:
            listen_key = self._keyspace.per_service(address, kind).listen_key(key)
            ok = self._outbound.submit(
                f"emit:{kind}/{key}",
                lambda k=listen_key: self._session.put(k, payload),
            )
            if not ok:
                self._logger.error(
                    "emit dropped (outbound unavailable): kind=%s key=%s address=%s",
                    kind, key, address,
                )

    # -- lifecycle -------------------------------------------------------

    async def __aenter__(self) -> Self:
        if self._started:
            return self
        self._started = True
        self._loop = asyncio.get_running_loop()
        self._outbound.start()
        self._sub_dispatcher.start(self._loop, self._dispatch_sub)
        self._liveness_dispatcher.start(self._loop, self._dispatch_liveness)
        await self._liveness_listener.__aenter__()
        self._logger.debug("ZenohOperator started: address=%s", self._this_address)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # 1. close terminals in reverse order
        for terminal in reversed(list(self._terminals.values())):
            try:
                await terminal.__aexit__(exc_type, exc_val, exc_tb)
            except Exception:
                self._logger.exception("terminal exit error")
        self._terminals.clear()

        # 2. stop producers: sub subscribers + liveness listener
        for h in list(self._sub_handles):
            try:
                h.close()
            except Exception:
                self._logger.exception("sub handle close error: %s", h.key)
        self._sub_handles.clear()
        await self._liveness_listener.__aexit__(exc_type, exc_val, exc_tb)

        # 3. stop pipelines: cancel consumers + in-flight, gather
        await self._sub_dispatcher.aclose()
        await self._liveness_dispatcher.aclose()

        # 4. stop the outbound worker (drain bounded)
        self._outbound.close()

        self._start_callbacks.clear()
        self._stop_callbacks.clear()
        self._meta_cache.clear()
        self._started = False

    # -- liveness → callback dispatch (zenoh thread → pipeline → loop) ----

    def _on_service_online(self, identity: str) -> None:
        """zenoh callback: enqueue (online, addr, kind)."""
        parsed = self._keyspace.parse_live_identity(identity)
        if parsed is None:
            return
        if not self._liveness_dispatcher.push_from_thread(('online',) + parsed):
            self._logger.error(
                "liveness queue full, online event lost: %s", identity,
            )

    def _on_service_offline(self, identity: str) -> None:
        """zenoh callback: enqueue (offline, addr, kind)."""
        parsed = self._keyspace.parse_live_identity(identity)
        if parsed is None:
            return
        if not self._liveness_dispatcher.push_from_thread(('offline',) + parsed):
            self._logger.error(
                "liveness queue full, offline event lost: %s", identity,
            )

    async def _dispatch_liveness(self, item: tuple[str, str, str]) -> None:
        """loop task (one per event): fetch/evict meta, fire callbacks."""
        direction, dotted_addr, kind = item
        if direction == 'online':
            meta = await self._fetch_meta(dotted_addr, kind)
            if meta is None:
                return  # fetch failure already logged
            callbacks = self._start_callbacks.get(kind, [])
            event = 'on_service_start'
        else:
            meta = self._meta_cache.pop((dotted_addr, kind), None)
            if meta is None:
                # Service died before its meta was ever fetched — no valid
                # ServiceDeclaration can be reconstructed, so K4 (from_meta
                # round-trip invariant) cannot be upheld.  Skip the callbacks
                # entirely rather than deliver a broken meta.
                self._logger.warning(
                    "service stopped before meta was cached, "
                    "on_service_stop callbacks will not fire: "
                    "kind=%s address=%s", kind, dotted_addr,
                )
                return
            callbacks = self._stop_callbacks.get(kind, [])
            event = 'on_service_stop'

        for cb, _ in list(callbacks):
            try:
                await invoke_handler(cb, meta)
            except asyncio.CancelledError:
                raise
            except Exception:
                self._logger.exception(
                    "%s callback error: kind=%s address=%s",
                    event, kind, dotted_addr,
                )

    async def _fetch_meta(self, dotted_addr: str, kind: str) -> ServiceMeta | None:
        """Query a service's meta queryable (callback-based, no thread held)."""
        keys = self._keyspace.per_service(
            dotted_addr.replace('.', '/'), kind,
        )
        try:
            fut = self._issue_get(keys.meta_query_key(), None)
            replies = await asyncio.wait_for(fut, timeout=_QUERY_WAIT_TIMEOUT)
            for reply in replies:
                if reply.ok:
                    meta = json.loads(reply.ok.payload.to_bytes())
                    self._meta_cache[(dotted_addr, kind)] = meta
                    return meta
        except Exception:
            pass  # fall through to the warning below
        # discovery degrades silently otherwise — a service whose meta
        # can't be fetched simply vanishes from the client's view.
        self._logger.warning(
            "fetch meta failed: kind=%s address=%s", kind, dotted_addr,
        )
        return None

    def _parse_pub_sample(self, key_expr: str, payload: bytes) -> Sample:
        """Parse a zenoh pub sample key/payload → operator Sample."""
        parsed = self._keyspace.parse_key(key_expr)
        if parsed is not None:
            dotted_addr, _kind, _slot, business_key = parsed
            return Sample(
                address=dotted_addr.replace('.', '/'),
                key=business_key,
                payload=payload,
                timestamp=time.time(),
            )
        return Sample(
            address='',
            key='',
            payload=payload,
            timestamp=time.time(),
        )
