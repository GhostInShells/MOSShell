"""
ZenohOperator — ServiceOperator ABC 的 zenoh 实现.

每个 cell 一个实例 (通过 ``matrix.service_operator()`` 惰性获取).
管理本 cell 的 service terminal 集合 + 全网 service 发现.

发现管线:
  ZenohLivenessListener({services_ns}/**) → on_online/offline
  → parse kind + dotted_addr → on_service_start/stop callbacks
  get_services_by_kind → liveness cache filter + meta queryable per service
"""

import asyncio
import json
import threading
import time
from typing import Callable, Awaitable

import janus

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
from ghoshell_moss.tools.zenoh_helper import ZenohLivenessListener

from ._utils import ServiceKeyspace, ServiceKeyExpr
from .zenoh_service_terminal import (
    ZenohServiceTerminal,
    _encode_query_payload,
    _ZenohHandle,
)

import logging

__all__ = ['ZenohOperator']

# Bound a single query's blocking time. ``session.get`` holds its iterator open
# until replies arrive or the zenoh default timeout — a stale target (no
# queryable at the key) would otherwise pin an executor thread for 10-30s.
_QUERY_TIMEOUT = 5.0


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
        self._start_callbacks: dict[str, list[tuple[Callable, _ZenohHandle]]] = {}
        self._stop_callbacks: dict[str, list[tuple[Callable, _ZenohHandle]]] = {}

        # subscriber handles for sub()
        self._sub_handles: list[_ZenohHandle] = []

        # liveness callback bridge (K2 janus.Queue pattern)
        self._liveness_queue: janus.Queue | None = None
        self._liveness_task: asyncio.Task | None = None

        self._loop: asyncio.AbstractEventLoop | None = None
        self._started = False

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

    async def get_services_by_kind(self, kind: str) -> list[ServiceMeta]:
        identities = self._liveness_listener.live_keys
        result: list[ServiceMeta] = []
        for identity in identities:
            parsed = self._keyspace.parse_live_identity(identity)
            if parsed is None:
                continue
            dotted_addr, k = parsed
            if k != kind:
                continue
            meta = await self._fetch_meta(dotted_addr, kind)
            if meta is not None:
                result.append(meta)
        return result

    async def get_services_by_address(self, address: str) -> list[ServiceMeta]:
        identities = self._liveness_listener.live_keys
        dotted_target = address.replace('/', '.')
        result: list[ServiceMeta] = []
        for identity in identities:
            parsed = self._keyspace.parse_live_identity(identity)
            if parsed is None:
                continue
            dotted_addr, kind = parsed
            if dotted_addr != dotted_target:
                continue
            meta = await self._fetch_meta(dotted_addr, kind)
            if meta is not None:
                result.append(meta)
        return result

    def on_service_start(
            self,
            kind: str,
            callback: Callable[[ServiceMeta], None],
    ) -> Handle:
        handle = _ZenohHandle(kind, lambda: self._remove_callback(
            self._start_callbacks, kind, callback,
        ))
        lst = self._start_callbacks.setdefault(kind, [])
        lst.append((callback, handle))
        return handle

    def on_service_stop(
            self,
            kind: str,
            callback: Callable[[ServiceMeta], None],
    ) -> Handle:
        handle = _ZenohHandle(kind, lambda: self._remove_callback(
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

    async def get(
            self,
            kind: str,
            key: str,
            params: bytes | None,
            *services: ServiceMeta,
    ) -> list[Reply]:
        targets = list(services) if services else await self.get_services_by_kind(kind)
        if not targets:
            return []

        payload = _encode_query_payload(self._this_address, params)

        async def _query_one(meta: ServiceMeta) -> Reply | None:
            keys = self._keyspace.per_service(meta['address'], kind)
            query_key = keys.query_key(key)
            try:
                # session.get returns a blocking iterator — consume it in the
                # worker thread so iteration doesn't stall the event loop.
                replies = await asyncio.to_thread(
                    lambda: list(self._session.get(
                        query_key, payload=payload, timeout=_QUERY_TIMEOUT,
                    )),
                )
                for reply in replies:
                    if reply.ok:
                        return Reply(
                            address=meta['address'],
                            key=key,
                            payload=reply.ok.payload.to_bytes(),
                            timestamp=time.time(),
                        )
                    else:
                        self._logger.warning(
                            "get query rejected: kind=%s key=%s address=%s error=%s",
                            kind, key, meta['address'],
                            reply.err.payload.to_string() if reply.err.payload else 'unknown',
                        )
            except Exception:
                self._logger.exception(
                    "get query failed: kind=%s key=%s address=%s",
                    kind, key, meta['address'],
                )
            return None

        results = await asyncio.gather(*[_query_one(m) for m in targets])
        return [r for r in results if r is not None]

    # -- ServiceOperator: sub --------------------------------------------

    def sub(
            self,
            kind: str,
            key: str,
            handler: Callable[[Sample], Awaitable[None]],
            *services: ServiceMeta,
    ) -> Handle:
        """Subscribe to a service's pub stream.

        No-callback subscriber + background-thread iterator + janus.Queue
        bridge.  Same pattern as ``ZenohTopicSubscriber`` — the zenoh
        subscriber's callback-based delivery is not reliable on single
        sessions; blocking iteration on a daemon thread is.
        """
        targets = list(services) if services else []
        if not targets:
            wildcard = self._keyspace.kind_pub_wildcard(kind, key)
            return self._start_sub_bridge(wildcard, handler, f"{kind}/{key}")
        else:
            handles: list[_ZenohHandle] = []
            for meta in targets:
                keys = self._keyspace.per_service(meta['address'], kind)
                pub_key = keys.pub_key(key)
                h = self._start_sub_bridge(
                    pub_key, handler,
                    f"{meta['address']}/{kind}/{key}",
                )
                handles.append(h)
            close_all = lambda: [h.close() for h in handles]
            return _ZenohHandle(f"{kind}/{key}", close_all)

    def _start_sub_bridge(
            self,
            key_expr: str,
            handler: Callable[[Sample], Awaitable[None]],
            handle_key: str,
    ) -> _ZenohHandle:
        zenoh_sub = self._session.declare_subscriber(key_expr)
        queue: janus.Queue = janus.Queue(maxsize=256)

        def _reader() -> None:
            try:
                for sample in zenoh_sub:
                    if sample.kind != zenoh.SampleKind.PUT:
                        continue
                    s = self._parse_pub_sample(sample)
                    try:
                        queue.sync_q.put_nowait(s)
                    except janus.SyncQueueFull:
                        pass
                    except janus.SyncQueueShutDown:
                        return
            except zenoh.ZError:
                pass  # subscriber undeclared — normal
            except Exception:
                self._logger.exception("sub reader error: %s", handle_key)

        t = threading.Thread(target=_reader, daemon=True, name=f"sub-{handle_key}")
        t.start()

        async def _consumer() -> None:
            while True:
                try:
                    s = await queue.async_q.get()
                    await handler(s)
                except janus.AsyncQueueShutDown:
                    return
                except asyncio.CancelledError:
                    return
                except Exception:
                    self._logger.exception("sub handler error: %s", handle_key)

        task: asyncio.Task = (
            self._loop.create_task(_consumer()) if self._loop
            else asyncio.create_task(_consumer())
        )

        def _close() -> None:
            try:
                zenoh_sub.undeclare()
            except Exception:
                self._logger.info(
                    "subscriber already undeclared (session likely closed): %s",
                    handle_key,
                )
            queue.shutdown(immediate=True)
            if not task.done():
                task.cancel()

        h = _ZenohHandle(handle_key, _close)
        self._sub_handles.append(h)
        return h

    # -- ServiceOperator: emit -------------------------------------------

    async def emit(
            self,
            kind: str,
            key: str,
            payload: bytes,
            *services: ServiceMeta,
    ) -> None:
        targets = list(services) if services else await self.get_services_by_kind(kind)
        for meta in targets:
            keys = self._keyspace.per_service(meta['address'], kind)
            listen_key = keys.listen_key(key)
            try:
                self._session.put(listen_key, payload)
            except Exception:
                self._logger.exception(
                    "emit failed: kind=%s key=%s address=%s",
                    kind, key, meta['address'],
                )

    # -- lifecycle -------------------------------------------------------

    async def __aenter__(self) -> Self:
        if self._started:
            return self
        self._started = True
        self._loop = asyncio.get_running_loop()
        self._liveness_queue = janus.Queue()
        self._liveness_task = self._loop.create_task(self._consume_liveness())
        await self._liveness_listener.__aenter__()
        self._logger.debug("ZenohOperator started: address=%s", self._this_address)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # close terminals in reverse order
        for terminal in reversed(list(self._terminals.values())):
            try:
                await terminal.__aexit__(exc_type, exc_val, exc_tb)
            except Exception:
                self._logger.exception("terminal exit error")
        self._terminals.clear()

        # undeclare all sub handles
        for h in list(self._sub_handles):
            try:
                h.close()
            except Exception:
                self._logger.exception("sub handle close error: %s", h.key)
        self._sub_handles.clear()

        # stop liveness listener
        await self._liveness_listener.__aexit__(exc_type, exc_val, exc_tb)

        # stop liveness consumer
        if self._liveness_task is not None:
            self._liveness_task.cancel()
            try:
                await self._liveness_task
            except asyncio.CancelledError:
                pass
            self._liveness_task = None
        if self._liveness_queue is not None:
            self._liveness_queue.shutdown(immediate=True)
            self._liveness_queue = None

        self._start_callbacks.clear()
        self._stop_callbacks.clear()
        self._started = False

    # -- liveness → callback dispatch (zenoh thread -→ janus -→ loop) ----

    def _on_service_online(self, identity: str) -> None:
        """zenoh callback: enqueue (addr, kind, online)."""
        parsed = self._keyspace.parse_live_identity(identity)
        if parsed is None:
            return
        try:
            self._liveness_queue.sync_q.put_nowait(('online',) + parsed)
        except janus.SyncQueueFull:
            pass
        except janus.SyncQueueShutDown:
            pass

    def _on_service_offline(self, identity: str) -> None:
        """zenoh callback: enqueue (addr, kind, offline)."""
        parsed = self._keyspace.parse_live_identity(identity)
        if parsed is None:
            return
        try:
            self._liveness_queue.sync_q.put_nowait(('offline',) + parsed)
        except janus.SyncQueueFull:
            pass
        except janus.SyncQueueShutDown:
            pass

    async def _consume_liveness(self) -> None:
        """asyncio task: single-point consumer for liveness events."""
        while True:
            try:
                event = await self._liveness_queue.async_q.get()
            except janus.AsyncQueueShutDown:
                return
            except asyncio.CancelledError:
                return

            direction, dotted_addr, kind = event
            if direction == 'online':
                meta = await self._fetch_meta(dotted_addr, kind)
                if meta is None:
                    continue
                for cb, _ in list(self._start_callbacks.get(kind, [])):
                    try:
                        cb(meta)
                    except Exception:
                        self._logger.exception(
                            "on_service_start callback error: kind=%s address=%s",
                            kind, dotted_addr,
                        )
            else:
                meta: ServiceMeta = {
                    'address': dotted_addr.replace('.', '/'),
                    'kind': kind,
                    'data': {},
                }
                for cb, _ in list(self._stop_callbacks.get(kind, [])):
                    try:
                        cb(meta)
                    except Exception:
                        self._logger.exception(
                            "on_service_stop callback error: kind=%s address=%s",
                            kind, dotted_addr,
                        )

    async def _fetch_meta(self, dotted_addr: str, kind: str) -> ServiceMeta | None:
        """Query a service's meta queryable."""
        keys = self._keyspace.per_service(
            dotted_addr.replace('.', '/'), kind,
        )
        meta_key = keys.meta_query_key()
        try:
            replies = await asyncio.to_thread(
                lambda: list(self._session.get(
                    meta_key, timeout=_QUERY_TIMEOUT,
                )),
            )
            for reply in replies:
                if reply.ok:
                    return json.loads(reply.ok.payload.to_bytes())
        except Exception:
            # discovery degrades silently otherwise — a service whose meta
            # can't be fetched simply vanishes from the client's view.
            self._logger.warning(
                "fetch meta failed: kind=%s address=%s", kind, dotted_addr,
            )
        return None

    def _parse_pub_sample(self, sample: zenoh.Sample) -> Sample:
        """Parse a zenoh pub sample → operator Sample (no side effects)."""
        raw_payload = sample.payload.to_bytes()
        parsed = self._keyspace.parse_key(str(sample.key_expr))
        if parsed is not None:
            dotted_addr, _kind, _slot, business_key = parsed
            return Sample(
                address=dotted_addr.replace('.', '/'),
                key=business_key,
                payload=raw_payload,
                timestamp=time.time(),
            )
        return Sample(
            address='',
            key='',
            payload=raw_payload,
            timestamp=time.time(),
        )
