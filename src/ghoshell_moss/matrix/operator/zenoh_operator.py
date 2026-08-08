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
from ghoshell_moss.tools.zenoh_helper import ZenohLivenessListener

from ._utils import ServiceKeyspace, ServiceKeyExpr
from .zenoh_service_terminal import (
    ZenohServiceTerminal,
    _encode_query_payload,
    _ZenohHandle,
)

import logging

__all__ = ['ZenohOperator']


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
                replies = await asyncio.to_thread(
                    self._session.get, query_key, payload,
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
        targets = list(services) if services else []
        if not targets:
            # no specific targets → subscribe to wildcard (all services of kind)
            wildcard = self._keyspace.kind_pub_wildcard(kind, key)
            sub = self._session.declare_subscriber(
                wildcard, self._make_sub_callback(handler),
            )
            handle = _ZenohHandle(f"{kind}/{key}",
                                   lambda s=sub: self._undeclare_sub(s))
            self._sub_handles.append(handle)
            # clean stale entries on close — kept in list for bulk cleanup at exit
            return handle
        else:
            handles: list[_ZenohHandle] = []
            for meta in targets:
                keys = self._keyspace.per_service(meta['address'], kind)
                pub_key = keys.pub_key(key)
                sub = self._session.declare_subscriber(
                    pub_key, self._make_sub_callback(handler),
                )
                h = _ZenohHandle(
                    f"{meta['address']}/{kind}/{key}",
                    lambda s=sub: self._undeclare_sub(s),
                )
                handles.append(h)
                self._sub_handles.append(h)

            def _close_all() -> None:
                for h in handles:
                    h.close()

            return _ZenohHandle(f"{kind}/{key}", _close_all)

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

        self._start_callbacks.clear()
        self._stop_callbacks.clear()
        self._started = False

    # -- liveness → callback dispatch (zenoh thread) ---------------------

    def _on_service_online(self, identity: str) -> None:
        parsed = self._keyspace.parse_live_identity(identity)
        if parsed is None:
            return
        dotted_addr, kind = parsed
        try:
            if self._loop is not None:
                asyncio.run_coroutine_threadsafe(
                    self._fire_start_callbacks(dotted_addr, kind), self._loop,
                )
        except RuntimeError:
            # loop closed — operator is shutting down
            pass

    def _on_service_offline(self, identity: str) -> None:
        parsed = self._keyspace.parse_live_identity(identity)
        if parsed is None:
            return
        dotted_addr, kind = parsed
        try:
            if self._loop is not None:
                asyncio.run_coroutine_threadsafe(
                    self._fire_stop_callbacks(dotted_addr, kind), self._loop,
                )
        except RuntimeError:
            # loop closed — operator is shutting down
            pass

    async def _fire_start_callbacks(self, dotted_addr: str, kind: str) -> None:
        meta = await self._fetch_meta(dotted_addr, kind)
        if meta is None:
            return
        for cb, _ in list(self._start_callbacks.get(kind, [])):
            try:
                cb(meta)
            except Exception:
                self._logger.exception(
                    "on_service_start callback error: kind=%s address=%s",
                    kind, dotted_addr,
                )

    async def _fire_stop_callbacks(self, dotted_addr: str, kind: str) -> None:
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

    # -- internal helpers ------------------------------------------------

    async def _fetch_meta(self, dotted_addr: str, kind: str) -> ServiceMeta | None:
        """Query a service's meta queryable."""
        keys = self._keyspace.per_service(
            dotted_addr.replace('.', '/'), kind,
        )
        meta_key = keys.meta_query_key()
        try:
            replies = await asyncio.to_thread(self._session.get, meta_key)
            for reply in replies:
                if reply.ok:
                    return json.loads(reply.ok.payload.to_bytes())
        except Exception:
            self._logger.debug(
                "fetch meta failed: kind=%s address=%s", kind, dotted_addr,
            )
        return None

    def _make_sub_callback(
            self,
            handler: Callable[[Sample], Awaitable[None]],
    ):
        """Return a zenoh-thread-safe callback that bridges to async handler.

        Parses the sample key_expr to recover the publishing service's
        address and the business key — otherwise wildcard subscribers
        cannot tell *which* service emitted the sample.
        """

        def _on_sample(sample: zenoh.Sample) -> None:
            if sample.kind != zenoh.SampleKind.PUT:
                return
            try:
                raw_payload = sample.payload.to_bytes()
            except Exception:
                self._logger.exception("sub sample payload decode error")
                return

            # parse key_expr → (dotted_addr, kind, slot, business_key)
            parsed = self._keyspace.parse_key(str(sample.key_expr))
            if parsed is not None:
                dotted_addr, _kind, _slot, business_key = parsed
                s = Sample(
                    address=dotted_addr.replace('.', '/'),
                    key=business_key,
                    payload=raw_payload,
                    timestamp=time.time(),
                )
            else:
                s = Sample(
                    address='',
                    key='',
                    payload=raw_payload,
                    timestamp=time.time(),
                )

            async def _run() -> None:
                try:
                    await handler(s)
                except Exception:
                    self._logger.exception("sub handler error")

            if self._loop is not None:
                try:
                    asyncio.run_coroutine_threadsafe(_run(), self._loop)
                except RuntimeError:
                    # loop closed — operator is shutting down
                    pass

    @staticmethod
    def _undeclare_sub(sub: zenoh.Subscriber) -> None:
        try:
            sub.undeclare()
        except RuntimeError:
            pass
