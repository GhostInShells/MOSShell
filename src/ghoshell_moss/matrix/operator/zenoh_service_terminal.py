"""
ZenohServiceTerminal — ServiceProvider ABC 的 zenoh 实现.

一个 terminal 对应一次 ``provide()`` 调用: 一个 cell + 一个 service kind 的
运行时接线端子。生命周期内持有 liveness token + meta queryable + 按需创建的
per-key queryable / publisher / subscriber。

异步桥: zenoh 回调线程 → janus.Queue → asyncio consumer task → async handler。
implementer 只写 ``async def handler(query) -> bytes``, 永远不碰 zenoh 线程。
"""

import asyncio
import json
import time
from typing import Callable, Awaitable

import janus
import zenoh

from ghoshell_moss.depends import depend_matrix

depend_matrix()

from ghoshell_moss.core.blueprint.service import (
    ServiceProvider,
    ServiceDeclaration,
    ServiceMeta,
    Query,
    Sample,
    Handle,
)
from ._utils import ServiceKeyExpr, _META_KEY

import logging

__all__ = ['ZenohServiceTerminal']

_QUERY_QUEUE_MAXSIZE = 1000
_LISTEN_QUEUE_MAXSIZE = 1000


# -- internal Handle impl ------------------------------------------------

class _ZenohHandle(Handle):

    def __init__(self, key: str, close_fn: Callable[[], None]):
        self._key = key
        self._close_fn = close_fn

    @property
    def key(self) -> str:
        return self._key

    def close(self) -> None:
        self._close_fn()


# -- query payload envelope (caller identity) ----------------------------
#
# 跨进程线格式: ZenohOperator.get() 打包, ZenohServiceTerminal 解包。
# 字段名是契约 — 另一侧进程实现 service 时必须解析同名 key。


def _encode_query_payload(caller: str, params: bytes | None) -> bytes:
    return json.dumps({
        'caller': caller,
        'params': params.hex() if params is not None else None,
    }).encode()


def _decode_query_payload(payload: bytes) -> tuple[str, bytes | None]:
    d = json.loads(payload)
    return d.get('caller', ''), bytes.fromhex(d['params']) if d.get('params') is not None else None


# -- ZenohServiceTerminal ------------------------------------------------

class ZenohServiceTerminal(ServiceProvider):
    """ServiceProvider 的 zenoh 实现 — 单 service 的运行时接线端子."""

    def __init__(
            self,
            *,
            session: zenoh.Session,
            keys: ServiceKeyExpr,
            declaration: ServiceDeclaration,
            logger: logging.Logger,
    ):
        self._session = session
        self._keys = keys
        self._declaration = declaration
        self._logger = logger
        self._meta = declaration.to_meta(keys.address)

        # -- zenoh handles -------------------------------------------------
        self._liveness_token: zenoh.LivelinessToken | None = None
        self._queryable: zenoh.Queryable | None = None
        self._publishers: dict[str, zenoh.Publisher] = {}

        # -- handler registries --------------------------------------------
        # business_key → async handler
        self._query_handlers: dict[str, Callable[[Query], Awaitable[bytes]]] = {}
        self._listen_handlers: dict[str, Callable[[Sample], Awaitable[None]]] = {}

        # -- async bridge --------------------------------------------------
        self._loop: asyncio.AbstractEventLoop | None = None
        self._query_queue: janus.Queue | None = None
        self._listen_queue: janus.Queue | None = None
        self._query_task: asyncio.Task | None = None
        self._listen_task: asyncio.Task | None = None

        self._started = False
        self._closed = False

    # -- ServiceProvider props -------------------------------------------

    @property
    def meta(self) -> ServiceMeta:
        return self._meta

    # -- ServiceProvider: queryable --------------------------------------

    def queryable(
            self,
            key: str,
            handler: Callable[[Query], Awaitable[bytes]],
    ) -> Handle:
        if key in self._query_handlers:
            raise RuntimeError(
                f"queryable handler already registered for key={key!r}"
            )
        self._query_handlers[key] = handler

        def _close() -> None:
            self._query_handlers.pop(key, None)

        return _ZenohHandle(key, _close)

    # -- ServiceProvider: pub --------------------------------------------

    def pub(self, key: str, payload: bytes) -> None:
        pub = self._publishers.get(key)
        if pub is None:
            pub_key = self._keys.pub_key(key)
            pub = self._session.declare_publisher(pub_key)
            self._publishers[key] = pub
        pub.put(payload)

    # -- ServiceProvider: listen -----------------------------------------

    def listen(
            self,
            key: str,
            handler: Callable[[Sample], Awaitable[None]],
    ) -> Handle:
        if key in self._listen_handlers:
            raise RuntimeError(
                f"listen handler already registered for key={key!r}"
            )

        listen_key = self._keys.listen_key(key)
        self._listen_handlers[key] = handler

        # declare subscriber (callback runs in zenoh thread)
        sub = self._session.declare_subscriber(
            listen_key,
            self._on_listen_sample,
        )

        def _close() -> None:
            self._listen_handlers.pop(key, None)
            try:
                sub.undeclare()
            except RuntimeError:
                pass

        return _ZenohHandle(key, _close)

    # -- lifecycle -------------------------------------------------------

    async def __aenter__(self):
        if self._started:
            return self
        self._started = True
        self._closed = False
        self._loop = asyncio.get_running_loop()

        # -- async bridge queues ------------------------------------------
        self._query_queue = janus.Queue(maxsize=_QUERY_QUEUE_MAXSIZE)
        self._listen_queue = janus.Queue(maxsize=_LISTEN_QUEUE_MAXSIZE)

        # -- wildcard queryable FIRST (TOCTOU: subscriber must see queryable) --
        queryable_wildcard = self._keys.query_prefix + '**'
        self._queryable = self._session.declare_queryable(
            queryable_wildcard,
            self._on_query,
        )

        # -- auto-register meta handler (replies with ServiceMeta) ---------
        self._query_handlers[_META_KEY] = self._meta_handler

        # -- liveness token -----------------------------------------------
        self._liveness_token = self._session.liveliness().declare_token(
            self._keys.live_key,
        )

        # -- consumer tasks -----------------------------------------------
        self._query_task = self._loop.create_task(self._consume_queries())
        self._listen_task = self._loop.create_task(self._consume_listen())

        self._logger.debug(
            "ZenohServiceTerminal started: address=%s kind=%s",
            self._keys.address, self._keys.kind,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._closed = True

        # cancel consumer tasks
        for task in (self._query_task, self._listen_task):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._query_task = None
        self._listen_task = None

        # undeclare all zenoh handles (sync → to_thread)
        async def _undeclare():
            if self._liveness_token is not None:
                try:
                    self._liveness_token.undeclare()
                except RuntimeError:
                    pass
                self._liveness_token = None

            if self._queryable is not None:
                try:
                    self._queryable.undeclare()
                except RuntimeError:
                    pass
                self._queryable = None

            for pub in list(self._publishers.values()):
                try:
                    pub.undeclare()
                except RuntimeError:
                    pass
            self._publishers.clear()

        await asyncio.to_thread(_undeclare)

        # close queues
        for q in (self._query_queue, self._listen_queue):
            if q is not None:
                q.shutdown(immediate=True)
        self._query_queue = None
        self._listen_queue = None

        self._query_handlers.clear()
        self._listen_handlers.clear()
        self._started = False

    async def _meta_handler(self, _query: Query) -> bytes:
        """Auto-registered handler for the 'meta' business key."""
        return json.dumps(self._meta).encode('utf-8')

    # -- queryable bridge (zenoh thread → asyncio) -----------------------

    def _on_query(self, query: zenoh.Query) -> None:
        """zenoh thread: extract business key, enqueue."""
        # key_expr is {query_prefix}{business_key}
        key_expr = str(query.key_expr)
        business_key = key_expr[len(self._keys.query_prefix):]
        try:
            self._query_queue.sync_q.put_nowait((query, business_key))
        except janus.SyncQueueFull:
            self._logger.error(
                "query queue full, dropping: key=%s", business_key,
            )
        except janus.SyncQueueShutDown:
            pass

    async def _consume_queries(self) -> None:
        """asyncio: dequeue query, run async handler, reply."""
        while not self._closed:
            try:
                query, business_key = await self._query_queue.async_q.get()
            except janus.AsyncQueueShutDown:
                return
            except asyncio.CancelledError:
                return

            handler = self._query_handlers.get(business_key)
            if handler is None:
                self._logger.warning(
                    "query dropped: no handler for key=%r (address=%s kind=%s)",
                    business_key, self._keys.address, self._keys.kind,
                )
                try:
                    query.reply(
                        query.key_expr,
                        json.dumps({
                            'error': f'no handler for key={business_key!r}',
                        }).encode(),
                    )
                except Exception:
                    pass
                continue

            try:
                raw = query.payload.to_bytes() if query.payload is not None else b'{}'
                caller, params = _decode_query_payload(raw)
                q = Query(
                    address=caller,
                    key=business_key,
                    payload=params,
                    timestamp=time.time(),
                )
                result = await handler(q)
                await asyncio.to_thread(query.reply, query.key_expr, result)
            except asyncio.CancelledError:
                raise
            except Exception:
                self._logger.exception(
                    "query handler error: address=%s kind=%s key=%s",
                    self._keys.address, self._keys.kind, business_key,
                )

    # -- listen bridge (zenoh thread → asyncio) --------------------------

    def _on_listen_sample(self, sample: zenoh.Sample) -> None:
        """zenoh thread: extract business key, enqueue sample."""
        if sample.kind != zenoh.SampleKind.PUT:
            return
        key_expr = str(sample.key_expr)
        business_key = key_expr[len(self._keys.listen_prefix):]
        try:
            self._listen_queue.sync_q.put_nowait((sample, business_key))
        except janus.SyncQueueFull:
            self._logger.error(
                "listen queue full, dropping: key=%s", business_key,
            )
        except janus.SyncQueueShutDown:
            pass

    async def _consume_listen(self) -> None:
        """asyncio: dequeue sample, run async handler."""
        while not self._closed:
            try:
                sample, business_key = await self._listen_queue.async_q.get()
            except janus.AsyncQueueShutDown:
                return
            except asyncio.CancelledError:
                return

            handler = self._listen_handlers.get(business_key)
            if handler is None:
                continue

            try:
                # zenoh pub/sub does not expose the publisher's identity.
                # For client→server emissions (emit→listen), the caller
                # address is unavailable without a payload envelope.
                # V2: wrap emit payload with caller identity, like query does.
                s = Sample(
                    address='',
                    key=business_key,
                    payload=sample.payload.to_bytes(),
                    timestamp=time.time(),
                )
                await handler(s)
            except asyncio.CancelledError:
                raise
            except Exception:
                self._logger.exception(
                    "listen handler error: address=%s kind=%s key=%s",
                    self._keys.address, self._keys.kind, business_key,
                )
