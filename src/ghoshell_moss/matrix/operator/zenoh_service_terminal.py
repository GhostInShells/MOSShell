"""
ZenohServiceTerminal — ServiceProvider ABC 的 zenoh 实现.

一个 terminal 对应一次 ``provide()`` 调用: 一个 cell + 一个 service kind 的
运行时接线端子。生命周期内持有 liveness token + meta queryable + 按需创建的
per-key queryable / publisher / subscriber。

异步桥纪律 (2026-09-01 实测定型, 见 _bridge.py 模块 docstring):

- zenoh 回调只入队, 立即返回。queryable 投递对同一 closure 串行,
  阻塞回调 = 队头阻塞整个 queryable。
- query 对象即回复信道 (deferred reply): 回调入队后, handler 在 loop 上
  create_task-per-query 并发执行, reply 经出站 worker 从 loop 外发出。
  zenoh 线程从不等待 loop 的计算结果。
- 错误永不静默: 无 handler / 解码失败 / handler 异常 / 队满 → error reply
  (``query.reply_err``), caller 不会挂到超时。
- handler 并发契约: handler 会被并发调用。有 await 点且共享状态的 handler
  必须自己加锁 (内核不串行化)。
"""

import asyncio
import json
import time
from typing import Callable, Awaitable

from ghoshell_moss.depends import depend_matrix

depend_matrix()

import zenoh

from ghoshell_moss.core.blueprint.service import (
    ServiceProvider,
    ServiceDeclaration,
    ServiceMeta,
    Query,
    Sample,
    Handle,
)
from ._utils import ServiceKeyExpr, _META_KEY
from ._bridge import ZenohHandle, LoopDispatcher, OutboundWorker, invoke_handler

import logging

__all__ = ['ZenohServiceTerminal']

_QUERY_QUEUE_MAXSIZE = 1000
_LISTEN_QUEUE_MAXSIZE = 1000

# Backward-compatible alias — operator historically imported this name.
_ZenohHandle = ZenohHandle


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
        # publishers are created and used only on the outbound worker thread
        self._publishers: dict[str, zenoh.Publisher] = {}
        # every queryable/listen registration — closed at exit so no zenoh
        # entity outlives the terminal (handles are idempotent)
        self._registered_handles: list[ZenohHandle] = []

        # -- handler registries --------------------------------------------
        # business_key → sync or async handler
        self._query_handlers: dict[str, Callable[[Query], Awaitable[bytes] | bytes]] = {}
        self._listen_handlers: dict[str, Callable[[Sample], Awaitable[None] | None]] = {}

        # -- async bridge --------------------------------------------------
        self._query_dispatcher: LoopDispatcher[tuple[zenoh.Query, str]] = LoopDispatcher(
            f"query-{keys.kind}", logger, maxsize=_QUERY_QUEUE_MAXSIZE,
        )
        self._listen_dispatcher: LoopDispatcher[tuple[str, bytes]] = LoopDispatcher(
            f"listen-{keys.kind}", logger, maxsize=_LISTEN_QUEUE_MAXSIZE,
        )
        self._outbound = OutboundWorker(f"terminal-{keys.kind}", logger)

        self._started = False
        self._closed = False

    # -- ServiceProvider props -------------------------------------------

    @property
    def meta(self) -> ServiceMeta:
        return self._meta

    def _require_started(self) -> None:
        if not self._started or self._closed:
            raise RuntimeError(
                f"terminal not running (kind={self._keys.kind!r}): "
                "register handlers between __aenter__ and __aexit__"
            )

    # -- ServiceProvider: queryable --------------------------------------

    def queryable(
            self,
            key: str,
            handler: Callable[[Query], Awaitable[bytes] | bytes],
    ) -> Handle:
        self._require_started()
        if key in self._query_handlers:
            raise RuntimeError(
                f"queryable handler already registered for key={key!r}"
            )
        self._query_handlers[key] = handler

        # declare a per-key queryable — wildcard queryable matching is
        # not guaranteed across zenoh versions; explicit keys are safe.
        query_key = self._keys.query_key(key)
        q = self._session.declare_queryable(query_key, self._on_query)

        def _close() -> None:
            self._query_handlers.pop(key, None)
            try:
                q.undeclare()
            except Exception:
                self._logger.info(
                    "queryable already undeclared: kind=%s key=%s",
                    self._keys.kind, key,
                )

        handle = ZenohHandle(key, _close)
        self._registered_handles.append(handle)
        return handle

    # -- ServiceProvider: pub --------------------------------------------

    def pub(self, key: str, payload: bytes) -> None:
        self._require_started()

        def _op() -> None:
            # publisher creation + put both happen on the worker thread —
            # zenoh write ops may block under congestion control.
            p = self._publishers.get(key)
            if p is None:
                p = self._session.declare_publisher(self._keys.pub_key(key))
                self._publishers[key] = p
            p.put(payload)

        if not self._outbound.submit(f"pub:{key}", _op):
            self._logger.error(
                "pub dropped (outbound unavailable): kind=%s key=%s",
                self._keys.kind, key,
            )

    # -- ServiceProvider: listen -----------------------------------------

    def listen(
            self,
            key: str,
            handler: Callable[[Sample], Awaitable[None] | None],
    ) -> Handle:
        self._require_started()
        if key in self._listen_handlers:
            raise RuntimeError(
                f"listen handler already registered for key={key!r}"
            )
        self._listen_handlers[key] = handler
        listen_key = self._keys.listen_key(key)

        def _on_sample(sample: zenoh.Sample) -> None:
            # zenoh callback thread: extract + enqueue, never block
            if sample.kind != zenoh.SampleKind.PUT:
                return
            biz_key = str(sample.key_expr)[len(self._keys.listen_prefix):]
            item = (biz_key, sample.payload.to_bytes())
            if not self._listen_dispatcher.push_from_thread(item):
                self._logger.warning(
                    "listen queue full, dropping sample: kind=%s key=%s",
                    self._keys.kind, biz_key,
                )

        sub = self._session.declare_subscriber(listen_key, _on_sample)

        def _close() -> None:
            self._listen_handlers.pop(key, None)
            try:
                sub.undeclare()
            except Exception:
                self._logger.info(
                    "listen subscriber already undeclared: kind=%s key=%s",
                    self._keys.kind, key,
                )

        handle = ZenohHandle(key, _close)
        self._registered_handles.append(handle)
        return handle

    # -- lifecycle -------------------------------------------------------

    async def __aenter__(self):
        if self._started:
            return self
        self._started = True
        self._closed = False
        loop = asyncio.get_running_loop()

        self._outbound.start()
        self._query_dispatcher.start(loop, self._dispatch_query)
        self._listen_dispatcher.start(loop, self._dispatch_listen)

        # -- auto-register meta queryable FIRST (TOCTOU: before liveness) ---
        self.queryable(_META_KEY, self._meta_handler)

        # -- liveness token -----------------------------------------------
        self._liveness_token = self._session.liveliness().declare_token(
            self._keys.live_key,
        )

        self._logger.debug(
            "ZenohServiceTerminal started: address=%s kind=%s",
            self._keys.address, self._keys.kind,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._closed = True

        # 1. stop pipelines: cancel consumers + all in-flight handlers.
        #    cancelled queries still push best-effort error replies into the
        #    outbound worker, so it must close AFTER this gather.
        await self._query_dispatcher.aclose()
        await self._listen_dispatcher.aclose()

        # 2. drain + stop the outbound worker (bounded)
        self._outbound.close()

        # 3. undeclare all zenoh entities off the loop
        def _undeclare() -> None:
            for handle in list(self._registered_handles):
                try:
                    handle.close()
                except Exception:
                    self._logger.exception(
                        "handle close error: kind=%s key=%s",
                        self._keys.kind, handle.key,
                    )
            if self._liveness_token is not None:
                try:
                    self._liveness_token.undeclare()
                except Exception:
                    self._logger.info(
                        "liveness token already undeclared (session likely closed): "
                        "address=%s kind=%s",
                        self._keys.address, self._keys.kind,
                    )
                self._liveness_token = None
            for pub in list(self._publishers.values()):
                try:
                    pub.undeclare()
                except Exception:
                    self._logger.info(
                        "publisher already undeclared: kind=%s",
                        self._keys.kind,
                    )
            self._publishers.clear()

        await asyncio.to_thread(_undeclare)

        self._registered_handles.clear()
        self._query_handlers.clear()
        self._listen_handlers.clear()
        self._started = False

    async def _meta_handler(self, _query: Query) -> bytes:
        """Auto-registered handler for the 'meta' business key."""
        return json.dumps(self._meta).encode('utf-8')

    # -- queryable bridge (zenoh thread → asyncio) -----------------------

    def _on_query(self, query: zenoh.Query) -> None:
        """zenoh thread: extract business key, enqueue, return."""
        # key_expr is {query_prefix}{business_key}
        key_expr = str(query.key_expr)
        business_key = key_expr[len(self._keys.query_prefix):]
        if not self._query_dispatcher.push_from_thread((query, business_key)):
            # backpressure breach — never silent, never hang the caller.
            # reply_err is a short zenoh op, acceptable on the callback thread.
            self._logger.error(
                "query queue full, rejecting: kind=%s key=%s",
                self._keys.kind, business_key,
            )
            try:
                query.reply_err(json.dumps({'error': 'service overloaded'}).encode())
            except Exception:
                self._logger.exception(
                    "overload reply failed: kind=%s key=%s",
                    self._keys.kind, business_key,
                )

    async def _dispatch_query(self, item: tuple[zenoh.Query, str]) -> None:
        """loop task (one per query): run handler, reply via outbound worker."""
        query, business_key = item

        handler = self._query_handlers.get(business_key)
        if handler is None:
            self._logger.warning(
                "query rejected: no handler for key=%r (address=%s kind=%s)",
                business_key, self._keys.address, self._keys.kind,
            )
            self._submit_error_reply(query, business_key, f'no handler for key={business_key!r}')
            return

        try:
            raw = query.payload.to_bytes() if query.payload is not None else b'{}'
            caller, params = _decode_query_payload(raw)
        except Exception:
            self._logger.exception(
                "malformed query payload: address=%s kind=%s key=%s",
                self._keys.address, self._keys.kind, business_key,
            )
            self._submit_error_reply(query, business_key, 'malformed query payload')
            return

        q = Query(
            address=caller,
            key=business_key,
            payload=params,
            timestamp=time.time(),
        )
        try:
            result = await invoke_handler(handler, q)
        except asyncio.CancelledError:
            # shutdown: best-effort error reply so the caller doesn't hang,
            # then propagate so the gather sees a clean cancellation.
            self._submit_error_reply(query, business_key, 'service shutting down')
            raise
        except Exception:
            self._logger.exception(
                "query handler error: address=%s kind=%s key=%s",
                self._keys.address, self._keys.kind, business_key,
            )
            self._submit_error_reply(query, business_key, 'handler error')
            return

        if not isinstance(result, (bytes, bytearray)):
            self._logger.error(
                "query handler returned %s, expected bytes: kind=%s key=%s",
                type(result).__name__, self._keys.kind, business_key,
            )
            self._submit_error_reply(query, business_key, 'handler returned non-bytes')
            return

        ok = self._outbound.submit(
            f"reply:{business_key}",
            lambda: query.reply(query.key_expr, bytes(result)),
        )
        if not ok:
            self._logger.error(
                "reply dropped (outbound unavailable): kind=%s key=%s",
                self._keys.kind, business_key,
            )

    def _submit_error_reply(self, query: zenoh.Query, business_key: str, message: str) -> None:
        payload = json.dumps({'error': message}).encode()
        ok = self._outbound.submit(
            f"reply_err:{business_key}",
            lambda: query.reply_err(payload),
        )
        if not ok:
            self._logger.error(
                "error reply dropped (outbound unavailable): kind=%s key=%s error=%s",
                self._keys.kind, business_key, message,
            )

    # -- listen bridge (zenoh thread → asyncio) ---------------------------

    async def _dispatch_listen(self, item: tuple[str, bytes]) -> None:
        """loop task (one per sample): run handler."""
        business_key, payload = item
        handler = self._listen_handlers.get(business_key)
        if handler is None:
            return
        # zenoh pub/sub does not expose the publisher's identity.
        # For client→server emissions (emit→listen), the caller
        # address is unavailable without a payload envelope.
        # V2: wrap emit payload with caller identity, like query does.
        s = Sample(
            address='',
            key=business_key,
            payload=payload,
            timestamp=time.time(),
        )
        try:
            await invoke_handler(handler, s)
        except asyncio.CancelledError:
            raise
        except Exception:
            self._logger.exception(
                "listen handler error: address=%s kind=%s key=%s",
                self._keys.address, self._keys.kind, business_key,
            )
