"""
Shared zenoh↔asyncio bridge primitives for the service operator layer.

Architecture discipline (settled 2026-09-01, probed on zenoh 1.9.0):

1. zenoh callbacks NEVER block — enqueue and return.  Queryable delivery is
   serial per closure: a blocking callback head-of-line blocks every later
   query on that queryable.
2. No point in the whole pipeline makes a zenoh thread wait for a loop-side
   result.  Server side decouples via deferred reply (the ``zenoh.Query``
   object itself is the reply channel); client side flows values one-way
   into the loop via ``loop.call_soon_threadsafe`` onto an asyncio Future.
3. Handler execution is create_task-per-event with in-flight governance —
   a slow or hung handler never freezes the pipeline, and shutdown cancels
   and gathers every in-flight task.
"""

import asyncio
import inspect
import queue
import threading
from typing import Any, Awaitable, Callable, Generic, TypeVar

import janus

from ghoshell_moss.core.blueprint.service import Handle

import logging

__all__ = ['ZenohHandle', 'LoopDispatcher', 'OutboundWorker', 'invoke_handler']

_T = TypeVar('_T')


class ZenohHandle(Handle):
    """Idempotent close handle wrapping a cleanup function."""

    def __init__(self, key: str, close_fn: Callable[[], None]):
        self._key = key
        self._close_fn = close_fn
        self._closed = False

    @property
    def key(self) -> str:
        return self._key

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._close_fn()


async def invoke_handler(handler: Callable[[_T], Any], arg: _T) -> Any:
    """Run a sync-or-async handler without blocking the event loop.

    Async handlers run on the loop; sync handlers are offloaded to a
    worker thread (they may block on I/O or locks).
    """
    if inspect.iscoroutinefunction(handler):
        return await handler(arg)
    result = await asyncio.to_thread(handler, arg)
    if inspect.isawaitable(result):
        return await result
    return result


class LoopDispatcher(Generic[_T]):
    """zenoh-thread → asyncio pipeline: janus queue, single consumer,
    create_task-per-event with in-flight governance.

    Producers call ``push_from_thread`` from zenoh callback threads (never
    blocks).  The consumer task dequeues and spawns one task per event, so a
    slow or hung dispatch never blocks later events.  ``aclose`` cancels the
    consumer and every in-flight task, then gathers them.
    """

    def __init__(self, name: str, logger: logging.Logger, *, maxsize: int):
        self._name = name
        self._logger = logger
        self._maxsize = maxsize
        self._queue: janus.Queue | None = None
        self._consumer: asyncio.Task | None = None
        self._inflight: set[asyncio.Task] = set()
        self._dispatch: Callable[[_T], Awaitable[None]] | None = None

    def start(
            self,
            loop: asyncio.AbstractEventLoop,
            dispatch: Callable[[_T], Awaitable[None]],
    ) -> None:
        """Must be called on the running loop (janus binds to it)."""
        self._queue = janus.Queue(maxsize=self._maxsize)
        self._dispatch = dispatch
        self._consumer = loop.create_task(
            self._consume(), name=f"dispatcher-{self._name}",
        )

    def push_from_thread(self, item: _T) -> bool:
        """Enqueue from any thread.  Returns False when full or shut down —
        the caller decides the overflow policy (error reply / drop + log)."""
        q = self._queue
        if q is None:
            return False
        try:
            q.sync_q.put_nowait(item)
            return True
        except (janus.SyncQueueFull, janus.SyncQueueShutDown):
            return False

    async def _consume(self) -> None:
        while True:
            try:
                item = await self._queue.async_q.get()
            except (janus.AsyncQueueShutDown, asyncio.CancelledError):
                return
            task = asyncio.create_task(self._safe_dispatch(item))
            self._inflight.add(task)
            task.add_done_callback(self._inflight.discard)
            # release our reference now — items may carry zenoh objects
            # (e.g. Query) whose finalization signals remote completion;
            # holding them until the next event would delay it.
            del item, task

    async def _safe_dispatch(self, item: _T) -> None:
        try:
            await self._dispatch(item)
        except asyncio.CancelledError:
            raise
        except Exception:
            # per-event fault boundary — dispatch functions do their own
            # domain-specific error handling (e.g. error replies); this net
            # only guarantees one event's failure never poisons the pipeline.
            self._logger.exception("dispatch error: pipeline=%s", self._name)

    async def aclose(self) -> None:
        if self._consumer is not None:
            self._consumer.cancel()
            try:
                await self._consumer
            except asyncio.CancelledError:
                pass
            self._consumer = None
        inflight = list(self._inflight)
        for task in inflight:
            task.cancel()
        if inflight:
            await asyncio.gather(*inflight, return_exceptions=True)
        self._inflight.clear()
        if self._queue is not None:
            self._queue.shutdown(immediate=True)
            self._queue = None


class OutboundWorker:
    """Single daemon thread executing sync zenoh ops (reply / put / declare)
    off the event loop.

    zenoh write ops can block under congestion control; routing them through
    one worker keeps the loop deterministically clean.  ``submit`` never
    blocks; a full queue is reported to the caller (never silent).
    """

    def __init__(self, name: str, logger: logging.Logger, *, maxsize: int = 1024):
        self._name = name
        self._logger = logger
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._thread = threading.Thread(
            target=self._run, daemon=True, name=f"outbound-{name}",
        )
        self._closed = False

    def start(self) -> None:
        self._thread.start()

    def submit(self, label: str, op: Callable[[], None]) -> bool:
        """Enqueue a sync zenoh op.  Returns False when full or closed."""
        if self._closed:
            return False
        try:
            self._queue.put_nowait((label, op))
            return True
        except queue.Full:
            self._logger.error(
                "outbound queue full, op dropped: worker=%s op=%s",
                self._name, label,
            )
            return False

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            label, op = item
            try:
                op()
            except Exception:
                self._logger.exception(
                    "outbound op failed: worker=%s op=%s", self._name, label,
                )
            finally:
                # release refs immediately — ops close over zenoh objects
                # (e.g. Query) whose finalization signals remote completion.
                del item, op

    def close(self, timeout: float = 2.0) -> None:
        """Drain pending ops, then stop the worker.  Bounded by ``timeout``."""
        if self._closed:
            return
        self._closed = True
        try:
            self._queue.put(None, timeout=timeout)
        except queue.Full:
            self._logger.error(
                "outbound worker close: queue full, pending ops abandoned: %s",
                self._name,
            )
            return
        self._thread.join(timeout)
        if self._thread.is_alive():
            self._logger.error(
                "outbound worker did not stop within %.1fs: %s",
                timeout, self._name,
            )
