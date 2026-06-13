"""In-process future routing — string protocol, thread-safe.

Issuer side creates a future and awaits it. Executor side registers an
on_create callback to learn about new futures, then calls resolve/reject/
cancel to settle them. Direct mutation of the underlying future is
supported — an internal done callback handles the pending→done migration
either way.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Optional

__all__ = [
    "FutureEntry",
    "FutureRouter",
    "wait_future",
]

logger = logging.getLogger(__name__)


@dataclass
class FutureEntry:
    id: str
    arguments: str
    future: Future


class FutureRouter:
    """Thread-safe in-process future routing keyed by string id.

    Done futures are retained in a bounded deque (max_done) for inspection;
    the oldest entry is evicted when full.
    """

    def __init__(self, max_done: int = 64) -> None:
        self._pending: dict[str, FutureEntry] = {}
        self._done: deque[FutureEntry] = deque(maxlen=max_done)
        self._on_create: list[Callable[[FutureEntry], None]] = []
        self._lock = threading.Lock()

    # --- Issuer side ---

    def create(self, arguments: str) -> tuple[str, Future]:
        future_id = uuid.uuid4().hex
        future: Future = Future()
        entry = FutureEntry(id=future_id, arguments=arguments, future=future)

        with self._lock:
            self._pending[future_id] = entry
            callbacks = list(self._on_create)

        future.add_done_callback(lambda _f, fid=future_id: self._archive(fid))

        for cb in callbacks:
            try:
                cb(entry)
            except Exception:
                logger.exception("FutureRouter on_create callback failed")

        return future_id, future

    async def call(self, arguments: str, timeout: float | None = None) -> str:
        """Reference path for the common issuer flow: create, await, settle.

        Spelled out as code-as-prompt — readers should be able to copy this
        body when they need cancellation/timeout to reach the executor side.
        Use ``create`` + ``wait_future`` directly when you need the id (for
        logging, external cancel, or correlation).
        """
        _id, future = self.create(arguments)
        return await wait_future(future, timeout)

    def get(self, future_id: str) -> Optional[Future]:
        with self._lock:
            entry = self._pending.get(future_id)
            if entry is not None:
                return entry.future
            for e in self._done:
                if e.id == future_id:
                    return e.future
        return None

    def list_pending(self) -> list[FutureEntry]:
        with self._lock:
            return list(self._pending.values())

    def list_done(self) -> list[FutureEntry]:
        with self._lock:
            return list(self._done)

    # --- Executor side ---

    def resolve(self, future_id: str, result: str) -> bool:
        with self._lock:
            entry = self._pending.get(future_id)
        if entry is None or entry.future.done():
            return False
        try:
            entry.future.set_result(result)
            return True
        except Exception:
            logger.exception("FutureRouter resolve failed for %s", future_id)
            return False

    def reject(self, future_id: str, reason: str) -> bool:
        with self._lock:
            entry = self._pending.get(future_id)
        if entry is None or entry.future.done():
            return False
        try:
            entry.future.set_exception(RuntimeError(reason))
            return True
        except Exception:
            logger.exception("FutureRouter reject failed for %s", future_id)
            return False

    def cancel(self, future_id: str) -> bool:
        with self._lock:
            entry = self._pending.get(future_id)
        if entry is None or entry.future.done():
            return False
        return entry.future.cancel()

    # --- Callback registration ---

    def on_create(self, callback: Callable[[FutureEntry], None]) -> None:
        with self._lock:
            self._on_create.append(callback)

    # --- Internal ---

    def _archive(self, future_id: str) -> None:
        with self._lock:
            entry = self._pending.pop(future_id, None)
            if entry is not None:
                self._done.append(entry)


async def wait_future(future: Future, timeout: float | None = None) -> str:
    """Await a concurrent.futures.Future from asyncio; propagate timeout/cancel.

    On timeout or external cancellation, the underlying future is cancelled
    so the executor side observes the abandonment via future.cancelled().
    """
    awaitable = asyncio.wrap_future(future)
    try:
        if timeout is not None and timeout > 0:
            return await asyncio.wait_for(awaitable, timeout)
        return await awaitable
    except (asyncio.TimeoutError, asyncio.CancelledError):
        future.cancel()
        raise
