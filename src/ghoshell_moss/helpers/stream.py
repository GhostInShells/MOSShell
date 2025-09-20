from typing import Generic, TypeVar, Tuple, Iterable, AsyncIterable, Iterator, AsyncIterator
from ghoshell_common.helpers import Timeleft
from queue import Queue
import threading
import asyncio

I = TypeVar("I")


class ThreadSafeStreamSender(Generic[I]):

    def __init__(
            self,
            failed: threading.Event,
            completed: threading.Event,
            queue: Queue[I | Exception],
    ):
        self._completed = completed
        self._failed = failed
        self._queue = queue

    def append(self, item: I) -> None:
        if self._failed.is_set() or self._completed.is_set():
            # todo: error type
            raise RuntimeError("ThreadStreamSender is already done")
        self._queue.put_nowait(item)

    def end(self) -> None:
        self._completed.set()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            self._failed.set()
            self._queue.put_nowait(exc_val)
        self.end()


class ThreadSafeStreamReceiver(Generic[I]):
    """
    thread-safe receiver that also implements AsyncIterable[I]
    """

    def __init__(
            self,
            failed: threading.Event,
            completed: threading.Event,
            queue: Queue[I],
            timeout: float | None = None,
    ):
        self._completed = completed
        self._failed = failed
        self._queue = queue
        self._timeleft = Timeleft(timeout or 0)

    def __iter__(self):
        return self

    def __next__(self) -> I:
        if self._failed.is_set():
            # todo
            raise RuntimeError("ThreadSafeStreamReceiver is already failed")
        if self._completed.is_set() and self._queue.empty():
            raise StopIteration

        if not self._timeleft.alive():
            raise TimeoutError(self._timeleft.timeout)

        left = self._timeleft.left()
        left = left if left > 0 else None
        item = self._queue.get(block=True, timeout=left)
        if isinstance(item, Exception):
            raise item
        return item

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            self._failed.set()
        self._completed.set()

    def __aiter__(self):
        return self

    async def __anext__(self) -> I:
        if self._failed.is_set():
            # todo
            raise RuntimeError("ThreadSafeStreamReceiver is already failed")
        if self._completed.is_set() and self._queue.empty():
            raise StopAsyncIteration

        if not self._timeleft.alive():
            raise TimeoutError(self._timeleft.timeout)

        left = self._timeleft.left()
        left = left if left > 0 else None
        item = await asyncio.to_thread(self._queue.get, block=True, timeout=left)
        if isinstance(item, Exception):
            raise item
        return item

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            self._failed.set()
        self._completed.set()


def create_thread_safe_stream(timeout: float | None = None) -> Tuple[ThreadSafeStreamSender, ThreadSafeStreamReceiver]:
    failed = threading.Event()
    completed = threading.Event()
    queue = Queue()
    return ThreadSafeStreamSender(failed, completed, queue), ThreadSafeStreamReceiver(failed, completed, queue, timeout)
