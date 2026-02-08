import asyncio
from collections import deque
from typing import Generic, TypeVar

from ghoshell_common.helpers import Timeleft

from ghoshell_moss.core.helpers.asyncio_utils import ThreadSafeEvent

ItemT = TypeVar("ItemT")


class ThreadSafeStreamSender(Generic[ItemT]):
    def __init__(
        self,
        added: ThreadSafeEvent,
        completed: ThreadSafeEvent,
        queue: deque[ItemT | Exception | None],
    ):
        self._added = added
        self._completed = completed
        self._queue = queue

    def append(self, item: ItemT | Exception | None) -> None:
        if self._completed.is_set():
            return
        if item is None or isinstance(item, Exception):
            self.commit()
            return
        self._queue.append(item)
        self._added.set()

    def commit(self) -> None:
        if not self._completed.is_set():
            self._queue.append(None)
            self._added.set()
            self._completed.set()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            self.append(exc_val)
        else:
            self.commit()


class ThreadSafeStreamReceiver(Generic[ItemT]):
    """
    thread-safe receiver that also implements AsyncIterable[ItemT]
    """

    def __init__(
        self,
        added: ThreadSafeEvent,
        completed: ThreadSafeEvent,
        queue: deque[ItemT | Exception | None],
        timeout: float | None = None,
    ):
        self._completed = completed
        self._added = added
        self._queue = queue
        self._timeleft = Timeleft(timeout or 0)

    def __iter__(self):
        return self

    def __next__(self) -> ItemT:
        if len(self._queue) > 0:
            item = self._queue.popleft()
            if isinstance(item, Exception):
                raise item
            elif item is None:
                raise StopIteration
            else:
                return item
        elif self._completed.is_set():
            # 已经拿到了所有的结果.
            raise StopIteration
        else:
            left = self._timeleft.left() or None
            if not self._added.wait_sync(left):
                raise TimeoutError(f"Timeout waiting for {self._timeleft.timeout}")
            item = self._queue.popleft()
            if len(self._queue) == 0:
                self._added.clear()

            if isinstance(item, Exception):
                raise item
            elif item is None:
                raise StopIteration
            else:
                return item

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._completed.set()

    def __aiter__(self):
        return self

    async def __anext__(self) -> ItemT:
        if len(self._queue) > 0:
            item = self._queue.popleft()
            if isinstance(item, Exception):
                raise item
            elif item is None:
                raise StopAsyncIteration
            else:
                return item
        elif self._completed.is_set():
            # 已经拿到了所有的结果.
            raise StopAsyncIteration
        else:
            left = self._timeleft.left() or None
            await asyncio.wait_for(self._added.wait(), timeout=left)
            item = self._queue.popleft()
            if len(self._queue) == 0:
                self._added.clear()

            if isinstance(item, Exception):
                raise item
            elif item is None:
                raise StopAsyncIteration
            else:
                return item

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._completed.set()


def create_thread_safe_stream(timeout: float | None = None) -> tuple[ThreadSafeStreamSender, ThreadSafeStreamReceiver]:
    added = ThreadSafeEvent()
    completed = ThreadSafeEvent()
    queue = deque()
    return ThreadSafeStreamSender(added, completed, queue), ThreadSafeStreamReceiver(added, completed, queue, timeout)
