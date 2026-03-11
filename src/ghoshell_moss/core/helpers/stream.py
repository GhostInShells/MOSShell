import asyncio
from collections import deque
from typing import Generic, TypeVar

from ghoshell_common.helpers import Timeleft

from ghoshell_moss.core.helpers.asyncio_utils import ThreadSafeEvent

__all__ = [
    "ThreadSafeStreamReceiver",
    "ThreadSafeStreamSender",
    "create_sender_and_receiver",
    "create_typed_sender_and_receiver",
    "ItemT",
]

ItemT = TypeVar("ItemT")


# 实现线程安全的 Stream 对象, 预计同时支持 asyncio 与 sync 两种调用方式.
# 能够支持阻塞逻辑.


class _Committed:
    pass


class ThreadSafeStreamSender(Generic[ItemT]):
    """
    实现线程安全的对象发送者.
    """

    def __init__(
        self,
        added: ThreadSafeEvent,
        completed: ThreadSafeEvent,
        queue: deque[ItemT | Exception | _Committed],
    ):
        self._added = added
        """通过一个 added event 来做发送 item 信号的通讯. 用于阻塞等待. """
        self._completed = completed
        """通过一个 completed event 来标记发送终结. """
        self._queue = queue
        """通过 deque 做线程安全的数据队列存储. """

    def fail(self, error: Exception):
        if self._completed.is_set():
            return
        self._queue.append(error)
        self._added.set()
        self._completed.set()

    def append(self, item: ItemT) -> None:
        if self._completed.is_set():
            # 当输入已经结束时, 不再接受新的对象.
            return
        # 通过 deque 做线程安全的 buffer.
        self._queue.append(item)
        # 标记已经有输入的新 item.
        # 注意永远是先入队, 再标记.
        self._added.set()

    def commit(self) -> None:
        if self._completed.is_set():
            # 可重入.
            return
        # 发送毒丸, 用来提示流的结束.
        self._queue.append(_Committed)
        # 毒丸也需要事件标记.
        self._added.set()
        self._completed.set()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            # 标记失败.
            self.fail(exc_val)
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

    def __next__(self):
        while True:
            if len(self._queue) > 0:
                item = self._queue.popleft()
                if item is _Committed:
                    raise StopIteration
                elif isinstance(item, Exception):
                    raise item
                return item
            else:
                if self._completed.is_set():
                    if len(self._queue) > 0:
                        continue
                    raise StopIteration
                self._added.wait_sync(self._timeleft.left() or None)
                continue

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._completed.set()

    def __aiter__(self):
        return self

    async def __anext__(self):
        while True:
            if len(self._queue) > 0:
                item = self._queue.popleft()
                if isinstance(item, Exception):
                    raise item
                elif item is _Committed:
                    raise StopAsyncIteration
                else:
                    return item
            else:
                if self._completed.is_set():
                    # 已经拿到了所有的结果.
                    raise StopAsyncIteration
                self._added.clear()
                left = self._timeleft.left() or None
                if left and left > 0.0:
                    await asyncio.wait_for(self._added.wait(), timeout=left)
                    continue
                else:
                    await self._added.wait()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._completed.set()


def create_sender_and_receiver(
    timeout: float | None = None,
) -> tuple[ThreadSafeStreamSender, ThreadSafeStreamReceiver]:
    added = ThreadSafeEvent()
    completed = ThreadSafeEvent()
    queue = deque()
    return ThreadSafeStreamSender(added, completed, queue), ThreadSafeStreamReceiver(added, completed, queue, timeout)


def create_typed_sender_and_receiver(
    item_type: type[ItemT],
    *,
    timeout: float | None = None,
) -> tuple[ThreadSafeStreamSender[ItemT], ThreadSafeStreamReceiver[ItemT]]:
    added = ThreadSafeEvent()
    completed = ThreadSafeEvent()
    queue = deque()
    return ThreadSafeStreamSender(added, completed, queue), ThreadSafeStreamReceiver(added, completed, queue, timeout)
