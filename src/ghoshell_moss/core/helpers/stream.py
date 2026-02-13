import asyncio
from collections import deque
from typing import Generic, TypeVar

from ghoshell_common.helpers import Timeleft

from ghoshell_moss.core.helpers.asyncio_utils import ThreadSafeEvent

ItemT = TypeVar("ItemT")


# 实现线程安全的 Stream 对象, 预计同时支持 asyncio 与 sync 两种调用方式.
# 能够支持阻塞逻辑.
#
# todo: 还需要大量的单元测试验证.


class ThreadSafeStreamSender(Generic[ItemT]):
    """
    实现线程安全的对象发送者.
    """

    def __init__(
            self,
            added: ThreadSafeEvent,
            completed: ThreadSafeEvent,
            queue: deque[ItemT | Exception | None],
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
        self._completed.set()
        self._queue.append(error)
        self._added.set()

    def append(self, item: ItemT | None) -> None:
        if self._completed.is_set():
            # 当输入已经结束时, 不再接受新的对象.
            return
        if item is None:
            # 异常和 None item 都用来表示发送流已经结束.
            # commit 函数可以重入.
            self.commit()
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
        self._completed.set()
        # 发送毒丸, 用来提示流的结束.
        self._queue.append(None)
        # 毒丸也需要事件标记.
        self._added.set()

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

    def __next__(self) -> ItemT:
        if len(self._queue) > 0:
            # 队列不为空的情况.
            item = self._queue.popleft()
            if isinstance(item, Exception):
                # 接受到异常, 抛出. 所以 ItemT 不支持用异常.
                raise item
            elif item is None:
                # 接受到毒丸, 结束遍历.
                raise StopIteration
            else:
                return item

        elif self._completed.is_set():
            # 已经拿到了所有的结果.
            raise StopIteration

        else:
            # 判断时间是否超时.
            left = self._timeleft.left() or None
            # 阻塞等待到下一个 item 输入.
            if not self._added.wait_sync(left):
                raise TimeoutError(f"Timeout waiting for {self._timeleft.timeout}")

            item = None
            if len(self._queue) > 0:
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
            item = None
            if len(self._queue) > 0:
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
