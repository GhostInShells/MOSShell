import asyncio
import threading
from typing import Optional, Generic, TypeVar, Tuple
from collections import deque

R = TypeVar("R")


class ThreadSafeResult(Generic[R]):
    """
    Thread-safe Result container that supports both synchronous and asynchronous waiting.

    This class allows setting a result from one thread and waiting for that result
    from multiple threads and coroutines simultaneously.

    Example:
        >>> result = ThreadSafeResult[str]()
        >>> # In one thread: result.resolve("Hello")
        >>> # In another thread/coroutine: value = result.wait() or await result.wait_async()

    Note: Written with the help from deepseek:v3.1
    """

    def __init__(self):
        self._waiting: deque[Tuple[asyncio.AbstractEventLoop, asyncio.Event]] = deque()
        self._event = threading.Event()
        self._result: Optional[R] = None
        self._cancelled_reason: str | None = None
        self._lock = threading.Lock()

    def resolve(self, result: R):
        """
        set the result
        """
        if self._event.is_set():
            raise RuntimeError(f"Already set result")

        self._result = result
        self._set_event()

    async def wait_async(self, timeout: float | None = None) -> R:
        """
        wait result in coroutine, thread-safe, non-blocking
        """
        if self._cancelled_reason is not None:
            raise asyncio.CancelledError(self._cancelled_reason)
        if self._event.is_set():
            return self._result

        with self._lock:
            loop = asyncio.get_running_loop()
            e = asyncio.Event()
            self._waiting.append((loop, e))

        if timeout:
            await asyncio.wait_for(e.wait(), timeout)
        else:
            await e.wait()
        if self._cancelled_reason is not None:
            raise asyncio.CancelledError(self._cancelled_reason)
        return self._result

    def wait(self, timeout: float | None = None) -> R:
        """
        wait result in thread level, block the thread
        """
        if self._cancelled_reason is not None:
            raise RuntimeError(self._cancelled_reason)
        if self._event.is_set():
            return self._result
        if not self._event.wait(timeout=timeout):
            raise TimeoutError(f"Timed out after {timeout} seconds while waiting for result")
        if self._cancelled_reason is not None:
            raise RuntimeError(self._cancelled_reason)

        return self._result

    def get_cancellation_reason(self) -> Optional[str]:
        """Get the reason for cancellation, if any."""
        return self._cancelled_reason

    def is_done(self) -> bool:
        return self._event.is_set()

    def cancel(self, reason: str) -> None:
        """
        cancel the task thread-safely
        """
        if self._event.is_set():
            return
        self._cancelled_reason = reason
        self._set_event()

    def _set_event(self):
        self._event.set()
        with self._lock:  # 获取锁
            self._event.set()
            waiting_copy = list(self._waiting)  # 复制数据
            self._waiting.clear()  # 清空原队列

        for loop, event in waiting_copy:
            if loop.is_running() and not event.is_set():
                loop.call_soon_threadsafe(event.set)
