import asyncio
from typing import List, Tuple
from ghoshell_common.helpers import Timeleft

import threading


class ThreadSafeEvent:
    """
    thread-safe event
    """

    def __init__(self):
        self.thread_event = threading.Event()
        self.awaits_events: List[Tuple[asyncio.AbstractEventLoop, asyncio.Event]] = []
        self._lock = threading.Lock()

    def is_set(self) -> bool:
        return self.thread_event.is_set()

    def wait_sync(self, timeout: float | None = None) -> bool:
        return self.thread_event.wait(timeout)

    def set(self) -> None:
        self.thread_event.set()
        with self._lock:
            for loop, event in self.awaits_events.copy():
                loop.call_soon_threadsafe(event.set)
            self.awaits_events.clear()

    def _add_awaits(self, loop: asyncio.AbstractEventLoop, event: asyncio.Event) -> None:
        with self._lock:
            if self.thread_event.is_set():
                loop.call_soon_threadsafe(event.set)
            else:
                self.awaits_events.append((loop, event))

    async def wait(self) -> None:
        loop = asyncio.get_running_loop()
        event = asyncio.Event()
        self._add_awaits(loop, event)
        await event.wait()

    async def wait_until_timeout(self, timeout: float | None) -> None:
        if timeout is None or timeout <= 0.0:
            await self.wait()
        else:
            wait_task = asyncio.create_task(self.wait())

            async def cancel_when_timeout():
                await asyncio.sleep(timeout)
                if not wait_task.done():
                    wait_task.cancel()
                    raise asyncio.TimeoutError(f"wait timeout after: {timeout}")

            # 测试未到时间直接结束.
            await asyncio.gather(wait_task, cancel_when_timeout())

    def clear(self) -> None:
        self.thread_event.clear()
        with self._lock:
            for loop, event in self.awaits_events.copy():
                event.clear()
            self.awaits_events.clear()
