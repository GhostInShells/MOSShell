import asyncio
from typing import List, Tuple, Protocol, Coroutine, Callable, Any, Optional
from typing_extensions import Self
from collections import deque

import threading


class ThreadSafeEvent:
    """
    thread-safe event
    """

    def __init__(self, debug: bool = False):
        self.thread_event = threading.Event()
        self.awaits_events: deque[Tuple[asyncio.AbstractEventLoop, asyncio.Event]] = deque()
        self.debug = debug
        self.set_at: Optional[str] = None
        self._lock = threading.Lock()

    def is_set(self) -> bool:
        return self.thread_event.is_set()

    def wait_sync(self, timeout: float | None = None) -> bool:
        return self.thread_event.wait(timeout)

    def set(self) -> None:
        with self._lock:
            if self.thread_event.is_set():
                return
            self.thread_event.set()
            for loop, event in self.awaits_events:
                if loop.is_running():
                    loop.call_soon_threadsafe(event.set)
            self.awaits_events.clear()

    def _add_awaits(self, loop: asyncio.AbstractEventLoop, event: asyncio.Event) -> None:
        with self._lock:
            is_set = self.thread_event.is_set()
            if is_set:
                loop.call_soon_threadsafe(event.set)
            else:
                self.awaits_events.append((loop, event))

    async def wait(self) -> bool:
        loop = asyncio.get_running_loop()
        event = asyncio.Event()
        self._add_awaits(loop, event)
        return await event.wait()

    async def wait_for(self, timeout: float) -> bool:
        if timeout is None or timeout <= 0.0:
            await self.wait()
        else:
            return await asyncio.wait_for(self.wait(), timeout)

    def clear(self) -> None:
        with self._lock:
            self.thread_event.clear()
            for loop, event in self.awaits_events:
                event.clear()
            self.awaits_events.clear()


async def ensure_tasks_done_or_cancel(
        *fts: asyncio.Task | Coroutine,
        timeout: float | None = None,
        cancel: Callable[[], Coroutine[None, None, Any]] | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
) -> List:
    """实现一个通用函数, 确保所有的 tasks or coroutines 必然会被执行或者 cancel """
    gathering = []
    for task in fts:
        if isinstance(task, asyncio.Task):
            gathering.append(task)
        else:
            task = asyncio.ensure_future(task, loop=loop)
            gathering.append(task)
    if len(gathering) == 0:
        return []

    gathered = asyncio.gather(*gathering, return_exceptions=False)
    ordered_tasks = [gathered]
    if cancel:
        ordered_tasks.append(asyncio.create_task(cancel()))

    done, pending = await asyncio.wait(ordered_tasks, return_when=asyncio.FIRST_COMPLETED, timeout=timeout)
    for t in pending:
        # 不 cancel gathered.
        if t is not gathered:
            t.cancel()
    if gathered in done:
        # 先正常完成了所有任务.
        # 确保任何异常被抛出.
        unfinished = []
        for t in gathering:
            if not t.done():
                t.cancel()
                unfinished.append(t)
        if len(unfinished) > 0:
            await asyncio.wait(unfinished, return_when=asyncio.ALL_COMPLETED)
        return list(gathered.result())
    elif len(done) == 0:
        for t in gathering:
            t.cancel()
        await gathered
        raise asyncio.TimeoutError(f'Timed out waiting for {timeout}')
    else:
        await gathered
        raise asyncio.CancelledError


class TreeNotify:
    """
    树形的 notify 工具. 当自己和自己所有的孩子都 set 时, 自己是 is_set() 的.
    当自己或自己的任何孩子 clear 时, 自己是 not is_set() 的.
    """

    def __init__(self, name: str, parent: Optional[Self] = None) -> None:
        self.event = ThreadSafeEvent()
        self._name = name
        self._parent = parent
        self._unset_names = {name}

    def set(self):
        if self._name in self._unset_names:
            self._unset_names.remove(self._name)
        if self._parent is not None:
            self._parent._set_by_child(self._name)
        self._check_self_event()

    def child(self, name: str) -> Self:
        self._unset_names.add(name)
        self.event.clear()
        return TreeNotify(name, self)

    def is_set(self) -> bool:
        return self.event.is_set()

    def is_self_set(self) -> bool:
        return self._name not in self._unset_names

    async def wait(self) -> None:
        await self.event.wait()

    def wait_sync(self, timeout: float | None = None) -> bool:
        return self.event.wait_sync(timeout)

    def clear(self) -> None:
        self._unset_names.add(self._name)
        if self._parent is not None:
            self._parent._clear_by_child(self._name)
        self._check_self_event()

    def _set_by_child(self, name: str):
        if name != self._name and name in self._unset_names:
            self._unset_names.remove(name)
            self._check_self_event()

    def _clear_by_child(self, name: str):
        if name != self._name:
            self._unset_names.add(name)
        self._check_self_event()

    def _check_self_event(self) -> None:
        if len(self._unset_names) == 0:
            self.event.set()
        elif self.event.is_set():
            self.event.clear()
