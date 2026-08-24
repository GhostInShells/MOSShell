import asyncio
import functools
import logging
from typing import Callable, Coroutine

__all__ = ['SimpleTaskGroup']


class SimpleTaskGroup:
    """
    实现一个极简的 task group.

    add_coroutine 用 asyncio.ensure_future 直接包用户协程, 不额外套一层 _wrap.
    这样任务天生持有用户协程, 即使取消落在首次调度之前也不产生 "never awaited" 孤儿,
    因而 add 之后无需先 yield 再 close. ignore_error 在 done-callback 里处理.

    on_exception 让持有者感知每个任务的非取消异常对象; 政策对异常是"吞"还是"关组"与感知正交,
    两者都可触发 on_exception. 任务被取消不算异常, 不触发 on_exception, 也不让组关闭.
    规定: 取消不影响生命周期, 只有绑定任务(bind)的非取消异常才关组.

    为了收紧取消语义, 这里的 policy 与 stdlib TaskGroup 不同: TaskGroup 把"子任务自取消"
    视为全局取消, 这里视为普通结束.

    为什么不用 anyio 呢? 因为太重了. 退出逻辑和这里的设想不一致.
    为什么不用 python 自带 asyncio TaskGroup ? 因为默认支持 python 3.10.
    """

    def __init__(
            self,
            *,
            logger: logging.Logger | None = None,
            on_exception: Callable[[BaseException], None] | None = None,
    ) -> None:
        self.tasks: set[asyncio.Future] = set()
        self._closed = False
        self._logger = logger or logging.getLogger(__name__)
        self._on_exception = on_exception

    def clear(self) -> None:
        """取消当前所有未完成的任务, 清空集合. 不改变 closed 状态. """
        tasks = list(self.tasks)
        self.tasks.clear()
        for t in tasks:
            if not t.done():
                t.cancel()

    def add_task(self, task: asyncio.Task) -> None:
        if self._closed:
            task.cancel()
            return
        self.tasks.add(task)
        task.add_done_callback(self._on_task_done)

    def add_coroutine(self, cor: Coroutine, ignore_error: bool = True) -> asyncio.Future:
        task = asyncio.ensure_future(cor)
        if self._closed:
            task.cancel()
            return task
        self.tasks.add(task)
        task.add_done_callback(functools.partial(self._on_task_done, ignore_error=ignore_error))
        return task

    def _on_task_done(self, task: asyncio.Future, *, ignore_error: bool = False) -> None:
        self.tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is None:
            return
        if ignore_error:
            self._logger.error(exc)
        else:
            self.close()
        if self._on_exception is not None:
            self._on_exception(exc)

    def close(self) -> None:
        """永久关闭: 取消所有剩余任务, 之后 add_task/add_coroutine 会立即 cancel. """
        if self._closed:
            return
        self._closed = True
        self.clear()

    async def aclose(self) -> None:
        tasks = list(self.tasks)
        self.close()
        pending = [t for t in tasks if not t.done()]
        if len(pending) > 0:
            await asyncio.gather(*pending, return_exceptions=True)

    async def __aenter__(self) -> 'SimpleTaskGroup':
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.aclose()
