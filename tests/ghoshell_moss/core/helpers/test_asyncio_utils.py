import asyncio
import threading
from threading import Thread

import pytest

from ghoshell_moss.core.helpers.asyncio_utils import (
    ThreadSafeEvent,
    ThreadSafeFuture,
    TreeNotify,
    ensure_tasks_done_or_cancel,
)


def test_event_set_and_wait():
    event = ThreadSafeEvent()

    done = []

    def set_thread():
        event.set()
        done.append(1)

    def wait_sync_thread():
        event.wait_sync()
        done.append(1)

    async def wait_async():
        await event.wait()
        done.append(1)

    def wait_async_thread():
        asyncio.run(wait_async())

    t = [Thread(target=set_thread)]
    for i in range(5):
        t.append(Thread(target=wait_sync_thread))
        t.append(Thread(target=wait_async_thread))

    for i in t:
        i.start()
    for i in t:
        i.join()

    assert len(done) == 11


@pytest.mark.asyncio
async def test_event_set_and_wait_in_same_loop():
    event = ThreadSafeEvent()
    assert not event.is_set()
    event.set()
    assert event.is_set()
    await event.wait()
    event.clear()
    assert not event.is_set()
    try:
        await event.wait_for(0.01)
    except asyncio.TimeoutError:
        pass


def test_event_set_and_wait_in_defer_loop():
    event = ThreadSafeEvent()

    async def call_event_clear(_e: ThreadSafeEvent):
        # 等待 1
        await _e.wait()
        # 清空 2
        _e.clear()
        # 等待设置 3
        await _e.wait()
        # 清空 4
        _e.clear()

    def _call_event_clear():
        asyncio.run(call_event_clear(event))

    async def call_event_set(_e: ThreadSafeEvent):
        # 设置 1
        _e.set()
        # 等待清空2
        while _e.is_set():
            await asyncio.sleep(0.01)
        # 设置3
        _e.set()

    def _call_event_set():
        asyncio.run(call_event_set(event))

    t1 = Thread(target=_call_event_clear)
    t2 = Thread(target=_call_event_set)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    # 最终结果是清空4
    assert not event.is_set()


@pytest.mark.asyncio
async def test_wait_timeout():
    event = ThreadSafeEvent()
    with pytest.raises(asyncio.TimeoutError):
        await event.wait_for(0.2)

    event = ThreadSafeEvent()
    event.set()
    await event.wait_for(10)
    assert event.is_set()


@pytest.mark.asyncio
async def test_wait_done_or_cancel():
    canceled = []

    async def foo() -> int:
        try:
            await asyncio.sleep(0.1)
            return 123
        except asyncio.CancelledError:
            canceled.append(1)

    async def bar() -> int:
        try:
            await asyncio.sleep(0.1)
            return 456
        except asyncio.CancelledError:
            canceled.append(2)

    async def baz() -> int:
        raise RuntimeError("failed")

    # 空执行无影响.
    assert await ensure_tasks_done_or_cancel() == []

    # 正常调用.
    assert await ensure_tasks_done_or_cancel(foo(), bar()) == [123, 456]
    # cancel 事件没触发.
    cancel = asyncio.Event()
    assert await ensure_tasks_done_or_cancel(foo(), bar(), cancel=cancel.wait) == [123, 456]
    # cancel 触发了.
    cancel.set()
    with pytest.raises(asyncio.CancelledError):
        assert await ensure_tasks_done_or_cancel(foo(), bar(), cancel=cancel.wait) == []

    # 设计超时.
    with pytest.raises(asyncio.TimeoutError):
        await ensure_tasks_done_or_cancel(foo(), bar(), timeout=0.01)
    assert len(canceled) == 2
    canceled.clear()

    # 确保就算 baz 抛出了 runtime error, foo 和 bar 也仍然被正确 cancel 了.
    with pytest.raises(RuntimeError):
        await ensure_tasks_done_or_cancel(foo(), bar(), baz())
    assert len(canceled) == 2


@pytest.mark.asyncio
async def test_notify_tree_baseline():
    order = []

    async def foo() -> None:
        notify = TreeNotify("foo")
        t1 = asyncio.create_task(bar(notify.child("bar")))
        t2 = asyncio.create_task(baz(notify.child("baz")))
        notify.set()
        await notify.event.wait()
        await asyncio.sleep(0.1)
        order.append("foo")

        #  准备开始第二轮.
        await asyncio.gather(t1, t2)
        child = notify.child("none")
        assert not child.event.is_set()
        # 自己的 event 也不是 set 了.
        assert not notify.event.is_set()
        # 自己 set 是没用的.
        notify.set()
        assert not notify.event.is_set()
        child.set()
        assert notify.event.is_set()

    async def bar(notify: TreeNotify) -> None:
        notify.set()
        await notify.event.wait()
        order.append("bar")

    async def baz(notify: TreeNotify) -> None:
        notify.set()
        await notify.event.wait()
        order.append("baz")

    await foo()
    assert order[2] == "foo"


def test_wait_the_event_timeout():
    event = ThreadSafeEvent()
    errors = []

    async def main():
        try:
            await asyncio.wait_for(event.wait(), timeout=0.2)
        except asyncio.TimeoutError as e:
            errors.append(e)

    asyncio.run(main())

    assert len(errors) == 1
    assert isinstance(errors[0], asyncio.TimeoutError)


def test_cancel_future_in_other_thread():
    from queue import Queue

    future_queue = Queue()
    done = []

    async def thread_a_produce_future():
        future = ThreadSafeFuture()
        future_queue.put_nowait(future)
        try:
            await future
        except asyncio.CancelledError:
            done.append(1)

    def thread_a_main():
        asyncio.run(thread_a_produce_future())

    def thread_b_consume_future():
        future: ThreadSafeFuture = future_queue.get()
        future.cancel()

    t_a = threading.Thread(target=thread_a_main)
    t_b = threading.Thread(target=thread_b_consume_future)
    t_a.start()
    t_b.start()
    t_a.join()
    t_b.join()
    assert done[0] == 1


def test_set_exp_future_in_other_thread():
    from queue import Queue

    future_queue = Queue()
    done = []

    async def thread_a_produce_future():
        future = ThreadSafeFuture()
        future_queue.put_nowait(future)
        try:
            await future
        except Exception as e:
            done.append(str(e))

    def thread_a_main():
        asyncio.run(thread_a_produce_future())

    def thread_b_consume_future():
        future: asyncio.Future = future_queue.get()
        future.set_exception(RuntimeError("hello"))

    t_a = threading.Thread(target=thread_a_main)
    t_b = threading.Thread(target=thread_b_consume_future)
    t_a.start()
    t_b.start()
    t_a.join()
    t_b.join()
    assert done[0] == "hello"


def test_set_result_future_in_other_thread():
    from queue import Queue

    future_queue = Queue()
    done = []

    async def thread_a_produce_future():
        future = ThreadSafeFuture()
        future_queue.put_nowait(future)
        r = await future
        assert r == 123
        done.append(r)

    def thread_a_main():
        asyncio.run(thread_a_produce_future())

    def thread_b_consume_future():
        future: asyncio.Future = future_queue.get()
        future.set_result(123)

    t_a = threading.Thread(target=thread_a_main)
    t_b = threading.Thread(target=thread_b_consume_future)
    t_a.start()
    t_b.start()
    t_a.join()
    t_b.join()
    assert done[0] == 123
