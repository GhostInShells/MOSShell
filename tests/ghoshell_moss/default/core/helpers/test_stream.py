import asyncio
import threading
from collections import deque

from ghoshell_moss.core.helpers.asyncio_utils import ThreadSafeEvent
from ghoshell_moss.core.helpers.stream import (
    create_sender_and_receiver,
    ThreadSafeStreamReceiver,
    ThreadSafeStreamSender,
)
import pytest


@pytest.mark.asyncio
async def test_sender_and_receiver_with_sleep():
    content = "hello world"
    done = []
    sender, receiver = create_sender_and_receiver()

    async def sending():
        with sender:
            for char in content:
                await asyncio.sleep(0.01)
                sender.append(char)

    async def receiving():
        async with receiver:
            async for char in receiver:
                await asyncio.sleep(0.01)
                done.append(char)

    t1 = asyncio.create_task(sending())
    t2 = asyncio.create_task(receiving())
    await asyncio.gather(t1, t2)
    assert len(done) == len(content)


def test_thread_send_async_receive():
    content = "hello world"
    done = []
    sender, receiver = create_sender_and_receiver()

    def sending():
        with sender:
            for char in content:
                sender.append(char)

    async def receiving():
        try:
            buffer = ""
            async with receiver:
                async for char in receiver:
                    buffer += char
            done.append(buffer)
        except Exception as e:
            done.append(str(e))

    def sync_receiving():
        asyncio.run(receiving())

    t1 = threading.Thread(target=sending)
    t2 = threading.Thread(target=sync_receiving)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert content == done[0]


def test_thread_send_and_receive():
    content = "hello world"
    done = []
    sender, receiver = create_sender_and_receiver()

    def sending():
        with sender:
            for char in content:
                sender.append(char)

    def sync_receiving():
        buffer = ""
        with receiver:
            for char in receiver:
                buffer += char
        done.append(buffer)

    t1 = threading.Thread(target=sending)
    t2 = threading.Thread(target=sync_receiving)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert content == done[0]


@pytest.mark.asyncio
async def test_fractal_stream():
    sender1, receiver1 = create_sender_and_receiver()

    async def sender1_func():
        nonlocal sender1
        with sender1:
            for i in "hello":
                await asyncio.sleep(0.01)
                sender1.append(i)

    sender2, receiver2 = create_sender_and_receiver()

    async def sender2_func():
        nonlocal sender2, receiver1
        with sender2:
            async for i in receiver1:
                await asyncio.sleep(0.01)
                sender2.append(i)

    got = []

    async def consume2():
        async for char in receiver2:
            got.append(char)

    await asyncio.gather(sender1_func(), sender2_func(), consume2())

    assert len(got) == len("hello")


@pytest.mark.asyncio
async def test_late_consumer_gets_text_in_one_chunk():
    """流在消费者到达之前就已经完备: 一次性交付, 不逐 item 走一遍."""
    sender, receiver = create_sender_and_receiver(merge="".join)
    with sender:
        for char in "hello world":
            sender.append(char)

    got = [chunk async for chunk in receiver]
    assert got == ["hello world"]


@pytest.mark.asyncio
async def test_late_consumer_keeps_one_to_one_without_merge():
    """没有装配 merge 的流, 行为与过去一致: N 个 item 逐一交付."""
    sender, receiver = create_sender_and_receiver()
    with sender:
        for char in "hello world":
            sender.append(char)

    got = [chunk async for chunk in receiver]
    assert got == list("hello world")


@pytest.mark.asyncio
async def test_streaming_consumer_keeps_one_to_one():
    """消费者先到, 生产还在进行 —— 生成与消费重叠, 塌缩不生效."""
    sender, receiver = create_sender_and_receiver(merge="".join)
    sender.append("he")
    assert await anext(receiver) == "he"
    with sender:
        sender.append("llo")
    assert await anext(receiver) == "llo"
    with pytest.raises(StopAsyncIteration):
        await anext(receiver)


@pytest.mark.asyncio
async def test_collapse_does_not_swallow_failure():
    """塌缩只合并普通 item, 失败仍按原语义抛出."""
    sender, receiver = create_sender_and_receiver(merge="".join)
    sender.append("he")
    sender.append("llo")
    sender.fail(ValueError("boom"))

    assert await anext(receiver) == "hello"
    with pytest.raises(ValueError):
        await anext(receiver)


class _AppendOnClear(ThreadSafeEvent):
    """把 append 精确投放到 receiver "查到空队列" 与 clear 之间的窗口里."""

    def __init__(self):
        super().__init__()
        self.on_clear = None

    def clear(self) -> None:
        callback, self.on_clear = self.on_clear, None
        if callback is not None:
            callback()
        super().clear()


@pytest.mark.asyncio
async def test_append_during_clear_is_not_lost():
    """落在 clear 之前的 append 不能被抹掉, 否则消费者会一直挂在这里."""
    added = _AppendOnClear()
    completed = ThreadSafeEvent()
    queue = deque()
    sender = ThreadSafeStreamSender(added, completed, queue)
    receiver = ThreadSafeStreamReceiver(added, completed, queue)
    added.on_clear = lambda: sender.append("x")

    assert await asyncio.wait_for(anext(receiver), timeout=1.0) == "x"
