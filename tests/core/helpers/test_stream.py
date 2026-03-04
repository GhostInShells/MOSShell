import asyncio
import threading

from ghoshell_moss.core.helpers.stream import (
    create_sender_and_receiver,
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
