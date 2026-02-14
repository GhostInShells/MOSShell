import asyncio
import threading

from ghoshell_moss.core.helpers.stream import (
    create_thread_safe_stream,
)


def test_thread_send_async_receive():
    content = "hello world"
    done = []
    sender, receiver = create_thread_safe_stream()

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
    sender, receiver = create_thread_safe_stream()

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
