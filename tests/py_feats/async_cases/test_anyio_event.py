import threading

import anyio
from anyio import create_memory_object_stream
from anyio import to_thread
import pytest


@pytest.mark.asyncio
async def test_anyio_stream():
    sender, receiver = create_memory_object_stream(max_buffer_size=11)
    with sender:
        for i in range(10):
            await sender.send(1)

    receiver.close()
    got = []
    with pytest.raises(anyio.ClosedResourceError):
        async for v in receiver:
            got.append(v)
    assert len(got) == 0


def test_thread_event():
    e = threading.Event()
    order = []

    def setter():
        order.append("setter")
        e.set()

    async def waiter():
        await to_thread.run_sync(e.wait)
        order.append("waiter")

    def main() -> None:
        anyio.run(waiter)

    t1 = threading.Thread(target=setter)
    t2 = threading.Thread(target=main)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert order == ["setter", "waiter"]
