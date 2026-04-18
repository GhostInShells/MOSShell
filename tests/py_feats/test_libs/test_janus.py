import threading
import janus
import asyncio
import uvloop


def test_janus_empty():
    queue = janus.Queue()
    queue.sync_q.put_nowait(1)
    assert not queue.sync_q.empty()
    assert not queue.async_q.empty()

    assert queue.sync_q.get_nowait() == 1
    assert queue.sync_q.empty()
    assert queue.async_q.empty()


def test_janus_async_q_in_differ_thread():
    queue = janus.Queue()
    got = []

    async def producer():
        # 不能两个queue 是 async
        for i in range(10):
            queue.sync_q.put_nowait(i)
        queue.sync_q.put_nowait(None)

    async def consumer():
        while True:
            item = await queue.async_q.get()
            if item is None:
                break
            got.append(item)

    def _producer_thread():
        asyncio.set_event_loop(uvloop.new_event_loop())
        asyncio.run(producer())

    def _consumer_thread():
        asyncio.set_event_loop(uvloop.new_event_loop())
        asyncio.run(consumer())

    t1 = threading.Thread(target=_producer_thread)
    t2 = threading.Thread(target=_consumer_thread)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert len(got) == 10
