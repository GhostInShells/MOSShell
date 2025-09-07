import threading

import anyio
from anyio import to_thread


def test_thread_event():
    e = threading.Event()
    order = []

    def setter():
        order.append('setter')
        e.set()

    async def waiter():
        await to_thread.run_sync(e.wait)
        order.append('waiter')

    def main() -> None:
        anyio.run(waiter)

    t1 = threading.Thread(target=setter)
    t2 = threading.Thread(target=main)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert order == ['setter', 'waiter']
