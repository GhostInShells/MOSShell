import asyncio
import time


def test_to_thread():
    order = []

    def foo():
        time.sleep(0.1)
        order.append("foo")

    async def bar():
        order.append("bar")

    async def main():
        t1 = asyncio.to_thread(foo)
        t2 = asyncio.create_task(bar())
        await asyncio.wait([t1, t2], return_when=asyncio.ALL_COMPLETED)

    asyncio.run(main())
    assert order == ["bar", "foo"]
