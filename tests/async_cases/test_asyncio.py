from typing import Awaitable
import asyncio
import time
import pytest


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
        await asyncio.gather(t1, t2)

    asyncio.run(main())
    assert order == ["bar", "foo"]
    # assert isinstance(bar(), Awaitable)


@pytest.mark.asyncio
async def test_gather():
    cancelled = []
    done = []

    async def foo():
        try:
            await asyncio.sleep(0.1)
            return "foo"
        except asyncio.CancelledError:
            cancelled.append("foo")
        finally:
            done.append("foo")

    async def bar():
        try:
            await asyncio.sleep(0.1)
            return "bar"
        except asyncio.CancelledError:
            cancelled.append("bar")
        finally:
            done.append("bar")

    async def baz():
        raise asyncio.CancelledError()

    # test 1
    result = await asyncio.gather(asyncio.shield(foo()), asyncio.shield(bar()))
    assert result == ["foo", "bar"]
    assert done == ["foo", "bar"]
    done.clear()
    cancelled.clear()

    # test 2: test cancel
    result = None
    gathered = asyncio.gather(asyncio.shield(foo()), asyncio.shield(bar()))
    gathered.cancel()
    with pytest.raises(asyncio.CancelledError):
        result = await gathered
    assert result is None
    done.clear()
    cancelled.clear()

    # test 3:
    result = None
    with pytest.raises(asyncio.CancelledError):
        result = await asyncio.gather(asyncio.shield(foo()), asyncio.shield(bar()), baz())

    assert result is None
    # the cancellation totally ignored
    assert done == []

    # test 4:
    with pytest.raises(asyncio.CancelledError):
        baz_future = asyncio.create_task(baz())
        _done, pending = await asyncio.wait(
            [asyncio.ensure_future(t) for t in [foo(), bar(), baz_future]],
            return_when=asyncio.FIRST_COMPLETED,
        )
        if baz_future in _done:
            for t in pending:
                t.cancel()
            raise asyncio.CancelledError()


@pytest.mark.asyncio
async def test_wait_for():
    event = asyncio.Event()

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(event.wait(), 0.01)

    async def foo():
        return 123

    assert await asyncio.wait_for(foo(), 0.01) == 123


@pytest.mark.asyncio
async def test_call_soon_thread_safe():
    queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    loop.call_soon_threadsafe(queue.put_nowait, 1)
    t = await queue.get()
    assert t == 1
