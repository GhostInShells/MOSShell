import asyncio
import threading
import time
import pytest
import contextlib


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
    baz_future = asyncio.create_task(baz())
    _done, pending = await asyncio.wait(
        [asyncio.ensure_future(t) for t in [foo(), bar(), baz_future]],
        return_when=asyncio.FIRST_COMPLETED,
    )
    if baz_future in _done:
        for t in pending:
            t.cancel()
        with pytest.raises(asyncio.CancelledError):
            raise asyncio.CancelledError


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


@pytest.mark.asyncio
async def test_task_await_in_tasks():
    async def foo():
        await asyncio.sleep(0.1)
        return 123

    task = asyncio.create_task(foo())

    async def cancel_foo():
        task.cancel()
        return 123

    async def wait_foo():
        return await task

    cancel_task = asyncio.create_task(cancel_foo())
    wait_task = asyncio.create_task(wait_foo())

    with pytest.raises(asyncio.CancelledError):
        await asyncio.gather(task, cancel_task, wait_task)

    assert task.done()
    assert wait_task.done()
    assert cancel_task.done()
    assert await cancel_task == 123


@pytest.mark.asyncio
async def test_first_exception_in_gathering():
    async def foo(a: int):
        await asyncio.sleep(0.01 * a)
        raise ValueError(a)

    e = None
    try:
        await asyncio.gather(foo(1), foo(2), foo(3), foo(4))
    except ValueError as err:
        e = err
    assert e is not None
    assert str(e) == str(ValueError(1))

    # all exceptions 返回.
    e = None
    try:
        done = await asyncio.gather(foo(1), foo(2), foo(3), foo(4), return_exceptions=True)
        i = 0
        for t in done:
            i += 1
            assert str(t) == str(ValueError(i))
    except ValueError as err:
        e = err
    assert e is None


@pytest.mark.asyncio
async def test_run_until_complete_in_loop():
    event = asyncio.Event()

    def foo():
        event.set()

    loop = asyncio.get_running_loop()
    loop.call_soon(foo)
    await event.wait()


@pytest.mark.asyncio
async def test_catch_cancelled_error():
    async def foo():
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            pass

    task = asyncio.create_task(foo())
    task.cancel()
    # 测试不会抛出异常. 实际上仍然会抛出.
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_asyncio_call_soon():
    event = asyncio.Event()
    done = []

    async def foo():
        await asyncio.sleep(0.05)
        done.append(1)
        event.set()

    loop = asyncio.get_running_loop()
    _ = loop.create_task(foo())
    await event.wait()
    assert done[0] == 1


@pytest.mark.asyncio
async def test_asyncio_future():
    fut = asyncio.Future()
    assert not fut.done()
    fut.set_result(123)
    assert fut.result() == 123
    assert fut.done()


@pytest.mark.asyncio
async def test_future_in_diff_thread():
    import threading

    fut = asyncio.Future()
    done = []

    def wait_fut():
        async def wait_futu():
            await fut
            assert fut.result() == 123
            done.append(1)

        asyncio.run(wait_futu())

    def set_fut_result():
        fut.set_result(123)
        time.sleep(0.3)

    t1 = threading.Thread(target=wait_fut)
    t2 = threading.Thread(target=wait_fut)
    t3 = threading.Thread(target=set_fut_result)
    t3.start()
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    t3.join()
    assert fut.result() == 123
    assert len(done) == 2


@pytest.mark.asyncio
async def test_future_cancel_is_done():
    fut = asyncio.Future()
    fut.cancel()
    assert fut.cancelled()
    assert fut.done()


@pytest.mark.asyncio
async def test_future_exception_is_done():
    fut = asyncio.Future()
    e = Exception("hello")
    fut.set_exception(e)
    assert fut.done()


@pytest.mark.asyncio
async def test_future_set_result_is_done():
    fut = asyncio.Future()
    fut.set_result(123)
    assert fut.done()


@pytest.mark.asyncio
async def test_future_cancel():
    fut = asyncio.Future()
    fut.cancel()
    with pytest.raises(asyncio.CancelledError):
        await fut


@pytest.mark.asyncio
async def test_future_call_done_callback_before_done():
    fut = asyncio.Future()
    check = []

    def done_callback(_fut: asyncio.Future):
        assert _fut.result() == 123
        if _fut.done():
            check.append(1)

    fut.add_done_callback(done_callback)
    fut.set_result(123)
    r = await fut
    assert r == 123

    assert len(check) == 0


def test_cancel_future_in_other_thread():
    from queue import Queue

    future_queue = Queue()
    done = []

    async def thread_a_produce_future():
        future = asyncio.Future()
        future_queue.put_nowait(future)
        try:
            await future
        except asyncio.CancelledError:
            done.append(1)
        except Exception:
            done.append(0)

    def thread_a_main():
        asyncio.run(thread_a_produce_future())
        done.append(1)

    def thread_b_consume_future():
        try:
            future: asyncio.Future = future_queue.get()
            future.get_loop().call_soon_threadsafe(future.cancel)
        except Exception:
            done.append(0)

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
        future = asyncio.Future()
        future_queue.put_nowait(future)
        try:
            await future
        except Exception as e:
            done.append(str(e))

    def thread_a_main():
        asyncio.run(thread_a_produce_future())

    def thread_b_consume_future():
        future: asyncio.Future = future_queue.get()
        future.get_loop().call_soon_threadsafe(future.set_exception, RuntimeError("hello"))

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
        future = asyncio.Future()
        future_queue.put_nowait(future)
        r = await future
        assert r == 123
        done.append(r)

    def thread_a_main():
        asyncio.run(thread_a_produce_future())

    def thread_b_consume_future():
        future: asyncio.Future = future_queue.get()
        future.get_loop().call_soon_threadsafe(future.set_result, 123)

    t_a = threading.Thread(target=thread_a_main)
    t_b = threading.Thread(target=thread_b_consume_future)
    t_a.start()
    t_b.start()
    t_a.join()
    t_b.join()
    assert done[0] == 123


@pytest.mark.asyncio
async def test_future_result_and_exception():
    future = asyncio.Future()

    def foo():
        exp = Exception("hello")
        future.set_exception(exp)

    foo()
    assert future.exception() is not None
    with pytest.raises(Exception, match="hello"):
        # result will always raise exception.
        future.result()


@pytest.mark.asyncio
async def test_async_iterable():
    from collections.abc import AsyncIterable

    async def foo() -> AsyncIterable[int]:
        for i in range(10):
            yield i

    arr = []
    async for k in foo():
        arr.append(k)

    assert len(arr) == 10
    r = foo()
    async for k in r:
        arr.append(k)
    assert len(arr) == 20


@pytest.mark.asyncio
async def test_async_iterable_item():
    from collections.abc import AsyncIterable

    class Int(int):
        pass

    async def foo(num: int) -> AsyncIterable[Int]:
        for i in range(num):
            yield Int(i)

    arr = []
    async for k in foo(10):
        arr.append(k)

    assert len(arr) == 10
    r = foo(5)
    async for k in r:
        arr.append(k)
    assert len(arr) == 15

    items = foo(10)
    arr = []
    async for k in items:
        arr.append(k)
    assert len(arr) == 10


@pytest.mark.asyncio
async def test_wait_for_exception():
    exp = []

    async def foo():
        try:
            await asyncio.sleep(1)
        except Exception as e:
            exp.append(e)

    catch = False
    foo_task = asyncio.ensure_future(foo())
    try:
        await asyncio.wait_for(foo_task, 0.01)
    except asyncio.TimeoutError:
        catch = True

    with pytest.raises(asyncio.CancelledError):
        await foo_task
    assert catch
    assert len(exp) == 0


@pytest.mark.asyncio
async def test_async_context_manager():
    log = []

    @contextlib.asynccontextmanager
    async def foo():
        idx = len(log)
        log.append("start_%s" % idx)
        yield
        log.append("end_%s" % idx)

    async with contextlib.AsyncExitStack() as stack:
        await stack.enter_async_context(foo())
        await stack.enter_async_context(foo())
        await stack.enter_async_context(foo())
        await stack.enter_async_context(foo())
        await stack.enter_async_context(foo())

    assert len(log) == 10


@pytest.mark.asyncio
async def test_async_iterable():
    from typing import AsyncIterable

    async def generator_method() -> AsyncIterable[int]:
        for i in range(10):
            yield i

    result = []
    async for k in generator_method():
        result.append(k)
    assert len(result) == 10


@pytest.mark.asyncio
async def test_raise_in_wait():
    async def foo():
        await asyncio.sleep(0.05)
        raise ValueError()

    async def bar():
        await asyncio.sleep(0.1)
        return 123

    t1 = asyncio.create_task(foo())
    t2 = asyncio.create_task(bar())

    done, pending = await asyncio.wait([t1, t2], return_when=asyncio.ALL_COMPLETED)
    # 抛出异常仍然会等待到结束.
    assert len(pending) == 0

    t3 = asyncio.create_task(bar())
    t4 = asyncio.create_task(bar())
    done, pending = await asyncio.wait([t3, t4], return_when=asyncio.FIRST_EXCEPTION)
    # 不抛出异常, 仍然是等待全部结束.
    assert len(pending) == 0


@pytest.mark.asyncio
async def test_gather_in_order():
    order = []

    async def foo():
        await asyncio.sleep(0.05)
        order.append("foo")

    async def bar():
        order.append("bar")

    await asyncio.gather(foo(), bar())
    assert order == ["bar", "foo"]


@pytest.mark.asyncio
async def test_task_done_with_callback():
    async def foo():
        return 123

    order = []

    def done(t):
        order.append(t)

    task = asyncio.create_task(foo())
    task.add_done_callback(done)
    await task
    assert len(order) == 1
    assert order[0].done()


@pytest.mark.asyncio
async def test_task_wait_in_many():
    async def foo():
        return 123

    task = asyncio.create_task(foo())

    order = []

    async def wait():
        order.append(await task)

    _ = await asyncio.gather(wait(), wait(), wait(), wait(), wait())
    assert len(order) == 5
    for t in order:
        assert t == 123


@pytest.mark.asyncio
async def test_async_iterator_generator_exit():
    class Sensor:
        def __init__(self, m: int):
            self.i = 0
            self.max = m

        async def aclose(self):
            self.i += 1

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.i < self.max:
                i = self.i
                self.i += 1
                return i
            else:
                raise StopAsyncIteration

    s = Sensor(3)
    async for val in s:
        pass
    assert s.i == 3

    s = Sensor(3)
    async for val in s:
        if val == 1:
            assert s.i == 2
            break
    assert s.i == 2

    s = Sensor(3)
    async with contextlib.aclosing(s):
        async for val in s:
            break
        assert s.i == 1
    assert s.i == 2
