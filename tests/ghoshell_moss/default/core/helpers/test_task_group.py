import asyncio
import gc
import warnings

import pytest

from ghoshell_moss.core.helpers.task_group import SimpleTaskGroup


@pytest.mark.asyncio
async def test_close_cancels_pending_tasks():
    # close() 应当取消组里尚未完成的任务, 让它们收到 CancelledError.
    cancelled = []
    group = SimpleTaskGroup()

    async def worker(token):
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.append(token)
            raise

    task1 = group.add_coroutine(worker(1))
    task2 = group.add_coroutine(worker(2))
    await asyncio.sleep(0)

    group.close()
    await asyncio.gather(task1, task2, return_exceptions=True)

    assert sorted(cancelled) == [1, 2]


@pytest.mark.asyncio
async def test_as_context_manager_closes_on_exit():
    # 退出 async with 时 aclose() 应取消所有未完成任务.
    cancelled = []
    group = SimpleTaskGroup()

    async def worker(token):
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.append(token)
            raise

    async with group:
        group.add_coroutine(worker(1))
        group.add_coroutine(worker(2))
        await asyncio.sleep(0)

    assert sorted(cancelled) == [1, 2]


@pytest.mark.asyncio
async def test_errored_task_closes_group_and_cancels_siblings():
    # 一个任务异常 (ignore_error=False) 应关掉整组, 连带取消其余任务.
    cancelled = []
    group = SimpleTaskGroup()

    async def worker(token):
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.append(token)
            raise

    async def boom():
        raise RuntimeError("boom")

    good = group.add_coroutine(worker(1))
    group.add_coroutine(boom(), ignore_error=False)
    await asyncio.sleep(0)
    await asyncio.gather(good, return_exceptions=True)

    assert cancelled == [1]


@pytest.mark.asyncio
async def test_ignore_error_keeps_group_open():
    # ignore_error=True 的任务异常被吞掉, 组保持打开, 后续任务仍能完成.
    result = []
    group = SimpleTaskGroup()

    async def ok_worker():
        await asyncio.sleep(0.01)
        result.append("ok")

    async def crash():
        raise ValueError("ignored")

    group.add_coroutine(crash(), ignore_error=True)
    ok = group.add_coroutine(ok_worker())
    await ok

    assert result == ["ok"]


@pytest.mark.asyncio
async def test_add_coroutine_without_aenter_completes():
    # add_coroutine 不依赖 __aenter__ 建立事件循环绑定.
    group = SimpleTaskGroup()

    async def compute():
        await asyncio.sleep(0)
        return 42

    task = group.add_coroutine(compute())

    assert await task == 42


@pytest.mark.asyncio
async def test_add_task_error_closes_group():
    # add_task 注册一条会异常的任务, 应同样触发整组关闭 (与 add_coroutine 同一套规则).
    cancelled = []
    group = SimpleTaskGroup()

    async def good():
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.append(1)
            raise

    async def boom():
        raise RuntimeError("boom")

    good_task = group.add_coroutine(good())
    group.add_task(asyncio.create_task(boom()))
    await asyncio.sleep(0)
    await asyncio.gather(good_task, return_exceptions=True)

    assert cancelled == [1]


@pytest.mark.asyncio
async def test_clear_cancels_pending_but_group_stays_open():
    # clear() 取消当前任务, 但不封闭组: 之后还能再添加并正常完成.
    cancelled = []
    result = []
    group = SimpleTaskGroup()

    async def blocker():
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.append(1)
            raise

    async def ok():
        await asyncio.sleep(0.01)
        result.append("ok")

    t1 = group.add_coroutine(blocker())
    await asyncio.sleep(0)
    group.clear()
    await asyncio.gather(t1, return_exceptions=True)
    assert cancelled == [1]

    t2 = group.add_coroutine(ok())
    await t2
    assert result == ["ok"]


@pytest.mark.asyncio
async def test_close_is_idempotent_and_future_adds_cancelled():
    # close() 可重复调用; close 之后 add_coroutine 的产物立即被取消, 组保持终态.
    group = SimpleTaskGroup()
    group.close()
    group.close()

    async def worker():
        await asyncio.sleep(30)

    task = group.add_coroutine(worker())
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_errored_task_exception_retrievable():
    # ignore_error=False 的任务异常没有被吞掉, 调用方可重新拿到它 (不产生"忽略了异常").
    group = SimpleTaskGroup()

    async def boom():
        raise RuntimeError("boom")

    task = group.add_coroutine(boom(), ignore_error=False)
    with pytest.raises(RuntimeError):
        await task


@pytest.mark.asyncio
async def test_aclose_cancels_pending():
    cancelled = []
    group = SimpleTaskGroup()

    async def worker(token):
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.append(token)
            raise

    group.add_coroutine(worker(1))
    group.add_coroutine(worker(2))
    await asyncio.sleep(0)
    await group.aclose()

    assert sorted(cancelled) == [1, 2]


@pytest.mark.asyncio
async def test_aclose_twice_is_safe():
    # 重复调用 aclose 应幂等: 所有 pending 任务被取消, 第二次是空操作.
    cancelled = []
    group = SimpleTaskGroup()

    async def worker(token):
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.append(token)
            raise

    group.add_coroutine(worker(1))
    group.add_coroutine(worker(2))
    await asyncio.sleep(0)

    await group.aclose()
    await group.aclose()

    assert sorted(cancelled) == [1, 2]


@pytest.mark.asyncio
async def test_aclose_concurrent_is_safe():
    # 并发 aclose 应安全: 任一 aclose 关闭后, 其余应为空操作, 不重复取消/不抛异常.
    cancelled = []
    group = SimpleTaskGroup()

    async def worker(token):
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.append(token)
            raise

    group.add_coroutine(worker(1))
    group.add_coroutine(worker(2))
    await asyncio.sleep(0)

    await asyncio.gather(group.aclose(), group.aclose())

    assert sorted(cancelled) == [1, 2]


def test_add_then_immediate_close_no_orphan_warning():
    # ensure_future 直接持有用户协程: add 后立即 close(不先 yield)也不产生 "never awaited" 孤儿.
    async def scenario():
        group = SimpleTaskGroup()

        async def worker():
            await asyncio.sleep(30)

        group.add_coroutine(worker())
        group.close()

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        asyncio.run(scenario())
        gc.collect()

    assert not any("was never awaited" in str(w.message) for w in rec)


@pytest.mark.asyncio
async def test_on_exception_fires_on_bound_error():
    seen = []
    cancelled = []
    group = SimpleTaskGroup(on_exception=seen.append)

    async def worker():
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.append(1)
            raise

    async def boom():
        raise ValueError("boom")

    good = group.add_coroutine(worker())
    group.add_coroutine(boom(), ignore_error=False)
    await asyncio.sleep(0)
    await asyncio.gather(good, return_exceptions=True)

    assert len(seen) == 1
    assert isinstance(seen[0], ValueError)
    assert str(seen[0]) == "boom"
    assert cancelled == [1]  # bind 错误仍关组


@pytest.mark.asyncio
async def test_on_exception_fires_on_ignore_error():
    seen = []
    result = []
    group = SimpleTaskGroup(on_exception=seen.append)

    async def ok():
        await asyncio.sleep(0.01)
        result.append("ok")

    async def crash():
        raise ValueError("ignored")

    group.add_coroutine(crash(), ignore_error=True)
    ok_task = group.add_coroutine(ok())
    await ok_task

    assert len(seen) == 1
    assert isinstance(seen[0], ValueError)
    assert result == ["ok"]  # ignore 错误不关组


@pytest.mark.asyncio
async def test_on_exception_not_fired_on_cancel():
    seen = []
    group = SimpleTaskGroup(on_exception=seen.append)

    async def worker():
        await asyncio.sleep(30)

    task = group.add_coroutine(worker())
    await asyncio.sleep(0)
    await group.aclose()
    await asyncio.gather(task, return_exceptions=True)

    assert seen == []


@pytest.mark.asyncio
async def test_on_exception_not_fired_on_normal_completion():
    seen = []
    group = SimpleTaskGroup(on_exception=seen.append)

    async def ok():
        await asyncio.sleep(0)
        return 42

    task = group.add_coroutine(ok())
    assert await task == 42

    assert seen == []


@pytest.mark.asyncio
async def test_on_exception_raising_still_closes_group():
    # on_exception 抛异常也不应打断 bind 的关组契约.
    asyncio.get_running_loop().set_exception_handler(lambda _loop, _ctx: None)

    def bad_handler(exc):
        raise RuntimeError("handler bug")

    cancelled = []
    group = SimpleTaskGroup(on_exception=bad_handler)

    async def worker():
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.append(1)
            raise

    async def boom():
        raise ValueError("boom")

    good = group.add_coroutine(worker())
    group.add_coroutine(boom(), ignore_error=False)
    await asyncio.sleep(0)
    await asyncio.gather(good, return_exceptions=True)

    assert cancelled == [1]
