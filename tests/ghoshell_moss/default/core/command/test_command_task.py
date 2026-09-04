import asyncio
import threading

import pytest

from ghoshell_moss.core.concepts.command import (
    BaseCommandTask,
    CommandStackResult,
    CommandTaskState,
    PyCommand,
    CommandTaskResult,
    CommandTask,
)
from ghoshell_moss.core.concepts.errors import CommandError, CommandErrorCode
from ghoshell_moss.core.concepts.channel import ChannelCtx
from ghoshell_moss.core.blueprint.channel_builder import CommandUtil


@pytest.mark.asyncio
async def test_command_task_baseline():
    async def foo() -> int:
        return 123

    command = PyCommand(foo)

    # from command create a basic command task
    task = BaseCommandTask.from_command(command, tokens_="<foo />")

    await task.run()

    assert task.result() == 123
    assert task.state == CommandTaskState.done.value
    assert len(task.trace) == 2
    assert task.tokens == "<foo />"
    assert task.done()

    assert task.wait_sync() == 123


def test_command_task_in_multi_thread():
    async def foo() -> int:
        return 123

    command = PyCommand(foo)
    task = BaseCommandTask.from_command(command, tokens_="<foo />")

    def thread_1():
        asyncio.run(task.run())

    result = []

    def thread_2():
        result.append(task.wait_sync())

    t1 = threading.Thread(target=thread_1)
    t2 = []
    # wait in different thread.
    for i in range(10):
        t2_thread = threading.Thread(target=thread_2)
        t2_thread.start()
        t2.append(t2_thread)
    # change start order
    t1.start()
    t1.join()
    for t in t2:
        t.join()

    assert len(result) == 10
    for r in result:
        assert r == 123


@pytest.mark.asyncio
async def test_command_task_in_parallel_tasks():
    async def foo() -> int:
        return 123

    command = PyCommand(foo)
    task = BaseCommandTask.from_command(command, tokens_="<foo />")
    run_task = asyncio.create_task(task.run())

    awaits_tasks = []
    for i in range(10):
        await_task = asyncio.create_task(task.wait())
        awaits_tasks.append(await_task)

    done = await asyncio.gather(run_task, *awaits_tasks)
    assert len(done) == 11
    for r in done:
        assert r == 123


@pytest.mark.asyncio
async def test_command_task_cancel():
    async def foo() -> int:
        await asyncio.sleep(0.1)
        return 123

    command = PyCommand(foo)
    task = BaseCommandTask.from_command(command, tokens_="<foo />")
    task.cancel("test")
    assert task.done()
    with pytest.raises(CommandError):
        await task.run()

    task2 = task.copy()
    # cancel come first
    await asyncio.gather(asyncio.to_thread(task2.cancel), task2.wait(throw=False))
    assert task2.errcode == CommandErrorCode.CANCELLED
    assert task2.cancelled()
    assert not task2.success()


@pytest.mark.asyncio
async def test_command_task_stack():
    import time

    start = time.time()

    async def foo() -> int:
        return 123

    stack = CommandStackResult(
        [
            BaseCommandTask.from_command(PyCommand(foo)),
            BaseCommandTask.from_command(PyCommand(foo)),
        ]
    )

    got = []
    async for i in stack:
        got.append(i)
    assert len(got) == 2

    async def iter_tasks():
        yield BaseCommandTask.from_command(PyCommand(foo))
        yield BaseCommandTask.from_command(PyCommand(foo))
        yield BaseCommandTask.from_command(PyCommand(foo))

    stack = CommandStackResult(iter_tasks())
    got = []
    async for i in stack:
        got.append(i)
    assert len(got) == 3
    end = time.time()

    async def bar() -> CommandStackResult:
        async def result(ran_tasks):
            count = 0
            # 计算有多少个子 task 被运行了.
            for t in ran_tasks:
                if t.done():
                    count += 1
            return count

        return CommandStackResult(iter_tasks(), callback=result)

    bar_task = BaseCommandTask.from_command(PyCommand(bar))
    # 返回的应该是一个 stack.
    stack = await bar_task.dry_run()
    assert isinstance(stack, CommandStackResult)
    # 把所有的 stack 再运行一次.
    i = 0
    async for r in stack:
        assert await r.run() == 123
        i += 1
        # 只运行两个子 task.
        if i == 2:
            break

    await stack.callback(bar_task)
    assert bar_task.result() == 2
    assert bar_task.done()
    assert bar_task.success()


@pytest.mark.asyncio
async def test_command_task_in_context():
    async def foo() -> str:
        task = ChannelCtx.task()
        return task.cid

    # 可以拿到外部传递的数据.
    foo_task = BaseCommandTask.from_command(PyCommand(foo))
    assert await foo_task.run() == foo_task.cid


def test_task_caller_name():
    async def foo() -> str:
        return ""

    task = BaseCommandTask.from_command(PyCommand(foo), chan_="a")
    task.call_id = "2"
    assert task.caller_name() == "a:foo:2"


def test_await_task_in_threads():
    async def foo() -> int:
        return 123

    # 跨线程创建.
    foo_task = BaseCommandTask.from_command(PyCommand(foo))

    done = []

    def thread_await_task():
        async def wait():
            await foo_task
            done.append(True)

        asyncio.run(wait())

    threads = []
    for i in range(10):
        t = threading.Thread(target=thread_await_task)
        t.start()
        threads.append(t)

    # 运行并且等待 task 结束.
    asyncio.run(foo_task.run())
    for t in threads:
        t.join()

    assert len(done) == 10


@pytest.mark.asyncio
async def test_command_task_result():
    class Bar:
        bar = 123

    async def foo() -> Bar:
        return Bar()

    command = PyCommand(foo)
    task = BaseCommandTask.from_command(command)
    task.call_id = "2"
    await task.run()
    task_result = task.task_result()
    assert task_result.caller == "foo:2"
    assert len(task_result.as_messages()) > 0

    async def baz():
        return CommandTaskResult(result="hello")

    command = PyCommand(baz)
    task = BaseCommandTask.from_command(command)
    await task.run()
    assert task.result() == "hello"
    assert task.task_result().caller is not None


@pytest.mark.asyncio
async def test_bare_and_magic_task():
    async def __foo__() -> int:
        return 123

    command = PyCommand(__foo__)

    task = BaseCommandTask.from_command(command)
    assert task.is_magical()
    task.func = None
    assert task.is_bare_task()
    r = await task.dry_run()
    assert r is None

    task = BaseCommandTask.from_command(command)
    assert not task.is_bare_task()
    assert task.is_magical()
    r = await task.dry_run()
    assert r is 123


@pytest.mark.asyncio
async def test_command_task_timeout():
    async def foo() -> int:
        await asyncio.sleep(1)
        return 123

    foo_command = PyCommand(foo, timeout=0.01)
    task = BaseCommandTask.from_command(foo_command)
    with pytest.raises(asyncio.TimeoutError):
        await task.run()

    task = BaseCommandTask.from_command(foo_command)
    task.func = None
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(task.wait(throw=True), 0.2)


@pytest.mark.asyncio
async def test_command_task_cancel_is_not_success():
    async def foo() -> int:
        await asyncio.sleep(1)
        return 123

    foo_command = PyCommand(foo, timeout=0.01)
    task = BaseCommandTask.from_command(foo_command)
    task.cancel()
    assert task.cancelled()
    assert not task.success()


@pytest.mark.asyncio
async def test_command_task_with_progresses():
    task = None

    async def foo() -> int:
        nonlocal task
        for i in range(10):
            if task is not None:
                task.set_progress(f"progress {i}")
            await asyncio.sleep(0.01)
        return 123

    foo_command = PyCommand(foo)
    progresses = []

    def count_progress(_t: CommandTask):
        nonlocal progresses
        progresses.append(_t.progress)

    task = BaseCommandTask.from_command(foo_command)
    task.on_progress_callback(count_progress)

    await task.run()
    assert len(progresses) == 10


@pytest.mark.asyncio
async def test_command_reraises_stopped_on_cancel():
    """命令被 asyncio cancel 时, catch CancelledError 后 reraise STOPPED(301) 携带进度."""
    async def cmd():
        try:
            await asyncio.sleep(60)   # 模拟播放中
        except asyncio.CancelledError:
            CommandUtil.reraise_stopped("played 3.2s, stopped at ...世界")

    runner = asyncio.create_task(cmd())
    await asyncio.sleep(0.02)
    runner.cancel()
    with pytest.raises(CommandError) as ei:
        await runner
    assert ei.value.code == CommandErrorCode.STOPPED
    assert "stopped at" in ei.value.message


@pytest.mark.asyncio
async def test_stopped_is_notifiable_without_observe():
    """STOPPED(301) 落 notify 档: fail 后生成可读 message, 不触发 observe, 非调度取消."""
    async def foo() -> int:
        return 123

    task = BaseCommandTask.from_command(PyCommand(foo))
    task.fail(CommandErrorCode.STOPPED.error("played 3.2s, stopped at ...世界"))
    assert task.done()
    assert task.errcode == CommandErrorCode.STOPPED
    assert not task.cancelled()     # state=failed, 不是 200 区调度取消
    assert not task.success()

    tr = task.task_result()
    assert tr is not None
    assert tr.observe is False      # 不触发 observe
    msgs = tr.as_messages()
    assert len(msgs) > 0
    assert "stopped at" in msgs[0].to_content_string()
