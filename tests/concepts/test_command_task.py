import threading

from ghoshell_moss.concepts.command import PyCommand, BaseCommandTask, CommandTaskState, CommandTaskStack, CommandTask
from ghoshell_moss.concepts.errors import CommandError, CommandErrorCode
import pytest
import asyncio


@pytest.mark.asyncio
async def test_command_task_baseline():
    async def foo() -> int:
        return 123

    command = PyCommand(foo)

    # from command create a basic command task
    task = BaseCommandTask.from_command(command, tokens_="<foo />")

    await task.run()

    assert task._result == 123
    assert task.state == CommandTaskState.DONE.value
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

    stack = CommandTaskStack([
        BaseCommandTask.from_command(PyCommand(foo)),
        BaseCommandTask.from_command(PyCommand(foo)),
    ])

    got = []
    async for i in stack:
        got.append(i)
    assert len(got) == 2

    async def iter_tasks():
        yield BaseCommandTask.from_command(PyCommand(foo))
        yield BaseCommandTask.from_command(PyCommand(foo))
        yield BaseCommandTask.from_command(PyCommand(foo))

    stack = CommandTaskStack(iter_tasks())
    got = []
    async for i in stack:
        got.append(i)
    assert len(got) == 3
    end = time.time()

    async def bar() -> CommandTaskStack:
        async def result(ran_tasks):
            count = 0
            # 计算有多少个子 task 被运行了.
            for t in ran_tasks:
                if t.done():
                    count += 1
            return count

        return CommandTaskStack(iter_tasks(), on_success=result)

    bar_task = BaseCommandTask.from_command(PyCommand(bar))
    # 返回的应该是一个 stack.
    stack = await bar_task.dry_run()
    assert isinstance(stack, CommandTaskStack)
    # 把所有的 stack 再运行一次.
    i = 0
    async for r in stack:
        assert await r.run() == 123
        i += 1
        # 只运行两个子 task.
        if i == 2:
            break

    await stack.success(bar_task)
    assert bar_task._result == 2


@pytest.mark.asyncio
async def test_command_task_in_context():
    async def foo() -> str:
        task = CommandTask.get_from_context()
        return task.cid

    # 可以拿到外部传递的数据.
    foo_task = BaseCommandTask.from_command(PyCommand(foo))
    assert await foo_task.run() == foo_task.cid
