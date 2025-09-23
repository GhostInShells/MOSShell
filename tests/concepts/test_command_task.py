import threading

from ghoshell_moss.concepts.command import PyCommand, BaseCommandTask, CommandTaskState
from ghoshell_moss.concepts.errors import CommandError
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

    assert task.result == 123
    assert task.state == CommandTaskState.DONE.value
    assert len(task.trace) == 2
    assert task.tokens == "<foo />"
    assert task.is_done()

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
    assert task.is_done()
    with pytest.raises(CommandError):
        await task.run()

    task2 = task.copy()
    # cancel come first
    await asyncio.gather(asyncio.to_thread(task2.cancel), task2.wait(throw=False))
    assert task2.errcode == CommandError.CANCEL_CODE
