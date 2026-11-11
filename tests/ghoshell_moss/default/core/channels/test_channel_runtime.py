import logging

import pytest

from ghoshell_moss.core import BaseCommandTask, PyChannel, new_channel
from ghoshell_moss.core.concepts.errors import CommandErrorCode
import asyncio


@pytest.mark.asyncio
async def test_channel_runtime_execution():
    chan = PyChannel(name="test")

    @chan.build.command()
    async def foo() -> int:
        return 123

    async with chan.bootstrap() as runtime:
        assert runtime.name == "test"
        assert runtime.is_running()
        assert runtime.is_available()
        await runtime.wait_idle()
        assert runtime.is_idle()

        foo_cmd = runtime.get_command("foo")
        assert foo_cmd is not None
        task = BaseCommandTask.from_command(foo_cmd)
        runtime.push_task(task)
        await task.wait()
    assert task.done()
    assert task.result() == 123


@pytest.mark.asyncio
async def test_channel_runtime_clear():
    chan = PyChannel(name="test")

    @chan.build.command()
    async def foo() -> int:
        await asyncio.sleep(1)
        return 123

    async with chan.bootstrap() as runtime:
        task = runtime.create_command_task("foo")
        assert task is not None
        runtime.push_task(task)
        assert runtime.is_idle()
        await asyncio.sleep(0.01)
        assert not runtime.is_idle()
        await runtime.clear()
        assert task.done()
        assert CommandErrorCode.CLEARED.match(task.exception())

    # assert pause also clear the channel.
    async with chan.bootstrap() as runtime:
        task = runtime.create_command_task("foo")
        assert task is not None
        runtime.push_task(task)
        await asyncio.sleep(0.001)
        assert not runtime.is_idle()
        await task
        assert task.done()


@pytest.mark.asyncio
async def test_child_channel_runtime_running():
    """
    由于现在 Channel Runtime 不再递归启动了, 所以不应该有任何子 channel 被启动.
    """
    main = PyChannel(name="test")

    @main.build.command()
    async def bar() -> int:
        return 123

    a = new_channel("a")
    main.import_channels(a)

    @a.build.command()
    async def foo() -> int:
        return 123

    async with main.bootstrap() as runtime:
        assert "a" in main.children()

        assert main.children().get("a") is a
        commands = runtime.own_commands()
        assert "bar" in commands

        bar_cmd = commands["bar"]
        assert await bar_cmd() == 123


@pytest.mark.asyncio
async def test_channel_runtime_non_blocking():
    chan = PyChannel(name="test")

    @chan.build.command(blocking=False)
    async def foo() -> int:
        await asyncio.sleep(0.2)
        return 234

    @chan.build.command(blocking=False)
    async def bar() -> int:
        await asyncio.sleep(0.05)
        return 123

    async with chan.bootstrap() as runtime:
        task1 = runtime.create_command_task("foo")
        task2 = runtime.create_command_task("bar")
        runtime.push_task(task1, task2)
        assert await task2 == 123
        # 估计 task1 还没执行完.
        assert not task1.done()
        # 仍然会执行完
        assert await task1 == 234

        task3 = runtime.create_command_task("foo")
        task4 = runtime.create_command_task("bar")
        runtime.push_task(task3, task4)
        # 直接清空.
        await runtime.clear()
        # 都被清空了.
        assert task3.done()
        assert task4.done()
        assert CommandErrorCode.CLEARED.match(task3.exception())


@pytest.fixture
def plain_chan() -> PyChannel:
    chan = PyChannel(name="test")

    @chan.build.command()
    async def foo() -> int:
        return 123

    return chan


@pytest.mark.asyncio
async def test_channel_runtime_create_asyncio_task_runs(plain_chan: PyChannel):
    """
    A detached task created on the runtime actually runs, and leaves the runtime's group.
    """
    started = asyncio.Event()

    async def coro() -> int:
        started.set()
        return 123

    async with plain_chan.bootstrap() as runtime:
        task = runtime.create_asyncio_task(coro(), name="unit-task")
        # registered on creation, so the runtime can still cancel it on close.
        assert task in runtime._runtime_asyncio_task_group
        assert await task == 123
        assert started.is_set()
        # the done callback hops through loop.call_soon.
        await asyncio.sleep(0.01)
        assert task not in runtime._runtime_asyncio_task_group, (
            "a finished detached task must leave the runtime group, otherwise it leaks"
        )


@pytest.mark.asyncio
async def test_channel_runtime_create_asyncio_task_error_is_logged(plain_chan: PyChannel, caplog):
    """
    A failing detached task must reach the log - silence here is what makes a
    background feature look like it never ran at all.
    """
    async def boom() -> None:
        raise ValueError("kaboom")

    async with plain_chan.bootstrap() as runtime:
        with caplog.at_level(logging.ERROR, logger="moss"):
            task = runtime.create_asyncio_task(boom(), name="boom-task")
            with pytest.raises(ValueError, match="kaboom"):
                await task
            await asyncio.sleep(0.01)

        assert task not in runtime._runtime_asyncio_task_group
        matched = [r for r in caplog.records if "boom-task" in r.getMessage()]
        assert matched, "the runtime must report a failed detached task"
        assert matched[-1].levelno >= logging.ERROR
        assert "kaboom" in matched[-1].getMessage()


@pytest.mark.asyncio
async def test_channel_runtime_create_asyncio_task_cancel_is_clean(plain_chan: PyChannel):
    """
    Cancelling a detached task is normal life, not an error: it must still leave the group.
    """
    gate = asyncio.Event()

    async def hang() -> None:
        await gate.wait()

    async with plain_chan.bootstrap() as runtime:
        task = runtime.create_asyncio_task(hang(), name="hang-task")
        await asyncio.sleep(0.01)
        assert task in runtime._runtime_asyncio_task_group

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.01)
        assert task not in runtime._runtime_asyncio_task_group


@pytest.mark.asyncio
async def test_channel_runtime_create_asyncio_task_dies_with_runtime(plain_chan: PyChannel):
    """
    A detached task outlives its command, but never outlives its channel runtime.
    """
    gate = asyncio.Event()

    async def hang() -> None:
        await gate.wait()

    async with plain_chan.bootstrap() as runtime:
        task = runtime.create_asyncio_task(hang(), name="hang-task")
        await asyncio.sleep(0.01)
        assert not task.done()

    assert task.done()
    assert task.cancelled()


@pytest.mark.asyncio
async def test_channel_runtime_create_asyncio_task_requires_running(plain_chan: PyChannel):
    """
    Without a running loop in this runtime, there is nothing to attach the task to.
    """
    runtime = plain_chan.bootstrap()
    coro = asyncio.sleep(0)
    try:
        with pytest.raises(RuntimeError):
            runtime.create_asyncio_task(coro)
    finally:
        coro.close()
