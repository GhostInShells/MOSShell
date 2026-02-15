import pytest
from ghoshell_container import Container

from ghoshell_moss import BaseCommandTask, Channel, CommandTask, PyChannel, new_chan
from ghoshell_moss.core.concepts.runtime import ChannelTreeRuntime
from ghoshell_moss.core.concepts.errors import CommandErrorCode
import asyncio


@pytest.mark.asyncio
async def test_channel_runtime_execution():
    chan = PyChannel(name="")

    @chan.build.command()
    async def foo() -> int:
        return 123

    async with ChannelTreeRuntime.bootstrap(chan) as runtime:
        assert runtime.name == ""
        assert runtime.is_running()
        assert runtime.is_available()
        await runtime.wait_blocking_task_done()
        assert runtime.is_blocking_task_empty()

        foo_cmd = runtime.get_command("foo")
        assert foo_cmd is not None
        assert foo_cmd.meta().chan == ""
        task = BaseCommandTask.from_command(foo_cmd)
        await runtime.put_task(task)
        await task.wait()
    assert task.done()
    assert task._result == 123


@pytest.mark.asyncio
async def test_channel_runtime_clear():
    chan = PyChannel(name="")

    paused = []

    @chan.build.command()
    async def foo() -> int:
        await asyncio.sleep(1)
        return 123

    @chan.build.pause
    async def pause():
        paused.append(True)

    async with ChannelTreeRuntime.bootstrap(chan) as runtime:
        task = runtime.create_command_task("foo")
        assert task is not None
        await runtime.put_task(task)
        assert not runtime.is_blocking_task_empty()
        await runtime.clear()
        assert task.done()
        assert CommandErrorCode.CLEARED.match(task.exception())

    # assert pause also clear the channel.
    async with ChannelTreeRuntime.bootstrap(chan) as runtime:
        task = runtime.create_command_task("foo")
        assert task is not None
        await runtime.put_task(task)
        assert not runtime.is_blocking_task_empty()
        await runtime.pause()
        assert task.done()
        assert CommandErrorCode.CLEARED.match(task.exception())


@pytest.mark.asyncio
async def test_child_channel_runtime_running():
    """
    由于现在 Channel Broker 不再递归启动了, 所以不应该有任何子 channel 被启动.
    """
    main = PyChannel(name="")

    @main.build.command()
    async def bar() -> int:
        return 123

    a = new_chan("a")
    main.import_channels(a)

    @a.build.command()
    async def foo() -> int:
        return 123

    async with ChannelTreeRuntime.bootstrap(main) as runtime:
        main_runtime = await runtime.fetch_node("")
        assert main_runtime.is_running()
        assert "a" in main.children()

        a_runtime = await runtime.fetch_node("a")
        assert a_runtime is not None
        assert a_runtime.is_running()
        assert main.children().get("a") is a
        commands = runtime.commands()
        assert "bar" in commands

        bar_cmd = commands["bar"]
        assert await bar_cmd() == 123


@pytest.mark.asyncio
async def test_channel_runtime_non_blocking():
    chan = PyChannel(name="")

    @chan.build.command(blocking=False)
    async def foo() -> int:
        await asyncio.sleep(0.2)
        return 234

    @chan.build.command(blocking=False)
    async def bar() -> int:
        await asyncio.sleep(0.05)
        return 123

    async with ChannelTreeRuntime.bootstrap(chan) as runtime:
        task1 = runtime.create_command_task("foo")
        task2 = runtime.create_command_task("bar")
        await runtime.put_task(task1, task2)
        assert await task2 == 123
        # 估计 task1 还没执行完.
        assert not task1.done()
        # 仍然会执行完
        assert await task1 == 234

        task3 = runtime.create_command_task("foo")
        task4 = runtime.create_command_task("bar")
        await runtime.put_task(task3, task4)
        # 直接清空.
        await runtime.clear()
        # 都被清空了.
        assert task3.done()
        assert task4.done()
        assert CommandErrorCode.CLEARED.match(task3.exception())
