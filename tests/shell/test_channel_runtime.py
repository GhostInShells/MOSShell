import logging

from typing import List
from ghoshell_moss.shell.channel_runtime import ChannelRuntime
from ghoshell_container import Container
from ghoshell_moss import PyChannel, PyCommand, BaseCommandTask, Channel, CommandTask
import pytest


async def callback(channel: Channel, paths: List[str], task: CommandTask):
    task.fail("test has no child runtime")


@pytest.mark.asyncio
async def test_channel_runtime_impl_baseline():
    chan = PyChannel(name="")

    @chan.build.command()
    async def foo() -> int:
        return 123

    runtime = ChannelRuntime(Container(), chan, callback)
    async with runtime:
        assert runtime.name == ""
        assert runtime.is_running()
        assert runtime.is_available()
        await runtime.wait_until_idle()
        assert not runtime.is_busy()

        foo_cmd = runtime.channel.client.get_command("foo")
        assert foo_cmd is not None
        assert foo_cmd.meta().chan == ""
        task = BaseCommandTask.from_command(foo_cmd)
        runtime.add_task(task)
        await task.wait()
    assert task.done()
    assert task._result is 123


@pytest.mark.asyncio
async def test_child_channel_runtime_is_running():
    main = PyChannel(name="")

    @main.build.command()
    async def bar() -> int:
        return 123

    a = main.new_child('a')

    @a.build.command()
    async def foo() -> int:
        return 123

    runtime = ChannelRuntime(Container(), main, callback)
    async with runtime:
        assert a.is_running()
        assert main.children().get('a') is a
        commands = runtime.commands()
        assert "bar" in commands
        bar_cmd = commands['bar']
        assert 123 == await bar_cmd()
