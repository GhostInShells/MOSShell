import pytest
from ghoshell_container import Container

from ghoshell_moss import BaseCommandTask, Channel, CommandTask, PyChannel, new_chan
from ghoshell_moss.core.shell.channel_runtime import ChannelRuntime


async def callback(channel: Channel, paths: list[str], task: CommandTask):
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

        foo_cmd = runtime.channel.broker.get_command("foo")
        assert foo_cmd is not None
        assert foo_cmd.meta().chan == ""
        task = BaseCommandTask.from_command(foo_cmd)
        runtime.add_task(task)
        await task.wait()
    assert task.done()
    assert task._result == 123


@pytest.mark.asyncio
async def test_child_channel_runtime_is_not_running():
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

    runtime = ChannelRuntime(Container(), main, callback)
    async with runtime:
        assert main.is_running()
        assert not a.is_running()
        assert main.children().get("a") is a
        commands = runtime.commands()
        assert "bar" in commands
        bar_cmd = commands["bar"]
        assert await bar_cmd() == 123
