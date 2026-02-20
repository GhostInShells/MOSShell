import asyncio

import pytest

from ghoshell_moss import PyChannel
from ghoshell_moss.message import Message


@pytest.mark.asyncio
async def test_shell_execution_baseline():
    from ghoshell_moss.core.shell import new_shell

    shell = new_shell()

    a_chan = PyChannel(name="a")
    b_chan = PyChannel(name="b")

    async def a_message() -> list[Message]:
        msg = Message.new(role="system").with_content("hello")
        return [msg]

    def b_message() -> list[Message]:
        msg = Message.new(role="system").with_content("world")
        return [msg]

    a_chan.build.context_messages(a_message)
    b_chan.build.context_messages(b_message)
    shell.main_channel.import_channels(a_chan, b_chan)

    @a_chan.build.command()
    async def foo() -> int:
        return 123

    @b_chan.build.command()
    async def bar() -> int:
        # 晚执行 0.1 秒.
        await asyncio.sleep(0.1)
        return 456

    async with shell:
        assert shell.is_running()
        await shell.wait_connected()
        interpreter = await shell.interpreter()
        metas = interpreter.channels()
        assert len(metas) == 3
        messages = interpreter.context_messages()
        assert len(messages) >= 2
