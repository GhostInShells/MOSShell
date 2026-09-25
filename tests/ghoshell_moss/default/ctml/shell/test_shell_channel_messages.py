import asyncio

import pytest

from ghoshell_moss.core import PyChannel
from ghoshell_moss.message import Message


@pytest.mark.asyncio
async def test_shell_execution_baseline():
    from ghoshell_moss.core.ctml.shell import new_ctml_shell

    shell = new_ctml_shell()

    a_chan = PyChannel(name="a")
    b_chan = PyChannel(name="b")

    async def a_message() -> list[Message]:
        msg = Message.new().with_content("hello")
        return [msg]

    def b_message() -> list[Message]:
        msg = Message.new().with_content("world")
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
        shell_metas = shell.channel_metas()
        assert len(shell_metas) == 3
        interpreter = await shell.interpreter()
        metas = interpreter.channels()
        assert len(metas) == 3

        messages = interpreter.merge_messages([], [])
        assert len(messages) > 0


@pytest.mark.asyncio
async def test_channel_metas_generation_callback_count():
    from ghoshell_moss.core.ctml.shell import new_ctml_shell

    shell = new_ctml_shell()
    a_chan = PyChannel(name="a")
    shell.main_channel.import_channels(a_chan)

    async with shell:
        await shell.wait_connected()

        calls = []

        def on_generation(metas) -> None:
            calls.append(metas)

        discard = shell.on_channel_metas_generation(on_generation)

        # 回归: refresh 完成后必须触发一次 metas 重建回调.
        assert await shell.refresh_metas() is True
        assert len(calls) == 1

        # 每次 refresh 都触发一次.
        assert await shell.refresh_metas() is True
        assert len(calls) == 2

        # discard 后不再触发.
        discard()
        assert await shell.refresh_metas() is True
        assert len(calls) == 2
