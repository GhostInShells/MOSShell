import contextvars

import pytest
from ghoshell_moss.core.concepts.channel import ChannelCtx
from ghoshell_moss.core.concepts.command import BaseCommandTask, PyCommand


@pytest.mark.asyncio
async def test_channel_ctx_not_effect_outside():
    async def foo():
        return ChannelCtx.task()

    foo_cmd = PyCommand(foo)

    assert ChannelCtx.runtime() is None
    assert ChannelCtx.task() is None
    assert await foo() is None
    assert await foo_cmd() is None

    assert ChannelCtx.runtime() is None
    assert ChannelCtx.task() is None
    assert await foo() is None
    assert await foo_cmd() is None

    task = BaseCommandTask.from_command(foo_cmd)
    ctx = ChannelCtx(task=task)
    assert await ctx.run(foo) is task
    assert await ctx.run(foo_cmd) is task

    assert ChannelCtx.runtime() is None
    assert ChannelCtx.task() is None
    assert await foo() is None
    assert await foo_cmd() is None
