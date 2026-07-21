import pytest
import asyncio

from ghoshell_moss.core.ctml.shell.primitives.thinking import thinking_command
from ghoshell_moss.core import PyChannel, new_ctml_shell


@pytest.mark.asyncio
async def test_thinking_consumes_content():
    """<thinking>text</thinking> should be consumed without parse error or prompt pollution."""
    shell = new_ctml_shell()

    async with shell:
        async with await shell.interpreter() as interpreter:
            interpreter.feed("<thinking>some inner monologue</thinking>")
            interpreter.commit()
            tasks = await interpreter.wait_tasks()
            assert len(tasks) == 1
            task = list(tasks.values())[0]
            assert task.success()
            assert task.meta.name == "thinking"


@pytest.mark.asyncio
async def test_thinking_invisible_to_model():
    """thinking is a protocol compat shim — model must not see it in static interface."""
    assert thinking_command.meta().visible is False


@pytest.mark.asyncio
async def test_thinking_with_other_command():
    """thinking should coexist with regular commands in the same feed."""
    shell = new_ctml_shell()

    chan = PyChannel(name="test")
    executed = []

    @chan.build.command()
    async def foo():
        executed.append(1)

    shell.main_channel.import_channels(chan)

    async with shell:
        async with await shell.interpreter() as interpreter:
            interpreter.feed("<thinking>reasoning here</thinking><test:foo/>")
            interpreter.commit()
            tasks = await interpreter.wait_tasks()
            assert len(tasks) == 2
            assert all(t.success() for t in tasks.values())
            assert len(executed) == 1
