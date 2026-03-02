import pytest
import asyncio
import time

from ghoshell_moss.core.ctml.shell.primitives.sleep import sleep
from ghoshell_moss.core.concepts.command import CommandStackResult
from ghoshell_moss.core import PyChannel, new_ctml_shell


@pytest.mark.asyncio
async def test_interrupt_in_ctml():
    """
    测试在 CTML 中调用 sleep（无 channel 参数）
    """
    shell = new_ctml_shell()

    async with shell:
        async with shell.interpreter_in_ctx() as interpreter:
            # 发送 CTML：先执行 foo，然后 sleep，再执行 foo
            interpreter.feed(f"<noop/>")
            interpreter.commit()
            tasks = await interpreter.wait_tasks()
            assert len(tasks) == 1
            noop_task = list(tasks.values())[0]
            assert noop_task.success()
            assert noop_task.meta.name == "noop"
