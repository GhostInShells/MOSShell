import pytest
import asyncio
import time

from ghoshell_moss.core import PyChannel, new_ctml_shell


@pytest.mark.asyncio
async def test_interrupt_in_ctml():
    """
    测试在 CTML 中调用 sleep（无 channel 参数）
    """
    shell = new_ctml_shell()

    cancelled = []

    async def foo():
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            cancelled.append(1)

    for i in range(10):
        chan = PyChannel(name=f"chan{i}")
        chan.build.command()(foo)
        shell.main_channel.import_channels(chan)

    async with shell:
        async with await shell.interpreter() as interpreter:
            # 发送 CTML：先执行 foo，然后 sleep，再执行 foo
            for i in range(10):
                interpreter.feed(f"<chan{i}:foo/>")
            interpreter.feed("<interrupt/>")
            interpreter.commit()
            await interpreter.wait_stopped()
            tasks = interpreter.compiled_tasks()
            assert len(tasks) == 11
            success = 0
            for task in tasks.values():
                if task.success():
                    success += 1
            assert success == 1
        assert len(cancelled) >= 9

        cancelled.clear()
        async with await shell.interpreter() as interpreter:
            # 发送 CTML：先执行 foo，然后 sleep，再执行 foo
            for i in range(10):
                interpreter.feed(f"<chan{i}:foo/>")
            # sleep 10 also cleared
            interpreter.feed("<sleep duration='10'/><interrupt/>")
            interpreter.commit()
            await interpreter.wait_stopped()
            tasks = interpreter.compiled_tasks()
            assert len(tasks) == 12
            success = 0
            for task in tasks.values():
                if task.success():
                    success += 1
            assert success == 1
        assert len(cancelled) == 10
