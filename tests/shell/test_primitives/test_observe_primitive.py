import pytest
import asyncio
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
            interpreter.feed("<observe/>")
            interpreter.commit()
            await interpreter.wait_compiled()
            assert len(interpreter.compiled_tasks()) == 11
            # when observe done, interpreter is stopped
            await interpreter.wait_stopped()
            # task not done while observe raise
            assert len(interpreter.done_tasks()) == 1
            await interpreter.close(cancel_executing=True)
            assert len(interpreter.done_tasks()) == 11
