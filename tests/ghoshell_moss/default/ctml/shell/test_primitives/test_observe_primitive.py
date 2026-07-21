import pytest
import asyncio
from ghoshell_moss.core import PyChannel, new_ctml_shell
from ghoshell_moss.core.concepts.command import ObserveError
from ghoshell_moss.core.concepts.errors import CommandErrorCode


@pytest.mark.asyncio
async def test_observe_non_interrupting():
    """
    Observe 返回不中断并行任务，所有命令都正常执行完毕。
    模型在关键帧统一观察所有结果。
    """
    shell = new_ctml_shell()
    done_count = []

    async def foo():
        await asyncio.sleep(0.1)
        done_count.append(1)

    for i in range(10):
        chan = PyChannel(name=f"chan{i}")
        chan.build.command()(foo)
        shell.main_channel.import_channels(chan)

    async with shell:
        async with await shell.interpreter() as interpreter:
            for i in range(10):
                interpreter.feed(f"<chan{i}:foo/>")
            interpreter.feed("<observe/>")
            interpreter.commit()
            await interpreter.wait_compiled()
            assert len(interpreter.compiled_tasks()) == 11
            await interpreter.wait_stopped()
            # 非中断：所有 11 个 task 都执行完毕
            assert len(interpreter.done_tasks()) == 11
            assert len(done_count) == 10
            # observe 标记被正确设置
            assert interpreter.interpretation().observe is True


@pytest.mark.asyncio
async def test_observe_error_interrupting():
    """
    ObserveError 中断解释流程，取消未完成的并行任务。
    注意：CommandTask.cancel() 标记任务状态为 cancelled，
    但不保证取消底层 asyncio 协程（dry_run 直接 await 函数，无 Task 包装）。
    """
    shell = new_ctml_shell()

    async def foo():
        await asyncio.sleep(1)

    async def raise_error():
        raise ObserveError("emergency interrupt")

    for i in range(5):
        chan = PyChannel(name=f"chan{i}")
        chan.build.command()(foo)
        shell.main_channel.import_channels(chan)

    error_chan = PyChannel(name="err")
    error_chan.build.command()(raise_error)
    shell.main_channel.import_channels(error_chan)

    async with shell:
        async with await shell.interpreter() as interpreter:
            for i in range(5):
                interpreter.feed(f"<chan{i}:foo/>")
            interpreter.feed("<err:raise_error/>")
            interpreter.commit()
            await interpreter.wait_stopped()
            # observe 标记被设置
            assert interpreter.interpretation().observe is True
            # error task 触发了 critical 中断
            critical_tasks = [
                t for t in interpreter.compiled_tasks().values()
                if CommandErrorCode.is_critical(t.errcode)
            ]
            assert len(critical_tasks) >= 1
            # 并行 task 被标记为 cancelled
            cancelled_tasks = [
                t for t in interpreter.managing_tasks().values()
                if t.cancelled()
            ]
            assert len(cancelled_tasks) > 0
