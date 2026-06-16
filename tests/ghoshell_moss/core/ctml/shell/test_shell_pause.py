"""Shell pause() 行为测试 — 急停级 feature.

测试纪律:
    - 只用公开 API, 不调私有方法.
    - 时序靠 ``asyncio.Event``, 不靠 ``asyncio.sleep`` 推测.
    - 所有 await 加 ``asyncio.wait_for(timeout=...)`` 防挂.
"""

import asyncio

import pytest
from ghoshell_moss.core import new_channel
from ghoshell_moss.core.concepts.errors import PausedError
from ghoshell_moss.core.ctml.shell import new_ctml_shell


@pytest.mark.asyncio
async def test_pause_stops_running_task():
    """pause(True) 取消正在运行的命令, 关闭当前 interpreter."""
    shell = new_ctml_shell()
    chan = new_channel("a")
    shell.main_channel.import_channels(chan)

    started = asyncio.Event()
    cancelled = asyncio.Event()

    @chan.build.command()
    async def long_task() -> str:
        started.set()
        try:
            await asyncio.sleep(10)
            return "never"
        except asyncio.CancelledError:
            cancelled.set()
            raise

    async with shell:
        async with await shell.interpreter() as interpreter:
            interpreter.feed("<a:long_task />")
            interpreter.commit()
            await interpreter.wait_compiled()
            await asyncio.wait_for(started.wait(), timeout=1.0)

            done = asyncio.Event()
            shell.pause(True, callback=lambda: done.set())

            await asyncio.wait_for(done.wait(), timeout=2.0)
            # callback fired → clear 完成 → task 已被 cancel
            assert cancelled.is_set()
            assert interpreter.is_closed()


@pytest.mark.asyncio
async def test_pause_blocks_new_interpreter():
    """pause(True) 后 shell.interpreter() 抛出 PausedError."""
    shell = new_ctml_shell()
    async with shell:
        shell.pause(True)
        with pytest.raises(PausedError):
            await shell.interpreter()


@pytest.mark.asyncio
async def test_pause_blocks_new_task():
    """pause(True) 后 push_task() 抛出 PausedError."""
    from ghoshell_moss.core.concepts.command import BaseCommandTask

    shell = new_ctml_shell()
    chan = new_channel("test")
    shell.main_channel.import_channels(chan)

    @chan.build.command()
    async def dummy() -> int:
        return 1

    async with shell:
        shell.pause(True)

        cmd = await shell.get_command("test", "dummy")
        task = BaseCommandTask.from_command(cmd, "test")
        with pytest.raises(PausedError):
            shell.push_task(task)


@pytest.mark.asyncio
async def test_pause_idempotent():
    """pause(True) 多次调用安全, 复用 interpreter 仍被阻塞."""
    shell = new_ctml_shell()
    async with shell:
        shell.pause(True)
        shell.pause(True)
        shell.pause(True)

        with pytest.raises(PausedError):
            await shell.interpreter()


@pytest.mark.asyncio
async def test_pause_not_running_noop():
    """shell 未启动时 pause(True) 不抛异常."""
    shell = new_ctml_shell()
    shell.pause(True)
