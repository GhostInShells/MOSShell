import asyncio

from ghoshell_moss.core.concepts.command import (
    CommandStackResult,
    PyCommand,
    BaseCommandTask,
)

__all__ = ["sleep"]


async def _sleep(duration: float):
    await asyncio.sleep(duration)


sleep_command = PyCommand(_sleep)


async def sleep(duration: float, chan: str = ""):
    """
    停止 duration 秒, 阻塞后续命令执行.
    :param duration: 单位是秒
    :param chan: 指定在哪个轨道进行等待, 默认在根轨道阻塞.
    """
    if duration <= 0.0:
        return
    if chan == "":
        await _sleep(duration)
        return

    task = BaseCommandTask.from_command(sleep_command, chan_=chan, kwargs=dict(duration=duration))
    return CommandStackResult(
        [task],
    )
