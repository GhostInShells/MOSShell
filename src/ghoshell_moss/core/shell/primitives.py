import asyncio

from ghoshell_moss.core.concepts.command import (
    CommandTask,
    CommandStackResult,
    PyCommand,
    BaseCommandTask,
)
from ghoshell_moss.core.concepts.errors import (
    CommandErrorCode
)
from ghoshell_moss.core import ChannelCtx, MOSSShell

__all__ = ['wait']


async def wait(
        tokens__,
        timeout: float | None = None,
        return_when: str = "ALL_COMPLETE",
        channels: str = "",
):
    """
    核心阻塞原语, 可以阻塞等待 一段 CTML 指令 彻底结束.
    用这种方式, 能把你输出的命令分成几组, 分段来执行, 保证其阻塞效果.

    :param tokens__: 嵌套的 CTML 指令, 会由 wait 原语统一管理.
    :param timeout: 超时时间, 不为空的话, 在时间到达后会主动中断所有的指令, 让执行继续.
    :param return_when: 定义 ctml__ 命令整体结束的时机:
        ALL_COMPLETE: 等待所有指令运行结束后, 才继续执行后续指令.
        FIRST_COMPLETE: 当有一个指令执行成功时, 将其它指令设置为取消.
    :param channels: 指定 return when 生效对应的 channel 名, 用 , 隔开. 为空的话, 则 return_when 针对所有指令.

    CTML 用法:
        等待一串命令执行完: `<wait><foo/><bar/></wait>`  所有参数不必填写. 默认值即可.
        等待一串命令到超时: `<wait timeout="0.5"><foo /><bar/></wait>` 当时间达到时, 未完成的命令都会被取消.
        第一个命令完成时退出: `<wait return_when="FIRST_COMPLETE"><a:foo/><b:bar/></wait>` 如果 b:bar 先完成, a:foo 会立刻被终止.
        指定生效的通道: `<wait channels="a" return_when="FIRST_COMPLETE"><a: foo/><b:bar></wait>` 这时 b:bar 先执行完, a:foo 也不会被终止.
    """
    shell = ChannelCtx.get_contract(MOSSShell)
    iterable_tasks = shell.parse_tokens_to_command_tasks(tokens__)

    channel_names = []
    if channels:
        channel_names = channels.split(",")
    timeout_task = None
    if timeout is not None and timeout > 0.0:
        timeout_task = asyncio.create_task(asyncio.sleep(timeout))

    async def _wait_for_done(tasks: list[CommandTask]) -> str | None:
        # 创建 wait task group.
        # 如果 channels 为空的话, 意味着对所有 tasks 生效.
        # 如果它为空的话, 意味着 return_when 的逻辑对所有 task 生效.
        _return_when = return_when
        try:
            wait_task_group = []
            if timeout_task:
                wait_task_group.append(timeout_task)
            if len(channel_names) == 0:
                wait_task_group = tasks
            else:
                for task in tasks:
                    if task.chan in channel_names:
                        wait_task_group.append(task)
                if len(wait_task_group) == 0:
                    raise CommandErrorCode.VALUE_ERROR.error(f"generated command not in channels: {channel_names}")

            if _return_when == "FIRST_COMPLETE":
                wait_done = asyncio.create_task(asyncio.wait(
                    [asyncio.create_task(t.wait(throw=False)) for t in wait_task_group],
                    return_when=asyncio.FIRST_COMPLETED,
                ))
            elif return_when == "ALL_COMPLETE":
                wait_done = asyncio.wait(
                    [asyncio.create_task(t.wait(throw=False)) for t in wait_task_group],
                    return_when=asyncio.ALL_COMPLETED,
                )
                _return_when = "ALL_COMPLETE"
            else:
                raise ValueError(f"invalid return_when: {return_when}")

            if not timeout_task:
                done, pending = await wait_done
                for t in pending:
                    t.cancel()
                return return_when
            else:
                wait_done_task = asyncio.create_task(wait_done)
                done, pending = await asyncio.wait(
                    [timeout_task, wait_done_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for t in pending:
                    t.cancel()
                if wait_done in done:
                    return _return_when
                else:
                    canceling = 0
                    for t in tasks:
                        if not t.done():
                            canceling += 1
                    return f"timeout and cancel {canceling} command"

        except asyncio.CancelledError:
            pass
        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()

    return CommandStackResult(
        iterable_tasks,
        _wait_for_done,
    )


async def _sleep(timeout: float):
    await asyncio.sleep(timeout)
    return


_sleep_command = PyCommand(_sleep)


async def sleep(seconds: float, chan: str = ""):
    """

    """
    if chan == "":
        await asyncio.sleep(seconds)
        return
    runtime = ChannelCtx.runtime()
    sub = await runtime.fetch_sub_runtime(chan)
    if sub is None:
        raise ValueError(f"invalid chan: {chan}")

    task = BaseCommandTask.from_command(_sleep_command, chan_=chan, kwargs={"seconds": seconds})
    return CommandStackResult(
        [task],
    )
