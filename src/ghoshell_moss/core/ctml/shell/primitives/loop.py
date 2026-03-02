import asyncio

from ghoshell_moss.core.concepts.command import (
    CommandTask,
    CommandStackResult,
    CommandTaskResult,
)
from ghoshell_moss.message import Message
from ghoshell_moss.core import ChannelCtx, MOSSShell

__all__ = ['loop']


async def loop(times: int, ctml__):
    """
    loop the given CTML until exception or observe
    the result of the commands are ignored

    the loop will always stop after 100 times

    :param times: the number of times to loop, if <0, means endless loop
    :param ctml__: the looping CTML
    """
    shell = ChannelCtx.get_contract(MOSSShell)
    iterable_tasks = shell.parse_tokens_to_command_tasks(ctml__)

    tasks = []
    async for task in iterable_tasks:
        tasks.append(task)

    if len(tasks) == 0:
        return
    if times == 0:
        return

    loop_times = 0

    async def on_result(got: list[CommandTask]) -> CommandStackResult | CommandTaskResult | None:
        nonlocal loop_times
        loop_times += 1
        if len(got) == 0:
            return None
        _ = await asyncio.gather(*[t.wait(throw=False) for t in got])
        for t in got:
            if not t.success() or t.observe():
                return CommandTaskResult().join_result(t.result())
        new_tasks = []
        for t in got:
            new_tasks.append(t.copy())
        if 0 < times == loop_times:
            return CommandTaskResult(
                observe=True,
                messages=[
                    Message.new(role="system").with_content("loop done at {}".format(times)),
                ]
            )
        if loop_times >= 100:
            return CommandTaskResult(
                observe=True,
                messages=[
                    Message.new(role="system").with_content("loop stopped after 100 times!")
                ]
            )
        return CommandStackResult(
            new_tasks,
            on_result,
        )

    return CommandStackResult(
        tasks,
        on_result,
    )
