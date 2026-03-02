import asyncio

from ghoshell_moss.core.concepts.command import (
    CommandTask,
    CommandStackResult,
    CommandTaskResult,
)
from ghoshell_moss.core import ChannelCtx, MOSSShell

__all__ = ['branch']


async def branch(ctml__):
    """
    Conditional branching primitive that selects execution path based on the first command's result.

    Accepts exactly three command tasks:
    1. Condition command: returns a boolean or value convertible to boolean
    2. True branch: executed when condition is truthy
    3. False branch: executed when condition is falsy

    CTML Usage Example:
    <branch>
    <check_something/>      <!-- condition command -->
    <do_if_true/>           <!-- true branch -->
    <do_if_false/>          <!-- false branch -->
    </branch>
    """
    shell = ChannelCtx.get_contract(MOSSShell)
    iterable_tasks = shell.parse_tokens_to_command_tasks(ctml__)

    tasks = []
    async for task in iterable_tasks:
        tasks.append(task)

    if len(tasks) != 3:
        raise ValueError(f"condition only accepts 3 command tasks, got {len(tasks)}")

    async def generate():
        try:
            condition_task = tasks[0]
            yield condition_task
            r = await condition_task
            if r:
                yield tasks[1]
            else:
                yield tasks[2]
        except Exception:
            raise StopAsyncIteration

    async def on_result(got: list[CommandTask]):
        result = CommandTaskResult()
        _ = await asyncio.gather(*[t.wait(throw=False) for t in got])
        for r in got:
            result.join_result(r.result())
        return result

    return CommandStackResult(
        generate(),
        on_result,
    )
