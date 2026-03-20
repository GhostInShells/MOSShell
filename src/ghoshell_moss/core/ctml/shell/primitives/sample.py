import asyncio
import random

from ghoshell_moss.core.concepts.command import (
    CommandTask,
    CommandStackResult,
    CommandTaskResult,
)
from ghoshell_moss.core import ChannelCtx, MOSShell

__all__ = ["sample"]


async def sample(ctml__, pick: int = 1):
    """
    Random selection primitive that randomly selects and executes N commands from the given CTML.

    Randomly selects 'pick' number of commands from the provided CTML and executes them.
    The selection is without replacement (each command can be selected at most once).
    Commands are executed sequentially in random order.

    CTML Usage Examples:
        1. Select and execute 1 random command from 3:
           <sample><task1/><task2/><task3/></sample>

        2. Select and execute 2 random commands from 5:
           <sample pick="2"><t1/><t2/><t3/><t4/><t5/></sample>

        3. Execute all tasks in random order (pick equals task count):
           <sample pick="3"><t1/><t2/><t3/></sample>
    """
    shell = ChannelCtx.get_contract(MOSShell)
    iterable_tasks = shell.parse_tokens_to_command_tasks(ctml__)

    tasks = []
    async for task in iterable_tasks:
        tasks.append(task)

    # 验证参数
    if pick < 1:
        raise ValueError(f"sample pick must be >= 1, got {pick}")

    if len(tasks) < pick:
        raise ValueError(f"sample requires at least {pick} tasks to pick from, but only got {len(tasks)} tasks")

    # 随机选择指定数量的任务（不放回抽样）
    selected_tasks = random.sample(tasks, pick)

    async def generate():
        """按随机顺序逐个生成选中的任务"""
        try:
            for task in selected_tasks:
                yield task
        except Exception:
            raise StopAsyncIteration

    async def on_result(got: list[CommandTask]):
        """等待所有执行的任务完成，合并结果"""
        result = CommandTaskResult()
        if len(got) == 0:
            return result

        # 等待所有任务完成（不抛出异常）
        _ = await asyncio.gather(*[t.wait(throw=False) for t in got])

        # 合并所有任务的结果
        for task in got:
            task_result = task.result()
            if task_result is not None:
                result.join_result(task_result)
        return result

    return CommandStackResult(
        generate(),
        on_result,
    )
