import asyncio

from typing import Literal
from ghoshell_moss.core.concepts.command import (
    CommandTask,
    CommandStackResult,
    CommandTaskResult,
    ObserveError,
)
from ghoshell_moss.core import ChannelCtx, MOSSShell
from ghoshell_common.helpers import Timeleft

__all__ = ["wait"]


async def wait(
        ctml__,
        timeout: float | None = None,
        return_when: Literal['ALL_COMPLETE', 'FIRST_COMPLETE', 'FIRST_EXCEPTION'] = "FIRST_EXCEPTION",
        chans: str | None = "",
):
    """
    Core blocking primitive for grouping and synchronizing CTML command execution.

    This primitive allows you to segment your command output into groups, ensuring
    that commands within a `<wait>` block complete according to the specified
    synchronization policy before proceeding.

    Args:
        ctml__: Nested CTML commands to be executed as a synchronized group.
               The commands will be parsed as sub-tasks and managed by the wait primitive.
        timeout: Optional timeout in seconds.
        return_when: same as asyncio.wait()
        chans: choose which channels to wait, separate by `,` . None means wait all. default wait for main channel done

    Returns:
        result of the commands

    CTML Usage Examples:
        1. Wait for a sequence of commands to complete:
           `<wait><foo/><bar/></wait>`

        2. Wait with timeout (0.5 seconds):
           `<wait timeout="0.5"><foo /><bar/></wait>`
           Unfinished commands will be cancelled when timeout is reached.

        3. Exit when first command completes:
           `<wait return_when="FIRST_COMPLETE"><a:foo/><b:bar/></wait>`
           If b:bar completes first, a:foo will be immediately terminated.

        4. Wait for specific channels done and terminate others
           `<wait chans="speech"><a:foo/><b:bar/><speech:say>something</speech:say></wait>
    """
    shell = ChannelCtx.get_contract(MOSSShell)
    iterable_tasks = shell.parse_tokens_to_command_tasks(ctml__)
    timeleft = Timeleft(timeout or 0.0)

    channel_names = []
    if chans:
        channel_names = chans.split(",")

    async def _wait_for_done(tasks: list[CommandTask]):
        # 创建 wait task group.
        # 如果 channels 为空的话, 意味着对所有 tasks 生效.
        # 如果它为空的话, 意味着 return_when 的逻辑对所有 task 生效.
        _return_when = return_when
        result = CommandTaskResult()
        try:
            if len(channel_names) > 0:
                wait_tasks = []
                for task in tasks:
                    if task.chan in channel_names:
                        wait_tasks.append(task)
            else:
                wait_tasks = tasks
            if len(wait_tasks) == 0:
                raise ValueError(f"No tasks to wait for channels: {chans}")

            wait_task_group = []
            for task in wait_tasks:
                wait_task_group.append(asyncio.create_task(task.wait(throw=True)))
            if len(wait_task_group) == 0:
                return

            if _return_when == "FIRST_COMPLETE":
                wait_done = asyncio.wait(
                    wait_task_group,
                    timeout=timeleft.left() or None,
                    return_when=asyncio.FIRST_COMPLETED,
                )
            elif _return_when == "ALL_COMPLETE":
                wait_done = asyncio.wait(
                    wait_task_group,
                    timeout=timeleft.left() or None,
                    return_when=asyncio.ALL_COMPLETED,
                )
            else:
                wait_done = asyncio.wait(
                    wait_task_group,
                    timeout=timeleft.left() or None,
                    return_when=asyncio.FIRST_EXCEPTION,
                )

            done, pending = await wait_done
            for t in pending:
                t.cancel()
            for task in tasks:
                if task.done():
                    result.join_result(task.task_result())
                else:
                    task.cancel("cancel by wait")
            await asyncio.gather(*[t.wait(throw=False) for t in tasks])
            return result
        except ObserveError as e:
            result.join_result(e.observe)
            return result
        except Exception as e:
            runtime = ChannelCtx.runtime()
            if runtime:
                runtime.logger.exception(e)
            raise
        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()

    return CommandStackResult(
        iterable_tasks,
        _wait_for_done,
    )
