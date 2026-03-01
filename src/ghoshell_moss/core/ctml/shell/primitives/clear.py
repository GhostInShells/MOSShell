import asyncio

from ghoshell_moss.core.concepts.channel import (
    ChannelCtx, ChannelRuntime,
)

__all__ = ["clear"]


async def _clear_children(runtime: ChannelRuntime):
    """
    由于执行的命令本身不需要清空, 所以 clear 本质上是清空子轨道.
    """
    runtime = ChannelCtx.runtime()
    if runtime is None:
        return
    children = runtime.sub_channels()
    if len(children) == 0:
        return None
    group_clear = []

    async def clear_child(_name: str):
        sub_runtime = await runtime.fetch_sub_runtime(_name)
        if sub_runtime and sub_runtime.is_running():
            await sub_runtime.clear()

    for name in children:
        sub_name = name
        group_clear.append(clear_child(sub_name))
    await asyncio.gather(*group_clear, return_exceptions=False)


async def clear(chan: str = ""):
    """
    清空指定 Channel 和所有子轨的运行状态, 会递归地清空.
    :param chan: 指定在清空哪个 Channel 的执行状态, 默认在根 Channel
    """
    runtime = ChannelCtx.runtime()
    if runtime is None:
        return
    if chan == "":
        await _clear_children(runtime)
        return
    children_runtime = await runtime.fetch_sub_runtime(chan)
    if children_runtime:
        await children_runtime.clear()
