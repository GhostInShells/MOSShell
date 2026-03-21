import asyncio

from ghoshell_moss.core.concepts.channel import (
    ChannelCtx,
    ChannelRuntime,
)

__all__ = ["clear"]


async def _clear_children(runtime: ChannelRuntime) -> None:
    """
    由于执行的命令本身不需要清空, 所以 clear 本质上是清空子轨道.
    """
    runtime = ChannelCtx.runtime()
    if runtime is None:
        return
    children = runtime.sub_channels()
    if len(children) == 0:
        return
    group_clear = []

    async def clear_child(_name: str):
        sub_runtime = await runtime.fetch_sub_runtime(_name)
        if sub_runtime and sub_runtime.is_running():
            await sub_runtime.clear()

    for name in children:
        sub_name = name
        group_clear.append(clear_child(sub_name))
    await asyncio.gather(*group_clear, return_exceptions=False)
    return


async def clear(chan: str = ""):
    """
    清空指定 Channel 和所有子轨的运行状态, 会递归地清空.
    :param chan: 指定在清空哪些 Channel 的执行状态, 用 `,` 隔开多个. 为空的话清空全部.
    """
    runtime = ChannelCtx.runtime()
    if runtime is None:
        return
    chans = chan.split(",")
    if not chans or "" in chans or "__main__" in chans:
        await _clear_children(runtime)
        return
    clear_all = []
    for chan in chans:
        children_runtime = await runtime.fetch_sub_runtime(chan)
        clear_all.append(children_runtime.clear())
    await asyncio.gather(*clear_all, return_exceptions=False)
