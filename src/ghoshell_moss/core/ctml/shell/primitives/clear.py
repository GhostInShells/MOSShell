import asyncio

from ghoshell_moss.core.concepts.channel import (
    ChannelCtx,
)

__all__ = ["clear"]


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
        await runtime.clear_children()
        return
    clear_all = []
    for chan in chans:
        children_runtime = runtime.fetch_sub_runtime(chan)
        clear_all.append(children_runtime.clear())
    await asyncio.gather(*clear_all, return_exceptions=False)
