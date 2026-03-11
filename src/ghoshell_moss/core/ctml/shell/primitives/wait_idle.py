import asyncio

from ghoshell_moss.core.concepts.channel import (
    ChannelCtx,
    ChannelRuntime,
)

__all__ = ["wait_idle"]


async def _wait_children_idle(runtime: ChannelRuntime, timeout: float | None):
    """
    由于执行的命令本身不需要清空, 所以 clear 本质上是清空子轨道.
    """
    runtime = ChannelCtx.runtime()
    if runtime is None:
        return
    children = runtime.sub_channels()
    if len(children) == 0:
        return None
    group_wait = []

    async def wait_child(_name: str):
        sub_runtime = await runtime.fetch_sub_runtime(_name)
        if sub_runtime and sub_runtime.is_running():
            if timeout is None:
                await sub_runtime.wait_idle()
            else:
                try:
                    await asyncio.wait_for(sub_runtime.wait_idle(), timeout)
                except asyncio.TimeoutError:
                    await sub_runtime.clear()

    for name in children:
        sub_name = name
        group_wait.append(wait_child(sub_name))
    await asyncio.gather(*group_wait, return_exceptions=False)


async def _wait_for_runtime(_runtime: ChannelRuntime, _timeout: float | None):
    if _timeout is not None and _timeout > 0.0:
        try:
            await asyncio.wait_for(_runtime.wait_idle(), _timeout)
        except asyncio.TimeoutError:
            # 直接清空子轨.
            await _runtime.clear()
    else:
        await _runtime.wait_idle()


async def wait_idle(chan: str = "", timeout: float | None = None):
    """
    等待 指定轨道和它的子轨道的命令执行结束.
    :param chan: 指定等待哪个轨道执行完毕. 为空在主轨等待. 多个轨道名用 `,` 隔开.
    :param timeout: 如果设置超时, 超时后会清空目标轨道.
    """
    if timeout is not None and timeout < 0:
        raise ValueError("timeout must be greater than or equal to 0.")

    runtime = ChannelCtx.runtime()
    if runtime is None:
        return
    chans = chan.split(",")
    if chan == "" or "" in chans or "__main__" in chans:
        # 之所以 wait children, 是因为当前 wait idle 就在主轨执行, 如果它等待自己 idle 会死锁.
        await _wait_children_idle(runtime, timeout)
        return

    wait_all = []
    for sub_chan in chans:
        children_runtime = await runtime.fetch_sub_runtime(sub_chan)
        if children_runtime:
            wait_all.append(_wait_for_runtime(children_runtime, timeout))
    await asyncio.gather(*wait_all, return_exceptions=False)
