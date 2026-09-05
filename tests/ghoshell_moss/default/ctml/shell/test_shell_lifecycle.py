"""shell 生命周期: refresh_metas 的并发/共享协议.

主题: ``shell.refresh_metas()`` 被**并发**调用两次时, shell 层的 meta 生成
(``_update_channel_metas`` → ``on_channel_metas_generation`` 回调) 应当只发生一次.

分层事实:
  - 底层 tree/runtime 各有 "一次只刷一次" 守卫 (``refreshing_task``), 所以 channel
    自身的 ``refresh_meta`` 钩子不会在并发下重复触发。
  - 但 shell 层 ``_refresh_channel_metas`` 此前只有 ``stale_time`` 时间窗去重
    (新鲜度优化, 非并发锁)。两个并发 ``refresh_metas`` 会各自跑一次
    ``_update_channel_metas``, 把 meta 生成回调 fire 两次。

本文件只测这个 shell 层行为: 并发两次 ``refresh_metas``, 断言 meta 生成回调
只 fire 一次 (共享同一轮刷新)。
"""
import asyncio

import pytest

from ghoshell_moss.core.ctml import new_ctml_shell


@pytest.mark.asyncio
async def test_refresh_metas_concurrent_shared_not_duplicated():
    """并发两次 refresh_metas, shell 层 meta 生成必须合并成一次.

    用 ``on_channel_metas_generation`` 观察 ``_update_channel_metas`` 触发的次数,
    这是 shell 层重入 (而非底层 tree 刷新) 的直接观测面。
    """
    shell = new_ctml_shell()

    fires: list = []
    shell.on_channel_metas_generation(lambda _metas: fires.append(1))

    async with shell:
        r1, r2 = await asyncio.gather(
            shell.refresh_metas(stale_time=0.0, timeout=1.0),
            shell.refresh_metas(stale_time=0.0, timeout=1.0),
        )
        assert r1 is True
        assert r2 is True

    assert len(fires) == 1, (
        "concurrent refresh_metas must share a single meta generation "
        f"(on_channel_metas_generation fires once), but it fired {len(fires)} times"
    )
