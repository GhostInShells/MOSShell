"""shell.dynamic_messages / channel_metas / refresh_metas 的 stale_time 缓存协议.

stale_time 语义订正:
    *不是* "过期时间", 而是 "距上次刷新的时间若小于 stale_time, 复用缓存".
    - stale_time = 0.0 (默认): now - refreshed_at > 0 几乎永真 → 每次都重建.
    - stale_time = T > 0: 在 T 秒内复用上次结果, 不重建.
    - 默认值在三处 (dynamic_messages / refresh_metas / channel_metas) 都是 0.0,
      与 "默认每次都新鲜" 的直觉一致.

只覆盖 ``available_only=True`` 路径 — 仅此分支走 cache,
``available_only=False`` 永远直拉 ``channel_metas(available_only=False)`` 不缓存.
这是 source-of-truth 与 fast-path 的明确分离, 也是测试主题.

测试风格: 探针 (probe) 风格 — stale_time 不是 CTML 协议的一部分, 而是
shell 内部缓存机制. 用 ``has_moss_dynamic_cache`` /
``moss_dynamic_refreshed_at`` 等只读 property 做反身性观察, 不窥视私有属性.
"""
import asyncio

import pytest

from ghoshell_moss.core.ctml import new_ctml_shell


@pytest.mark.asyncio
async def test_dynamic_messages_default_stale_time_rebuilds_each_call():
    """默认 stale_time=0.0: 每次调用都重建缓存.

    `now - refreshed_at > 0` 在两次相邻调用之间几乎永真 (除非 monotonic 分辨率内同帧).
    用 `moss_dynamic_refreshed_at` 时间戳推进作为重建的反推依据.
    """
    shell = new_ctml_shell()
    async with shell:
        msgs1 = shell.dynamic_messages()
        ts1 = shell.moss_dynamic_refreshed_at
        assert ts1 > 0
        assert shell.has_moss_dynamic_cache

        # 让 monotonic 推进, 避免 now == ts1 → now - ts1 == 0 → 不重建.
        await asyncio.sleep(0.01)

        msgs2 = shell.dynamic_messages()
        ts2 = shell.moss_dynamic_refreshed_at
        assert ts2 > ts1, "default stale_time must rebuild on each call"

        # 内容应保持等价 (channel metas 没变).
        assert len(msgs1) == len(msgs2)


@pytest.mark.asyncio
async def test_dynamic_messages_stale_time_reuses_cache():
    """stale_time > 0 且距上次刷新 < stale_time: 复用缓存.

    `moss_dynamic_refreshed_at` 时间戳保持不变 + 返回同一 list 对象身份,
    两者共同钉死 "未重建" 的反推依据.
    """
    shell = new_ctml_shell()
    async with shell:
        msgs1 = shell.dynamic_messages()
        ts1 = shell.moss_dynamic_refreshed_at

        await asyncio.sleep(0.01)

        # stale_time=10s 充分覆盖 10ms 的窗口, 应复用.
        msgs2 = shell.dynamic_messages(stale_time=10.0)
        ts2 = shell.moss_dynamic_refreshed_at
        assert ts2 == ts1, "within stale_time window must not refresh timestamp"
        assert msgs1 is msgs2, "should return cached list object (identity check)"


@pytest.mark.asyncio
async def test_dynamic_messages_stale_time_rebuilds_after_window():
    """stale_time 窗口过去后: 必须重建. 防 "永久缓存" 类型 bug."""
    shell = new_ctml_shell()
    async with shell:
        shell.dynamic_messages()
        ts1 = shell.moss_dynamic_refreshed_at

        # 窗口设置为 50ms, 等待 100ms 确保超过窗口.
        await asyncio.sleep(0.10)

        shell.dynamic_messages(stale_time=0.05)
        ts2 = shell.moss_dynamic_refreshed_at
        assert ts2 > ts1, "must rebuild after stale_time window expires"


@pytest.mark.asyncio
async def test_dynamic_messages_available_only_false_bypasses_cache():
    """available_only=False 旁路 cache: source-of-truth, 永远不走 fast-path.

    可观察的反推: ``moss_dynamic_refreshed_at`` 在 ``available_only=False`` 调用后
    不变 (cache 没动). 两次 available_only=False 返回不同的 list 对象 (无缓存).
    """
    shell = new_ctml_shell()
    async with shell:
        # 先用 available_only=True 建一次 cache.
        cached = shell.dynamic_messages()
        ts_baseline = shell.moss_dynamic_refreshed_at
        assert shell.has_moss_dynamic_cache

        await asyncio.sleep(0.01)

        # available_only=False, 即使设置巨大的 stale_time 也不走 cache 路径.
        fresh1 = shell.dynamic_messages(available_only=False, stale_time=999.0)
        assert isinstance(fresh1, list)
        # cache 时间戳没动.
        assert shell.moss_dynamic_refreshed_at == ts_baseline
        # 不复用缓存对象身份.
        assert fresh1 is not cached

        # 再来一次, 仍然新建对象.
        fresh2 = shell.dynamic_messages(available_only=False, stale_time=999.0)
        assert fresh2 is not fresh1, "available_only=False must not memoize"


@pytest.mark.asyncio
async def test_refresh_metas_stale_time_skip_in_window():
    """refresh_metas 的 stale_time: 在窗口内直接 return True, 不触发底层 runtime 刷新.

    用 `channel_metas_refreshed_at` 时间戳作为 "底层是否被调" 的反推依据.
    """
    shell = new_ctml_shell()
    async with shell:
        # 先做一次真实刷新, 给 baseline 时间戳.
        ok1 = await shell.refresh_metas()
        assert ok1
        ts_baseline = shell.channel_metas_refreshed_at
        assert ts_baseline > 0

        await asyncio.sleep(0.01)

        # 窗口 10s 内, 应直接跳过, 时间戳不动.
        ok2 = await shell.refresh_metas(stale_time=10.0)
        assert ok2
        assert shell.channel_metas_refreshed_at == ts_baseline, (
            "refresh_metas within stale_time must skip backing refresh"
        )

        # 等过窗口, 同样的 stale_time 应触发真刷新.
        await asyncio.sleep(0.10)
        ok3 = await shell.refresh_metas(stale_time=0.05)
        assert ok3
        assert shell.channel_metas_refreshed_at > ts_baseline


@pytest.mark.asyncio
async def test_channel_metas_stale_time_reuses_built():
    """channel_metas 的 stale_time: 同 dynamic_messages 模式.

    内部用 ``channel_metas_built_at`` 控制重建. 注意这条路径与
    ``refresh_metas`` 是两层缓存 —
    refresh_metas 控制是否拉 runtime 远端刷新,
    channel_metas 控制是否重建本地 meta 字典.
    """
    shell = new_ctml_shell()
    async with shell:
        # 触发首次构建.
        shell.channel_metas()
        built_at_1 = shell.channel_metas_built_at
        assert built_at_1 > 0

        await asyncio.sleep(0.01)

        # 窗口内复用.
        shell.channel_metas(stale_time=10.0)
        built_at_2 = shell.channel_metas_built_at
        assert built_at_2 == built_at_1

        # 默认 stale_time=0 几乎必然重建.
        shell.channel_metas()
        built_at_3 = shell.channel_metas_built_at
        assert built_at_3 > built_at_1