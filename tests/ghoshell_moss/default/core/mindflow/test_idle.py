"""Mindflow idle 治理测试 — 验证 is_idle / when_idle / moments 协议承诺的行为.

idle 的语义 (blueprint): ``is_idle`` 表示 "没有 attention 且没有 impulse".
本文件只测这份协议契约:

1. 无信号时启动 -> 立刻进入 idle, 回调携带 moments 触发.
2. 信号进入处理 -> 处理中非 idle; 处理完成且无新 impulse -> 回到 idle, 回调再次触发.
   这同时验证 moments 里已折进处理帧.
3. disposer 生效 -> 取消注册后不再触发.
4. 同一段 idle 只触发一次回调 (不在 idle 内反复 fire).
5. 发 signal 生成 attention 时, 阻塞的 idle 回调被取消, 证明 idle 反转.
6. close 应清理仍在跑的 idle 回调 (回归: 开发时丢失 close 时的取消).

驱动方式: 每个需要"处理信号"的测试, 都开一个真实的 thinking loop task —
消费到第一帧 thinking 时 set ``processing_started``, 主流程阻塞在其上, 断言后
``release_processing.set()`` 让 loop break. 收尾统一 cancel 两个 task (thinking/action
loop 会阻塞在 AsyncIterator 上, 不依赖 close 自然退出). 风格遵循 tests/CLAUDE.md:
测行为而非实现, 不访问 _private 成员.
"""
import asyncio

import pytest

from ghoshell_moss.core.blueprint.mindflow import Priority, Signal
from ghoshell_moss.core.mindflow import BaseMindflow
from ghoshell_moss.core.mindflow.buffer_nucleus import BufferNucleus
from ghoshell_moss.contracts.logger import get_console_logger


def make_mindflow() -> BaseMindflow:
    mf = BaseMindflow(logger=get_console_logger())
    mf.with_nucleus(
        BufferNucleus(name="sensor", description="sensor unit", target_signal="vision_event")
    )
    return mf


def make_thinking_loop(mf: BaseMindflow, processing_started: asyncio.Event, release_processing: asyncio.Event):
    """真实 thinking loop: 消费到第一帧 set processing_started, 阻塞到 release_processing 后 break."""
    async def _loop():
        async for think in mf.thinking_loop():
            async with think:
                think.attention.draw_from()
                processing_started.set()
                await release_processing.wait()
                break
    return _loop


def make_action_loop(mf: BaseMindflow):
    async def _loop():
        async for action in mf.action_loop():
            async with action:
                async for _delta in action.logos():
                    pass
    return _loop


async def stop_loops(mf: BaseMindflow, *tasks: asyncio.Task) -> None:
    """收尾: close + 显式 cancel thinking/action loop task.

    loop task 阻塞在 AsyncIterator 的 await 上, close 不保证它们退出, 直接 cancel.
    """
    mf.close()
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_idle_fires_on_startup_when_no_signal():
    """无信号启动 -> 立刻 idle, 回调携带 moments (与 mindflow.moments 同一实例)."""
    mf = make_mindflow()
    seen = []

    def _on_idle(moments):
        seen.append(moments)

    async def _on_idle_async(moments):
        seen.append(moments)

    mf.when_idle(_on_idle)
    mf.when_idle(_on_idle_async)

    async with mf:
        await mf.wait_started()
        # 无信号: 应该已经进入 idle (startup 即触发回调).
        await asyncio.sleep(0.2)
        assert mf.is_idle() is True
        assert len(seen) == 2
        # 回调收到的 moments 即 mindflow 持有的 moments 观测器.
        for observer in seen:
            assert observer is mf.moments


@pytest.mark.asyncio
async def test_idle_is_false_while_processing_and_refires_after_complete():
    """信号处理中非 idle; 处理完成且无新 impulse 后回到 idle, 回调携带处理帧."""
    mf = make_mindflow()
    idle_moments = []
    idle_refired = asyncio.Event()
    processing_started = asyncio.Event()
    release_processing = asyncio.Event()

    def _on_idle(moments):
        idle_moments.append(moments)
        idle_refired.set()

    mf.when_idle(_on_idle)

    thinking_loop = make_thinking_loop(mf, processing_started, release_processing)
    action_loop = make_action_loop(mf)

    async with mf:
        await mf.wait_started()
        # 启动时首次 idle 已触发, 清掉, 只观察本轮.
        await asyncio.wait_for(idle_refired.wait(), 2)
        idle_moments.clear()
        idle_refired.clear()

        t_think = asyncio.create_task(thinking_loop())
        t_action = asyncio.create_task(action_loop())

        mf.add_signal(Signal.new("vision_event", priority=Priority.NOTICE))
        # 主流程阻塞在"拿到第一个 thinking" -> 说明 attention 已生产.
        await asyncio.wait_for(processing_started.wait(), 2)
        assert mf.is_idle() is False
        assert idle_moments == []

        release_processing.set()
        # 处理完成, 无新 impulse, 应回到 idle 并再次触发回调.
        await asyncio.wait_for(idle_refired.wait(), 2)
        assert mf.is_idle() is True
        assert len(idle_moments) == 1
        # 回调的时刻已折进处理帧.
        assert len(idle_moments[0].moments()) >= 1

        await stop_loops(mf, t_think, t_action)


@pytest.mark.asyncio
async def test_when_idle_disposer_stops_future_callbacks():
    """when_idle 返回 disposer; 调用后后续 idle 不再触发该回调."""
    mf = make_mindflow()
    calls = []
    idle_refired = asyncio.Event()
    processing_started = asyncio.Event()
    release_processing = asyncio.Event()

    def _on_idle(moments):
        calls.append("fired")
        idle_refired.set()

    disposer = mf.when_idle(_on_idle)

    thinking_loop = make_thinking_loop(mf, processing_started, release_processing)
    action_loop = make_action_loop(mf)

    async with mf:
        await mf.wait_started()
        # 首次 idle 已触发.
        await asyncio.wait_for(idle_refired.wait(), 2)
        assert len(calls) == 1

        # 取消注册.
        disposer()
        idle_refired.clear()

        t_think = asyncio.create_task(thinking_loop())
        t_action = asyncio.create_task(action_loop())
        mf.add_signal(Signal.new("vision_event", priority=Priority.NOTICE))
        await asyncio.wait_for(processing_started.wait(), 2)
        release_processing.set()
        # 处理完成回到 idle, 但已 disposer, 不再触发.
        await asyncio.sleep(0.3)
        assert mf.is_idle() is True
        assert len(calls) == 1
        assert not idle_refired.is_set()

        await stop_loops(mf, t_think, t_action)


@pytest.mark.asyncio
async def test_idle_callback_fires_once_per_idle_episode():
    """同一段 idle 内回调只触发一次, 不重复fire."""
    mf = make_mindflow()
    calls = []
    idle_refired = asyncio.Event()
    processing_started = asyncio.Event()
    release_processing = asyncio.Event()

    def _on_idle(moments):
        calls.append(mf.is_idle())
        idle_refired.set()

    mf.when_idle(_on_idle)

    thinking_loop = make_thinking_loop(mf, processing_started, release_processing)
    action_loop = make_action_loop(mf)

    async with mf:
        await mf.wait_started()
        await asyncio.wait_for(idle_refired.wait(), 2)
        calls.clear()
        idle_refired.clear()

        t_think = asyncio.create_task(thinking_loop())
        t_action = asyncio.create_task(action_loop())

        # 第一段信号: 处理 -> idle.
        mf.add_signal(Signal.new("vision_event", priority=Priority.NOTICE))
        await asyncio.wait_for(processing_started.wait(), 2)
        release_processing.set()
        await asyncio.wait_for(idle_refired.wait(), 2)
        assert len(calls) == 1
        # 回调运行时 is_idle() 通道为 True — 说明回调确实发生在 idle 已建立的时刻.
        assert calls[0] is True

        # 停在 idle 上再等一拍, 不应重复触发.
        idle_refired.clear()
        await asyncio.sleep(0.2)
        assert len(calls) == 1
        assert not idle_refired.is_set()

        await stop_loops(mf, t_think, t_action)


@pytest.mark.asyncio
async def test_signal_turns_off_idle_by_cancelling_blocked_callback():
    """attention 生产时 idle 已停止: 阻塞死的 idle 回调被取消, 其 finally 得证.

    用一个永不 ok 的 await 占住回调, 回调一旦开始便是"仍在跑". 发 signal 生成
    attention -> ``_stop_idling`` 取消该回调 -> finally 记录, 证明 idle 反转.
    """
    mf = make_mindflow()
    callback_started = asyncio.Event()
    callback_cancelled = asyncio.Event()
    processing_started = asyncio.Event()
    release_processing = asyncio.Event()

    async def _on_idle(moments):
        try:
            callback_started.set()
            # 阻塞到被 cancel; 回调一旦停留在此, 即证明 idle 回调仍在执行.
            await asyncio.Event().wait()
        finally:
            callback_cancelled.set()

    mf.when_idle(_on_idle)

    thinking_loop = make_thinking_loop(mf, processing_started, release_processing)
    action_loop = make_action_loop(mf)

    async with mf:
        await mf.wait_started()
        # idle 触发 -> 回调进入阻塞.
        await asyncio.wait_for(callback_started.wait(), 2)
        assert mf.is_idle() is True
        assert not callback_cancelled.is_set()

        t_think = asyncio.create_task(thinking_loop())
        t_action = asyncio.create_task(action_loop())
        # 发送 signal -> attention 生产, _loop_attention 在 yield 前调用 _stop_idling.
        mf.add_signal(Signal.new("vision_event", priority=Priority.NOTICE))
        # 主流程阻塞在"拿到第一个 thinking" -> attention 已生产 -> _stop_idling 已执行.
        await asyncio.wait_for(processing_started.wait(), 2)
        assert callback_cancelled.is_set()
        assert mf.is_idle() is False

        release_processing.set()
        await stop_loops(mf, t_think, t_action)


@pytest.mark.asyncio
async def test_close_cancels_blocked_idle_callback():
    """close 应清理仍在跑的 idle 回调 (回归: 开发时丢失 close 时的取消).

    idle 触发后, 一个阻塞的 async 回调不能被放养到事件循环关闭. exit 后
    该回调必须被取消, finally 得证.
    """
    mf = make_mindflow()
    callback_started = asyncio.Event()
    callback_cancelled = asyncio.Event()

    async def _on_idle(moments):
        try:
            callback_started.set()
            await asyncio.Event().wait()
        finally:
            callback_cancelled.set()

    mf.when_idle(_on_idle)

    async with mf:
        await mf.wait_started()
        await asyncio.wait_for(callback_started.wait(), 2)
        assert not callback_cancelled.is_set()
        # 退出 async with -> __aexit__ 应清理阻塞的 idle 回调.
        mf.close()
    await asyncio.wait_for(callback_cancelled.wait(), 2)
    assert callback_cancelled.is_set()
