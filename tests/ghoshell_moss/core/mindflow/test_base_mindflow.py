import pytest
import asyncio

from ghoshell_moss.core.mindflow.buffer_nucleus import BufferNucleus
from ghoshell_moss.core.mindflow.base_mindflow import BaseMindflow
from ghoshell_moss.core.concepts.mindflow import Mindflow, Signal, Impulse, Priority, Attention


def make_base_mindflow() -> BaseMindflow:
    return BaseMindflow()


@pytest.mark.asyncio
async def test_full_link_signal_to_impulse():
    """测试全链路：Signal -> Nucleus -> Mindflow.on_impulse"""
    mindflow = make_base_mindflow()
    nucleus = BufferNucleus(
        name="test_sensor",
        description="Sensor unit",
        target_signal="vision_event"
    )

    # 会自动注册 bus. 而且启动前不能用 add .
    mindflow.with_nucleus(nucleus)

    async with mindflow:
        sig = Signal.new(name="vision_event", priority=Priority.NOTICE)
        mindflow.on_signal(sig)
        async for attention in mindflow.loop():
            async with attention:
                impulse = attention.peek()
                assert impulse.source == "test_sensor"
                assert impulse.priority == Priority.NOTICE
                break


@pytest.mark.asyncio
async def test_suppress_and_stale_race_condition():
    """验证 suppress 和 stale 结合后的行为"""
    mindflow = make_base_mindflow()
    # 冷静期 0.1s, beat 0.05s
    nucleus = BufferNucleus(
        name="test_sensor",
        description="Sensor unit",
        target_signal="vision_event",
        # 每次 suppress 要 0.1 秒后才能继续.
        suppress_seconds=0.1,
        # 高频尝试 pulse, 实际上会阻塞到 suppress.
        pulse_beat_interval=0.01
    )

    mindflow.with_nucleus(nucleus)

    count = 0

    wait_started = asyncio.Event()

    async def _counter_task():
        nonlocal count
        async for attention in mindflow.loop():
            async with attention:
                wait_started.set()
                assert not attention.peek().is_stale()
                # 模拟 Attention 耗时处理
                await asyncio.sleep(0.15)
                count += 1

    async with mindflow:
        task = asyncio.create_task(_counter_task())

        # 1. 第一个信号，正常通过
        mindflow.on_signal(Signal.new(name="vision_event", priority=Priority.NOTICE))

        # 2. 紧接着发第二个信号，它在 suppress 期间，且 stale 为 0.09s
        await wait_started.wait()
        mindflow.on_signal(Signal.new(name="vision_event", priority=Priority.NOTICE, stale_timeout=0.09))

        # 3. 等待足够久，让冷静期过期，让第二个信号 Stale
        await asyncio.sleep(0.15)

        mindflow.close()
        await task

    # 结果验证：只有第一个信号成功了，第二个被 suppress 压制并因 Stale 被丢弃
    assert count == 1


@pytest.mark.asyncio
async def test_mindflow_able_to_close():
    """测试全链路：Signal -> Nucleus -> Mindflow.on_impulse"""
    mindflow = make_base_mindflow()
    nucleus = BufferNucleus(
        name="test_sensor",
        description="Sensor unit",
        target_signal="vision_event"
    )

    # 会自动注册 bus. 而且启动前不能用 add .
    mindflow.with_nucleus(nucleus)
    async with mindflow:
        sig = Signal.new(name="vision_event", priority=Priority.NOTICE)
        mindflow.on_signal(sig)
        async for attention in mindflow.loop():
            async with attention:
                impulse = attention.peek()
                assert impulse.source == "test_sensor"
                assert impulse.priority == Priority.NOTICE
                # 调用之后应该不会阻塞, 都会退出.
                mindflow.close()


@pytest.mark.asyncio
async def test_mindflow_run_in_task():
    """测试全链路：Signal -> Nucleus -> Mindflow.on_impulse"""
    mindflow = make_base_mindflow()
    nucleus = BufferNucleus(
        name="test_sensor",
        description="Sensor unit",
        target_signal="vision_event"
    )

    count = 0

    async def _run_in_task():
        nonlocal count
        # 会自动注册 bus. 而且启动前不能用 add .
        mindflow.with_nucleus(nucleus)
        async with mindflow:
            sig = Signal.new(name="vision_event", priority=Priority.NOTICE)
            mindflow.on_signal(sig)
            async for attention in mindflow.loop():
                async with attention:
                    impulse = attention.peek()
                    assert impulse.source == "test_sensor"
                    assert impulse.priority == Priority.NOTICE
                    # 验证完 impulse 直接退出.
                    count += 1
                assert attention.is_aborted()
        assert not mindflow.is_running()

    task = asyncio.create_task(_run_in_task())
    await asyncio.sleep(0.1)
    assert not task.done()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    # 只有一个信号, 不会有第二个行为.
    assert count == 1


@pytest.mark.asyncio
async def test_mindflow_run_with_multi_signal():
    """测试全链路：Signal -> Nucleus -> Mindflow.on_impulse"""
    mindflow = make_base_mindflow()
    nucleus = BufferNucleus(
        name="test_sensor",
        description="Sensor unit",
        target_signal="vision_event",
    )

    count = 0

    done_flag = asyncio.Event()

    mindflow.with_nucleus(nucleus)

    async def _run_in_task():
        nonlocal count
        # 会自动注册 bus. 而且启动前不能用 add .
        async for attention in mindflow.loop():
            async with attention:
                impulse = attention.peek()
                assert impulse.priority == Priority.NOTICE
                count += 1
            done_flag.set()
            assert attention.is_aborted()

    async def _main():
        await asyncio.sleep(0.0)
        sig = Signal.new(name="vision_event", priority=Priority.NOTICE)
        mindflow.on_signal(sig)
        await asyncio.sleep(0.0)
        await done_flag.wait()
        assert count == 1
        done_flag.clear()
        # 尝试发送第二个信号.
        sig = Signal.new(name="vision_event", priority=Priority.NOTICE)
        mindflow.on_signal(sig)
        await asyncio.sleep(0.1)
        await done_flag.wait()
        # 然后就直接退出.
        mindflow.close()

    async with mindflow:
        task = asyncio.create_task(_run_in_task())
        main_task = asyncio.create_task(_main())
        await asyncio.wait([task, main_task], return_when=asyncio.FIRST_COMPLETED)
        await task
        await main_task

    # 只有一个信号, 不会有第二个行为.
    assert count == 2
