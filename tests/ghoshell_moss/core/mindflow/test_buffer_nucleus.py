import pytest
import asyncio
import time
from ghoshell_moss.core.mindflow.buffer_nucleus import BufferNucleus
from ghoshell_moss.core.concepts.mindflow import Signal, Priority, Impulse


# 简单的 Mock 信号对象
def create_mock_signal(name: str, priority: Priority = Priority.INFO, stale: float = 0.0) -> Signal:
    # 假设 Signal 的构造函数满足这些参数
    return Signal(
        id=f"test_id_{time.time()}",
        name=name,
        priority=priority,
        messages=[],
        prompt="test prompt",
        stale_timeout=stale,
    )


@pytest.mark.asyncio
async def test_buffer_nucleus_basic_flow():
    """验证最基本的：收到信号 -> 推送 Impulse"""
    nucleus = BufferNucleus(
        name="test_nucleus",
        description="test",
        target_signal="test_signal"
    )

    notified_impulses = []

    def mock_notify(impulse):
        notified_impulses.append(impulse)

    nucleus.with_bus(lambda s: None, mock_notify)

    async with nucleus:
        sig = create_mock_signal("test_signal")
        nucleus.add_signal(sig)

        # 等待异步任务执行
        await asyncio.sleep(0.1)

        assert len(notified_impulses) == 1
        assert notified_impulses[0].source == "test_nucleus"
        assert nucleus.peek() is not None


@pytest.mark.asyncio
async def test_buffer_nucleus_suppress():
    """验证压制逻辑：在冷静期内不推送"""
    nucleus = BufferNucleus(
        name="test_nucleus",
        description="test",
        target_signal="test_signal",
        suppress_seconds=1.0
    )

    notified_count = 0

    def mock_notify(impulse):
        nonlocal notified_count
        notified_count += 1

    nucleus.with_bus(lambda s: None, mock_notify)

    higher_impulse = Impulse(
        priority=2,
    )

    async with nucleus:
        # 第一次信号正常触发
        nucleus.add_signal(create_mock_signal("test_signal"))
        await asyncio.sleep(0.1)
        assert notified_count == 1

        # 压制
        nucleus.suppress(higher_impulse)

        # 第二次信号，被压制，count 不应该增加
        nucleus.add_signal(create_mock_signal("test_signal"))
        await asyncio.sleep(0.1)
        assert notified_count == 1


@pytest.mark.asyncio
async def test_buffer_nucleus_buffer_limit():
    """验证 Buffer 限制：超过 size 后 FIFO"""
    nucleus = BufferNucleus(
        name="test_nucleus",
        description="test",
        target_signal="test_signal",
        buffer_size=2
    )

    async with nucleus:
        nucleus.add_signal(create_mock_signal("test_signal"))
        nucleus.add_signal(create_mock_signal("test_signal"))
        nucleus.add_signal(create_mock_signal("test_signal"))

        await asyncio.sleep(0.1)
        # 检查内部 buffer 长度
        assert len(nucleus._signals) == 2


@pytest.mark.asyncio
async def test_pop_clears_buffer():
    """验证 pop_impulse 后缓冲会被清空"""
    nucleus = BufferNucleus(
        name="test_nucleus",
        description="test",
        target_signal="test_signal"
    )

    async with nucleus:
        nucleus.add_signal(create_mock_signal("test_signal"))
        await asyncio.sleep(0.1)
        assert nucleus.peek() is not None

        nucleus.pop_impulse(nucleus.peek())
        await asyncio.sleep(0.1)
        assert nucleus.peek() is None

        await asyncio.sleep(0.1)
        assert nucleus.peek() is None


@pytest.mark.asyncio
async def test_signal_and_impulse_stale():
    """验证 pop_impulse 后缓冲会被清空"""
    nucleus = BufferNucleus(
        name="test_nucleus",
        description="test",
        target_signal="test_signal",
        pulse_beat_interval=0.03,
    )

    async with nucleus:
        nucleus.add_signal(create_mock_signal("test_signal", stale=0.05))
        await asyncio.sleep(0.01)
        assert nucleus.peek() is not None
        await asyncio.sleep(0.1)
        assert nucleus.peek() is None
        assert nucleus.peek(no_stale=False) is None
