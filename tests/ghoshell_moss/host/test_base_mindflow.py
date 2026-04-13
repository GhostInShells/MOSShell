import asyncio
import pytest
import pytest_asyncio
from ghoshell_moss.host.abcd.mindflow import InputSignal, Impulse
from ghoshell_moss.host.base_mindflow import MindflowBus, PriorityMindPulse


# 定义一个异步 fixture 来自动化管理 Mindflow 的启动和关闭
@pytest_asyncio.fixture
async def mindflow_bus():
    # 1. 初始化一个测试用的 Pulse，监听输入信号
    pulse = PriorityMindPulse(
        pulse_name="test_input_pulse",
        description="Testing Pulse",
        signals=[InputSignal.signal_name()],
        instruction="Instruction for AI"
    )

    # 2. 初始化 Bus
    bus = MindflowBus(pulse)

    # 3. 启动上下文（执行 __aenter__）
    async with bus as started_bus:
        yield started_bus
    # 退出时会自动执行 __aexit__


@pytest.mark.asyncio
async def test_input_signal_flow(mindflow_bus: MindflowBus):
    """
    测试基础链路：发送信号 -> 等待脉冲 -> 弹出脉冲
    """
    # 1. 构造信号
    test_content = "Hello, Moss"
    # 使用 InputSignal 协议生成 Signal
    signal = InputSignal().to_signal(test_content, priority=5)

    # 2. 开启异步监听
    # wait_impulse 会返回一个 Future
    wait_fut = mindflow_bus.wait_impulse(priority=0, wait_new=True)

    # 3. 投递信号
    mindflow_bus.on_signal(signal)

    # 4. 验证 Future 是否被正确填充
    # 因为 PriorityMindPulse 内部是 create_task 异步入队，这里给一点调度时间
    impulse = await asyncio.wait_for(wait_fut, timeout=1.0)

    assert impulse is not None
    assert impulse.belongs_to == "test_input_pulse"
    assert impulse.priority == 5
    assert len(impulse.messages[0].contents) > 0

    # 5. 测试 pop 逻辑
    popped = mindflow_bus.pop_impulse()
    assert popped is not None
    assert popped.priority == 5

    # 再次 pop 应该是 None
    assert mindflow_bus.pop_impulse() is None


@pytest.mark.asyncio
async def test_priority_preemption(mindflow_bus: MindflowBus):
    """
    测试优先级抢占逻辑
    """
    # 同时发送两个信号，一个低优，一个高优
    low_sig = InputSignal().to_signal("Low", priority=1)
    high_sig = InputSignal().to_signal("High", priority=100)

    mindflow_bus.on_signal(low_sig)
    mindflow_bus.on_signal(high_sig)

    # 等待异步队列处理完成
    await asyncio.sleep(0.1)

    # 弹出最优先的脉冲
    best_imp = mindflow_bus.pop_impulse()

    assert best_imp is not None
    assert best_imp.priority == 100
    assert len(best_imp.messages[0].contents) > 0


@pytest.mark.asyncio
async def test_stale_signal_ignored(mindflow_bus: MindflowBus):
    """
    测试过期信号是否被丢弃
    """
    # 构造一个已经过期的信号 (stale_timeout 设为极小值并等待)
    stale_sig = InputSignal().to_signal("Old News", priority=10, stale_timeout=0.001)
    await asyncio.sleep(0.01)

    mindflow_bus.on_signal(stale_sig)
    await asyncio.sleep(0.1)

    # 应该拿不到任何脉冲
    assert mindflow_bus.pop_impulse() is None


@pytest.mark.asyncio
async def test_multiple_waiters_and_concurrent_notification(mindflow_bus: MindflowBus):
    """
    用例 1: 测试多个 Future 同时等待。
    当一个脉冲产生时，所有符合优先级的等待者都应该被唤醒。
    """
    # 创建三个等待者
    fut1 = mindflow_bus.wait_impulse(priority=10)
    fut2 = mindflow_bus.wait_impulse(priority=20)
    fut3 = mindflow_bus.wait_impulse(priority=50)

    # 投递一个优先级为 30 的信号
    signal = InputSignal().to_signal("Priority 30", priority=30)
    mindflow_bus.on_signal(signal)

    # fut1 和 fut2 应该被唤醒（因为 30 >= 10 且 30 >= 20）
    # fut3 应该还在等待（因为 30 < 50）
    res1 = await asyncio.wait_for(fut1, timeout=0.5)
    res2 = await asyncio.wait_for(fut2, timeout=0.5)

    assert res1.priority == 30
    assert res2.priority == 30
    assert not fut3.done()

    # 清理 fut3 避免影响后续测试
    fut3.cancel()


@pytest.mark.asyncio
async def test_supress_logic(mindflow_bus: MindflowBus):
    """
    用例 2: 测试压制逻辑。
    当 pop_impulse 弹出最优先脉冲时，其他 Pulse 应该触发 supress。
    """
    from unittest.mock import MagicMock

    # 额外注册一个 Pulse 用来观察 supress 是否被调用
    mock_pulse = MagicMock(spec=PriorityMindPulse)
    mock_pulse.name.return_value = "mock_pulse"
    mock_pulse.receiving.return_value = ["test_signal"]
    # 模拟它当前有一个低优脉冲
    mock_pulse.peek.return_value = Impulse(belongs_to="mock_pulse", priority=1)

    mindflow_bus.with_pulse(mock_pulse)

    # 投递一个高优信号给原有的 test_input_pulse
    high_sig = InputSignal().to_signal("High", priority=100)
    mindflow_bus.on_signal(high_sig)
    await asyncio.sleep(0.1)

    # 弹出高优脉冲
    popped = mindflow_bus.pop_impulse()
    assert popped.priority == 100

    # 验证 mock_pulse 是否被压制了
    # 注意：在你的实现中，_suppress_all 会遍历所有 pulse 触发 supress
    mock_pulse.supress.assert_called()


@pytest.mark.asyncio
async def test_wait_future_cancellation(mindflow_bus: MindflowBus):
    """
    用例 3: 测试 Future 取消后的清理。
    确保不会因为外部取消等待而导致 Mindflow 内部引用残留。
    """
    # 创建一个等待
    fut = mindflow_bus.wait_impulse(priority=10)
    assert fut in mindflow_bus._wait_impulse_futures

    # 外部取消这个 future
    fut.cancel()

    # 给一点时间让 callback 执行 (_remove_done_wait_impulse_future)
    await asyncio.sleep(0)

    # 验证字典已经干净了
    assert fut not in mindflow_bus._wait_impulse_futures