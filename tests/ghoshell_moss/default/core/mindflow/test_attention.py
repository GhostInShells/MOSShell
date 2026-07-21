import pytest
from ghoshell_moss.core.blueprint.mindflow import Impulse, Priority, Reaction, ObserveError
from ghoshell_moss.core.mindflow.base_attention import BaseAttention
from ghoshell_moss.message import Message
import time
import asyncio


@pytest.mark.asyncio
async def test_attention_lifecycle_and_loop():
    """测试 Attention 的完整运行循环是否能正常产出 Articulate 和 Action"""
    # 1. 准备初始状态
    initial_impulse = Impulse(source="test", priority=Priority.INFO, messages=[Message.new().with_content("init")])
    outcome = Reaction()

    attention = BaseAttention(previous=outcome, impulse=initial_impulse)

    # 2. 启动 Attention
    async with attention:
        # 验证是否启动
        assert attention.is_started() is True

        # 3. 运行 loop
        loop_gen = attention.loop()
        articulate, action = await anext(loop_gen)

        assert articulate is not None
        assert action is not None

        # 4. 模拟 Articulate 和 Action 的生命周期
        async with articulate, action:
            articulate.send_nowait("Hello")
            # 消费 Action 的 logos
            async for delta in action.received_logos():
                assert delta == "Hello"
                break  # 简单测试

    assert attention.is_closed()


@pytest.mark.asyncio
async def test_attention_preemption_by_priority():
    """测试不同优先级的 impulse 挑战是否会引发 aborted"""
    current = Impulse(source="main", priority=Priority.INFO, strength=100)
    attention = BaseAttention(previous=Reaction(), impulse=current)

    async with attention:
        # 模拟 CRITICAL 挑战
        challenger = Impulse(source="emergency", priority=Priority.CRITICAL, strength=100)
        result = attention.challenge(challenger)

        assert result is True  # 应该返回抢占成功
        attention.abort("preempted")
        assert attention.is_aborted()


@pytest.mark.asyncio
async def test_observe_error_propagation():
    """测试 ObserveError 如何正确导致下一轮循环"""
    initial = Impulse(source="test", priority=Priority.INFO)
    attention = BaseAttention(previous=Reaction(), impulse=initial)

    async with attention:
        loop_gen = attention.loop()
        articulate, action = await anext(loop_gen)

        # 模拟 Articulate 抛出 ObserveError
        async with articulate:
            articulate.raise_observe("need more info")

        # 注意：BaseAttention.__aexit__ 会捕获这个异常并调用 ctx.capture_error
        # 应该验证 observe_messages 是否被记录
        observe_msgs = attention._ctx.get_observe_messages()
        assert observe_msgs is not None
        assert len(observe_msgs) > 0


@pytest.mark.asyncio
async def test_attention_strength_decay():
    # 时间缩放到 1s 量级, 留出 100ms+ 的事件循环抖动余量.
    # 原 100ms 量级在 0.09 边界对 asyncio.sleep 精度过于敏感, 偶发 strength 已归零.
    impulse = Impulse(
        source="test",
        priority=Priority.INFO,
        strength=100,
        strength_decay_seconds=1.0,  # 1s
    )
    attention = BaseAttention(previous=Reaction(), impulse=impulse)
    await asyncio.sleep(0.5)
    # 中段衰减: protection (0.2s) 已过, 进度 ~ 0.375, strength ~ 62. 抖动余量充足.
    assert attention.current_strength() > 0
    await asyncio.sleep(0.6)
    # 累计 1.1s, 已过 TTL.
    assert attention.current_strength() == 0


@pytest.mark.asyncio
async def test_attention_rapid_timeout_aborted():
    """
    测试 Impulse 强度过期时间极短 (100ms) 时，
    Attention 是否能在启动后立即进入超时 aborted 状态。
    """
    # 1. 构造一个 0.1 秒后失效的 Impulse
    impulse = Impulse(
        source="test",
        priority=Priority.INFO,
        strength=100,
        strength_decay_seconds=0.1  # 100ms
    )

    attention = BaseAttention(previous=Reaction(), impulse=impulse)
    start_time = time.perf_counter()
    async with attention:
        # 2. 等待直到生命周期被触发超时
        # 这里的等待逻辑应该是内部生命周期感知到强度衰减为 0
        await attention.wait_aborted()

    duration = time.perf_counter() - start_time
    # 3. 验证结果
    # 验证是否是因为 TimeoutError 导致的 abort (或其他方式标记的 aborted)
    assert attention.is_aborted() is True

    # 4. 验证时间精度：应该在 0.1s 到 0.5s 之间（考虑异步调度开销）
    # 如果 duration 远大于 1s，说明计时逻辑有问题；若小于 0.05s，说明没有触发衰减逻辑
    assert 0.05 <= duration <= 0.6


@pytest.mark.asyncio
async def test_attention_homologous_escalation():
    """
    测试同源信号在保护期内外对 Attention 的影响：
    1. 保护期内：同源信号无法接力刷新时间（保持原过期时间）
    2. 保护期外：同源信号成功接力刷新时间
    """
    ttl = 2.0  # 设置 2s 的 TTL
    impulse = Impulse(
        source="engine",
        priority=Priority.NOTICE,
        strength=100,
        strength_decay_seconds=ttl
    )
    # 保护区: min(2.0 * 0.2, 3.0) = 0.4s
    attention = BaseAttention(
        previous=Reaction(),
        impulse=impulse,
        # 保护期时间 0.1
        protection_duration_ratio=0.1,
        max_protection_time=3.0
    )

    async with attention:
        # 1. 初始状态
        start_time = attention.strength_refreshed_at

        # 2. 模拟保护期内 (2.0 * 0.1 = 0.2s) 信号进入
        await asyncio.sleep(0.19)
        challenger = Impulse(source="engine", priority=Priority.NOTICE, strength=100)

        # 保护期内，on_challenge 返回 None (表示吸收，但不打断/不重置)
        # 注意：这里需要确保你 on_challenge 逻辑里检查了 protection_time
        result = attention.challenge(challenger)
        assert result is False

        # 3. 模拟保护期外 (2.0 * 0.1) 信号进入
        await asyncio.sleep(0.01)
        # 此时已经超过了 0.4s 保护期，同源信号应该能刷新时间
        assert attention.challenge(challenger) is True
        async for articulate, action in attention.loop():
            # 刷新了.
            assert attention.strength_refreshed_at > start_time
            break


@pytest.mark.asyncio
async def test_attention_max_protection_time():
    """
    测试同源信号在保护期内外对 Attention 的影响：
    1. 保护期内：同源信号无法接力刷新时间（保持原过期时间）
    2. 保护期外：同源信号成功接力刷新时间
    """
    impulse = Impulse(
        source="engine",
        priority=Priority.NOTICE,
        strength=100,
        strength_decay_seconds=100,
    )
    # 保护区: min(2.0 * 0.2, 3.0) = 0.4s
    attention = BaseAttention(
        previous=Reaction(),
        impulse=impulse,
        # 保护期比例 100%
        protection_duration_ratio=1.0,
        max_protection_time=0.05
    )

    async with attention:
        # 所以在整个周期里都是被保护的.
        # 但是我们测最大的保护期 0.05 是否生效.
        await asyncio.sleep(0.04)
        challenger = Impulse(source="engine", priority=Priority.NOTICE, strength=100, stale_timeout=0.1)

        # 保护期内，on_challenge 返回 None (表示吸收，但不打断/不重置)
        # 注意：这里需要确保你 on_challenge 逻辑里检查了 protection_time
        result = attention.challenge(challenger)
        assert result is False
        # 这时应该过了保护期.
        await asyncio.sleep(0.01)
        assert attention.challenge(challenger) is True
        assert not attention.is_aborted()
        await asyncio.sleep(0.095)
        assert challenger.is_stale()
        assert attention.challenge(challenger) is False


@pytest.mark.asyncio
async def test_attention_not_aborted_during_active_articulation():
    """
    协议: 强度衰减到零不应导致正在活跃工作的 attention 自杀.
    只要 loop 已经产出过帧 (articulator/action 正在运行),
    _inner_attention_lifecycle 不应因为强度跌零而 abort.
    """
    impulse = Impulse(
        source="test",
        priority=Priority.NOTICE,
        strength=100,
        strength_decay_seconds=0.15,
    )
    attention = BaseAttention(
        previous=Reaction(),
        impulse=impulse,
        protection_duration_ratio=0.0,
    )

    async with attention:
        loop_gen = attention.loop()
        art, act = await anext(loop_gen)

        async with art, act:
            # 模拟长时工作: 跨越 TTL 的多次活动
            for _ in range(6):
                await asyncio.sleep(0.05)
                art.send_nowait("working...")

            # 强度可能已经衰减到零 (取决于被刷新的频率),
            # 但 attention 不应被 abort.
            assert not attention.is_aborted()


@pytest.mark.asyncio
async def test_strength_refreshed_during_active_articulation():
    """
    协议: articulator.send_nowait 和 action 消费 logos 时,
    通过 on_active 回调刷新 _strength_refreshed_at,
    强度不会在活跃期间衰减到零.
    """
    impulse = Impulse(
        source="test",
        priority=Priority.NOTICE,
        strength=100,
        strength_decay_seconds=0.3,
    )
    attention = BaseAttention(
        previous=Reaction(),
        impulse=impulse,
        protection_duration_ratio=0.0,
    )

    async with attention:
        loop_gen = attention.loop()
        art, act = await anext(loop_gen)

        async with art, act:
            initial_strength = attention.current_strength()
            assert initial_strength > 0

            # 跨越 TTL 期间持续活动 — 每次 send_nowait 都刷新强度
            for _ in range(10):
                await asyncio.sleep(0.04)
                art.send_nowait("keep alive")

            # 强度应仍然 > 0 (被持续刷新, 不衰减)
            assert attention.current_strength() > 0


def test_strength_zero_yields_to_any_positive_challenger():
    """
    协议: 强度为零时任意正强度的同优先级 challenger 都能抢占成功.
    强度跌零意味着 "可以被随意打断", 不是 "必须自杀".
    challenge() 返回 True (Preempted).
    """
    impulse = Impulse(
        source="defender",
        priority=Priority.NOTICE,
        strength=100,
        strength_decay_seconds=0.1,
    )
    attention = BaseAttention(
        previous=Reaction(),
        impulse=impulse,
        protection_duration_ratio=0.0,
    )

    # 等待强度衰减到零
    time.sleep(0.2)
    assert attention.current_strength() == 0

    # 即使 strength=1 的 challenger 也能抢占
    challenger = Impulse(
        source="other",
        priority=Priority.NOTICE,
        strength=1,
    )
    result = attention.challenge(challenger)
    assert result is True
