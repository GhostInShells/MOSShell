import asyncio

import pytest

from ghoshell_moss.core.blueprint.mindflow import Impulse, Priority
from ghoshell_moss.core.mindflow import BaseAttention


@pytest.mark.asyncio
async def test_attention_preemption_by_priority():
    """测试不同优先级的 impulse 挑战是否会引发 aborted"""
    current = Impulse(source="main", priority=Priority.INFO, strength=100)
    attention = BaseAttention(impulse=current)

    async with attention:
        # 模拟 CRITICAL 挑战
        challenger = Impulse(source="emergency", priority=Priority.CRITICAL, strength=100)
        result = await attention.challenge(challenger)

        assert result == 'win'  # 应该返回抢占成功
        attention.abort("preempted")
        assert attention.is_aborted()


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
    attention = BaseAttention(impulse=impulse)
    await asyncio.sleep(0.5)
    # 中段衰减: protection (0.2s) 已过, 进度 ~ 0.375, strength ~ 62. 抖动余量充足.
    assert attention.current_strength() > 0
    await asyncio.sleep(0.6)
    # 累计 1.1s, 已过 TTL.
    assert attention.current_strength() == 0


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

        # 保护期内，challenge 返回 lose (表示压制, 但不打断/不重置)
        # 注意：这里需要确保 challenge 逻辑里检查了 protection_time
        result = await attention.challenge(challenger)
        assert result == 'lose'
        # 这时应该过了保护期.
        await asyncio.sleep(0.01)
        assert await attention.challenge(challenger) == 'win'
        assert not attention.is_aborted()
        await asyncio.sleep(0.095)
        assert challenger.is_stale()
        assert await attention.challenge(challenger) == 'lose'


@pytest.mark.asyncio
async def test_strength_zero_yields_to_any_positive_challenger():
    """
    协议: 强度为零时任意正强度的同优先级 challenger 都能抢占成功.
    强度跌零意味着 "可以被随意打断", 不是 "必须自杀".
    challenge() 返回 'win' (Preempted).
    """
    impulse = Impulse(
        source="defender",
        priority=Priority.NOTICE,
        strength=100,
        strength_decay_seconds=0.1,
    )
    attention = BaseAttention(
        impulse=impulse,
        protection_duration_ratio=0.0,
    )

    # 等待强度衰减到零
    await asyncio.sleep(0.2)
    assert attention.current_strength() == 0

    # 即使 strength=1 的 challenger 也能抢占
    challenger = Impulse(
        source="other",
        priority=Priority.NOTICE,
        strength=1,
    )
    result = await attention.challenge(challenger)
    assert result == 'win'


@pytest.mark.asyncio
async def test_same_id_tail_buffers_only_once(monkeypatch):
    """同 id 的 'absorb' 只应发生在 attention 仍未 complete 之前.

    partial 首包占据 attention 时, 同 id 尾包被 absorb (把 attention 变成 complete);
    一旦 attention 已经 complete (尾包已到), 同 id 的低优包不应再无限 absorb,
    而应走正常仲裁 (按优先级/强度决定 win/lose).
    """
    clock = [1000.0]
    monkeypatch.setattr("ghoshell_moss.core.mindflow._attention.time.monotonic", lambda: clock[0])
    partial = Impulse(id="a", source="s", priority=Priority.NOTICE, complete=False)
    attention = BaseAttention(impulse=partial)
    # 吸收 complete 尾包 -> attention 变 complete.
    attention.absorb_impulse(Impulse(id="a", source="s", priority=Priority.NOTICE, complete=True))
    assert attention.impulse().complete is True
    # 再来一个同 id 的 BACKGROUND 包: 不应再触发 'absorb', 而应按低优被压制.
    result = await attention.challenge(
        Impulse(id="a", source="s", priority=Priority.BACKGROUND, complete=True)
    )
    assert result == 'lose'


# ============================================================================
# 同步仲裁测试 (无需事件循环) — 去生命周期化后, 仲裁状态 / 强度数学 / 吸收路由
# 对 BaseAttention 而言是纯同步可测的, 不依赖 loop 或 sleep.
# ============================================================================

def test_current_strength_boosted_inside_protection_window():
    """保护窗内 current_strength = seed * source_escalation (1.1), 而非 seed."""
    impulse = Impulse(
        source='s', priority=Priority.NOTICE, strength=100,
        strength_decay_seconds=100,  # 长 TTL, 确保 elapsed 落在保护窗内
    )
    attention = BaseAttention(impulse=impulse, protection_duration_ratio=0.2)
    # protection_time = min(100*0.2, 3.0) = 3.0s; elapsed ~0 在窗内 -> 100*1.1 = 110.
    assert attention.current_strength() == 110


def test_current_strength_decay_uses_live_completeness(monkeypatch):
    """partial 起源的 attention 吸收 complete 尾包后, 衰减应按 '完整' 曲线 (factor=1.0) 而非 1.5.

    draw_from 是冻结的创建 impulse (恒 incomplete), 但 current_strength 的衰减因子
    应读活量 impulse().complete — 否则 partial 起源的 attention 即使尾包已到也永远快速衰减.
    """
    clock = [1000.0]
    monkeypatch.setattr("ghoshell_moss.core.mindflow._attention.time.monotonic", lambda: clock[0])
    partial = Impulse(id="a", source="s", strength=100, strength_decay_seconds=10, complete=False)
    attention = BaseAttention(
        impulse=partial,
        protection_duration_ratio=0.0,
        max_protection_time=0.0,
    )
    attention.absorb_impulse(
        Impulse(id="a", source="s", strength=100, strength_decay_seconds=10, complete=True)
    )
    assert attention.draw_from().complete is False  # 冻结创建者仍是 partial
    assert attention.impulse().complete is True     # 活量已 complete
    clock[0] += 5.0  # 进度 = 5 / 10 = 0.5
    # factor=1.0 -> 100 * (1 - 0.5) = 50; 若误用 draw_from.complete -> factor=1.5 -> 25.
    assert attention.current_strength() == 50


def test_arbit_same_source_escalates_and_can_flip_outcome():
    """同源 challenger 被放大 1.1 才能同级抢赢 (92*1.1=101>100); 异源不放大 (92<100)."""
    attention = BaseAttention(
        impulse=Impulse(source='a', priority=Priority.NOTICE, strength=100, strength_decay_seconds=100),
        protection_duration_ratio=0.0,
    )
    same = Impulse(source='a', priority=Priority.NOTICE, strength=92)
    assert attention.arbit_challenge_by_strength(same) is True
    other = Impulse(source='b', priority=Priority.NOTICE, strength=92)
    assert attention.arbit_challenge_by_strength(other) is False


def test_absorb_impulse_same_id_updates_seed_and_returns_none():
    """同 id 的 impulse (尾包) 折进当前 attention, 强化种子; 返回 None 表示已吸收."""
    init = Impulse(source='a', priority=Priority.NOTICE, strength=100, id='x')
    attention = BaseAttention(impulse=init)
    tail = Impulse(source='a', priority=Priority.NOTICE, strength=120, id='x', complete=True)
    result = attention.absorb_impulse(tail)
    assert result is None
    assert attention.impulse().strength == 120


def test_absorb_impulse_diff_id_returns_impulse_for_routing():
    """异 id 的 impulse 不被内部吸收, 返回给调用方路由."""
    init = Impulse(source='a', priority=Priority.NOTICE, strength=100, id='x')
    attention = BaseAttention(impulse=init)
    other = Impulse(source='b', priority=Priority.NOTICE, strength=100, id='y')
    assert attention.absorb_impulse(other) is other


def test_absorb_diff_id_does_not_replace_current_impulse():
    """异 id 的 impulse 被路由出去 (返回 impulse) 时, 不应篡改 attention 的当前 impulse.

    否则 effort() 会翻到那个被 buffer 的异 id 包的值, 破坏 '当前 impulse' 语义.
    """
    attention = BaseAttention(impulse=Impulse(source='a', priority=Priority.NOTICE, id='x'))
    other = Impulse(source='b', priority=Priority.NOTICE, id='y')
    assert attention.absorb_impulse(other) is other
    assert attention.impulse().id == 'x'


def test_priority_and_set_priority_override():
    """priority() 优先返回 attention 级覆盖值; set_priority(None) 回退到 impulse 优先级."""
    attention = BaseAttention(impulse=Impulse(source='a', priority=Priority.NOTICE))
    assert attention.priority() == Priority.NOTICE
    attention.set_priority(Priority.CRITICAL)
    assert attention.priority() == Priority.CRITICAL
    attention.set_priority(None)
    assert attention.priority() == Priority.NOTICE


def test_abort_sets_aborted_and_reason():
    """abort 置位 aborted/closed, 并记录 abort_reason."""
    attention = BaseAttention(impulse=Impulse(source='a', priority=Priority.INFO))
    assert attention.is_aborted() is False
    assert attention.is_closed() is False
    attention.abort('preempted')
    assert attention.is_aborted() is True
    assert attention.is_closed() is True
    assert attention.abort_reason() == 'preempted'


def test_is_protected_within_protection_window():
    """配置 protection_time 后, is_protected() 在窗内立即为真."""
    attention = BaseAttention(impulse=Impulse(source='a', priority=Priority.NOTICE, protection_time=10))
    assert attention.is_protected() is True


def test_draw_from_returns_seed_impulse():
    """draw_from 返回播种的 impulse (id 可溯源)."""
    init = Impulse(source='a', priority=Priority.INFO, id='abc')
    attention = BaseAttention(impulse=init)
    assert attention.draw_from() is init
    assert attention.draw_from().id == 'abc'
