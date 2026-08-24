"""
Attention challenge 协议基线.

完全隔离 attention 层 (不依赖 mindflow 调度, 不依赖 nucleus), 只测 challenge() 协议:
- 6 路径仲裁 (stale / same-id / FATAL / higher / lower / BACKGROUND / same-priority+strength)
- protection_time 协议 (作用于"同优先级"层级, 同源不豁免)

challenge() 是 async, 返回约定字面量: 'win' / 'lose' / 'absorb'.
强度衰减曲线、escalation 比率等实现细节不在此覆盖.
"""
import time
import pytest

from ghoshell_moss.core.blueprint.mindflow import (
    Impulse, Priority,
)
from ghoshell_moss.core.mindflow import BaseAttention
from ghoshell_moss.message import Message


def _imp(
        *,
        source: str = 'src',
        priority: int = Priority.NOTICE,
        strength: int = 100,
        complete: bool = True,
        stale_timeout: float = 0.0,
        protection_time: float = 0.0,
        id: str | None = None,
        messages: list[Message] | None = None,
) -> Impulse:
    """构造 challenge 测试用的 impulse. 字段都暴露成命名参数, 测试意图直白."""
    kwargs = dict(
        source=source,
        priority=priority,
        strength=strength,
        complete=complete,
        stale_timeout=stale_timeout,
        protection_time=protection_time,
        messages=messages or [Message.new().with_content('m')],
    )
    if id is not None:
        kwargs['id'] = id
    return Impulse(**kwargs)


def _attention(
        defender: Impulse,
        *,
        protection_duration_ratio: float = 0.2,
        max_protection_time: float = 3.0,
        source_escalation: float = 1.1,
) -> BaseAttention:
    """
    构建一个强度/保护期已就绪的 attention.

    构造函数现在会对初始 impulse 调 absorb_impulse 播种强度, 因此无需再手动播种.
    """
    return BaseAttention(
        impulse=defender,
        protection_duration_ratio=protection_duration_ratio,
        max_protection_time=max_protection_time,
        source_escalation=source_escalation,
    )


# ============================================================
# 路径 1: stale challenger 直接 lose
# ============================================================

@pytest.mark.asyncio
async def test_stale_challenger_is_suppressed():
    """过期 impulse 永远不进入仲裁, 直接 lose (不论优先级如何)."""
    att = _attention(_imp(priority=Priority.NOTICE))
    # 构造一个已过期的 challenger.
    stale = _imp(priority=Priority.FATAL, stale_timeout=0.01)
    time.sleep(0.02)
    assert stale.is_stale()
    assert await att.challenge(stale) == 'lose'


# ============================================================
# 路径 2: same-id challenger
# 协议: 仅当当前 held impulse 为 partial (complete=False) 时, 同 id 走 absorb;
# 若 held 已 complete, 同 id 被视为新一轮仲裁 (按优先级/强度判定).
# ============================================================

@pytest.mark.asyncio
async def test_same_id_partial_holder_absorbs():
    """同 id 且当前为 partial 时, 一律 absorb, 与 challenger 优先级无关."""
    partial = _imp(complete=False, id='shared-id')
    att = _attention(partial)
    same_id_high = _imp(id='shared-id', priority=Priority.WARNING)
    assert await att.challenge(same_id_high) == 'absorb'


@pytest.mark.asyncio
async def test_same_id_complete_holder_is_not_auto_absorbed():
    """已 complete 的 held impulse, 同 id 不再自动 absorb, 走正常优先级仲裁."""
    defender = _imp(priority=Priority.NOTICE, id='shared-id')
    att = _attention(defender)
    same_id_high = _imp(id='shared-id', priority=Priority.WARNING)
    assert await att.challenge(same_id_high) == 'win'


@pytest.mark.asyncio
async def test_partial_held_absorbs_same_id_complete_tail():
    """协议: partial 占据注意力, 同 id complete 抵达 → absorb, 随后 draw_from 拿到 complete 尾包."""
    partial = _imp(complete=False, id='shared-id')
    att = _attention(partial)
    assert att.draw_from().id == 'shared-id'
    complete = _imp(complete=True, id='shared-id', messages=[Message.new().with_content('final')])
    # challenge 判 absorb, 不另起 attention.
    assert await att.challenge(complete) == 'absorb'
    # mindflow 随后调 absorb_impulse 折叠尾包, draw_from 反映 complete.
    assert att.absorb_impulse(complete) is None
    assert att.draw_from().id == 'shared-id'
    assert att.draw_from().complete


# ============================================================
# 路径 3: 高优先级抢占成功
# ============================================================

@pytest.mark.asyncio
async def test_higher_priority_preempts():
    att = _attention(_imp(priority=Priority.NOTICE))
    challenger = _imp(priority=Priority.WARNING)
    assert await att.challenge(challenger) == 'win'


@pytest.mark.asyncio
async def test_fatal_always_preempts():
    """FATAL 是约定的最高级别, 应总是抢占成功, 不受任何 defender 优先级影响."""
    att = _attention(_imp(priority=Priority.CRITICAL))
    challenger = _imp(priority=Priority.FATAL)
    assert await att.challenge(challenger) == 'win'


@pytest.mark.asyncio
async def test_fatal_preempts_even_within_protection():
    """FATAL 应穿透保护期 (协议级承诺: 永远抢占成功)."""
    att = _attention(_imp(
        priority=Priority.NOTICE,
        protection_time=10.0,  # 长保护期
    ))
    # 即便 defender 处于保护期内, FATAL 仍应抢占.
    assert await att.challenge(_imp(priority=Priority.FATAL)) == 'win'


# ============================================================
# 路径 4: 低优先级被压制
# ============================================================

@pytest.mark.asyncio
async def test_lower_priority_is_suppressed():
    att = _attention(_imp(priority=Priority.WARNING))
    challenger = _imp(priority=Priority.NOTICE)
    assert await att.challenge(challenger) == 'lose'


@pytest.mark.asyncio
async def test_background_loses_to_info():
    """BACKGROUND (-1) 低于任何普通信号, 不论强度多高."""
    att = _attention(_imp(priority=Priority.INFO, strength=10))
    challenger = _imp(priority=Priority.BACKGROUND, strength=999)
    assert await att.challenge(challenger) == 'lose'


# ============================================================
# 路径 5: 保护期 — 作用于"同优先级"层级
# 协议: 在 _protected_until 之前, 同优先级的 challenger 一律被压制, 不论源.
# ============================================================

@pytest.mark.asyncio
async def test_protection_blocks_same_priority_different_source():
    """同优先级 + 不同 source + 保护期内 → 压制."""
    att = _attention(_imp(
        source='vision',
        priority=Priority.NOTICE,
        protection_time=10.0,
    ))
    challenger = _imp(source='audio', priority=Priority.NOTICE, strength=999)
    assert await att.challenge(challenger) == 'lose'


@pytest.mark.asyncio
async def test_protection_blocks_same_priority_same_source():
    """同优先级 + 同 source + 保护期内 → 也被压制 (同源不豁免保护期).
    协议立场: 保护期为防止同级抢占抖动, 同源的连续性由 same-id absorb 路径单独保护.
    同源不同 id 仍属新一轮独立输入, 不应即时刷新当前 attention."""
    defender = _imp(source='vision', priority=Priority.NOTICE, protection_time=10.0)
    att = _attention(defender)
    challenger = _imp(source='vision', priority=Priority.NOTICE, strength=999)
    # 不同 id, 同 source, 同优先级, 保护期内 → lose.
    assert challenger.id != defender.id
    assert await att.challenge(challenger) == 'lose'


@pytest.mark.asyncio
async def test_protection_does_not_block_higher_priority():
    """保护期只在"同优先级"层级生效, 高于 defender 的 challenger 应正常抢占."""
    att = _attention(_imp(
        priority=Priority.NOTICE,
        protection_time=10.0,
    ))
    assert await att.challenge(_imp(priority=Priority.WARNING)) == 'win'


@pytest.mark.asyncio
async def test_protection_does_not_apply_to_lower_priority():
    """低于 defender 的 challenger, 不论保护期, 都被压制 (走 priority 短路, 不走 protection 分支)."""
    att = _attention(_imp(
        priority=Priority.WARNING,
        protection_time=10.0,
    ))
    # 这条更多是行为一致性 — 结果与"没有保护期"等价.
    assert await att.challenge(_imp(priority=Priority.INFO)) == 'lose'


@pytest.mark.asyncio
async def test_protection_expires_then_strength_arbitration_takes_over():
    """协议: 保护期到期后, 同优先级 challenge 进入 strength 仲裁."""
    defender = _imp(priority=Priority.NOTICE, strength=100, protection_time=0.05)
    att = _attention(defender)
    # 保护期内: 同级别强度更高也压制.
    strong_within = _imp(source='other', priority=Priority.NOTICE, strength=200)
    assert await att.challenge(strong_within) == 'lose'
    # 等保护期过.
    time.sleep(0.08)
    strong_after = _imp(source='other', priority=Priority.NOTICE, strength=200)
    assert await att.challenge(strong_after) == 'win'


# ============================================================
# 路径 6: 同优先级 + 出保护期 → 强度仲裁
# 协议: 强度大者胜 (具体曲线是实现, 不测).
# 影响协议表面的策略点: 同源 challenger 享 source_escalation 加成 (BaseAttention 实现).
# ============================================================

@pytest.mark.asyncio
async def test_strength_arbitration_higher_wins_after_protection():
    """同优先级、不同源、出保护期: 高强度 challenger 胜."""
    # 用 protection_time=0 + protection_duration_ratio=0 完全消除保护期.
    att = _attention(
        _imp(priority=Priority.NOTICE, strength=100),
        protection_duration_ratio=0.0,
    )
    assert await att.challenge(_imp(source='other', priority=Priority.NOTICE, strength=200)) == 'win'


@pytest.mark.asyncio
async def test_strength_arbitration_lower_loses_after_protection():
    """同优先级、不同源、出保护期: 低强度 challenger 输."""
    att = _attention(
        _imp(priority=Priority.NOTICE, strength=200),
        protection_duration_ratio=0.0,
    )
    assert await att.challenge(_imp(source='other', priority=Priority.NOTICE, strength=50)) == 'lose'


@pytest.mark.asyncio
async def test_same_source_escalation_helps_marginal_challenger():
    """同源 challenger 享 1.1x escalation, 等强度仍能赢同优先级 (出保护期).
    这条钉的是 BaseAttention 的策略选择: 同源比异源更容易接力."""
    defender = _imp(source='vision', priority=Priority.NOTICE, strength=100)
    att = _attention(defender, protection_duration_ratio=0.0)
    # 同 source 同 strength 的 challenger: escalation 后 strength=110 > defender_current.
    same_source_eq = _imp(source='vision', priority=Priority.NOTICE, strength=100)
    assert await att.challenge(same_source_eq) == 'win'
