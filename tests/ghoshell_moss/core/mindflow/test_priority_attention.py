import pytest
from ghoshell_moss.core.blueprint.mindflow import Impulse, Priority, Reaction
from ghoshell_moss.core.mindflow.priority_attention import PriorityProtectionAttention
import time
import asyncio


def _make_impulse(id: str = "a", source: str = "test", priority: Priority = Priority.NOTICE, strength: int = 100):
    return Impulse(id=id, source=source, priority=priority, strength=strength)


@pytest.mark.asyncio
async def test_higher_priority_preempts():
    """规则 5: 高优先级抢占."""
    current = _make_impulse(id="current", priority=Priority.NOTICE)
    attn = PriorityProtectionAttention(previous=Reaction(), impulse=current)
    challenger = _make_impulse(id="chal", priority=Priority.WARNING)
    assert attn.challenge(challenger) is True


@pytest.mark.asyncio
async def test_lower_priority_suppressed():
    """规则 6: 低优先级被压制."""
    current = _make_impulse(id="current", priority=Priority.WARNING)
    attn = PriorityProtectionAttention(previous=Reaction(), impulse=current)
    challenger = _make_impulse(id="chal", priority=Priority.NOTICE)
    assert attn.challenge(challenger) is False


@pytest.mark.asyncio
async def test_same_priority_within_protection_suppressed():
    """规则 7a: 同级保护期内压制."""
    current = _make_impulse(id="current", priority=Priority.NOTICE, strength=100)
    attn = PriorityProtectionAttention(
        previous=Reaction(), impulse=current, protection_seconds=0.5,
    )
    challenger = _make_impulse(id="chal", priority=Priority.NOTICE, strength=200)
    assert attn.challenge(challenger) is False


@pytest.mark.asyncio
async def test_same_priority_after_protection_higher_strength_preempts():
    """规则 7b: 同级保护期外 + 强度更高 → 抢占."""
    current = _make_impulse(id="current", priority=Priority.NOTICE, strength=100)
    attn = PriorityProtectionAttention(
        previous=Reaction(), impulse=current, protection_seconds=0.1,
    )
    await asyncio.sleep(0.11)
    challenger = _make_impulse(id="chal", priority=Priority.NOTICE, strength=150)
    assert attn.challenge(challenger) is True


@pytest.mark.asyncio
async def test_same_priority_after_protection_lower_strength_suppressed():
    """规则 7c: 同级保护期外 + 强度更低 → 压制."""
    current = _make_impulse(id="current", priority=Priority.NOTICE, strength=100)
    attn = PriorityProtectionAttention(
        previous=Reaction(), impulse=current, protection_seconds=0.1,
    )
    await asyncio.sleep(0.11)
    challenger = _make_impulse(id="chal", priority=Priority.NOTICE, strength=80)
    assert attn.challenge(challenger) is False


@pytest.mark.asyncio
async def test_same_id_absorbed():
    """规则 3: 同 ID 被吸收, 更新 complete."""
    current = _make_impulse(id="same", priority=Priority.NOTICE)
    attn = PriorityProtectionAttention(previous=Reaction(), impulse=current)
    challenger = _make_impulse(id="same", priority=Priority.CRITICAL)
    assert attn.challenge(challenger) is None
    assert attn._init_impulse.priority == Priority.CRITICAL


@pytest.mark.asyncio
async def test_fatal_always_preempts():
    """规则 4: FATAL 无论如何抢占, 即使保护期内."""
    current = _make_impulse(id="current", priority=Priority.NOTICE)
    attn = PriorityProtectionAttention(
        previous=Reaction(), impulse=current, protection_seconds=60.0,
    )
    challenger = _make_impulse(id="chal", priority=Priority.FATAL)
    assert attn.challenge(challenger) is True


@pytest.mark.asyncio
async def test_debug_absorbed():
    """规则 2: DEBUG 被吸收."""
    current = _make_impulse(id="current", priority=Priority.NOTICE)
    attn = PriorityProtectionAttention(previous=Reaction(), impulse=current)
    challenger = _make_impulse(id="debug", priority=Priority.DEBUG)
    assert attn.challenge(challenger) is None


@pytest.mark.asyncio
async def test_stale_suppressed():
    """规则 1: 过期 challenger 被压制."""
    current = _make_impulse(id="current")
    attn = PriorityProtectionAttention(previous=Reaction(), impulse=current)
    challenger = _make_impulse(id="stale")
    challenger.stale_timeout = 0.001
    await asyncio.sleep(0.002)
    assert attn.challenge(challenger) is False


@pytest.mark.asyncio
async def test_escalation_on_active_is_noop():
    """_escalation_on_active 是 no-op, 保护期不刷新."""
    current = _make_impulse(id="current", priority=Priority.NOTICE, strength=100)
    attn = PriorityProtectionAttention(
        previous=Reaction(), impulse=current, protection_seconds=0.3,
    )
    await asyncio.sleep(0.15)
    # 模拟一轮 loop 中的 escalation
    attn._escalation_on_active()
    await asyncio.sleep(0.2)
    # 总耗时 0.35s > 0.3s 保护期, escalation 没刷新时间
    challenger = _make_impulse(id="chal", priority=Priority.NOTICE, strength=150)
    assert attn.challenge(challenger) is True
