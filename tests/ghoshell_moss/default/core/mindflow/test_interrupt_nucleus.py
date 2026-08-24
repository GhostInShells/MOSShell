"""InterruptNucleus + InterruptSignalMeta + ImpulsePrimitive.broadcast/interrupt
unit tests.

只测协议层与单元行为, 不依赖 mindflow 主循环.

覆盖范围:
- ImpulsePrimitive.broadcast / interrupt 字段卸载 + 对偶性验证
- InterruptSignalMeta 协议往返 + 默认 FATAL
- InterruptNucleus.build_impulse 四字段卸载 (FATAL + notify + effort=none + interrupt=True)
- priority 被强制覆盖为 FATAL (interrupt 承诺不可降级)
- 反向 suppress: attended 触发冷静期, suppress() 不触发
- 冷静期内 add_signal 静默丢
- 冷静期到期后恢复
- lifecycle / 反身性接口
- InterruptNucleusMeta 自解释发现
"""
import asyncio
import logging
import time

import pytest
from ghoshell_container import Container

from ghoshell_moss.contracts.logger import LoggerItf
from ghoshell_moss.core.blueprint.mindflow import (
    ChallengeMode, Impulse, ImpulsePrimitive, Priority, Signal,
)
from ghoshell_moss.core.mindflow.interrupt_nucleus import (
    InterruptNucleus, InterruptNucleusMeta, InterruptSignalMeta, new_interrupt_signal,
)
from ghoshell_moss.message import Message


# ============================================================
# ImpulsePrimitive — broadcast / interrupt 对偶
# 协议级原语单测, 钉住对偶关系: 同 priority + effort, 不同 mode + interrupt 字段.
# ============================================================

def _empty_impulse() -> Impulse:
    return Impulse(messages=[Message.new().with_content('x')])


def test_broadcast_primitive_field_unload():
    """broadcast: FATAL + silent + effort='none', interrupt=False."""
    imp = ImpulsePrimitive.broadcast(_empty_impulse())
    assert imp.priority == Priority.FATAL
    assert imp.mode == ChallengeMode.silent.value
    assert imp.thinking_effort == 'none'
    assert imp.interrupt is False  # broadcast 不触发 stop_interpretation


def test_interrupt_primitive_field_unload():
    """interrupt: FATAL + notify + effort='none' + interrupt=True."""
    imp = ImpulsePrimitive.interrupt(_empty_impulse())
    assert imp.priority == Priority.FATAL
    assert imp.mode == ChallengeMode.notify.value
    assert imp.thinking_effort == 'none'
    assert imp.interrupt is True  # 核心承诺: 让 ghost_runtime 调 stop_interpretation


def test_broadcast_interrupt_share_priority_and_effort():
    """对偶性验证: 共享 FATAL + effort='none', 在 mode + interrupt 二维上分裂."""
    b = ImpulsePrimitive.broadcast(_empty_impulse())
    i = ImpulsePrimitive.interrupt(_empty_impulse())
    assert b.priority == i.priority == Priority.FATAL
    assert b.thinking_effort == i.thinking_effort == 'none'
    # 分裂点:
    assert b.mode != i.mode
    assert b.interrupt != i.interrupt


def test_interrupt_primitive_does_not_set_logos():
    """interrupt 不带 logos — 中断的本质是停旧, 不是发新."""
    imp = ImpulsePrimitive.interrupt(_empty_impulse())
    assert imp.logos == ''


# ============================================================
# InterruptSignalMeta — 协议往返
# ============================================================

def test_signal_meta_name_is_interrupt():
    assert InterruptSignalMeta.signal_name() == 'interrupt'


def test_signal_meta_default_priority_fatal():
    """interrupt 的承诺就是 '必送达 + 必接管', 默认 FATAL."""
    assert InterruptSignalMeta.priority() == Priority.FATAL
    signal = InterruptSignalMeta().to_signal()
    assert signal.priority == Priority.FATAL


def test_signal_meta_match_rejects_wrong_name():
    fake = Signal.new('other')
    assert InterruptSignalMeta.match(fake) is False


def test_new_interrupt_signal_helper():
    sig = new_interrupt_signal(Message.new().with_content('halt'))
    assert sig.name == 'interrupt'
    assert sig.priority == Priority.FATAL


# ============================================================
# InterruptNucleus.build_impulse — 字段卸载
# ============================================================

def _signal(text: str = 'stop') -> Signal:
    return new_interrupt_signal(Message.new().with_content(text))


def test_build_impulse_sets_all_interrupt_fields():
    """build_impulse 应一次性卸载 4 个字段."""
    nuc = InterruptNucleus()
    impulse = nuc.build_impulse(_signal())
    assert impulse is not None
    assert impulse.priority == Priority.FATAL
    assert impulse.mode == ChallengeMode.notify.value
    assert impulse.thinking_effort == 'none'
    assert impulse.interrupt is True


def test_build_impulse_overrides_low_priority_signal():
    """即便 caller 显式降级 signal, primitive 强制 FATAL — 承诺不可降级."""
    nuc = InterruptNucleus()
    signal = InterruptSignalMeta().to_signal(priority=Priority.INFO)
    assert signal.priority == Priority.INFO  # signal 携带 INFO
    impulse = nuc.build_impulse(signal)
    assert impulse.priority == Priority.FATAL  # primitive 升级回 FATAL


def test_build_impulse_drops_wrong_signal_name():
    nuc = InterruptNucleus()
    fake = Signal.new('input', priority=Priority.FATAL)
    assert nuc.build_impulse(fake) is None


def test_build_impulse_preserves_messages():
    nuc = InterruptNucleus()
    signal = new_interrupt_signal(
        Message.new().with_content('reason 1'),
        Message.new().with_content('reason 2'),
    )
    impulse = nuc.build_impulse(signal)
    assert len(impulse.messages) == 2


# ============================================================
# InterruptNucleus.add_signal — bus 投递
# ============================================================

@pytest.mark.asyncio
async def test_add_signal_fires_impulse_via_bus():
    notified: list[Impulse] = []
    async with InterruptNucleus() as nuc:
        nuc.with_bus(
            signal_broadcast=lambda s: None,
            fire_impulse=lambda imp: notified.append(imp),
        )
        nuc.add_signal(_signal('emergency'))
    assert len(notified) == 1
    impulse = notified[0]
    assert impulse.priority == Priority.FATAL
    assert impulse.interrupt is True


@pytest.mark.asyncio
async def test_add_signal_does_not_fire_when_not_running():
    notified: list[Impulse] = []
    nuc = InterruptNucleus()
    nuc.with_bus(
        signal_broadcast=lambda s: None,
        fire_impulse=lambda imp: notified.append(imp),
    )
    nuc.add_signal(_signal())
    assert notified == []


@pytest.mark.asyncio
async def test_add_signal_drops_wrong_name():
    notified: list[Impulse] = []
    async with InterruptNucleus() as nuc:
        nuc.with_bus(
            signal_broadcast=lambda s: None,
            fire_impulse=lambda imp: notified.append(imp),
        )
        nuc.add_signal(Signal.new('input'))
    assert notified == []


# ============================================================
# 反向 suppress — 胜利侧冷静期
# 核心命题: attended 触发冷静期, suppress() 不触发.
# ============================================================

@pytest.mark.asyncio
async def test_suppress_callback_does_not_start_cooldown():
    """协议: FATAL 仲裁失败仅可能是 same-id absorb 或 stale, 不应进冷静期.
    InterruptNucleus.suppress() 是空 op."""
    notified: list[Impulse] = []
    async with InterruptNucleus(suppress_seconds=10.0) as nuc:
        nuc.with_bus(
            signal_broadcast=lambda s: None,
            fire_impulse=lambda imp: notified.append(imp),
        )
        # 直接调 suppress, 模拟"被同 id absorb 通知"场景.
        nuc.suppress(Impulse(source='other'))
        # 后续 signal 应正常通过 (没有进入冷静期).
        nuc.add_signal(_signal())
    assert len(notified) == 1


@pytest.mark.asyncio
async def test_attended_starts_cooldown_and_blocks_subsequent_signals():
    """协议: 仲裁胜利 (attended) 后冷静期内 add_signal 静默丢."""
    notified: list[Impulse] = []
    async with InterruptNucleus(suppress_seconds=0.2) as nuc:
        nuc.with_bus(
            signal_broadcast=lambda s: None,
            fire_impulse=lambda imp: notified.append(imp),
        )
        nuc.add_signal(_signal('first'))
        assert len(notified) == 1
        # 模拟仲裁胜利后 mindflow 调 attended.
        nuc.attended(notified[0])
        # 冷静期内.
        nuc.add_signal(_signal('second'))
        nuc.add_signal(_signal('third'))
        assert len(notified) == 1  # 被冷静期吞了


@pytest.mark.asyncio
async def test_cooldown_expires_and_signals_flow_again():
    """协议: 冷静期到期后恢复正常."""
    notified: list[Impulse] = []
    async with InterruptNucleus(suppress_seconds=0.1) as nuc:
        nuc.with_bus(
            signal_broadcast=lambda s: None,
            fire_impulse=lambda imp: notified.append(imp),
        )
        nuc.add_signal(_signal('first'))
        nuc.attended(notified[0])
        await asyncio.sleep(0.15)  # 冷静期过.
        nuc.add_signal(_signal('second'))
    assert len(notified) == 2


@pytest.mark.asyncio
async def test_clear_resets_cooldown():
    """clear() 应重置冷静期 — 极限故障恢复后立刻可响应."""
    notified: list[Impulse] = []
    async with InterruptNucleus(suppress_seconds=10.0) as nuc:
        nuc.with_bus(
            signal_broadcast=lambda s: None,
            fire_impulse=lambda imp: notified.append(imp),
        )
        nuc.add_signal(_signal('first'))
        nuc.attended(notified[0])
        # 冷静期 10s, 但 clear 强制重置.
        nuc.clear()
        nuc.add_signal(_signal('second'))
    assert len(notified) == 2


# ============================================================
# 反身性接口
# ============================================================

def test_signals_returns_interrupt_name():
    assert InterruptNucleus().signals() == ['interrupt']


def test_peek_returns_none_before_any_signal():
    """初始 cache 空."""
    assert InterruptNucleus().peek() is None


@pytest.mark.asyncio
async def test_peek_returns_cached_impulse_after_signal():
    async with InterruptNucleus() as nuc:
        nuc.with_bus(lambda s: None, lambda imp: None)
        nuc.add_signal(_signal('halt'))
        peeked = nuc.peek()
        assert peeked is not None
        assert peeked.interrupt is True


@pytest.mark.asyncio
async def test_attended_clears_cache_and_starts_cooldown():
    """attended 既清 cache 也启动冷静期 (反向 suppress)."""
    async with InterruptNucleus(suppress_seconds=10.0) as nuc:
        nuc.with_bus(lambda s: None, lambda imp: None)
        nuc.add_signal(_signal())
        cached = nuc.peek()
        nuc.attended(cached)
        # cache 清空.
        assert nuc.peek() is None
        # 冷静期内新 signal 被静默丢, peek 仍空.
        nuc.add_signal(_signal('another'))
        assert nuc.peek() is None


# ============================================================
# InterruptNucleusMeta — 自解释发现
# ============================================================

def test_nucleus_meta_name():
    assert InterruptNucleusMeta().name() == InterruptNucleus.NAME


def test_nucleus_meta_exposes_signal_meta():
    metas = list(InterruptNucleusMeta().signals())
    assert InterruptSignalMeta in metas


def test_nucleus_meta_factory_returns_interrupt_nucleus():
    container = Container()
    container.set(LoggerItf, logging.getLogger(__name__))
    nuc = InterruptNucleusMeta().factory(container)
    assert isinstance(nuc, InterruptNucleus)
