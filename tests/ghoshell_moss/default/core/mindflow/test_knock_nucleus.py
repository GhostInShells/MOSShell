"""KnockNucleus + KnockSignalMeta unit tests.

只测协议层与单元行为, 不依赖 mindflow 主循环.

覆盖范围:
- KnockSignalMeta.to_signal / from_signal 往返 + priority 继承
- build_impulse 走 default mode: 无 logos、无 effort override、无 mode 标记
- 空消息体的 knock 被丢弃 (knock 的硬协议)
- 错误 signal name 被忽略
- suppress 清 cache — 输后丢弃 (knock 的核心承诺, 与 input/aside 相反)
- attended 清 cache
- last-wins 覆盖 + peek stale 过滤
- lifecycle: 未运行时 add_signal 不投递
- KnockNucleusMeta 自解释发现
"""
import asyncio
import logging

import pytest
from ghoshell_container import Container

from ghoshell_moss.contracts.logger import LoggerItf
from ghoshell_moss.core.blueprint.mindflow import Impulse, Priority, Signal
from ghoshell_moss.core.mindflow.knock_nucleus import (
    KnockNucleus, KnockNucleusMeta, KnockSignalMeta, new_knock_signal,
)
from ghoshell_moss.message import Message


# ============================================================
# KnockSignalMeta — 协议往返
# ============================================================

def test_signal_meta_signal_name_is_knock():
    assert KnockSignalMeta.signal_name() == 'knock'


def test_signal_meta_default_priority_is_notice():
    """普通 knock 是 NOTICE — knock 必须可输, 默认值不能高到必赢."""
    signal = new_knock_signal(Message.new().with_content('come and get it'))
    assert signal.priority == Priority.NOTICE


def test_signal_meta_caller_override_priority():
    signal = KnockSignalMeta().to_signal(priority=Priority.WARNING)
    assert signal.priority == Priority.WARNING


def test_signal_meta_body_survives_roundtrip():
    """knock 的载荷是消息体, 不是 metadata — 往返不丢消息."""
    signal = new_knock_signal(Message.new().with_content('payload'))
    assert KnockSignalMeta.from_signal(signal) is not None
    assert len(signal.messages) == 1


def test_signal_meta_match_rejects_wrong_name():
    fake = Signal.new('input', priority=Priority.NOTICE)
    assert KnockSignalMeta.match(fake) is False
    assert KnockSignalMeta.from_signal(fake) is None


# ============================================================
# KnockNucleus.build_impulse — default mode 的忠实实装
# ============================================================

def _signal(text: str = 'something waiting', **kwargs) -> Signal:
    return new_knock_signal(Message.new().with_content(text), **kwargs)


def test_build_impulse_is_default_mode():
    """knock 不偏离 default: 不标 mode, 不设 effort, 不带 logos —
    赢了就走真实思考, 这是它与 command 的唯一差别."""
    nuc = KnockNucleus()
    impulse = nuc.build_impulse(_signal())
    assert impulse is not None
    assert impulse.mode == ''
    assert impulse.thinking_effort == ''
    assert impulse.logos == ''


def test_build_impulse_carries_messages():
    """knock 必须有消息体 — 消息要随 impulse 送达."""
    nuc = KnockNucleus()
    impulse = nuc.build_impulse(_signal('three results are ready'))
    assert impulse is not None
    assert len(impulse.messages) == 1


def test_build_impulse_inherits_signal_priority():
    nuc = KnockNucleus()
    impulse = nuc.build_impulse(_signal(priority=Priority.WARNING))
    assert impulse.priority == Priority.WARNING


def test_build_impulse_drops_signal_without_messages():
    """空消息体的 knock 是无效 knock — 没有可宣告的东西, 直接丢."""
    nuc = KnockNucleus()
    assert nuc.build_impulse(KnockSignalMeta().to_signal()) is None


def test_build_impulse_drops_wrong_signal_name():
    nuc = KnockNucleus()
    assert nuc.build_impulse(Signal.new('input')) is None


# ============================================================
# KnockNucleus — 输后丢弃 (核心承诺)
# ============================================================

@pytest.mark.asyncio
async def test_suppress_clears_cache():
    """输后丢弃: suppress 后 peek 必须回到 None.

    与 input (输时等待) / aside (输后保留) 相反 — knock 不重试, 门没敲开就没了.
    """
    async with KnockNucleus() as nuc:
        nuc.with_bus(lambda s: None, lambda imp: None)
        nuc.add_signal(_signal())
        assert nuc.peek() is not None
        nuc.suppress(Impulse(source='other_nucleus'))
        assert nuc.peek() is None


@pytest.mark.asyncio
async def test_attended_clears_cache():
    async with KnockNucleus() as nuc:
        nuc.with_bus(lambda s: None, lambda imp: None)
        nuc.add_signal(_signal())
        cached = nuc.peek()
        nuc.attended(cached)
        assert nuc.peek() is None


@pytest.mark.asyncio
async def test_add_signal_last_wins_overwrites_cache():
    """连续 knock, last-wins — 门铃只保留最后一次, 不排队."""
    notified: list[Impulse] = []
    async with KnockNucleus() as nuc:
        nuc.with_bus(signal_broadcast=lambda s: None, fire_impulse=notified.append)
        nuc.add_signal(_signal('first'))
        nuc.add_signal(_signal('second'))
        peeked = nuc.peek()
    assert peeked is notified[1]
    assert peeked is not notified[0]


@pytest.mark.asyncio
async def test_peek_filters_stale():
    async with KnockNucleus() as nuc:
        nuc.with_bus(lambda s: None, lambda imp: None)
        nuc.add_signal(_signal(stale_timeout=0.01))
        await asyncio.sleep(0.02)
        assert nuc.peek() is None


# ============================================================
# KnockNucleus.add_signal — bus 投递
# ============================================================

@pytest.mark.asyncio
async def test_add_signal_fires_impulse_via_bus():
    notified: list[Impulse] = []
    async with KnockNucleus() as nuc:
        nuc.with_bus(signal_broadcast=lambda s: None, fire_impulse=notified.append)
        nuc.add_signal(_signal('come look'))
    assert len(notified) == 1
    assert notified[0].priority == Priority.NOTICE


@pytest.mark.asyncio
async def test_add_signal_does_not_fire_without_messages():
    notified: list[Impulse] = []
    async with KnockNucleus() as nuc:
        nuc.with_bus(signal_broadcast=lambda s: None, fire_impulse=notified.append)
        nuc.add_signal(KnockSignalMeta().to_signal())
    assert notified == []


@pytest.mark.asyncio
async def test_add_signal_does_not_fire_when_not_running():
    notified: list[Impulse] = []
    nuc = KnockNucleus()
    nuc.with_bus(signal_broadcast=lambda s: None, fire_impulse=notified.append)
    nuc.add_signal(_signal())
    assert notified == []


def test_signals_declares_knock():
    assert KnockNucleus().signals() == ['knock']


def test_peek_returns_none_before_any_signal():
    assert KnockNucleus().peek() is None


# ============================================================
# KnockNucleusMeta — 自解释发现
# ============================================================

def test_nucleus_meta_name():
    assert KnockNucleusMeta().name() == KnockNucleus.NAME


def test_nucleus_meta_exposes_signal_meta():
    assert KnockSignalMeta in list(KnockNucleusMeta().signals())


def test_nucleus_meta_factory_returns_knock_nucleus():
    container = Container()
    container.set(LoggerItf, logging.getLogger(__name__))
    assert isinstance(KnockNucleusMeta().factory(container), KnockNucleus)
