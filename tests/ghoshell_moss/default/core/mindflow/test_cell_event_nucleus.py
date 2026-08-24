"""CellEventNucleus + CellEventSignalMeta unit tests.

只测协议层与单元行为, 不依赖 Matrix/mesh/session 等外部抽象.

覆盖范围:
- CellEventSignalMeta.signal_name / priority / match
- CellEventNucleus.build_impulse → background_notice
- CellEventNucleus.add_signal → bus 投递
- priority 继承 signal (BACKGROUND)
- 错误 signal name 被忽略
- lifecycle: 未运行时 add_signal 不投递
- CellEventNucleusMeta 自解释发现
"""
import logging
import pytest

from ghoshell_container import Container

from ghoshell_moss.contracts.logger import LoggerItf
from ghoshell_moss.core.blueprint.mindflow import (
    ChallengeMode, Impulse, Priority, Signal,
)
from ghoshell_moss.signals import CellEventSignalMeta
from ghoshell_moss.core.mindflow.cell_event_nucleus import (
    CellEventNucleus, CellEventNucleusMeta, NAME,
)


# ============================================================
# CellEventSignalMeta — 协议
# ============================================================

def test_signal_name_is_cell_event():
    assert CellEventSignalMeta.signal_name() == 'cell_event'


def test_signal_priority_is_background():
    assert CellEventSignalMeta.priority() == Priority.BACKGROUND


def test_signal_meta_match_rejects_wrong_name():
    fake = Signal.new('other')
    assert CellEventSignalMeta.match(fake) is False
    assert CellEventSignalMeta.from_signal(fake) is None


# ============================================================
# CellEventNucleus.build_impulse — 字段卸载
# ============================================================

def _signal(priority: Priority = Priority.BACKGROUND) -> Signal:
    return CellEventSignalMeta().to_signal(
        description='node/test spawned', priority=priority,
    )


def test_build_impulse_sets_background_notice():
    nuc = CellEventNucleus()
    impulse = nuc.build_impulse(_signal())
    assert impulse is not None
    assert impulse.priority == Priority.BACKGROUND
    assert impulse.mode == ChallengeMode.notify.value


def test_build_impulse_always_background_priority():
    """background_notice primitive 强制 BACKGROUND, 不继承 signal priority."""
    nuc = CellEventNucleus()
    impulse = nuc.build_impulse(_signal(priority=Priority.INFO))
    assert impulse.priority == Priority.BACKGROUND


def test_build_impulse_drops_wrong_signal_name():
    nuc = CellEventNucleus()
    fake = Signal.new('input', priority=Priority.NOTICE)
    assert nuc.build_impulse(fake) is None


# ============================================================
# CellEventNucleus.add_signal — bus 投递
# ============================================================

@pytest.mark.asyncio
async def test_add_signal_fires_impulse_via_bus():
    notified: list[Impulse] = []
    async with CellEventNucleus() as nuc:
        nuc.with_bus(
            signal_broadcast=lambda s: None,
            fire_impulse=lambda imp: notified.append(imp),
        )
        nuc.add_signal(_signal())
    assert len(notified) == 1
    assert notified[0].mode == ChallengeMode.notify.value
    assert notified[0].priority == Priority.BACKGROUND


@pytest.mark.asyncio
async def test_add_signal_does_not_fire_when_not_running():
    notified: list[Impulse] = []
    nuc = CellEventNucleus()
    nuc.with_bus(
        signal_broadcast=lambda s: None,
        fire_impulse=lambda imp: notified.append(imp),
    )
    nuc.add_signal(_signal())
    assert notified == []


@pytest.mark.asyncio
async def test_add_signal_drops_wrong_name():
    notified: list[Impulse] = []
    async with CellEventNucleus() as nuc:
        nuc.with_bus(
            signal_broadcast=lambda s: None,
            fire_impulse=lambda imp: notified.append(imp),
        )
        nuc.add_signal(Signal.new('input'))
    assert notified == []


# ============================================================
# CellEventNucleus — 反身性接口
# ============================================================

def test_signals_returns_cell_event():
    assert CellEventNucleus().signals() == ['cell_event']


def test_peek_returns_none_before_any_signal():
    assert CellEventNucleus().peek() is None


@pytest.mark.asyncio
async def test_peek_returns_cached_impulse_after_signal():
    async with CellEventNucleus() as nuc:
        nuc.with_bus(lambda s: None, lambda imp: None)
        nuc.add_signal(_signal())
        peeked = nuc.peek()
        assert peeked is not None
        assert peeked.mode == ChallengeMode.notify.value


@pytest.mark.asyncio
async def test_attended_clears_cache():
    async with CellEventNucleus() as nuc:
        nuc.with_bus(lambda s: None, lambda imp: None)
        nuc.add_signal(_signal())
        cached = nuc.peek()
        nuc.attended(cached)
        assert nuc.peek() is None


@pytest.mark.asyncio
async def test_add_signal_last_wins_overwrites_cache():
    async with CellEventNucleus() as nuc:
        nuc.with_bus(lambda s: None, lambda imp: None)
        nuc.add_signal(_signal())
        nuc.add_signal(_signal())
        assert nuc.peek() is not None


# ============================================================
# CellEventNucleusMeta — 自解释发现
# ============================================================

def test_nucleus_meta_name():
    assert CellEventNucleusMeta().name() == NAME


def test_nucleus_meta_exposes_signal_meta():
    metas = list(CellEventNucleusMeta().signals())
    assert CellEventSignalMeta in metas


def test_nucleus_meta_factory_returns_cell_event_nucleus():
    container = Container()
    container.set(LoggerItf, logging.getLogger(__name__))
    nuc = CellEventNucleusMeta().factory(container)
    assert isinstance(nuc, CellEventNucleus)
