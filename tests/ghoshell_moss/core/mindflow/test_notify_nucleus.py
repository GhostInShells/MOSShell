"""NotifyNucleus + NotifySignalMeta unit tests.

只测协议层与单元行为, 不依赖 mindflow 主循环 (集成测试在
test_impulse_primitive_integration.py 与四元 nucleus 集成测试覆盖).

覆盖范围:
- NotifySignalMeta.to_signal / from_signal 往返
- NotifyNucleus.build_impulse 字段卸载 (mode='notify')
- NotifyNucleus.add_signal 在 bus 上 fire-and-forget
- priority 完全继承 Signal.priority (NOTICE / WARNING / BACKGROUND 验证)
- 错误 signal name 被忽略
- lifecycle: 未运行时 add_signal 不投递
- NotifyNucleusMeta.signals() 暴露 NotifySignalMeta
"""
import logging
import pytest

from ghoshell_container import Container

from ghoshell_moss.contracts.logger import LoggerItf
from ghoshell_moss.core.blueprint.mindflow import (
    ChallengeMode, Impulse, Priority, Signal,
)
from ghoshell_moss.core.mindflow.notify_nucleus import (
    NotifyNucleus, NotifyNucleusMeta, NotifySignalMeta, new_notify_signal,
)
from ghoshell_moss.message import Message


# ============================================================
# NotifySignalMeta — 协议往返
# ============================================================

def test_signal_meta_name_is_notify():
    assert NotifySignalMeta.signal_name() == 'notify'


def test_signal_meta_default_priority_notice():
    signal = NotifySignalMeta().to_signal()
    assert signal.priority == Priority.NOTICE


def test_signal_meta_caller_override_priority():
    signal = NotifySignalMeta().to_signal(priority=Priority.WARNING)
    assert signal.priority == Priority.WARNING


def test_signal_meta_match_rejects_wrong_name():
    fake = Signal.new('other')
    assert NotifySignalMeta.match(fake) is False
    assert NotifySignalMeta.from_signal(fake) is None


def test_new_notify_signal_helper_carries_message():
    signal = new_notify_signal(Message.new().with_content('hi'))
    assert signal.name == 'notify'
    assert len(signal.messages) == 1


# ============================================================
# NotifyNucleus.build_impulse — 字段卸载
# ============================================================

def _signal(
        text: str = 'msg',
        *,
        priority: Priority = Priority.NOTICE,
) -> Signal:
    return new_notify_signal(Message.new().with_content(text), priority=priority)


def test_build_impulse_sets_notify_mode():
    """build_impulse 应产出 mode='notify' 的 impulse."""
    nuc = NotifyNucleus()
    impulse = nuc.build_impulse(_signal())
    assert impulse is not None
    assert impulse.mode == ChallengeMode.notify.value


def test_build_impulse_does_not_set_thinking_effort():
    """notify primitive 单纯设 mode, 不动 thinking_effort (保持默认空值)."""
    nuc = NotifyNucleus()
    impulse = nuc.build_impulse(_signal())
    assert impulse.thinking_effort == ''


def test_build_impulse_does_not_set_logos():
    """notify primitive 不动 logos (notify 的承诺是 messages 不丢, 不带反射弧)."""
    nuc = NotifyNucleus()
    impulse = nuc.build_impulse(_signal())
    assert impulse.logos == ''


def test_build_impulse_inherits_signal_priority_notice():
    nuc = NotifyNucleus()
    impulse = nuc.build_impulse(_signal(priority=Priority.NOTICE))
    assert impulse.priority == Priority.NOTICE


def test_build_impulse_inherits_signal_priority_warning():
    """WARNING signal 应得到 WARNING impulse — 不强制 floor."""
    nuc = NotifyNucleus()
    impulse = nuc.build_impulse(_signal(priority=Priority.WARNING))
    assert impulse.priority == Priority.WARNING


def test_build_impulse_inherits_signal_priority_background():
    """BACKGROUND signal 也原样继承 — 等价于 background_notice primitive."""
    nuc = NotifyNucleus()
    impulse = nuc.build_impulse(_signal(priority=Priority.BACKGROUND))
    assert impulse.priority == Priority.BACKGROUND


def test_build_impulse_drops_wrong_signal_name():
    """非 notify signal 应被 match 短路, build_impulse 返回 None."""
    nuc = NotifyNucleus()
    fake = Signal.new('input', priority=Priority.NOTICE)
    assert nuc.build_impulse(fake) is None


def test_build_impulse_preserves_messages():
    """signal.messages 应原样 copy 到 impulse.messages."""
    nuc = NotifyNucleus()
    signal = new_notify_signal(
        Message.new().with_content('a'),
        Message.new().with_content('b'),
    )
    impulse = nuc.build_impulse(signal)
    assert len(impulse.messages) == 2


# ============================================================
# NotifyNucleus.add_signal — bus 投递
# ============================================================

@pytest.mark.asyncio
async def test_add_signal_fires_impulse_via_bus():
    notified: list[Impulse] = []
    async with NotifyNucleus() as nuc:
        nuc.with_bus(
            signal_broadcast=lambda s: None,
            impulse_notify=lambda imp: notified.append(imp),
        )
        nuc.add_signal(_signal('hello'))
    assert len(notified) == 1
    assert notified[0].mode == ChallengeMode.notify.value


@pytest.mark.asyncio
async def test_add_signal_does_not_fire_when_not_running():
    notified: list[Impulse] = []
    nuc = NotifyNucleus()
    nuc.with_bus(
        signal_broadcast=lambda s: None,
        impulse_notify=lambda imp: notified.append(imp),
    )
    nuc.add_signal(_signal())
    assert notified == []


@pytest.mark.asyncio
async def test_add_signal_drops_wrong_name():
    notified: list[Impulse] = []
    async with NotifyNucleus() as nuc:
        nuc.with_bus(
            signal_broadcast=lambda s: None,
            impulse_notify=lambda imp: notified.append(imp),
        )
        nuc.add_signal(Signal.new('input'))
    assert notified == []


# ============================================================
# NotifyNucleus 反身性接口
# ============================================================

def test_signals_returns_notify_name():
    assert NotifyNucleus().signals() == ['notify']


def test_peek_returns_none_before_any_signal():
    assert NotifyNucleus().peek() is None


@pytest.mark.asyncio
async def test_peek_returns_cached_impulse_after_signal():
    async with NotifyNucleus() as nuc:
        nuc.with_bus(lambda s: None, lambda imp: None)
        nuc.add_signal(_signal('user_msg'))
        peeked = nuc.peek()
        assert peeked is not None
        assert peeked.mode == ChallengeMode.notify.value


@pytest.mark.asyncio
async def test_pop_impulse_clears_cache():
    async with NotifyNucleus() as nuc:
        nuc.with_bus(lambda s: None, lambda imp: None)
        nuc.add_signal(_signal())
        cached = nuc.peek()
        nuc.pop_impulse(cached)
        assert nuc.peek() is None


@pytest.mark.asyncio
async def test_add_signal_last_wins_overwrites_cache():
    async with NotifyNucleus() as nuc:
        nuc.with_bus(lambda s: None, lambda imp: None)
        nuc.add_signal(_signal('first'))
        nuc.add_signal(_signal('second'))
        peeked = nuc.peek()
        # 取最新一条的 messages.
        texts = [c['text'] for m in peeked.messages for c in m.contents if 'text' in c]
        assert 'second' in texts


# ============================================================
# NotifyNucleusMeta — 自解释发现
# ============================================================

def test_nucleus_meta_name():
    assert NotifyNucleusMeta().name() == NotifyNucleus.NAME


def test_nucleus_meta_exposes_signal_meta():
    metas = list(NotifyNucleusMeta().signals())
    assert NotifySignalMeta in metas


def test_nucleus_meta_factory_returns_notify_nucleus():
    container = Container()
    container.set(LoggerItf, logging.getLogger(__name__))
    nuc = NotifyNucleusMeta().factory(container)
    assert isinstance(nuc, NotifyNucleus)
