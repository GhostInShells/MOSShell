"""CommandNucleus + CommandSignalMeta unit tests.

只测协议层与单元行为, 不依赖 mindflow 主循环 (集成测试在
test_impulse_primitive_integration.py 与后续集成测试覆盖).

覆盖范围:
- CommandSignalMeta.to_signal / from_signal 往返
- CommandNucleus.build_impulse 字段卸载 (logos / priority / thinking_effort / mode)
- CommandNucleus.add_signal 在 bus 上 fire-and-forget
- priority 完全继承 Signal.priority (NOTICE / FATAL / WARNING 三档验证)
- meta.logos 为空时不发 impulse
- 错误 signal name 被忽略
- lifecycle: 未运行时 add_signal 不投递
- CommandNucleusMeta.signals() 暴露 CommandSignalMeta
"""
import logging
import pytest

from ghoshell_container import Container

from ghoshell_moss.contracts.logger import LoggerItf
from ghoshell_moss.core.blueprint.mindflow import (
    ChallengeMode, Impulse, Priority, Signal,
)
from ghoshell_moss.core.mindflow.command_nucleus import (
    CommandNucleus, CommandNucleusMeta, CommandSignalMeta, new_command_signal,
)
from ghoshell_moss.message import Message


# ============================================================
# CommandSignalMeta — 协议往返
# ============================================================

def test_signal_meta_roundtrip_carries_logos():
    """to_signal -> from_signal 应保留 logos 字段."""
    meta = CommandSignalMeta(logos='do_x')
    signal = meta.to_signal()
    parsed = CommandSignalMeta.from_signal(signal)
    assert parsed is not None
    assert parsed.logos == 'do_x'


def test_signal_meta_default_priority_is_notice():
    """CommandSignalMeta 默认 priority = NOTICE (普通命令)."""
    signal = CommandSignalMeta(logos='x').to_signal()
    assert signal.priority == Priority.NOTICE


def test_signal_meta_caller_override_priority():
    """to_signal(priority=FATAL) 应覆盖类默认值."""
    signal = CommandSignalMeta(logos='x').to_signal(priority=Priority.FATAL)
    assert signal.priority == Priority.FATAL


def test_signal_meta_match_rejects_wrong_name():
    """match() / from_signal 应按 signal.name 短路."""
    fake = Signal.new('other', priority=Priority.NOTICE)
    assert CommandSignalMeta.match(fake) is False
    assert CommandSignalMeta.from_signal(fake) is None


# ============================================================
# CommandNucleus.build_impulse — 字段卸载
# ============================================================

def _signal(
        logos: str,
        *,
        priority: Priority = Priority.NOTICE,
) -> Signal:
    return new_command_signal(logos, priority=priority)


def test_build_impulse_unloads_command_only_fields():
    """build_impulse 应产出 command_only 组合: logos + thinking_effort='none'."""
    nuc = CommandNucleus()
    impulse = nuc.build_impulse(_signal('exec_me'))
    assert impulse is not None
    assert impulse.logos == 'exec_me'
    assert impulse.thinking_effort == 'none'
    # mode 留默认 (command_only 不设 mode)
    assert impulse.mode == ''


def test_build_impulse_inherits_signal_priority_notice():
    """priority 完全继承 — NOTICE 进, NOTICE 出."""
    nuc = CommandNucleus()
    impulse = nuc.build_impulse(_signal('x', priority=Priority.NOTICE))
    assert impulse.priority == Priority.NOTICE


def test_build_impulse_inherits_signal_priority_fatal():
    """FATAL signal 应得到 FATAL impulse (等价于 fatal_command primitive).
    不设 priority floor — 这是 CommandNucleus 的核心承诺."""
    nuc = CommandNucleus()
    impulse = nuc.build_impulse(_signal('halt', priority=Priority.FATAL))
    assert impulse.priority == Priority.FATAL


def test_build_impulse_inherits_signal_priority_warning():
    """WARNING (介于 NOTICE 和 FATAL 之间) 也应原样继承."""
    nuc = CommandNucleus()
    impulse = nuc.build_impulse(_signal('warn_act', priority=Priority.WARNING))
    assert impulse.priority == Priority.WARNING


def test_build_impulse_drops_signal_with_empty_logos():
    """meta.logos 为空时 build_impulse 返回 None — 不让空命令落到 ghost."""
    # 用 SignalMeta 直接构造一个 logos='' 的 signal.
    nuc = CommandNucleus()
    signal = CommandSignalMeta(logos='').to_signal()
    assert nuc.build_impulse(signal) is None


def test_build_impulse_drops_wrong_signal_name():
    """非 command signal 应被 from_signal 短路, build_impulse 返回 None."""
    nuc = CommandNucleus()
    fake = Signal.new('input', priority=Priority.NOTICE)
    assert nuc.build_impulse(fake) is None


# ============================================================
# CommandNucleus.add_signal — bus 投递
# ============================================================

@pytest.mark.asyncio
async def test_add_signal_fires_impulse_via_bus():
    """add_signal 应立刻通过 fire_impulse 投递 (fire-and-forget)."""
    notified: list[Impulse] = []
    async with CommandNucleus() as nuc:
        nuc.with_bus(
            signal_broadcast=lambda s: None,
            fire_impulse=lambda imp: notified.append(imp),
        )
        nuc.add_signal(_signal('go'))
    assert len(notified) == 1
    assert notified[0].logos == 'go'
    assert notified[0].thinking_effort == 'none'


@pytest.mark.asyncio
async def test_add_signal_does_not_fire_when_not_running():
    """生命周期外的 add_signal 应静默丢弃, 不投递."""
    notified: list[Impulse] = []
    nuc = CommandNucleus()
    nuc.with_bus(
        signal_broadcast=lambda s: None,
        fire_impulse=lambda imp: notified.append(imp),
    )
    # 不进入 __aenter__ → not running.
    nuc.add_signal(_signal('ghost'))
    assert notified == []


@pytest.mark.asyncio
async def test_add_signal_does_not_fire_on_empty_logos():
    """logos 为空的 signal 不应产生 impulse 投递."""
    notified: list[Impulse] = []
    async with CommandNucleus() as nuc:
        nuc.with_bus(
            signal_broadcast=lambda s: None,
            fire_impulse=lambda imp: notified.append(imp),
        )
        empty = CommandSignalMeta(logos='').to_signal()
        nuc.add_signal(empty)
    assert notified == []


@pytest.mark.asyncio
async def test_add_signal_does_not_fire_on_wrong_signal_name():
    """非 command signal 不应触发投递."""
    notified: list[Impulse] = []
    async with CommandNucleus() as nuc:
        nuc.with_bus(
            signal_broadcast=lambda s: None,
            fire_impulse=lambda imp: notified.append(imp),
        )
        nuc.add_signal(Signal.new('input'))
    assert notified == []


# ============================================================
# CommandNucleus 反身性接口
# ============================================================

def test_signals_returns_command_name():
    """signals() 应声明监听 'command' 信号."""
    nuc = CommandNucleus()
    assert nuc.signals() == ['command']


def test_peek_returns_none_before_any_signal():
    """初始无 signal 时 peek 应为 None — last-impulse cache 起步为空."""
    nuc = CommandNucleus()
    assert nuc.peek() is None


@pytest.mark.asyncio
async def test_peek_returns_cached_impulse_after_signal():
    """add_signal 后 peek 应返回 cached impulse (mindflow pull-based 协议要求)."""
    async with CommandNucleus() as nuc:
        nuc.with_bus(lambda s: None, lambda imp: None)
        nuc.add_signal(_signal('cmd'))
        peeked = nuc.peek()
        assert peeked is not None
        assert peeked.logos == 'cmd'


@pytest.mark.asyncio
async def test_attended_clears_cache():
    """attended 后 peek 应回到 None — 模拟 mindflow 仲裁后通知清状态."""
    async with CommandNucleus() as nuc:
        nuc.with_bus(lambda s: None, lambda imp: None)
        nuc.add_signal(_signal('cmd'))
        cached = nuc.peek()
        nuc.attended(cached)
        assert nuc.peek() is None


@pytest.mark.asyncio
async def test_add_signal_last_wins_overwrites_cache():
    """连续 add_signal, last-wins 覆盖未消费的旧 impulse."""
    async with CommandNucleus() as nuc:
        nuc.with_bus(lambda s: None, lambda imp: None)
        nuc.add_signal(_signal('first'))
        nuc.add_signal(_signal('second'))
        peeked = nuc.peek()
        assert peeked.logos == 'second'


# ============================================================
# CommandNucleusMeta — 自解释发现
# ============================================================

def test_nucleus_meta_name():
    assert CommandNucleusMeta().name() == CommandNucleus.NAME


def test_nucleus_meta_exposes_signal_meta():
    """signals() 应暴露 CommandSignalMeta, 让 ``moss manifests`` 能发现协议."""
    meta = CommandNucleusMeta()
    signal_metas = list(meta.signals())
    assert CommandSignalMeta in signal_metas


def test_nucleus_meta_factory_returns_command_nucleus():
    """factory() 应返回 CommandNucleus 实例."""
    container = Container()
    container.set(LoggerItf, logging.getLogger(__name__))
    nuc = CommandNucleusMeta().factory(container)
    assert isinstance(nuc, CommandNucleus)
