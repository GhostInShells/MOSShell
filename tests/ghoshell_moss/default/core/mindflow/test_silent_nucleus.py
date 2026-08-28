"""SilentNucleus + SilentSignalMeta unit tests.

只测协议层与单元行为, 不依赖 mindflow 主循环.

覆盖范围:
- SilentSignalMeta 协议往返
- 单 signal: silent mode 卸载 + priority 继承
- 多 signal 聚合: max priority / max strength / 全 messages
- buffer_size 上限 (溢出丢最早)
- stale 过滤 (加入前 + rebuild 时)
- suppress 冷静期阻断 notify_cb (signal 仍累积)
- min_priority 过滤
- attended 清空 buffer (下游 attention 已消费)
- lifecycle / 反身性接口
- SilentNucleusMeta 自解释发现
"""
import asyncio
import logging
import time

import pytest
from ghoshell_container import Container

from ghoshell_moss.contracts.logger import LoggerItf
from ghoshell_moss.core.blueprint.mindflow import (
    ChallengeMode, Impulse, Priority, Signal,
)
from ghoshell_moss.core.mindflow.silent_nucleus import (
    SilentNucleus, SilentNucleusMeta, SilentSignalMeta, new_silent_signal,
)
from ghoshell_moss.message import Message


# ============================================================
# SilentSignalMeta — 协议往返
# ============================================================

def test_signal_meta_name_is_silent():
    assert SilentSignalMeta.signal_name() == 'silent'


def test_signal_meta_default_priority_notice():
    assert SilentSignalMeta.priority() == Priority.NOTICE


def test_signal_meta_match_rejects_wrong_name():
    fake = Signal.new('other')
    assert SilentSignalMeta.match(fake) is False


def test_new_silent_signal_helper():
    sig = new_silent_signal(Message.new().with_content('data'))
    assert sig.name == 'silent'
    assert sig.priority == Priority.NOTICE
    assert len(sig.messages) == 1


# ============================================================
# 单 signal 行为
# ============================================================

def _signal(
        text: str = 'data',
        *,
        priority: Priority = Priority.NOTICE,
        strength: int = 100,
        stale_timeout: float = 0,
        complete: bool = True,
) -> Signal:
    return new_silent_signal(
        Message.new().with_content(text),
        priority=priority,
        stale_timeout=stale_timeout,
    ).model_copy(update={'strength': strength, 'complete': complete})


@pytest.mark.asyncio
async def test_single_signal_produces_silent_mode_impulse():
    """单 signal 进入 → peek 拿到的 impulse 应标记 mode='silent'."""
    async with SilentNucleus() as nuc:
        nuc.add_signal(_signal())
        impulse = nuc.peek()
        assert impulse is not None
        assert impulse.mode == ChallengeMode.silent.value


@pytest.mark.asyncio
async def test_single_signal_priority_inherited():
    async with SilentNucleus() as nuc:
        nuc.add_signal(_signal(priority=Priority.WARNING))
        assert nuc.peek().priority == Priority.WARNING


# ============================================================
# 多 signal 聚合 — 优先级提取
# ============================================================

@pytest.mark.asyncio
async def test_aggregate_picks_max_priority():
    """多 signal buffer 后, 输出 impulse 的 priority 是 buffer 内 max."""
    async with SilentNucleus() as nuc:
        nuc.add_signal(_signal(priority=Priority.INFO))
        nuc.add_signal(_signal(priority=Priority.WARNING))
        nuc.add_signal(_signal(priority=Priority.NOTICE))
        assert nuc.peek().priority == Priority.WARNING


@pytest.mark.asyncio
async def test_aggregate_picks_max_strength():
    async with SilentNucleus() as nuc:
        nuc.add_signal(_signal(strength=50))
        nuc.add_signal(_signal(strength=200))
        nuc.add_signal(_signal(strength=100))
        assert nuc.peek().strength == 200


@pytest.mark.asyncio
async def test_aggregate_concatenates_all_messages():
    """messages 累积 — silent 是数据流, 全部保留供下游 attention drain."""
    async with SilentNucleus() as nuc:
        nuc.add_signal(_signal('m1'))
        nuc.add_signal(_signal('m2'))
        nuc.add_signal(_signal('m3'))
        impulse = nuc.peek()
        texts = [c['text'] for m in impulse.messages for c in m.contents if 'text' in c]
        assert texts == ['m1', 'm2', 'm3']


@pytest.mark.asyncio
async def test_aggregate_complete_when_all_complete():
    async with SilentNucleus() as nuc:
        nuc.add_signal(_signal(complete=True))
        nuc.add_signal(_signal(complete=True))
        assert nuc.peek().complete is True


@pytest.mark.asyncio
async def test_aggregate_incomplete_if_any_partial():
    async with SilentNucleus() as nuc:
        nuc.add_signal(_signal(complete=True))
        nuc.add_signal(_signal(complete=False))
        assert nuc.peek().complete is False


# ============================================================
# Buffer 上限
# ============================================================

@pytest.mark.asyncio
async def test_buffer_size_limit_drops_oldest():
    async with SilentNucleus(buffer_size=3) as nuc:
        for i in range(5):
            nuc.add_signal(_signal(f'msg{i}'))
        impulse = nuc.peek()
        texts = [c['text'] for m in impulse.messages for c in m.contents if 'text' in c]
        # 旧的 msg0/msg1 被丢弃, 保留最近 3 条.
        assert texts == ['msg2', 'msg3', 'msg4']


# ============================================================
# Stale 过滤
# ============================================================

@pytest.mark.asyncio
async def test_stale_signal_dropped_on_add():
    """已 stale 的 signal 不应入 buffer."""
    async with SilentNucleus() as nuc:
        stale = _signal(stale_timeout=0.01)
        time.sleep(0.02)
        assert stale.is_stale()
        nuc.add_signal(stale)
        # 没有有效 signal → peek 应空.
        assert nuc.peek() is None


@pytest.mark.asyncio
async def test_stale_signals_filtered_on_rebuild():
    """rebuild 时旧 stale signal 被过滤, 新 valid signal 仍累积."""
    async with SilentNucleus() as nuc:
        # 先放一条短期会 stale 的.
        nuc.add_signal(_signal('expire_me', stale_timeout=0.01))
        time.sleep(0.02)
        # 再放一条新鲜的, 触发 rebuild — 旧的应被过滤.
        nuc.add_signal(_signal('fresh'))
        impulse = nuc.peek()
        texts = [c['text'] for m in impulse.messages for c in m.contents if 'text' in c]
        assert texts == ['fresh']


# ============================================================
# Suppress 冷静期
# ============================================================

@pytest.mark.asyncio
async def test_suppress_blocks_notify_cb_but_buffer_continues():
    """suppress 冷静期内 notify_cb 不被调; 但 signal 仍持续入 buffer."""
    notified: list[Impulse] = []
    async with SilentNucleus(suppress_seconds=0.2) as nuc:
        nuc.with_bus(
            signal_broadcast=lambda s: None,
            fire_impulse=lambda imp: notified.append(imp),
        )
        nuc.add_signal(_signal('first'))
        assert len(notified) == 1  # 首次通过

        # 模拟仲裁失败 → mindflow 调 suppress.
        nuc.suppress(Impulse(source='other'), None)

        nuc.add_signal(_signal('second'))
        assert len(notified) == 1  # 冷静期内不再通知

        # 但 buffer 里 signal 累积了.
        impulse = nuc.peek()
        texts = [c['text'] for m in impulse.messages for c in m.contents if 'text' in c]
        assert texts == ['first', 'second']

        # 冷静期过.
        await asyncio.sleep(0.21)
        nuc.add_signal(_signal('third'))
        assert len(notified) == 2  # 重新可通知


# ============================================================
# Min priority 过滤
# ============================================================

@pytest.mark.asyncio
async def test_min_priority_filter():
    """低于 min_priority 的 signal 直接丢弃, 不入 buffer."""
    async with SilentNucleus(min_priority=Priority.NOTICE) as nuc:
        nuc.add_signal(_signal(priority=Priority.INFO))
        assert nuc.peek() is None
        nuc.add_signal(_signal(priority=Priority.NOTICE))
        assert nuc.peek() is not None


# ============================================================
# attended — buffer 清空
# ============================================================

@pytest.mark.asyncio
async def test_attended_clears_buffer():
    """attended 后, buffer 与 cache 都应清空 — 下游 attention 已消费, 不重复."""
    async with SilentNucleus() as nuc:
        nuc.add_signal(_signal('a'))
        nuc.add_signal(_signal('b'))
        impulse = nuc.peek()
        assert impulse is not None
        nuc.attended(impulse)
        assert nuc.peek() is None


# ============================================================
# 错误 signal name + lifecycle
# ============================================================

@pytest.mark.asyncio
async def test_drops_wrong_signal_name():
    async with SilentNucleus() as nuc:
        nuc.add_signal(Signal.new('input', Message.new().with_content('x')))
        assert nuc.peek() is None


@pytest.mark.asyncio
async def test_does_not_buffer_when_not_running():
    nuc = SilentNucleus()
    # 不进入 __aenter__.
    nuc.add_signal(_signal())
    assert nuc.peek() is None


# ============================================================
# 反身性接口
# ============================================================

def test_signals_returns_silent_name():
    assert SilentNucleus().signals() == ['silent']


@pytest.mark.asyncio
async def test_status_shows_buffered_count_and_top_description():
    async with SilentNucleus() as nuc:
        assert nuc.status() == ""
        nuc.add_signal(_signal('low_pri').model_copy(
            update={'priority': Priority.INFO, 'description': 'low'},
        ))
        nuc.add_signal(_signal('high_pri').model_copy(
            update={'priority': Priority.WARNING, 'description': 'critical thing'},
        ))
        status = nuc.status()
        assert 'buffered: 2' in status
        # 高优 description 应出现在 status.
        assert 'critical thing' in status


# ============================================================
# SilentNucleusMeta — 自解释发现
# ============================================================

def test_nucleus_meta_name():
    assert SilentNucleusMeta().name() == SilentNucleus.NAME


def test_nucleus_meta_exposes_signal_meta():
    metas = list(SilentNucleusMeta().signals())
    assert SilentSignalMeta in metas


def test_nucleus_meta_factory_returns_silent_nucleus():
    container = Container()
    container.set(LoggerItf, logging.getLogger(__name__))
    nuc = SilentNucleusMeta().factory(container)
    assert isinstance(nuc, SilentNucleus)
