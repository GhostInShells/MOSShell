"""ListenerNucleus + ListenerSignal protocol tests.

只测协议层与单元行为, 不依赖 mindflow 主循环 (同 test_interrupt_nucleus).

覆盖范围:
- ListenerSignal 协议: signal_name / 默认 NOTICE / 往返 (from_signal) / 拒绝异名
- ListenerNucleus 信号面: signals() 监听 listener
- 完整 turn (首包→分句→尾包): impulse id 一致 (same-id absorb 前提), complete 相位正确
- 首包打断: complete=False + 高强, attended 降回正常强度 (用户钦定 attended 语义)
- 首包打断关闭: 首包不抢 attention, 分句 buffer, 尾包提交全量
- 分句响应开启: 分句逐包 impulse, 消息体为累计 buffer
- 尾包 diff: 内容完全由分句 sent 态决定 (已送达不重发; 尾包自身 text 不参与)
- 冷却语义: 尾包 commit 不受冷却约束; 分句在冷却期内 buffer 不发射
- 首包 priority 参数化: set_first_packet_priority
- ListenerNucleusMeta: name / signal meta 暴露 / factory
"""
import asyncio
import logging

import pytest
from ghoshell_container import Container

from ghoshell_moss.contracts.logger import LoggerItf
from ghoshell_moss.core.blueprint.mindflow import Priority, Signal
from ghoshell_moss.core.mindflow.listener_nucleus import (
    ListenerNucleus, ListenerNucleusMeta, ListenerSignal, ListenerPacket, new_listener_signal,
)


# ============================================================
# ListenerSignal — 协议往返
# ============================================================

def test_signal_name_is_listener():
    assert ListenerSignal.signal_name() == 'listener'


def test_signal_default_priority_notice():
    assert ListenerSignal.priority() == Priority.NOTICE
    signal = ListenerSignal(packet=ListenerPacket.CLAUSE).to_signal('x')
    assert signal.priority == Priority.NOTICE


def test_signal_roundtrip_preserves_fields():
    sig = new_listener_signal(
        ListenerPacket.CLAUSE, '今天天气不错',
        turn_id='t1', clause_index=1, start_ms=100, end_ms=800, confidence=0.9,
    )
    meta = ListenerSignal.from_signal(sig)
    assert meta is not None
    assert meta.packet == ListenerPacket.CLAUSE
    assert meta.text == '今天天气不错'
    assert meta.turn_id == 't1'
    assert meta.clause_index == 1
    assert meta.start_ms == 100
    assert meta.end_ms == 800
    assert meta.confidence == 0.9


def test_signal_match_rejects_wrong_name():
    assert ListenerSignal.match(Signal.new('input')) is False


# ============================================================
# ListenerNucleus — 信号面
# ============================================================

def test_signals_listens_listener():
    assert ListenerNucleus().signals() == ['listener']


# ============================================================
# 完整 turn — impulse id 一致 + complete 相位
# ============================================================

@pytest.mark.asyncio
async def test_full_turn_ids_consistent_and_complete_flags():
    """首包(打断开) → 尾包; 同 id 才能 same-id absorb."""
    notified: list = []
    async with ListenerNucleus() as nuc:
        nuc.with_bus(lambda s: None, lambda imp: notified.append(imp))
        nuc.add_signal(new_listener_signal(ListenerPacket.FIRST, turn_id='t1'))
        nuc.add_signal(new_listener_signal(ListenerPacket.CLAUSE, '今天天气不错', turn_id='t1', clause_index=1))
        nuc.add_signal(new_listener_signal(ListenerPacket.TAIL, '', turn_id='t1', clause_index=1))
    # 分句响应默认关: 首包 + 尾包 两个 impulse.
    assert len(notified) == 2
    first, tail = notified
    assert first.complete is False
    assert tail.complete is True
    assert first.id == tail.id  # 同 id — same-id absorb 前提
    assert first.strength > 100  # 首包高强
    assert _message_text(tail) == '今天天气不错'  # 分句未送达, 尾包发全量


# ============================================================
# 首包打断 — attended 降回正常强度
# ============================================================

@pytest.mark.asyncio
async def test_first_packet_attended_flattens_to_run_params():
    notified: list = []
    async with ListenerNucleus() as nuc:
        nuc.with_bus(lambda s: None, lambda imp: notified.append(imp))
        nuc.add_signal(new_listener_signal(ListenerPacket.FIRST, turn_id='t1'))
        first = notified[0]
        assert first.complete is False
        assert first.priority == Priority.WARNING  # challenge priority
        assert first.strength > 100
        result = nuc.attended(first)
        # attended 降回运行参数 (priority→INFO, strength→正常), 返回改动后的 impulse.
        assert result is first
        assert first.priority == Priority.INFO
        assert first.strength == 100


@pytest.mark.asyncio
async def test_clause_attended_flattens_to_run_params():
    """分句抢到 attention 后也降为运行参数, 让下一句能打断 (连续思考帧前提)."""
    notified: list = []
    async with ListenerNucleus(clause_response=True) as nuc:
        nuc.with_bus(lambda s: None, lambda imp: notified.append(imp))
        nuc.add_signal(new_listener_signal(ListenerPacket.FIRST, turn_id='t1'))
        nuc.add_signal(new_listener_signal(ListenerPacket.CLAUSE, '句1', turn_id='t1', clause_index=1))
        clause = notified[1]
        assert clause.complete is True
        assert clause.priority == Priority.NOTICE  # challenge priority
        result = nuc.attended(clause)
        assert result is clause
        assert clause.priority == Priority.INFO  # run priority


@pytest.mark.asyncio
async def test_first_packet_interrupt_off_holds_until_tail():
    """首包打断关: 首包不 fire, 分句只 buffer, 尾包提交全量."""
    notified: list = []
    async with ListenerNucleus(first_packet_interrupt=False) as nuc:
        nuc.with_bus(lambda s: None, lambda imp: notified.append(imp))
        nuc.add_signal(new_listener_signal(ListenerPacket.FIRST, turn_id='t1'))
        assert notified == []
        nuc.add_signal(new_listener_signal(ListenerPacket.CLAUSE, '你好', turn_id='t1', clause_index=1))
        assert notified == []  # clause_response 关, 分句只 buffer
        nuc.add_signal(new_listener_signal(ListenerPacket.TAIL, '', turn_id='t1'))
        assert len(notified) == 1
        assert notified[0].complete is True
        assert _message_text(notified[0]) == '你好'


# ============================================================
# 分句响应 — 逐包 impulse + 累计 buffer
# ============================================================

@pytest.mark.asyncio
async def test_clause_response_cumulative_buffer():
    notified: list = []
    async with ListenerNucleus(clause_response=True) as nuc:
        nuc.with_bus(lambda s: None, lambda imp: notified.append(imp))
        nuc.add_signal(new_listener_signal(ListenerPacket.FIRST, turn_id='t1'))
        nuc.add_signal(new_listener_signal(ListenerPacket.CLAUSE, '句1', turn_id='t1', clause_index=1))
        nuc.add_signal(new_listener_signal(ListenerPacket.CLAUSE, '句2', turn_id='t1', clause_index=2))
        nuc.add_signal(new_listener_signal(ListenerPacket.TAIL, '', turn_id='t1', clause_index=2))
    # 首包 + 句1 + 句2 + 尾包.
    assert len(notified) == 4
    first, clause1, clause2, tail = notified[0], notified[1], notified[2], notified[3]
    # 分句 impulse 携带累计 buffer (句1 = "句1"; 句2 = "句1\n句2").
    assert _message_text(clause1) == '句1'
    assert '句1' in _message_text(clause2) and '句2' in _message_text(clause2)
    # 分句用独立 id + complete=True (ghost 逐句思考, 互相打断的前提).
    assert first.complete is False
    assert clause1.complete is True
    assert clause2.complete is True
    assert clause1.id != clause2.id  # 不同 id 才能互相打断 (走 challenge 而非 absorb)
    assert clause1.id != first.id   # 分句不走 turn 的 same-id absorb
    assert tail.complete is True


# ============================================================
# 尾包 diff — 内容完全由分句 sent 态决定
# ============================================================

@pytest.mark.asyncio
async def test_tail_diff_empty_when_all_clauses_sent():
    """分句已送达 → 尾包 diff 为空, 不重发 (尾包自身 text 不参与递送)."""
    notified: list = []
    async with ListenerNucleus(clause_response=True) as nuc:
        nuc.with_bus(lambda s: None, lambda imp: notified.append(imp))
        nuc.add_signal(new_listener_signal(ListenerPacket.FIRST, turn_id='t1'))
        nuc.add_signal(new_listener_signal(ListenerPacket.CLAUSE, '句1', turn_id='t1', clause_index=1))
        nuc.attended(notified[1])  # 句1 已送达
        nuc.add_signal(new_listener_signal(ListenerPacket.TAIL, '尾巴', turn_id='t1', clause_index=1))
        tail = notified[-1]
        assert _message_text(tail) == ''


# ============================================================
# 冷却语义 — 尾包不受约束, 分句 buffer 不发射
# ============================================================

@pytest.mark.asyncio
async def test_tail_not_gated_by_suppress():
    """首包抢占失败 (suppress) 进入冷却, 但尾包 commit 仍提交."""
    notified: list = []
    async with ListenerNucleus(suppress_seconds=10.0) as nuc:
        nuc.with_bus(lambda s: None, lambda imp: notified.append(imp))
        nuc.add_signal(new_listener_signal(ListenerPacket.FIRST, turn_id='t1'))
        nuc.suppress(notified[0])  # 进入冷却
        nuc.add_signal(new_listener_signal(ListenerPacket.TAIL, '', turn_id='t1'))
        assert len(notified) == 2  # 首包 + 尾包
        assert notified[-1].complete is True


@pytest.mark.asyncio
async def test_clause_firing_gated_by_cooldown():
    """冷却期内分句 buffer 不发射, 尾包仍提交全量 (内容不丢)."""
    notified: list = []
    async with ListenerNucleus(clause_response=True, suppress_seconds=10.0) as nuc:
        nuc.with_bus(lambda s: None, lambda imp: notified.append(imp))
        nuc.add_signal(new_listener_signal(ListenerPacket.FIRST, turn_id='t1'))
        nuc.suppress(notified[0])  # 进入冷却
        nuc.add_signal(new_listener_signal(ListenerPacket.CLAUSE, '句1', turn_id='t1', clause_index=1))
        assert len(notified) == 1  # 冷却期内分句不发射 (但已 buffer)
        nuc.add_signal(new_listener_signal(ListenerPacket.TAIL, '', turn_id='t1'))
        assert len(notified) == 2
        assert _message_text(notified[-1]) == '句1'  # buffer 未丢, 尾包兜底


# ============================================================
# 首包 priority 参数化
# ============================================================

@pytest.mark.asyncio
async def test_set_first_packet_priority_takes_effect():
    notified: list = []
    async with ListenerNucleus() as nuc:
        nuc.with_bus(lambda s: None, lambda imp: notified.append(imp))
        nuc.set_first_packet_priority(Priority.ERROR)
        nuc.add_signal(new_listener_signal(ListenerPacket.FIRST, turn_id='t1'))
        assert notified[0].priority == Priority.ERROR


# ============================================================
# ListenerNucleusMeta — 自解释发现
# ============================================================

def test_nucleus_meta_name():
    assert ListenerNucleusMeta().name() == ListenerNucleus.NAME


def test_nucleus_meta_exposes_signal_meta():
    metas = list(ListenerNucleusMeta().signals())
    assert ListenerSignal in metas


def test_nucleus_meta_factory_returns_listener_nucleus():
    container = Container()
    container.set(LoggerItf, logging.getLogger(__name__))
    nuc = ListenerNucleusMeta().factory(container)
    assert isinstance(nuc, ListenerNucleus)


def _message_text(impulse) -> str:
    """汇总 impulse 所有消息体的文本."""
    texts = []
    for msg in impulse.messages:
        for c in msg.contents:
            if isinstance(c, dict) and c.get('type') == 'text':
                texts.append(c.get('text', ''))
    return '\n'.join(texts)
