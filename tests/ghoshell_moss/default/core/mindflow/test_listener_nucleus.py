"""ListenerNucleus + ListenerSignal protocol tests.

只测协议层与单元行为, 不依赖 mindflow 主循环 (同 test_interrupt_nucleus).

覆盖范围:
- ListenerSignal 协议: signal_name / 默认 INFO / 往返 (segment_id/interrupt/mode/logos/source)
- ListenerNucleus 信号面: signals() 监听 listener
- 完整 turn (打断包 → 发送包): same-id (segment_id) + complete 相位
- 打断包: complete=False + interrupt + 高强 + effort=none, attended 降回 + 小冷却
- 发送包: complete=True + notify + 消息体 <listen> tag 包装
- 冷却双档: 打断包受 suppress/attended 冷却约束, 发送包不受 (内容不丢)
- ListenerNucleusMeta: name / signal meta 暴露 / factory
"""
import asyncio
import logging

import pytest
from ghoshell_container import Container

from ghoshell_moss.contracts.logger import LoggerItf
from ghoshell_moss.core.blueprint.mindflow import ChallengeMode, Impulse, Priority, Signal
from ghoshell_moss.core.mindflow.listener_nucleus import (
    ListenerNucleus, ListenerNucleusMeta, ListenerSignal,
    new_listener_signal,
)


# ============================================================
# ListenerSignal — 协议往返
# ============================================================

def test_signal_name_is_listener():
    assert ListenerSignal.signal_name() == 'listener'


def test_signal_default_priority_info():
    assert ListenerSignal.priority() == Priority.INFO
    signal = ListenerSignal().to_signal('x')
    assert signal.priority == Priority.INFO


def test_signal_roundtrip_preserves_fields():
    sig = new_listener_signal(
        '今天天气不错',
        segment_id='t1', interrupt=True, mode='notify', logos='ok', source='wake_word',
    )
    assert sig.complete is True  # complete 是 Signal 通用字段, 不进 metadata
    meta = ListenerSignal.from_signal(sig)
    assert meta is not None
    assert meta.segment_id == 't1'
    assert meta.interrupt is True
    assert meta.mode == 'notify'
    assert meta.logos == 'ok'
    assert meta.source == 'wake_word'


def test_signal_roundtrip_complete_false():
    sig = new_listener_signal(segment_id='t1', interrupt=True, complete=False, priority=Priority.WARNING)
    assert sig.complete is False


def test_signal_match_rejects_wrong_name():
    assert ListenerSignal.match(Signal.new('input')) is False


def test_signal_source_defaults_to_asr():
    meta = ListenerSignal.from_signal(new_listener_signal('你好'))
    assert meta is not None
    assert meta.source == 'asr'


# ============================================================
# ListenerNucleus — 信号面
# ============================================================

def test_signals_listens_listener():
    assert ListenerNucleus().signals() == ['listener']


# ============================================================
# 完整 turn — same-id + complete 相位
# ============================================================

@pytest.mark.asyncio
async def test_full_turn_interrupt_then_deliver():
    """打断包(complete=False) → 发送包(complete=True); 同 id 才能 same-id absorb."""
    notified: list = []
    async with ListenerNucleus() as nuc:
        nuc.with_bus(lambda s: None, lambda imp: notified.append(imp))
        nuc.add_signal(new_listener_signal(segment_id='t1', interrupt=True, complete=False, priority=Priority.WARNING))
        nuc.add_signal(new_listener_signal('今天天气不错', segment_id='t1', complete=True))
    assert len(notified) == 2
    first, tail = notified
    assert first.complete is False
    assert tail.complete is True
    assert first.id == tail.id == 't1'  # same-id = segment_id
    assert first.strength > 100  # 首包高强
    assert _message_text(tail) == '今天天气不错'


# ============================================================
# 打断包 — interrupt + 高强 + 占坑不思考
# ============================================================

@pytest.mark.asyncio
async def test_interrupt_packet_flags():
    notified: list = []
    async with ListenerNucleus() as nuc:
        nuc.with_bus(lambda s: None, lambda imp: notified.append(imp))
        nuc.add_signal(new_listener_signal(segment_id='t1', interrupt=True, complete=False, priority=Priority.WARNING))
        imp = notified[0]
        assert imp.complete is False
        assert imp.interrupt is True
        assert imp.thinking_effort == 'none'
        assert imp.priority == Priority.WARNING
        assert imp.strength > 100
        assert imp.messages == []  # 占坑不携带内容


@pytest.mark.asyncio
async def test_interrupt_attended_flattens_and_cools():
    """attended 降回运行参数 + 进入小冷却 (下一打断包不 fire)."""
    notified: list = []
    async with ListenerNucleus(attended_seconds=10.0) as nuc:
        nuc.with_bus(lambda s: None, lambda imp: notified.append(imp))
        nuc.add_signal(new_listener_signal(segment_id='t1', interrupt=True, complete=False, priority=Priority.WARNING))
        first = notified[0]
        result = nuc.attended(first)
        assert result is first
        assert first.priority == Priority.INFO  # run priority
        assert first.strength == 100
        # attended 后小冷却生效: 下一打断包不 fire.
        nuc.add_signal(new_listener_signal(segment_id='t2', interrupt=True, complete=False, priority=Priority.WARNING))
        assert len(notified) == 1


# ============================================================
# 发送包 — notify + 消息体 <listen> tag
# ============================================================

@pytest.mark.asyncio
async def test_deliver_packet_notify():
    notified: list = []
    async with ListenerNucleus() as nuc:
        nuc.with_bus(lambda s: None, lambda imp: notified.append(imp))
        nuc.add_signal(new_listener_signal('你好', segment_id='t1', complete=True))
        imp = notified[0]
        assert imp.complete is True
        assert imp.mode == ChallengeMode.notify.value
        assert imp.priority == Priority.INFO
        assert _message_text(imp) == '你好'


@pytest.mark.asyncio
async def test_deliver_boost_raises_the_challenge_then_falls_back():
    """提权包: 挑战按 max(priority, boost), 抢到 attention 后降回运行档 — 只赢一局."""
    notified: list = []
    async with ListenerNucleus() as nuc:
        nuc.with_bus(lambda s: None, lambda imp: notified.append(imp))
        sig = new_listener_signal('你好', segment_id='t1', complete=True, boost=Priority.WARNING)
        assert ListenerSignal.from_signal(sig).boost == Priority.WARNING
        nuc.add_signal(sig)
        imp = notified[0]
        assert imp.priority == Priority.WARNING
        nuc.attended(imp)
    assert imp.priority == Priority.INFO


@pytest.mark.asyncio
async def test_deliver_without_boost_challenges_at_its_own_priority():
    """不填提权 (默认空) → 挑战档位就是 signal 自己的 priority, 不受影响."""
    notified: list = []
    async with ListenerNucleus() as nuc:
        nuc.with_bus(lambda s: None, lambda imp: notified.append(imp))
        sig = new_listener_signal('你好', segment_id='t1', complete=True, priority=Priority.NOTICE)
        assert ListenerSignal.from_signal(sig).boost is None
        nuc.add_signal(sig)
    assert notified[0].priority == Priority.NOTICE


@pytest.mark.asyncio
async def test_impulse_message_wrapped_in_listen_tag():
    notified: list = []
    async with ListenerNucleus() as nuc:
        nuc.with_bus(lambda s: None, lambda imp: notified.append(imp))
        nuc.add_signal(new_listener_signal('今天天气不错', segment_id='t1', complete=True))
    xml = _message_xml(notified[-1])
    assert xml.startswith('<listen source="asr" created="')
    assert '今天天气不错' in xml
    assert xml.endswith('</listen>')


@pytest.mark.asyncio
async def test_impulse_tag_source_attribute_follows_signal_source():
    """source 差异只走 attribute, tag 名保持共性."""
    notified: list = []
    async with ListenerNucleus() as nuc:
        nuc.with_bus(lambda s: None, lambda imp: notified.append(imp))
        nuc.add_signal(new_listener_signal('小莫', segment_id='t2', source='wake_word', complete=True))
    assert _message_xml(notified[-1]).startswith('<listen source="wake_word" created="')


# ============================================================
# 冷却双档 — 打断包受约束, 发送包不受
# ============================================================

@pytest.mark.asyncio
async def test_deliver_not_gated_by_cooldown():
    """打断包 suppress 后进入冷却, 但发送包仍提交 (内容不丢)."""
    notified: list = []
    async with ListenerNucleus(suppress_seconds=10.0) as nuc:
        nuc.with_bus(lambda s: None, lambda imp: notified.append(imp))
        nuc.add_signal(new_listener_signal(segment_id='t1', interrupt=True, complete=False, priority=Priority.WARNING))
        nuc.suppress(notified[0])  # 进入冷却
        nuc.add_signal(new_listener_signal('你好', segment_id='t1', complete=True))
        assert len(notified) == 2
        assert notified[-1].complete is True


@pytest.mark.asyncio
async def test_interrupt_gated_by_suppress_cooldown():
    """suppress 冷却期内打断包不 fire (丢弃)."""
    notified: list = []
    async with ListenerNucleus(suppress_seconds=10.0) as nuc:
        nuc.with_bus(lambda s: None, lambda imp: notified.append(imp))
        nuc.add_signal(new_listener_signal(segment_id='t1', interrupt=True, complete=False, priority=Priority.WARNING))
        nuc.suppress(notified[0])
        nuc.add_signal(new_listener_signal(segment_id='t2', interrupt=True, complete=False, priority=Priority.WARNING))
        assert len(notified) == 1  # 冷却期内打断包不 fire


@pytest.mark.asyncio
async def test_suppress_retains_impulse_peekable():
    """契约: suppressed 后 impulse 仍保留、可 peek — rank 输掉不等于完结 (不丢).

    落选的若是**携带内容**的发送包, 清缓存会把内容整包抹掉; 更硬的是它占着的那个 attention
    从此永远等不到 complete (尾包补不齐), thinking 通路会停摆.
    """
    async with ListenerNucleus() as nuc:
        nuc.with_bus(lambda s: None, lambda imp: None)
        nuc.add_signal(new_listener_signal('你好', segment_id='t1', complete=True))
        nuc.suppress(Impulse(), None)
        peeked = nuc.peek()
        assert peeked is not None, "suppress 后 impulse 仍应可 peek"
        assert peeked.complete is True
        joined = ' '.join(
            c['text'] for m in peeked.messages for c in m.contents if isinstance(c, dict) and 'text' in c
        )
        assert '你好' in joined, f"发送包内容不得因 rank 落选而丢失 — {joined!r}"


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


def _message_xml(impulse) -> str:
    return '\n'.join(msg.to_xml() for msg in impulse.messages)


def _message_text(impulse) -> str:
    """汇总 impulse 所有消息体的文本."""
    texts = []
    for msg in impulse.messages:
        for c in msg.contents:
            if isinstance(c, dict) and c.get('type') == 'text':
                texts.append(c.get('text', ''))
    return '\n'.join(texts)


@pytest.mark.asyncio
async def test_impulse_identity_is_planned_by_the_listener():
    """身份由听侧规划 (impulse_id): 同一话语的两个包共 id, 分句各自一枚.

    segment_id 是地址, 不是身份 —— 只在没给 impulse_id 时才兼任.
    """
    notified: list = []
    async with ListenerNucleus() as nuc:
        nuc.with_bus(lambda s: None, lambda imp: notified.append(imp))
        nuc.add_signal(new_listener_signal(
            segment_id='seg', impulse_id='seg#utterance', complete=False, priority=Priority.WARNING,
        ))
        nuc.add_signal(new_listener_signal('第一句', impulse_id='seg#c1', complete=True))
        nuc.add_signal(new_listener_signal('第二句', impulse_id='seg#c2', complete=True))
        nuc.add_signal(new_listener_signal('', segment_id='seg', impulse_id='seg#utterance', complete=True, paused=True))
    ids = [imp.id for imp in notified]
    assert ids[0] == ids[-1] == 'seg#utterance', 'onset 与尾包是同一次话语'
    assert len({ids[1], ids[2]}) == 2, '分句各自一枚身份, 谁也不吸收谁'


@pytest.mark.asyncio
async def test_a_clause_packet_reads_as_not_paused_and_the_tail_as_paused():
    """模型看到的是 paused: 中间分句 false, 段尾 true; 空正文的尾包只留标记."""
    messages: list = []
    async with ListenerNucleus() as nuc:
        nuc.with_bus(lambda s: None, lambda imp: messages.append(imp))
        nuc.add_signal(new_listener_signal('还没说完', impulse_id='a', complete=True, paused=False))
        nuc.add_signal(new_listener_signal('', segment_id='seg', impulse_id='b', complete=True, paused=True))
    clause_imp, tail_imp = messages
    assert 'paused="false"' in _message_xml(clause_imp)
    assert 'paused="true"' in _message_xml(tail_imp)
    assert _message_text(tail_imp) == '', '没有新内容时尾包正文为空 —— 标记全在属性上, 不填任何替代文字'
