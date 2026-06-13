"""
Moment + Impulse 连动单测.

覆盖 mindflow 生命周期中 impulse → moment 的数据落地链路:
- Impulse.update_moment 单 impulse 的合并字段语义
- 多 impulse 合并到同一 moment 的累加 / 覆盖语义 (按设计)
- AbsAttention init 不应让首个 impulse 被双重应用
- _prepare_moment 在首帧 / 后续帧的行为
- _buffer_impulse dedup 与 partial→complete 接力
- perspectives_func / percepts_func 集成与异常隔离
"""
from ghoshell_moss.core.blueprint.mindflow import Impulse, Reaction, Priority
from ghoshell_moss.core.blueprint.memento import Moment
from ghoshell_moss.core.mindflow.base_attention import BaseAttention
from ghoshell_moss.message import Message
import pytest


def _text(msg: Message) -> str:
    for c in msg.contents:
        if "text" in c:
            return c["text"]
    return ""


def _texts(msgs) -> list[str]:
    return [_text(m) for m in msgs]


def _make_attention(impulse: Impulse, previous: Reaction | None = None) -> BaseAttention:
    return BaseAttention(previous=previous or Reaction(), impulse=impulse)


# ============================================================
# Impulse.update_moment — 单 impulse 合并语义
# ============================================================

def test_update_moment_extends_percepts():
    moment = Moment()
    imp = Impulse(messages=[Message.new().with_content('a'), Message.new().with_content('b')])
    imp.update_moment(moment)
    assert _texts(moment.percepts) == ['a', 'b']


def test_update_moment_sets_hint_and_command_logos():
    moment = Moment()
    imp = Impulse(hint='think hard', logos='cmd!')
    imp.update_moment(moment)
    assert moment.hint == 'think hard'
    assert moment.command_logos == 'cmd!'


def test_update_moment_perspective_keyed_by_source():
    moment = Moment()
    imp = Impulse(
        source='vision',
        perspective=[Message.new().with_content('frame snapshot')],
    )
    imp.update_moment(moment)
    assert 'vision' in moment.perspectives
    assert _texts(moment.perspectives['vision']) == ['frame snapshot']


def test_update_moment_empty_logos_does_not_overwrite():
    """空 logos 不应清空 / 不应追加 command_logos."""
    moment = Moment(command_logos='preset')
    imp = Impulse(logos='')
    imp.update_moment(moment)
    assert moment.command_logos == 'preset'


def test_update_moment_empty_perspective_skipped():
    """空 perspective 不写入 moment.perspectives."""
    moment = Moment()
    imp = Impulse(source='vision', perspective=[])
    imp.update_moment(moment)
    assert 'vision' not in moment.perspectives


# ============================================================
# Impulse.update_moment — 多 impulse 累加 / 覆盖语义
# 设计意图: 同一帧可能合并多个 buffered impulses, 各字段的合并方式不同.
# ============================================================

def test_update_moment_twice_accumulates():
    """同一 impulse 两次 update_moment: percepts.extend 与 command_logos += 都是累加."""
    moment = Moment()
    imp = Impulse(
        messages=[Message.new().with_content('msg')],
        logos='cmd',
    )
    imp.update_moment(moment)
    imp.update_moment(moment)
    assert _texts(moment.percepts) == ['msg', 'msg']
    assert moment.command_logos == 'cmdcmd'


def test_update_moment_different_sources_perspectives_coexist():
    moment = Moment()
    Impulse(source='vision', perspective=[Message.new().with_content('v')]).update_moment(moment)
    Impulse(source='audio', perspective=[Message.new().with_content('a')]).update_moment(moment)
    assert _texts(moment.perspectives['vision']) == ['v']
    assert _texts(moment.perspectives['audio']) == ['a']


def test_update_moment_same_source_perspective_overwrites():
    moment = Moment()
    Impulse(source='vision', perspective=[Message.new().with_content('old')]).update_moment(moment)
    Impulse(source='vision', perspective=[Message.new().with_content('new')]).update_moment(moment)
    assert _texts(moment.perspectives['vision']) == ['new']


def test_update_moment_hint_overwrites_unconditionally():
    """hint 按设计跟随最新 impulse, 包括空字符串清空前者.
    理由: hint 只在 buffer/notify 系统级合并场景 (目前仅同 id 接力) 才有"多 impulse 合一帧"语义,
    其他场景下最新 impulse 的 hint 就是本轮的 hint."""
    moment = Moment()
    Impulse(hint='careful').update_moment(moment)
    Impulse(hint='').update_moment(moment)
    assert moment.hint == ''


# ============================================================
# AbsAttention init + _prepare_moment — 首个 impulse 不应被双重应用
# ============================================================

def test_attention_init_buffers_initial_impulse_exactly_once():
    """AbsAttention init 中, 首个 complete impulse 只入 buffer 一次."""
    imp = Impulse(
        source='nucleus_a',
        priority=Priority.NOTICE,
        messages=[Message.new().with_content('hello')],
    )
    att = _make_attention(imp)
    assert att.buffered_impulses == [imp]


def test_prepare_moment_first_frame_no_duplication():
    """首帧 _prepare_moment 后, moment 字段应只反映 impulse 一次, 不被双重应用."""
    imp = Impulse(
        source='nucleus_a',
        priority=Priority.NOTICE,
        messages=[
            Message.new().with_content('hello'),
            Message.new().with_content('world'),
        ],
        logos='cmd!',
        hint='think',
    )
    att = _make_attention(imp)
    moment = att._ctx.moment
    att._prepare_moment(moment)
    assert _texts(moment.percepts) == ['hello', 'world']
    assert moment.command_logos == 'cmd!'
    assert moment.hint == 'think'


def test_prepare_moment_second_frame_no_residual_impulse():
    """首帧 drain 后, 下一帧 _prepare_moment 不应残留首个 impulse 的内容."""
    imp = Impulse(source='nucleus_a', messages=[Message.new().with_content('hello')])
    att = _make_attention(imp)
    att._prepare_moment(att._ctx.moment)  # 首帧
    moment2 = Moment()
    att._prepare_moment(moment2)
    assert moment2.percepts == []
    assert moment2.command_logos == ''


def test_attention_incomplete_initial_impulse_not_in_buffer():
    """语义: 只有 complete 才进 buffer. incomplete 首包应等待 complete 接力."""
    imp = Impulse(
        source='nucleus_a',
        complete=False,
        messages=[Message.new().with_content('partial')],
    )
    att = _make_attention(imp)
    assert att.buffered_impulses == []


# ============================================================
# _buffer_impulse — dedup 与 partial → complete 接力
# ============================================================

def test_buffer_impulse_dedup_by_id():
    """相同 id 的 complete impulse 二次 _buffer_impulse 不重复入 buffer."""
    imp = Impulse(
        source='nucleus_a',
        priority=Priority.NOTICE,
        messages=[Message.new().with_content('a')],
    )
    att = _make_attention(imp)
    att._buffer_impulse(imp)  # 再来一次
    assert len(att.buffered_impulses) == 1


def test_buffer_impulse_partial_then_complete_same_id_enters_buffer():
    """partial 首包不在 buffer, complete 接力时进 buffer."""
    partial = Impulse(
        source='nucleus_a',
        complete=False,
        messages=[Message.new().with_content('partial')],
    )
    att = _make_attention(partial)
    assert att.buffered_impulses == []
    complete = Impulse(
        id=partial.id,
        source='nucleus_a',
        complete=True,
        messages=[Message.new().with_content('final')],
    )
    att._buffer_impulse(complete)
    buf = att.buffered_impulses
    assert len(buf) == 1
    assert buf[0].complete


def test_buffer_impulse_different_ids_both_queued():
    """不同 id 的 complete impulse 都进 buffer, 合并到下一帧."""
    imp1 = Impulse(source='nucleus_a', messages=[Message.new().with_content('m1')])
    att = _make_attention(imp1)
    imp2 = Impulse(source='nucleus_b', messages=[Message.new().with_content('m2')])
    att._buffer_impulse(imp2)
    assert {d.id for d in att.buffered_impulses} == {imp1.id, imp2.id}


def test_prepare_moment_merges_multiple_buffered_impulses():
    """多 impulse 合并到同一帧: percepts 按入队顺序拼接, 不同 source perspective 共存."""
    imp1 = Impulse(
        source='vision',
        messages=[Message.new().with_content('frame1')],
        perspective=[Message.new().with_content('snapshot1')],
    )
    att = _make_attention(imp1)
    imp2 = Impulse(
        source='audio',
        messages=[Message.new().with_content('audio1')],
        perspective=[Message.new().with_content('clip1')],
        logos='greet',
    )
    att._buffer_impulse(imp2)
    moment = Moment()
    att._prepare_moment(moment)
    assert _texts(moment.percepts) == ['frame1', 'audio1']
    assert _texts(moment.perspectives['vision']) == ['snapshot1']
    assert _texts(moment.perspectives['audio']) == ['clip1']
    assert moment.command_logos == 'greet'


# ============================================================
# _prepare_moment drain — buffer 在首帧后清空
# ============================================================

def test_prepare_moment_drains_buffer():
    """_prepare_moment 调用后, 已应用的 buffered_impulses 应被消费."""
    imp = Impulse(source='nucleus_a', messages=[Message.new().with_content('m')])
    att = _make_attention(imp)
    assert len(att.buffered_impulses) == 1
    att._prepare_moment(Moment())
    assert att.buffered_impulses == []


# ============================================================
# _prepare_moment — perspectives_func / percepts_func 集成与异常隔离
# ============================================================

def test_prepare_moment_calls_perspectives_func():
    imp = Impulse(source='nucleus_a', messages=[Message.new().with_content('m')])
    att = _make_attention(imp)
    att.with_perspective_func('snapshot', lambda: [Message.new().with_content('live')])
    moment = Moment()
    att._prepare_moment(moment)
    assert _texts(moment.perspectives['snapshot']) == ['live']


def test_prepare_moment_calls_percepts_func_before_impulse_drain():
    """percepts_func 产出在 impulse drain 之前, 顺序: func messages → impulse messages."""
    imp = Impulse(source='nucleus_a', messages=[Message.new().with_content('from_impulse')])
    att = _make_attention(imp)
    att.with_percepts_func('extra', lambda: [Message.new().with_content('from_func')])
    moment = Moment()
    att._prepare_moment(moment)
    # 先 func 再 impulse.
    assert _texts(moment.percepts) == ['from_func', 'from_impulse']


def test_prepare_moment_perspectives_func_exception_isolated():
    """单个 perspectives_func 抛错不影响其他 func 或 impulse drain."""
    imp = Impulse(source='nucleus_a', messages=[Message.new().with_content('m')])
    att = _make_attention(imp)

    def _boom() -> list[Message]:
        raise RuntimeError('boom')

    att.with_perspective_func('broken', _boom)
    att.with_perspective_func('ok', lambda: [Message.new().with_content('ok')])
    moment = Moment()
    att._prepare_moment(moment)
    # broken 未写入, ok 写入, impulse 仍正常 drain.
    assert 'broken' not in moment.perspectives
    assert _texts(moment.perspectives['ok']) == ['ok']
    assert _texts(moment.percepts) == ['m']


def test_prepare_moment_percepts_func_exception_isolated():
    """单个 percepts_func 抛错不影响其他 func 或 impulse drain."""
    imp = Impulse(source='nucleus_a', messages=[Message.new().with_content('m')])
    att = _make_attention(imp)

    def _boom() -> list[Message]:
        raise RuntimeError('boom')

    att.with_percepts_func('broken', _boom)
    att.with_percepts_func('ok', lambda: [Message.new().with_content('extra')])
    moment = Moment()
    att._prepare_moment(moment)
    # broken 无产出, ok 产出, impulse 正常 drain.
    assert 'extra' in _texts(moment.percepts)
    assert 'm' in _texts(moment.percepts)


# ============================================================
# Impulse → Attention 协议字段卸载
# Impulse 上的若干字段属于 mindflow 调度协议, init / _buffer_impulse 时应正确落到 Attention 内部状态.
# 这一节钉住 "字段去向" — 协议演进时, 若改动这些钩子会先红.
# ============================================================

def test_attention_init_unloads_thinking_effort():
    imp = Impulse(thinking_effort='high', messages=[Message.new().with_content('m')])
    att = _make_attention(imp)
    assert att.thinking_effort == 'high'


def test_attention_init_unloads_strength_protection_decay():
    imp = Impulse(
        strength=180,
        protection_time=2.0,
        strength_decay_seconds=15.0,
        messages=[Message.new().with_content('m')],
    )
    att = _make_attention(imp)
    assert att.strength_start_value == 180
    assert att.strength_decay_time == 15.0
    # _protected_until = strength_refreshed_at + protection_time, 留一点时钟容差.
    assert abs(att.protected_until - att.strength_refreshed_at - 2.0) < 0.05


def test_attention_zero_decay_seconds_clamped_to_one():
    """strength_decay_seconds=0 时, Attention 内部 clamp 到 1 防除零."""
    imp = Impulse(strength_decay_seconds=0, messages=[Message.new().with_content('m')])
    att = _make_attention(imp)
    assert att.strength_decay_time == 1


def test_buffer_impulse_refreshes_strength_and_protection_keeps_decay():
    """同一 attention 接力新 impulse:
    - strength / protection_time 被新 impulse 刷新 (反映最新调度状态)
    - strength_decay_seconds 钉在 init impulse, 不被刷新 (attention 生命周期时钟稳定)"""
    init_imp = Impulse(
        strength=100,
        protection_time=1.0,
        strength_decay_seconds=20.0,
        messages=[Message.new().with_content('init')],
    )
    att = _make_attention(init_imp)
    initial_decay = att.strength_decay_time
    new_imp = Impulse(
        strength=200,
        protection_time=3.0,
        strength_decay_seconds=99.0,  # 应被忽略.
        messages=[Message.new().with_content('next')],
    )
    att._buffer_impulse(new_imp)
    assert att.strength_start_value == 200
    assert abs(att.protected_until - att.strength_refreshed_at - 3.0) < 0.05
    assert att.strength_decay_time == initial_decay


def test_buffer_impulse_complete_refreshes_thinking_effort():
    """complete impulse 进 buffer 时, _thinking_effort 更新为新 impulse 的值."""
    att = _make_attention(Impulse(
        thinking_effort='low',
        messages=[Message.new().with_content('init')],
    ))
    assert att.thinking_effort == 'low'
    att._buffer_impulse(Impulse(
        thinking_effort='high',
        messages=[Message.new().with_content('next')],
    ))
    assert att.thinking_effort == 'high'


def test_buffer_impulse_incomplete_does_not_change_thinking_effort():
    """incomplete impulse 不更新 _thinking_effort — partial 不能接管思考强度决策."""
    att = _make_attention(Impulse(
        thinking_effort='high',
        messages=[Message.new().with_content('init')],
    ))
    att._buffer_impulse(Impulse(
        complete=False,
        thinking_effort='none',
        messages=[Message.new().with_content('partial')],
    ))
    assert att.thinking_effort == 'high'


# ============================================================
# End-to-end: thinking_effort 卸载到 yielded Articulator
# ============================================================

@pytest.mark.asyncio
async def test_articulator_receives_thinking_effort_from_impulse():
    """完整生命周期: Impulse.thinking_effort 应卸载到 attention.loop() yield 的 Articulator.
    按设计, grab 了 articulator/action 就有义务展开它们的生命周期, 否则 attention 会等 stop events."""
    imp = Impulse(
        source='nucleus_a',
        thinking_effort='medium',
        messages=[Message.new().with_content('m')],
    )
    att = _make_attention(imp)
    async with att:
        articulator, action = await anext(att.loop())
        async with articulator, action:
            assert articulator.thinking_effort() == 'medium'
            assert att.thinking_effort == 'medium'


@pytest.mark.asyncio
async def test_articulator_thinking_effort_reflects_latest_complete_impulse():
    """接力 complete impulse 后, 新 yield 的 Articulator 反映最新 effort."""
    att = _make_attention(Impulse(
        source='nucleus_a',
        thinking_effort='low',
        messages=[Message.new().with_content('init')],
    ))
    att._buffer_impulse(Impulse(
        source='nucleus_a',
        thinking_effort='high',
        messages=[Message.new().with_content('next')],
    ))
    async with att:
        articulator, action = await anext(att.loop())
        async with articulator, action:
            assert articulator.thinking_effort() == 'high'
