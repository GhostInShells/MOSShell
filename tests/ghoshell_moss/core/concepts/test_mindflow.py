import time
from ghoshell_moss.message import Message
from ghoshell_moss.core.blueprint.mindflow import (
    Signal, Impulse, Moment, Reaction, Priority,
)
from ghoshell_moss.core.mindflow.base_attention import AttentionContext
from ghoshell_moss.core.helpers import ThreadSafeEvent


# 1. 测试 Signal 到 Impulse 的转换逻辑
def test_signal_to_impulse_conversion():
    # 创建一个原始信号
    msg = Message.new().with_content("Hello MOSS")
    signal = Signal.new(
        "test_signal",
        msg,
        priority=Priority.WARNING,
        description="test",
        stale_timeout=2.0
    )

    # 执行转换
    impulse = Impulse.from_signal(signal, source="test_nucleus")

    # 验证数据对齐
    assert impulse.source == "test_nucleus"
    assert impulse.priority == Priority.WARNING
    assert impulse.messages[0].contents[0]['text'] == "Hello MOSS"
    assert impulse.stale_timeout > 0
    # 验证 trace_id 继承
    assert impulse.trace_id == signal.id


# 2. 测试 Observation 与 Outcome 的缝合 (核心认知流)
def test_moment_outcome_stitching():
    # 模拟第一轮 Observation
    obs = Moment()
    obs.percepts = {"test": [Message.new().with_content("Input 1")]}

    # 生成 Outcome
    outcome = obs.new_reaction()
    outcome.executed_logos = "MoveForward"
    outcome.messages = [Message.new().with_content("Action Done")]

    # 缝合到下一轮 Observation
    obs2 = outcome.new_moment()

    # 验证上下文连贯性
    assert obs2.previous is not None
    assert obs2.previous.executed_logos == "MoveForward"
    assert obs2.previous.messages[0].contents[0]['text'] == "Action Done"

    # 验证 as_request_messages 结构
    msgs = list(obs2.as_request_messages())
    # 应该包含 <outcomes> 标签及内部消息
    content_tags = [m.meta.tag for m in msgs if m.meta.tag]
    assert 'stop_reason' not in content_tags  # 此时 stop_reason 应为空


# 3. 测试 Impulse 的保鲜逻辑 (Stale Timeout)
def test_impulse_stale_logic():
    signal = Signal.new("test", stale_timeout=0.1)
    impulse = Impulse.from_signal(signal, source="test")

    assert impulse.is_stale() is False
    time.sleep(0.2)
    assert impulse.is_stale() is True


# 4. 测试优先级抢占判定逻辑 (on_challenge 核心模拟)
def test_attention_preemption_logic():
    # 模拟一个正在运行的 Attention 的 Impulse
    current_impulse = Impulse(source="nucleus_a", priority=Priority.INFO, strength=100)

    # 模拟一个高优先级的挑战
    challenge = Impulse(source="nucleus_b", priority=Priority.CRITICAL, strength=100)

    # 模拟 Attention 内部的仲裁 (simplified)
    # 规则：CRITICAL > INFO -> 必须被抢占
    assert challenge.priority > current_impulse.priority

    # 模拟同优先级，强弱对抗
    weak_challenge = Impulse(source="nucleus_b", priority=Priority.INFO, strength=50)
    assert weak_challenge.strength < current_impulse.strength


def test_signal_impulse_direct_set():
    signal = Signal.new("test", complete=False)
    impulse = Impulse.from_signal(signal, source="test")
    assert not impulse.complete


# ============================================================
# Moment / Reaction 参数传递链路单测
# 验证 percepts, reaction_instruction, command_logos 在
# new_moment() → _loop() → next_frame() 全链路不会重复或遗漏
# ============================================================

def test_new_moment_passes_all_params():
    """Reaction.new_moment() 将三个关键参数完整传递到 Moment."""
    reaction = Reaction(executed_logos="test logos", stop_reason="done")
    percept_msg = Message.new().with_content("percept content")
    moment = reaction.new_moment(
        percepts={"test": [percept_msg]},
        hint="handle this",
        command_logos="reflex!",
    )
    assert moment.previous is reaction
    assert len(moment.percepts) == 1
    assert next(iter(moment.percepts_messages())).contents[0]["text"] == "percept content"
    assert moment.hint == "handle this"
    assert moment.command_logos == "reflex!"


def test_new_moment_without_params_creates_empty_moment():
    """不带参数的 new_moment() 创建空的 Moment — observe 轮次应走此路径."""
    reaction = Reaction(executed_logos="prev")
    moment = reaction.new_moment()
    assert moment.percepts == {}
    assert moment.hint == ""
    assert moment.command_logos == ""


def test_new_moment_percepts_none_treated_as_empty():
    """percepts=None 时转为空列表，不抛异常."""
    reaction = Reaction()
    moment = reaction.new_moment(percepts=None)
    assert moment.percepts == {}


def test_moment_inputs_messages_yields_percepts_and_instruction():
    """inputs_messages() 按序产出 percepts → reaction_instruction."""
    percept = Message.new().with_content("p1")
    moment = Moment(
        percepts={"test": [percept]},
        hint="do it",
    )
    msgs = list(moment.inputs_messages(with_hint=True))
    assert len(msgs) == 2
    assert msgs[0].contents[0]["text"] == "p1"
    assert msgs[1].meta.tag == "hint"
    assert msgs[1].contents[0]["text"] == "do it"


def test_moment_inputs_messages_without_instruction():
    """with_hint=False 时不产出 instruction."""
    moment = Moment(
        percepts={"test": [Message.new().with_content("p1")]}, hint="skip me",
    )
    msgs = list(moment.inputs_messages(with_hint=False))
    assert len(msgs) == 1


def test_moment_inputs_messages_skips_empty_instruction():
    """reaction_instruction 为空时不产出多余消息."""
    moment = Moment(percepts={"test": [Message.new().with_content("p1")]})
    msgs = list(moment.inputs_messages(with_hint=True))
    assert len(msgs) == 1


def test_moment_previous_reaction_messages_includes_outcomes_and_stop_reason():
    """previous_reaction_messages() 产出 outcomes 包装 + stop_reason."""
    prev = Reaction(
        executed_logos="prev logos",
        messages=[Message.new().with_content("action result")],
        stop_reason="fade out",
    )
    moment = Moment(previous=prev)
    msgs = list(moment.previous_reaction_messages())
    # <outcomes>, action result, </outcomes>, stop_reason
    assert len(msgs) >= 2
    content_texts = []
    for m in msgs:
        for c in m.contents:
            if "text" in c:
                content_texts.append(c["text"])
    assert "action result" in content_texts


def test_moment_previous_reaction_messages_empty_when_no_previous():
    """没有 previous Reaction 时不产出消息."""
    moment = Moment()
    msgs = list(moment.previous_reaction_messages())
    assert len(msgs) == 0


def test_moment_is_empty_and_is_empty_request():
    """is_empty / is_empty_request 判断."""
    empty = Moment()
    assert empty.is_empty()
    assert empty.is_empty_request()

    with_percept = Moment(percepts={"test": [Message.new().with_content("x")]})
    assert not with_percept.is_empty()
    assert not with_percept.is_empty_request()

    with_prev = Moment(previous=Reaction())
    assert not with_prev.is_empty()
    assert with_prev.is_empty_request()  # 有 previous 但没有新 percepts


def test_moment_as_request_messages_full_structure():
    """as_request_messages() 按序组装: previous → perspectives → inputs."""
    prev = Reaction(
        messages=[Message.new().with_content("outcome 1")],
    )
    moment = Moment(
        previous=prev,
        percepts={"test": [Message.new().with_content("percept 1")]}, hint="react!",
    )
    moment.perspectives["moss_dynamic"] = [Message.new().with_content("dynamic ctx")]
    msgs = list(moment.as_request_messages(with_perspectives=True, with_hint=True))
    # 应该有: outcomes 包装 + perspective + percept + instruction
    texts = []
    for m in msgs:
        for c in m.contents:
            if "text" in c:
                texts.append(c["text"])
    assert "outcome 1" in texts
    assert "dynamic ctx" in texts
    assert "percept 1" in texts
    assert "react!" in texts


def test_moment_as_request_messages_without_perspectives():
    """with_perspectives=False 时完全不产出 perspectives."""
    moment = Moment(percepts={"test": [Message.new().with_content("p1")]})
    moment.perspectives["ctx"] = [Message.new().with_content("ctx1")]
    msgs = list(moment.as_request_messages(with_perspectives=False, with_hint=False))
    texts = []
    for m in msgs:
        for c in m.contents:
            if "text" in c:
                texts.append(c["text"])
    # perspectives 不应出现
    assert "ctx1" not in texts
    # 只有 percepts
    assert "p1" in texts


def test_moment_perspective_messages_compact_mode():
    """compact=True 且有 compact_perspectives 时优先使用压缩结果."""
    moment = Moment()
    moment.perspectives["ctx"] = [Message.new().with_content("long context")]
    moment.compacted_perspectives = [Message.new().with_content("compressed version")]
    msgs = list(moment.perspective_messages(compact_first=True))
    assert len(msgs) == 1
    assert msgs[0].contents[0]["text"] == "compressed version"


def test_moment_command_logos_preserved_in_new_moment():
    """command_logos 从 Reaction.new_moment() 正确传递，不被后续操作丢失."""
    reaction = Reaction()
    moment = reaction.new_moment(command_logos="hello!")
    assert moment.command_logos == "hello!"
    # 验证 new_reaction 后再 new_moment, command_logos 不自动继承
    reaction2 = moment.new_reaction()
    moment2 = reaction2.new_moment()
    assert moment2.command_logos == ""  # command_logos 不应跨轮次自动继承


# ============================================================
# AttentionContext 单测
# 验证 next_frame() (observe 链路) 不会携带上一轮的 percepts
# ============================================================

def _make_attention_ctx(
        attention_id: str = "test_attn",
        percepts: dict[str, list[Message]] | None = None,
        hint: str = "",
        command_logos: str = "",
) -> AttentionContext:
    """构造 AttentionContext 的测试夹具."""
    reaction = Reaction()
    moment = reaction.new_moment(
        percepts=percepts,
        hint=hint,
        command_logos=command_logos,
    )
    return AttentionContext(
        attention_id=attention_id,
        moment=moment,
        aborted_event=ThreadSafeEvent(),
        flags={},
    )


def test_attention_ctx_moment_has_percepts_on_first_creation():
    """首次创建时 Moment 携带 percepts."""
    percept = Message.new().with_content("input signal")
    ctx = _make_attention_ctx(percepts={"test": [percept]}, hint="go")
    assert len(ctx.moment.percepts) == 1
    assert ctx.moment.percepts_texts() == ["input signal"]
    assert ctx.moment.hint == "go"


def test_attention_ctx_new_moment_creates_empty_percepts():
    """ctx.new_moment() 调用 Reaction.new_moment() 无参数 — percepts 应为空."""
    ctx = _make_attention_ctx(
        percepts={"test": [Message.new().with_content("original")]},
        hint="original instruction",
    )
    new_moment = ctx.new_moment()
    # new_moment() → stop_at_outcome().new_moment() 不传参数
    assert new_moment.percepts == {}
    assert new_moment.hint == ""


def test_attention_ctx_next_frame_does_not_carry_percepts():
    """next_frame() 创建新 ctx，其 Moment 的 percepts 应为空 — observe 不应带入新感知."""
    ctx = _make_attention_ctx(
        percepts={"test": [Message.new().with_content("first round")]}, hint="first instruction",
    )
    # 模拟第一轮执行后触发 observe
    ctx.observe("")  # 标记 observe
    next_ctx = ctx.next_frame()
    # 新 ctx 应继承相同 attention_id
    assert next_ctx.attention_id == ctx.attention_id
    # percepts 不应被带入 observe 轮
    assert next_ctx.moment.percepts == {}
    assert next_ctx.moment.hint == ""
    # previous 应指向前一轮的 Reaction
    assert next_ctx.moment.previous is not None
    assert next_ctx.moment.previous.moment_id == ctx.moment.id


def test_attention_ctx_next_frame_preserves_moment_chain():
    """多次 next_frame() 构成连续的 Moment 链."""
    ctx = _make_attention_ctx(percepts={"test": [Message.new().with_content("init")]})
    ctx.observe("")
    ctx2 = ctx.next_frame()
    ctx2.observe("")
    ctx3 = ctx2.next_frame()
    # 链式引用
    assert ctx3.moment.previous is not None
    assert ctx3.moment.previous.moment_id == ctx2.moment.id
    assert ctx2.moment.previous.moment_id == ctx.moment.id
    # 所有 observe 轮的 percepts 均为空
    assert ctx2.moment.percepts == {}
    assert ctx3.moment.percepts == {}


def test_attention_ctx_stop_at_outcome_captures_logos_and_outcomes():
    """stop_at_outcome() 正确捕获 logos + outcomes + stop_reason."""
    ctx = _make_attention_ctx()
    ctx.buffer_executed_logos("model said hello")
    ctx.outcome(Message.new().with_content("action done"), observe=False)
    ctx.abort("fade out")
    reaction = ctx.stop_at_outcome()
    assert reaction.executed_logos == "model said hello"
    assert len(reaction.messages) == 1
    assert reaction.stop_reason == "fade out"


def test_attention_ctx_outcome_with_observe_triggers_next_frame():
    """outcome(observe=True) → get_observe_messages() 非 None → 可以 next_frame."""
    ctx = _make_attention_ctx()
    assert ctx.get_observe_messages() is None
    ctx.outcome(Message.new().with_content("result"), observe=True)
    assert ctx.get_observe_messages() is not None


def test_attention_ctx_command_logos_preserved_in_moment():
    """command_logos 在首次 Moment 中保留."""
    ctx = _make_attention_ctx(command_logos="conditional reflex")
    assert ctx.moment.command_logos == "conditional reflex"
