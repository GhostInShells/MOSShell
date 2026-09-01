import time
from ghoshell_moss.message import Message
from ghoshell_moss.core.blueprint.mindflow import (
    Signal, Impulse, Moment, Echoes, Priority,
)
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


# 2. 测试 数据 的缝合 (核心认知流)
def test_moment_echoes_stitching():
    # 模拟第一轮 Observation
    obs = Moment()
    obs.percepts = {"test": [Message.new().with_content("Input 1")]}

    # 生成 Echoes
    echoes = obs.new_echoes_container()
    echoes.executed_logos = "MoveForward"
    echoes.messages = [Message.new().with_content("Action Done")]

    # 缝合到下一轮 Observation
    obs2 = echoes.new_moment()

    # 验证上下文连贯性
    assert obs2.previous is not None
    assert obs2.previous.executed_logos == "MoveForward"
    assert obs2.previous.messages[0].contents[0]['text'] == "Action Done"

    # 验证 as_request_messages 结构
    msgs = list(obs2.full_moment_messages())
    # 应该包含 <echoes> 标签及内部消息
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
# Moment / Echoes 参数传递链路单测
# 验证 percepts, hint, command_logos 在
# new_moment() → _loop() → next_frame() 全链路不会重复或遗漏
# ============================================================

def test_new_moment_passes_all_params():
    """Echoes.new_moment() 将三个关键参数完整传递到 Moment."""
    echoes = Echoes(executed_logos="test logos", stop_reason="done")
    percept_msg = Message.new().with_content("percept content")
    moment = echoes.new_moment(
        percepts={"test": [percept_msg]},
        hint="handle this",
        command_logos="reflex!",
    )
    assert moment.previous is echoes
    assert len(moment.percepts) == 1
    assert next(iter(moment.percepts_messages())).contents[0]["text"] == "percept content"
    assert moment.hint == "handle this"
    assert moment.command_logos == "reflex!"


def test_new_moment_without_params_creates_empty_moment():
    """不带参数的 new_moment() 创建空的 Moment — observe 轮次应走此路径."""
    echoes = Echoes(executed_logos="prev")
    moment = echoes.new_moment()
    assert moment.percepts == {}
    assert moment.hint == ""
    assert moment.command_logos == ""


def test_new_moment_percepts_none_treated_as_empty():
    """percepts=None 时转为空列表，不抛异常."""
    echoes = Echoes()
    moment = echoes.new_moment(percepts=None)
    assert moment.percepts == {}
    r2 = moment.new_echoes_container()
    assert r2.moment_id == moment.id


def test_moment_inputs_messages_yields_percepts_and_instruction():
    """inputs_messages() 按序产出 percepts → hint."""
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
    """hint 为空时不产出多余消息."""
    moment = Moment(percepts={"test": [Message.new().with_content("p1")]})
    msgs = list(moment.inputs_messages(with_hint=True))
    assert len(msgs) == 1


def test_moment_previous_echoes_messages_includes_echoes_and_stop_reason():
    """previous_echoes_messages() 产出 echoes 包装 + stop_reason."""
    prev = Echoes(
        executed_logos="prev logos",
        messages=[Message.new().with_content("action result")],
        stop_reason="fade out",
    )
    moment = Moment(previous=prev)
    msgs = list(moment.previous_echoes_messages())
    # <echoes>, action result, </echoes>, stop_reason
    assert len(msgs) >= 2
    content_texts = []
    for m in msgs:
        for c in m.contents:
            if "text" in c:
                content_texts.append(c["text"])
    assert "action result" in content_texts


def test_moment_previous_echoes_messages_empty_when_no_previous():
    """没有 previous Echoes 时不产出消息."""
    moment = Moment()
    msgs = list(moment.previous_echoes_messages())
    assert len(msgs) == 0


def test_moment_is_empty_and_is_empty_request():
    """is_empty / is_empty_request 判断."""
    empty = Moment()
    assert empty.is_empty()
    assert empty.is_percepts_empty()

    with_percept = Moment(percepts={"test": [Message.new().with_content("x")]})
    assert not with_percept.is_empty()
    assert not with_percept.is_percepts_empty()

    with_prev = Moment(previous=Echoes(need_observe=True))
    assert with_prev.is_empty()  # 没有消息, 所以是 empty
    assert with_prev.is_percepts_empty()  # 有 previous 但没有新 percepts


def test_as_moment_message_wraps_content_into_container():
    """as_moment_message 返回自解释容器: 携带 moment id 锚点, 包裹全部内容子段."""
    prev = Echoes(
        messages=[Message.new().with_content("echoes 1")],
    )
    moment = Moment(
        previous=prev,
        percepts={"test": [Message.new().with_content("percept 1")]}, hint="react!",
    )
    moment.dynamic_context["moss_dynamic"] = [Message.new().with_content("dynamic ctx")]
    msg = moment.as_moment_message()
    assert msg is not None
    assert msg.meta.tag == "moment"
    assert msg.meta.attributes["moment_id"] == moment.id
    text = msg.to_content_string()
    assert "<echoes>" in text and "echoes 1" in text
    assert "<dynamic_context>" in text and "dynamic ctx" in text
    assert "<percepts>" in text and "percept 1" in text
    assert "<hint>" in text and "react!" in text


def test_moment_as_request_messages_without_perspectives():
    """with_dynamic_context=False 时动态上下文不进单条消息, percepts 保留."""
    moment = Moment(percepts={"test": [Message.new().with_content("p1")]})
    moment.dynamic_context["ctx"] = [Message.new().with_content("ctx1")]
    msgs = moment.full_moment_messages(with_dynamic_context=False, with_hint=False)
    assert len(msgs) == 1
    text = msgs[0].to_content_string()
    # 动态上下文被排除.
    assert "ctx1" not in text
    assert "<dynamic_context>" not in text
    # percepts 保留.
    assert "<percepts>" in text and "p1" in text


def test_moment_command_logos_preserved_in_new_moment():
    """command_logos 从 Echoes.new_moment() 正确传递，不被后续操作丢失."""
    echoes = Echoes()
    moment = echoes.new_moment(command_logos="hello!")
    assert moment.command_logos == "hello!"
    # 验证 new_echoes 后再 new_moment, command_logos 不自动继承
    echoes2 = moment.new_echoes_container()
    moment2 = echoes2.new_moment()
    assert moment2.command_logos == ""  # command_logos 不应跨轮次自动继承


def test_as_moment_message_excludes_percepts_and_hint_when_requested():
    """as_moment_message(with_percepts=False, with_hint=False) 只折叠 context 半 (echoes/dynamic/executing)."""
    prev = Echoes(messages=[Message.new().with_content("echo")])
    moment = Moment(
        previous=prev,
        percepts={"test": [Message.new().with_content("percept")]},
        hint="hint text",
        command_logos="cmd!",
    )
    moment.dynamic_context["ctx"] = [Message.new().with_content("dyn")]
    msg = moment.as_moment_message(with_percepts=False, with_hint=False)
    assert msg is not None
    assert msg.meta.tag == "moment"
    text = msg.to_content_string()
    assert "<echoes>" in text and "echo" in text
    assert "<dynamic_context>" in text and "dyn" in text
    assert "<executing>" in text and "cmd!" in text
    assert "percept" not in text
    assert "hint text" not in text


def test_as_moment_message_keeps_percepts_and_hint_by_default():
    """默认 as_moment_message 仍含 percepts + hint (向后兼容)."""
    moment = Moment(
        percepts={"test": [Message.new().with_content("percept")]},
        hint="hint text",
    )
    msg = moment.as_moment_message()
    assert msg is not None
    text = msg.to_content_string()
    assert "percept" in text
    assert "hint text" in text


def test_as_moment_message_returns_none_when_empty_and_always_return_false():
    """空 moment + always_return=False → None."""
    assert Moment().as_moment_message(always_return=False) is None
