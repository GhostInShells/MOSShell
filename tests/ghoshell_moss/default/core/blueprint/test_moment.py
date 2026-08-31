"""
moment.py 纯数据结构单测.

Echoes / Moment 是 mindflow→ghost 协议转换面的核心数据载体, 无 IO、无并发,
所有行为都应能通过构造 + 方法调用直接暴露. 本套件覆盖:

- Echoes: 默认值、new_moment 参数传递
- Moment: 访问器、is_empty 判定、dynamic_context 组合、各 messages 视图
- 序列化: to_dict / to_json / for_saving
- 历史缝合: as_history_messages / to_history_turns (回合切分核心)
"""

from ghoshell_moss.core.blueprint.moment import Moment, Echoes
from ghoshell_moss.message import Message


def _text(msg: Message) -> str:
    """取一条消息的首个文本内容, 简化断言."""
    for c in msg.as_contents():
        if "text" in c:
            return c["text"]
    return ""


def _texts(msgs) -> list[str]:
    return [_text(m) for m in msgs]


# ============================================================
# Echoes
# ============================================================

def test_echoes_defaults():
    r = Echoes()
    assert r.moment_id  # auto unique_id
    assert r.executed_logos == ""
    assert r.messages == []
    assert r.stop_reason == ""


def test_echoes_new_moment_passes_all_params():
    r = Echoes(executed_logos="prev logos", stop_reason="done")
    moment = r.new_moment(
        percepts={"test": [Message.new().with_content("p")]},
        hint="hint text",
        command_logos="reflex!",
    )
    assert moment.previous is r
    assert _text(next(iter(moment.percepts_messages()))) == "p"
    assert moment.hint == "hint text"
    assert moment.command_logos == "reflex!"


def test_echoes_new_moment_defaults_empty():
    r = Echoes(executed_logos="prev")
    moment = r.new_moment()
    assert moment.percepts == {}
    assert moment.hint == ""
    assert moment.command_logos == ""
    assert moment.previous is r


def test_echoes_new_moment_percepts_none_is_empty_dict():
    moment = Echoes().new_moment(percepts=None)
    assert moment.percepts == {}


# ============================================================
# Echoes — 边界行为
# ============================================================

def test_echoes_new_moment_passes_dynamic_context():
    r = Echoes()
    moment = r.new_moment(dynamic_context={"ctx": [Message.new().with_content("dyn")]})
    assert _texts(moment.dynamic_context["ctx"]) == ["dyn"]


def test_echoes_add_echoes_str_becomes_message():
    r = Echoes()
    appended = r.add_echoes(["hi"])
    assert len(appended) == 1
    assert _text(appended[0]) == "hi"
    assert _text(r.messages[0]) == "hi"


def test_echoes_add_echoes_message_appended_as_is():
    r = Echoes()
    msg = Message.new().with_content("m")
    assert r.add_echoes([msg]) == [msg]
    assert r.messages == [msg]


def test_echoes_add_echoes_mixed_str_and_message():
    r = Echoes()
    appended = r.add_echoes(["x", Message.new().with_content("y")])
    assert _texts(appended) == ["x", "y"]
    assert _texts(r.messages) == ["x", "y"]


def test_echoes_add_echoes_empty_list_still_sets_observe_signal():
    """空 list 不 append 消息, 但 need_observe=True 的观察信号仍应置位."""
    r = Echoes()
    assert r.add_echoes([], need_observe=True) == []
    assert r.need_observe is True
    assert r.messages == []


def test_echoes_add_echoes_sets_need_observe_only_on_append():
    r = Echoes()
    appended = r.add_echoes(["x"], need_observe=True)
    assert len(appended) == 1
    assert r.need_observe is True


def test_echoes_is_empty_matrix():
    # 无消息为空; 有真实消息不为空. (旧实现漏了括号恒返 True, 这条能兜住.)
    assert Echoes().is_empty() is True
    assert Echoes(messages=[Message.new().with_content("x")]).is_empty() is False


# ============================================================
# Moment — 基础访问器
# ============================================================

def test_moment_defaults():
    m = Moment()
    assert m.id  # auto unique_id
    assert m.previous is None
    assert m.dynamic_context == {}
    assert m.percepts == {}
    assert m.hint == ""
    assert m.command_logos == ""
    assert m.logos == ""
    assert m.created is not None


def test_moment_new_echoes_links_to_self_id():
    m = Moment()
    r = m.new_echoes_container()
    assert r.moment_id == m.id


def test_moment_previous_executed_logos():
    assert Moment().previous_executed_logos() == ""
    m = Moment(previous=Echoes(executed_logos="ran this"))
    assert m.previous_executed_logos() == "ran this"


def test_moment_last_moment_id():
    assert Moment().last_moment_id() is None
    prev = Echoes(moment_id="prev-id")
    assert Moment(previous=prev).last_moment_id() == "prev-id"


def test_moment_with_perspective_sets_and_dedups_by_key():
    m = Moment()
    m.with_dynamic_context("vision", [Message.new().with_content("v1")])
    assert _texts(m.dynamic_context["vision"]) == ["v1"]
    # 同 key 再次写入应覆盖, 不累加.
    m.with_dynamic_context("vision", [Message.new().with_content("v2")])
    assert _texts(m.dynamic_context["vision"]) == ["v2"]
    assert len(m.dynamic_context) == 1
    # with_perspective 返回 self, 支持链式.
    assert m.with_dynamic_context("audio", []) is m


def test_with_percepts_only_records_when_source_has_messages():
    # 与 with_dynamic_context 不对称: 空列表不落 key, 即不记录空 source.
    m = Moment()
    m.with_percepts("cam", [])
    assert "cam" not in m.percepts
    # 同 source 覆盖写, 不累加.
    m.with_percepts("cam", [Message.new().with_content("frame")])
    assert _texts(m.percepts["cam"]) == ["frame"]
    m.with_percepts("cam", [Message.new().with_content("frame2")])
    assert _texts(m.percepts["cam"]) == ["frame2"]


# ============================================================
# Moment — is_empty / is_percepts_empty
# ============================================================

def test_moment_is_empty_matrix():
    empty = Moment()
    assert empty.is_empty()
    assert empty.is_percepts_empty()

    with_percept = Moment(percepts={"test": [Message.new().with_content("x")]})
    assert not with_percept.is_empty()
    assert not with_percept.is_percepts_empty()

    # 有 previous 但无新 percepts: 不算 empty, 但算 empty_request.
    with_prev = Moment(previous=Echoes(messages=[Message.new().with_content("prev")]))
    assert not with_prev.is_empty()
    assert with_prev.is_percepts_empty()


# ============================================================
# Moment — dynamic_context_messages
# ============================================================

def test_dynamic_context_messages_empty():
    assert list(Moment().dynamic_context_messages()) == []


def test_dynamic_context_messages_flattens_all_keys_in_order():
    m = Moment()
    m.with_dynamic_context("a", [Message.new().with_content("a1")])
    m.with_dynamic_context("b", [Message.new().with_content("b1"), Message.new().with_content("b2")])
    assert _texts(m.dynamic_context_messages()) == ["a1", "b1", "b2"]


# ============================================================
# Moment — previous_echoes_messages
# ============================================================

def test_previous_echoes_messages_empty_when_no_previous():
    assert list(Moment().previous_echoes_messages()) == []


def test_previous_echoes_messages_with_messages_and_stop_reason():
    prev = Echoes(
        messages=[Message.new().with_content("result")],
        stop_reason="faded",
    )
    msgs = list(Moment(previous=prev).previous_echoes_messages())
    assert "result" in _texts(msgs)
    # stop_reason 作为独立 tag 消息追加.
    stop_msgs = [m for m in msgs if m.meta.tag == "stop_reason"]
    assert len(stop_msgs) == 1
    assert "faded" in _text(stop_msgs[0])


def test_previous_echoes_messages_no_stop_reason():
    prev = Echoes(messages=[Message.new().with_content("result")])
    msgs = list(Moment(previous=prev).previous_echoes_messages())
    assert all(m.meta.tag != "stop_reason" for m in msgs)


def test_previous_echoes_messages_stop_reason_only():
    prev = Echoes(stop_reason="just stopped")
    msgs = list(Moment(previous=prev).previous_echoes_messages())
    assert len(msgs) == 1
    assert msgs[0].meta.tag == "stop_reason"


# ============================================================
# Moment — inputs_messages
# ============================================================

def test_inputs_messages_order_percepts_executing_hint():
    m = Moment(
        percepts={"test": [Message.new().with_content("p1")]},
        command_logos="cmd",
        hint="do it",
    )
    msgs = list(m.inputs_messages(with_hint=True, with_command_executing=True))
    assert "p1" in _text(msgs[0])
    assert msgs[1].meta.tag == "executing"
    assert "cmd" in _text(msgs[1])
    assert msgs[2].meta.tag == "hint"
    assert "do it" in _text(msgs[2])


def test_inputs_messages_without_command_executing():
    m = Moment(percepts={"test": [Message.new().with_content("p1")]}, command_logos="cmd")
    msgs = list(m.inputs_messages(with_command_executing=False))
    assert all(mm.meta.tag != "executing" for mm in msgs)


def test_inputs_messages_without_hint():
    m = Moment(percepts={"test": [Message.new().with_content("p1")]}, hint="skip")
    msgs = list(m.inputs_messages(with_hint=False))
    assert all(mm.meta.tag != "hint" for mm in msgs)


def test_inputs_messages_skips_empty_command_and_hint():
    m = Moment(percepts={"test": [Message.new().with_content("p1")]})
    msgs = list(m.inputs_messages(with_hint=True, with_command_executing=True))
    assert len(msgs) == 1


# ============================================================
# Moment — full_moment_message
# ============================================================


def test_moment_messages_without_dynamic_context():
    m = Moment(percepts={"test": [Message.new().with_content("p1")]})
    m.with_dynamic_context("ctx", [Message.new().with_content("full")])
    texts = _texts(m.full_moment_messages(with_dynamic_context=False, with_hint=False))
    # with_dynamic_context=False 时, 动态上下文不进输入.
    assert "full" not in "\n".join(texts)
    assert "p1" in "\n".join(texts)


# ============================================================
# Moment — as_history_messages
# ============================================================

def test_as_history_messages_keeps_previous_and_percepts():
    """as_history_messages 遗忘 dynamic_context, 只保留 previous + percepts."""
    prev = Echoes(messages=[Message.new().with_content("outcome")])
    m = Moment(previous=prev, percepts={"test": [Message.new().with_content("p1")]})
    texts = _texts(m.as_history_messages())
    # 历史视图遗忘 dynamic_context, 只保留 previous + percepts.
    assert texts == ["outcome", "p1"]


def test_as_history_messages_forgets_live_dynamic_context():
    prev = Echoes(messages=[Message.new().with_content("outcome")])
    m = Moment(previous=prev, percepts={"test": [Message.new().with_content("p1")]})
    m.with_dynamic_context("ctx", [Message.new().with_content("live perspective")])
    texts = _texts(m.as_history_messages())
    # 实时 dynamic_context 被遗忘, previous + percepts 保留.
    assert "live perspective" not in texts
    assert texts == ["outcome", "p1"]


def test_as_history_messages_forgets_dynamic_context_even_when_present():
    m = Moment(percepts={"test": [Message.new().with_content("p1")]})
    m.with_dynamic_context("ctx", [Message.new().with_content("never in history")])
    texts = _texts(m.as_history_messages())
    assert "never in history" not in texts


# ============================================================
# Moment — 序列化
# ============================================================

def test_to_dict_excludes_defaults_and_none():
    m = Moment(percepts={"test": [Message.new().with_content("p1")]})
    d = m.to_dict()
    assert isinstance(d, dict)
    # 默认值字段不应出现.
    assert "hint" not in d
    assert "command_logos" not in d
    assert "dynamic_context" not in d
    # 非默认字段应出现.
    assert "percepts" in d


def test_to_json_excludes_dynamic_context_and_hint_by_default():
    """默认 exclude_dynamic_context=True + exclude_hint=True: 两者都不应泄漏."""
    m = Moment(percepts={"test": [Message.new().with_content("p1")]}, hint="secret hint")
    m.with_dynamic_context("ctx", [Message.new().with_content("secret perspective")])
    j = m.to_json()
    assert "secret perspective" not in j
    assert "secret hint" not in j


def test_to_json_can_keep_dynamic_context():
    m = Moment()
    m.with_dynamic_context("ctx", [Message.new().with_content("keep me")])
    j = m.to_json(exclude_dynamic_context=False, exclude_hint=True)
    assert "keep me" in j


def test_to_json_can_keep_hint():
    m = Moment(percepts={"test": [Message.new().with_content("p1")]}, hint="keep hint")
    j = m.to_json(exclude_dynamic_context=True, exclude_hint=False)
    assert "keep hint" in j


def test_for_saving_clears_dynamic_context_and_hint():
    prev = Echoes(executed_logos="ran")
    m = Moment(
        previous=prev,
        percepts={"test": [Message.new().with_content("p1")]}, hint="ephemeral",
        command_logos="cmd",
        logos="model output",
    )
    m.with_dynamic_context("ctx", [Message.new().with_content("live")])
    saved = m.for_saving()
    # dynamic_context / hint 被清空.
    assert saved.dynamic_context == {}
    assert saved.hint == ""
    # 其余字段保留.
    assert saved.previous is prev
    assert _texts(saved.percepts_messages()) == ["p1"]
    assert saved.command_logos == "cmd"
    assert saved.logos == "model output"
    # 原 moment 不被修改 (model_copy).
    assert m.hint == "ephemeral"
    assert "ctx" in m.dynamic_context


# ============================================================
# Moment.to_history_turns — 回合切分核心
# ============================================================

def _moment_with_logos(logos: str, percept: str = "", previous: Echoes | None = None) -> Moment:
    return Moment(
        previous=previous,
        percepts={"test": [Message.new().with_content(percept)]} if percept else {}, logos=logos,
    )


def test_to_history_turns_empty_iterable():
    assert list(Moment.to_history_turns([])) == []


def test_to_history_turns_single_moment_with_logos():
    m = _moment_with_logos("model said hi", percept="user input")
    turns = list(Moment.to_history_turns([m]))
    assert len(turns) == 1
    messages, logos = turns[0]
    assert logos == "model said hi"
    assert "user input" in _texts(messages)


def test_to_history_turns_splits_on_logos():
    m1 = _moment_with_logos("logos 1", percept="input 1")
    r1 = m1.new_echoes_container()
    m2 = m1.new_echoes_container().new_moment(percepts={"test": [Message.new().with_content("input 2")]})
    m2.logos = "logos 2"
    turns = list(Moment.to_history_turns([m1, m2]))
    assert len(turns) == 2
    assert turns[0][1] == "logos 1"
    assert turns[1][1] == "logos 2"


def test_to_history_turns_stitches_executed_logos_when_no_model_logos():
    """某轮模型未产 logos 但系统执行了 command, executed_logos 应缝合进下一回合."""
    # m1: 无 model logos, 但执行了 command.
    m1 = Moment(percepts={"test": [Message.new().with_content("input 1")]}, logos="")
    r1 = m1.new_echoes_container()
    r1.executed_logos = "command ran"
    # m2: 承接 r1, 模型产出 logos.
    m2 = r1.new_moment(percepts={"test": [Message.new().with_content("input 2")]})
    m2.logos = "model logos"
    turns = list(Moment.to_history_turns([m1, m2]))
    # m1 无 logos → 不切回合, buffer 继续; m2 有 logos → 切一个回合.
    assert len(turns) == 1
    messages, logos = turns[0]
    assert logos == "model logos"
    texts = _texts(messages)
    # m1 的 input、缝合的 executed_logos、m2 的 input 都在同一回合.
    assert "input 1" in "\n".join(texts)
    assert "command ran" in "\n".join(texts)
    assert "input 2" in "\n".join(texts)


def test_to_history_turns_trailing_buffer_yields_none_logos():
    """末尾若有未被 logos 切分的 buffer, 以 (messages, None) 收尾."""
    m1 = _moment_with_logos("logos 1", percept="input 1")
    # m2 无 logos, 末尾残留.
    m2 = m1.new_echoes_container().new_moment(percepts={"test": [Message.new().with_content("trailing")]})
    turns = list(Moment.to_history_turns([m1, m2]))
    assert len(turns) == 2
    assert turns[0][1] == "logos 1"
    assert turns[1][1] is None
    assert "trailing" in _texts(turns[1][0])


def test_to_history_turns_executed_logos_not_duplicated_after_model_logos():
    """上一轮有 model logos 时, 其 executed_logos 不应再缝合 (避免重复)."""
    m1 = _moment_with_logos("model logos 1", percept="input 1")
    r1 = m1.new_echoes_container()
    r1.executed_logos = "executed for m1"
    m2 = r1.new_moment(percepts={"test": [Message.new().with_content("input 2")]})
    m2.logos = "model logos 2"
    turns = list(Moment.to_history_turns([m1, m2]))
    assert len(turns) == 2
    # m1 有 model logos → last_moment_has_logos=True → m2 不缝合 executed_logos.
    second_texts = _texts(turns[1][0])
    assert "executed for m1" not in second_texts


def test_to_history_turns_single_moment_without_logos():
    """单条无 logos 但有消息 → 以 (messages, None) 收尾, 不丢消息."""
    m = Moment(percepts={"test": [Message.new().with_content("only input")]})
    turns = list(Moment.to_history_turns([m]))
    assert len(turns) == 1
    messages, logos = turns[0]
    assert logos is None
    assert "only input" in _texts(messages)


def test_to_history_turns_all_moments_without_logos_merge_into_one():
    """n 条全程无 logos → 合并为单个 (messages, None) 回合."""
    m1 = Moment(percepts={"test": [Message.new().with_content("a")]})
    m2 = m1.new_echoes_container().new_moment(percepts={"test": [Message.new().with_content("b")]})
    m3 = m2.new_echoes_container().new_moment(percepts={"test": [Message.new().with_content("c")]})
    turns = list(Moment.to_history_turns([m1, m2, m3]))
    assert len(turns) == 1
    messages, logos = turns[0]
    assert logos is None
    assert _texts(messages) == ["a", "b", "c"]


def test_to_history_turns_logos_without_messages_inserts_placeholder():
    """有 logos 但本帧无可入史消息 (dynamic_context 触发): 补占位, logos 不被丢弃."""
    # 无 previous / 无 percepts / 仅有 logos — 模拟 dynamic_context 触发的响应.
    first = Moment()
    second = first.new_echoes_container().new_moment()
    second.logos = "model spoke from perspective"
    third = second.new_echoes_container().new_moment()
    turns = list(Moment.to_history_turns([first, second, third]))
    assert len(turns) == 1
    messages, logos = turns[0]
    assert logos == "model spoke from perspective"
    assert len(messages) == 0
    forth_echoes = third.new_echoes_container()
    forth_echoes.add_echoes(["hello"], need_observe=True)
    forth = forth_echoes.new_moment()
    turns = list(Moment.to_history_turns([first, second, third, forth]))
    assert len(turns) == 2
    messages, logos = turns[1]
    assert logos is None


def test_to_history_turns_fully_empty_moment_yields_nothing():
    """完全空的 moment (无 logos 无消息) 不产出任何回合."""
    assert list(Moment.to_history_turns([Moment()])) == []
