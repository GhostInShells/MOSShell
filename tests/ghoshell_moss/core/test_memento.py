"""
memento.py 纯数据结构单测.

Reaction / Moment 是 mindflow→ghost 协议转换面的核心数据载体, 无 IO、无并发,
所有行为都应能通过构造 + 方法调用直接暴露. 本套件覆盖:

- Reaction: 默认值、new_moment 参数传递
- Moment: 访问器、is_empty 判定、perspectives 组合、各 messages 视图
- 序列化: to_dict / to_json / for_saving
- 历史缝合: as_history_messages / to_history_turns (回合切分核心)
"""
from ghoshell_moss.core.blueprint.memento import Moment, Reaction
from ghoshell_moss.message import Message


def _text(msg: Message) -> str:
    """取一条消息的首个文本内容, 简化断言."""
    for c in msg.contents:
        if "text" in c:
            return c["text"]
    return ""


def _texts(msgs) -> list[str]:
    return [_text(m) for m in msgs]


# ============================================================
# Reaction
# ============================================================

def test_reaction_defaults():
    r = Reaction()
    assert r.moment_id  # auto unique_id
    assert r.executed_logos == ""
    assert r.messages == []
    assert r.stop_reason == ""


def test_reaction_new_moment_passes_all_params():
    r = Reaction(executed_logos="prev logos", stop_reason="done")
    moment = r.new_moment(
        percepts={"test": [Message.new().with_content("p")]},
        hint="hint text",
        command_logos="reflex!",
    )
    assert moment.previous is r
    assert _text(next(iter(moment.percepts_messages()))) == "p"
    assert moment.hint == "hint text"
    assert moment.command_logos == "reflex!"


def test_reaction_new_moment_defaults_empty():
    r = Reaction(executed_logos="prev")
    moment = r.new_moment()
    assert moment.percepts == {}
    assert moment.hint == ""
    assert moment.command_logos == ""
    assert moment.previous is r


def test_reaction_new_moment_percepts_none_is_empty_dict():
    moment = Reaction().new_moment(percepts=None)
    assert moment.percepts == {}


# ============================================================
# Moment — 基础访问器
# ============================================================

def test_moment_defaults():
    m = Moment()
    assert m.id  # auto unique_id
    assert m.previous is None
    assert m.perspectives == {}
    assert m.compacted_perspectives is None
    assert m.percepts == {}
    assert m.hint == ""
    assert m.command_logos == ""
    assert m.logos == ""
    assert m.created is not None


def test_moment_new_reaction_links_to_self_id():
    m = Moment()
    r = m.new_reaction()
    assert r.moment_id == m.id


def test_moment_previous_executed_logos():
    assert Moment().previous_executed_logos() == ""
    m = Moment(previous=Reaction(executed_logos="ran this"))
    assert m.previous_executed_logos() == "ran this"


def test_moment_last_moment_id():
    assert Moment().last_moment_id() is None
    prev = Reaction(moment_id="prev-id")
    assert Moment(previous=prev).last_moment_id() == "prev-id"


def test_moment_with_perspective_sets_and_dedups_by_key():
    m = Moment()
    m.with_perspective("vision", [Message.new().with_content("v1")])
    assert _texts(m.perspectives["vision"]) == ["v1"]
    # 同 key 再次写入应覆盖, 不累加.
    m.with_perspective("vision", [Message.new().with_content("v2")])
    assert _texts(m.perspectives["vision"]) == ["v2"]
    assert len(m.perspectives) == 1
    # with_perspective 返回 self, 支持链式.
    assert m.with_perspective("audio", []) is m


# ============================================================
# Moment — is_empty / is_empty_request
# ============================================================

def test_moment_is_empty_matrix():
    empty = Moment()
    assert empty.is_empty()
    assert empty.is_empty_request()

    with_percept = Moment(percepts={"test": [Message.new().with_content("x")]})
    assert not with_percept.is_empty()
    assert not with_percept.is_empty_request()

    # 有 previous 但无新 percepts: 不算 empty, 但算 empty_request.
    with_prev = Moment(previous=Reaction())
    assert not with_prev.is_empty()
    assert with_prev.is_empty_request()


# ============================================================
# Moment — perspective_messages
# ============================================================

def test_perspective_messages_empty():
    assert list(Moment().perspective_messages()) == []


def test_perspective_messages_flattens_all_keys_in_order():
    m = Moment()
    m.with_perspective("a", [Message.new().with_content("a1")])
    m.with_perspective("b", [Message.new().with_content("b1"), Message.new().with_content("b2")])
    assert _texts(m.perspective_messages()) == ["a1", "b1", "b2"]


def test_perspective_messages_compact_first_uses_compacted():
    m = Moment()
    m.with_perspective("ctx", [Message.new().with_content("long")])
    m.compacted_perspectives = [Message.new().with_content("short")]
    assert _texts(m.perspective_messages(compact_first=True)) == ["short"]


def test_perspective_messages_compact_first_falls_back_when_no_compacted():
    m = Moment()
    m.with_perspective("ctx", [Message.new().with_content("long")])
    # compacted 为 None 时, compact_first=True 仍回退到全量.
    assert _texts(m.perspective_messages(compact_first=True)) == ["long"]


def test_perspective_messages_compact_false_ignores_compacted():
    m = Moment()
    m.with_perspective("ctx", [Message.new().with_content("long")])
    m.compacted_perspectives = [Message.new().with_content("short")]
    assert _texts(m.perspective_messages(compact_first=False)) == ["long"]


# ============================================================
# Moment — previous_reaction_messages
# ============================================================

def test_previous_reaction_messages_empty_when_no_previous():
    assert list(Moment().previous_reaction_messages()) == []


def test_previous_reaction_messages_with_messages_and_stop_reason():
    prev = Reaction(
        messages=[Message.new().with_content("result")],
        stop_reason="faded",
    )
    msgs = list(Moment(previous=prev).previous_reaction_messages())
    assert "result" in _texts(msgs)
    # stop_reason 作为独立 tag 消息追加.
    stop_msgs = [m for m in msgs if m.meta.tag == "stop_reason"]
    assert len(stop_msgs) == 1
    assert _text(stop_msgs[0]) == "faded"


def test_previous_reaction_messages_no_stop_reason():
    prev = Reaction(messages=[Message.new().with_content("result")])
    msgs = list(Moment(previous=prev).previous_reaction_messages())
    assert all(m.meta.tag != "stop_reason" for m in msgs)


def test_previous_reaction_messages_stop_reason_only():
    prev = Reaction(stop_reason="just stopped")
    msgs = list(Moment(previous=prev).previous_reaction_messages())
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
    assert _text(msgs[0]) == "p1"
    assert msgs[1].meta.tag == "executing"
    assert _text(msgs[1]) == "cmd"
    assert msgs[2].meta.tag == "hint"
    assert _text(msgs[2]) == "do it"


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
# Moment — as_request_messages
# ============================================================

def test_as_request_messages_full_order():
    prev = Reaction(messages=[Message.new().with_content("outcome")])
    m = Moment(
        previous=prev,
        percepts={"test": [Message.new().with_content("percept")]}, hint="react!",
    )
    m.with_perspective("moss_dynamic", [Message.new().with_content("dynamic")])
    texts = _texts(m.as_request_messages(with_perspectives=True, with_hint=True))
    # 顺序: previous(outcome) → perspectives(dynamic) → percepts → hint
    assert texts.index("outcome") < texts.index("dynamic") < texts.index("percept")
    assert "react!" in texts


def test_as_request_messages_without_perspectives_uses_compacted():
    m = Moment(percepts={"test": [Message.new().with_content("p1")]})
    m.with_perspective("ctx", [Message.new().with_content("full")])
    m.compacted_perspectives = [Message.new().with_content("compact")]
    texts = _texts(m.as_request_messages(with_perspectives=False, with_hint=False))
    # with_perspectives=False 走 elif: compacted 非空时产出 compacted.
    assert "full" not in texts
    assert "compact" in texts
    assert "p1" in texts


def test_as_request_messages_without_perspectives_and_no_compacted():
    m = Moment(percepts={"test": [Message.new().with_content("p1")]})
    m.with_perspective("ctx", [Message.new().with_content("full")])
    texts = _texts(m.as_request_messages(with_perspectives=False, with_hint=False))
    assert "full" not in texts
    assert texts == ["p1"]


# ============================================================
# Moment — as_history_messages (含 compacted None 守卫)
# ============================================================

def test_as_history_messages_with_none_compacted_does_not_crash():
    """compacted_perspectives 默认 None 时, as_history_messages 不应崩溃."""
    prev = Reaction(messages=[Message.new().with_content("outcome")])
    m = Moment(previous=prev, percepts={"test": [Message.new().with_content("p1")]})
    assert m.compacted_perspectives is None
    texts = _texts(m.as_history_messages())
    # 历史视图遗忘 perspectives, 只保留 previous + percepts.
    assert texts == ["outcome", "p1"]


def test_as_history_messages_with_compacted():
    prev = Reaction(messages=[Message.new().with_content("outcome")])
    m = Moment(previous=prev, percepts={"test": [Message.new().with_content("p1")]})
    m.with_perspective("ctx", [Message.new().with_content("live perspective")])
    m.compacted_perspectives = [Message.new().with_content("compacted")]
    texts = _texts(m.as_history_messages())
    # 实时 perspectives 被遗忘, 压缩快照保留.
    assert "live perspective" not in texts
    assert texts == ["outcome", "compacted", "p1"]


def test_as_history_messages_forgets_perspectives_even_when_present():
    m = Moment(percepts={"test": [Message.new().with_content("p1")]})
    m.with_perspective("ctx", [Message.new().with_content("never in history")])
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
    assert "compacted_perspectives" not in d
    # 非默认字段应出现.
    assert "percepts" in d


def test_to_json_excludes_perspectives_and_hint_by_default():
    """默认 exclude_perspectives=True + exclude_hint=True: 两者都不应泄漏."""
    m = Moment(percepts={"test": [Message.new().with_content("p1")]}, hint="secret hint")
    m.with_perspective("ctx", [Message.new().with_content("secret perspective")])
    j = m.to_json()
    assert "secret perspective" not in j
    assert "secret hint" not in j


def test_to_json_can_keep_perspectives():
    m = Moment()
    m.with_perspective("ctx", [Message.new().with_content("keep me")])
    j = m.to_json(exclude_perspectives=False, exclude_hint=True)
    assert "keep me" in j


def test_to_json_can_keep_hint():
    m = Moment(percepts={"test": [Message.new().with_content("p1")]}, hint="keep hint")
    j = m.to_json(exclude_perspectives=True, exclude_hint=False)
    assert "keep hint" in j


def test_for_saving_clears_perspectives_and_hint():
    prev = Reaction(executed_logos="ran")
    m = Moment(
        previous=prev,
        percepts={"test": [Message.new().with_content("p1")]}, hint="ephemeral",
        command_logos="cmd",
        logos="model output",
    )
    m.with_perspective("ctx", [Message.new().with_content("live")])
    saved = m.for_saving(compacted_perspectives=[Message.new().with_content("compact")])
    # perspectives / hint 被清空.
    assert saved.perspectives == {}
    assert saved.hint == ""
    # compacted 被写入.
    assert _texts(saved.compacted_perspectives) == ["compact"]
    # 其余字段保留.
    assert saved.previous is prev
    assert _texts(saved.percepts_messages()) == ["p1"]
    assert saved.command_logos == "cmd"
    assert saved.logos == "model output"
    # 原 moment 不被修改 (model_copy).
    assert m.hint == "ephemeral"
    assert "ctx" in m.perspectives


def test_for_saving_compacted_none_becomes_empty_list():
    m = Moment(percepts={"test": [Message.new().with_content("p1")]})
    saved = m.for_saving()
    assert saved.compacted_perspectives == []


# ============================================================
# Moment.to_history_turns — 回合切分核心
# ============================================================

def _moment_with_logos(logos: str, percept: str = "", previous: Reaction | None = None) -> Moment:
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
    r1 = m1.new_reaction()
    m2 = m1.new_reaction().new_moment(percepts={"test": [Message.new().with_content("input 2")]})
    m2.logos = "logos 2"
    turns = list(Moment.to_history_turns([m1, m2]))
    assert len(turns) == 2
    assert turns[0][1] == "logos 1"
    assert turns[1][1] == "logos 2"


def test_to_history_turns_stitches_executed_logos_when_no_model_logos():
    """某轮模型未产 logos 但系统执行了 command, executed_logos 应缝合进下一回合."""
    # m1: 无 model logos, 但执行了 command.
    m1 = Moment(percepts={"test": [Message.new().with_content("input 1")]}, logos="")
    r1 = m1.new_reaction()
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
    assert "input 1" in texts
    assert "command ran" in texts
    assert "input 2" in texts


def test_to_history_turns_trailing_buffer_yields_none_logos():
    """末尾若有未被 logos 切分的 buffer, 以 (messages, None) 收尾."""
    m1 = _moment_with_logos("logos 1", percept="input 1")
    # m2 无 logos, 末尾残留.
    m2 = m1.new_reaction().new_moment(percepts={"test": [Message.new().with_content("trailing")]})
    turns = list(Moment.to_history_turns([m1, m2]))
    assert len(turns) == 2
    assert turns[0][1] == "logos 1"
    assert turns[1][1] is None
    assert "trailing" in _texts(turns[1][0])


def test_to_history_turns_executed_logos_not_duplicated_after_model_logos():
    """上一轮有 model logos 时, 其 executed_logos 不应再缝合 (避免重复)."""
    m1 = _moment_with_logos("model logos 1", percept="input 1")
    r1 = m1.new_reaction()
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
    m2 = m1.new_reaction().new_moment(percepts={"test": [Message.new().with_content("b")]})
    m3 = m2.new_reaction().new_moment(percepts={"test": [Message.new().with_content("c")]})
    turns = list(Moment.to_history_turns([m1, m2, m3]))
    assert len(turns) == 1
    messages, logos = turns[0]
    assert logos is None
    assert _texts(messages) == ["a", "b", "c"]


def test_to_history_turns_logos_without_messages_inserts_placeholder():
    """有 logos 但本帧无可入史消息 (perspectives 触发): 补占位, logos 不被丢弃."""
    # 无 previous / 无 percepts / 无 compacted, 仅有 logos — 模拟 perspectives 触发的响应.
    bare = Moment(logos="model spoke from perspective")
    turns = list(Moment.to_history_turns([bare]))
    assert len(turns) == 1
    messages, logos = turns[0]
    assert logos == "model spoke from perspective"
    # 回合不为空: 插入了一条 perspective 占位消息.
    assert len(messages) == 1
    assert messages[0].meta.tag == "perspective"


def test_to_history_turns_fully_empty_moment_yields_nothing():
    """完全空的 moment (无 logos 无消息) 不产出任何回合."""
    assert list(Moment.to_history_turns([Moment()])) == []

