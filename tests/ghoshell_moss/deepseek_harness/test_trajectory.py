"""trajectory 行为证据 — seed 尾部截断重建 + transcript 折叠.

覆盖:
- seed_from_log: turn 主坐标反查 seq (无 end_seq 也能定位 end_turn 的 turn/end);
  end_seq 直接指定; 切在 turn/end; 吞 trailing standalone 到下一个 turn/start.
- seed_from_log 失败面: end turn 未闭合 / end_seq 越界 / 边界非 turn/end.
- render_transcript: > ~ @ 三符号; 非 injection 过滤; tool/result 与空 assistant 跳过;
  limit_turns 保留最近 N turn.
"""

import pytest

from ghoshell_moss.deepseek_harness.trajectory import (
    SeedUnavailable,
    render_transcript,
    seed_from_log,
)
from ghoshell_moss.deepseek_harness.types.refs import DshSessionRef
from ghoshell_moss.deepseek_harness.types.session_events import SessionEvent

TS = "turn/start"
TE = "turn/end"
UM = "user/message"
AM = "assistant/message"
TC = "tool/call"
TR = "tool/result"
TT = "session/title"


def _log(*specs):
    """(type, data) → seq 连续的 SessionEvent 列表 (seq == index)."""
    return [
        SessionEvent.from_dict({"type": t, "seq": i, "data": d})
        for i, (t, d) in enumerate(specs)
    ]


def _user(text, kind="user"):
    return {"source": {"kind": kind}, "content": [{"type": "text", "text": text}]}


def _assistant(text):
    return {"message": {"content": [{"type": "text", "text": text}]}}


# ---- seed_from_log ---- #


def test_seed_resolves_end_turn_without_seq():
    events = _log(
        (TS, {"turn": 0}),
        (UM, _user("hi")),
        (AM, _assistant("yo")),
        (TE, {"turn": 0}),
        (TS, {"turn": 1}),
        (UM, _user("again")),
        (AM, _assistant("ok")),
        (TE, {"turn": 1}),
    )
    ref = DshSessionRef(session_id="s", start_turn=0, end_turn=1)
    seed = seed_from_log(events, ref)
    # 切到 turn 1 的 turn/end, 前缀完整
    assert [e.meta.seq for e in seed] == [0, 1, 2, 3, 4, 5, 6, 7]
    assert seed[-1].meta.type == TE
    assert seed[-1].data["turn"] == 1


def test_seed_end_seq_overrides_reverse_lookup():
    events = _log(
        (TS, {"turn": 0}),
        (UM, _user("hi")),
        (TE, {"turn": 0}),
        (TS, {"turn": 1}),
        (UM, _user("again")),
        (TE, {"turn": 1}),
    )
    ref = DshSessionRef(session_id="s", start_turn=0, end_turn=0, end_seq=2)
    seed = seed_from_log(events, ref)
    assert [e.meta.seq for e in seed] == [0, 1, 2]


def test_seed_swallows_trailing_standalone_until_next_turn_start():
    events = _log(
        (TS, {"turn": 0}),
        (UM, _user("hi")),
        (TE, {"turn": 0}),
        (TT, {"title": "late title"}),
        (TS, {"turn": 1}),
        (UM, _user("next")),
    )
    ref = DshSessionRef(session_id="s", start_turn=0, end_turn=0)
    seed = seed_from_log(events, ref)
    # 边界 turn/end 之后的 session/title 是 standalone, 被吞进 seed; 停在下一个 turn/start 前
    assert [e.meta.type for e in seed] == [TS, UM, TE, TT]
    assert [e.meta.seq for e in seed] == [0, 1, 2, 3]


def test_seed_open_end_turn_raises():
    events = _log(
        (TS, {"turn": 0}),
        (UM, _user("hi")),
        (TE, {"turn": 0}),
        (TS, {"turn": 1}),
        (UM, _user("no end")),
    )
    ref = DshSessionRef(session_id="s", start_turn=0, end_turn=1)
    with pytest.raises(SeedUnavailable):
        seed_from_log(events, ref)


def test_seed_end_seq_out_of_range_raises():
    events = _log((TS, {"turn": 0}), (UM, _user("hi")), (TE, {"turn": 0}))
    ref = DshSessionRef(session_id="s", start_turn=0, end_turn=0, end_seq=99)
    with pytest.raises(SeedUnavailable):
        seed_from_log(events, ref)


def test_seed_boundary_not_turn_end_raises():
    events = _log((TS, {"turn": 0}), (UM, _user("hi")), (TE, {"turn": 0}))
    ref = DshSessionRef(session_id="s", start_turn=0, end_turn=0, end_seq=1)
    with pytest.raises(SeedUnavailable):
        seed_from_log(events, ref)


# ---- render_transcript ---- #


def test_transcript_symbols_and_filters():
    events = _log(
        (TS, {"turn": 0}),
        (UM, _user("hello")),
        (UM, _user("injected", kind="plugin")),
        (AM, _assistant("hi")),
        (TC, {"name": "search", "arguments": "{}"}),
        (TR, {"message": {"content": [{"type": "text", "text": "result"}]}}),
        (AM, {"message": {"content": []}}),  # 空 content 的 usage 帧
        (TE, {"turn": 0}),
    )
    out = render_transcript(events)
    assert out == "  > hello\n  ~ hi\n  @ search({})"


def test_transcript_limit_turns_keeps_tail():
    events = _log(
        (TS, {"turn": 0}),
        (UM, _user("first")),
        (TE, {"turn": 0}),
        (TS, {"turn": 1}),
        (UM, _user("second")),
        (TE, {"turn": 1}),
        (TS, {"turn": 2}),
        (UM, _user("third")),
        (TE, {"turn": 2}),
    )
    out = render_transcript(events, limit_turns=2)
    assert out == "  > second\n  > third"


def test_transcript_empty():
    assert render_transcript([]) == ""
