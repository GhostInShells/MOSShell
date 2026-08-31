"""
Porcelain layer — Moment <-> MomentRecord codec + merge message.
主权层测试, 覆盖信封与 MOSS 强类型之间的桥 (v2 API).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ghoshell_moss.core.blueprint.moment import Moment, Echoes
from ghoshell_moss.memento import (
    MOSS_MOMENT_TYPE,
    CommitNotFoundError,
    MementoRef,
    MomentRecord,
    make_merge_message,
    moment_to_record,
    new_filesystem_memento,
    record_to_moment,
    update_moment,
    window_messages,
)
from ghoshell_moss.message import Message


def test_moment_codec_round_trip():
    m = Moment(id="mx", logos="hello", command_logos="run")
    m = m.with_percepts("input", [Message.new(tag="user").with_content("hi")])
    rec = moment_to_record(m, threads=["chat"])
    assert rec.type == MOSS_MOMENT_TYPE
    assert rec.id == "mx"
    assert rec.threads == ["chat"]
    back = record_to_moment(rec)
    assert back.id == "mx"
    assert back.logos == "hello"
    assert back.command_logos == "run"
    assert "hi" in back.percepts_texts()[0]


def test_record_to_moment_rejects_wrong_type():
    rec = MomentRecord(id="a", type="other/v1", payload={"x": 1})
    with pytest.raises(ValueError):
        record_to_moment(rec)


def test_update_moment_convenience(tmp_path: Path):
    m = new_filesystem_memento(tmp_path / "memento", "o")
    b = m.create_line("main")
    update_moment(b, Moment(id="m1", logos="one"), threads=["chat"])
    update_moment(b, Moment(id="m2", logos="two"))
    view = b.commit("s", kind="semantic")
    records = m.show(view.id).moments
    assert [r.id for r in records] == ["m1", "m2"]
    stored = next(r for r in records if r.id == "m1")
    assert stored.threads == ["chat"]
    assert record_to_moment(stored).logos == "one"


def test_merge_message_carries_ref(tmp_path: Path):
    m = new_filesystem_memento(tmp_path / "memento", "o")
    b = m.create_line("main")
    update_moment(b, Moment(id="m1"))
    view = b.commit("summary text", kind="semantic")

    msg = make_merge_message(m, view.id)
    ref = MementoRef.read(msg)
    assert ref is not None
    assert ref.commit_id == view.id
    assert ref.origin == "o"
    assert ref.note_seq == 0

    # Message content = commit summary (孔径一: 主路收 Message 而非展开原文)
    assert "summary text" in msg.to_content_string()

    with pytest.raises(CommitNotFoundError):
        make_merge_message(m, "cmt_nonexistent")


def test_merge_message_ref_tracks_reinterpret(tmp_path: Path):
    """MementoRef.note_seq 是渲染打戳: 释义再改写后, 新 merge 拿到新版本号."""
    m = new_filesystem_memento(tmp_path / "memento", "o")
    b = m.create_line("main")
    update_moment(b, Moment(id="m1"))
    view = b.commit("v0", kind="semantic")
    m.annotate(view.id, body="v1\n\nKind: semantic")

    msg = make_merge_message(m, view.id)
    ref = MementoRef.read(msg)
    assert ref.note_seq == 1
    assert "v1" in msg.to_content_string()

    # 通过 (commit_id, note_seq) 回溯到 v0 的原释义
    versions = m.notes(view.id)
    assert versions[0].text() == "v0"
    assert versions[1].text() == "v1"


def test_window_messages_expand(tmp_path: Path):
    m = new_filesystem_memento(tmp_path / "memento", "o")
    b = m.create_line("main")
    prev = Echoes(moment_id="prev0")
    prev.executed_logos = "cmd"
    m0 = prev.new_moment(percepts={"user": [Message.new().with_content("hi")]})
    m0.id = "m0"
    m0.logos = "hello"
    update_moment(b, m0)
    b.commit("first", kind="semantic")

    update_moment(b, Moment(id="m1"))

    window = b.window(detail_n=1, summary_m=-1)
    msgs = list(window_messages(b, window))
    # 折叠区: commit-summary + MementoRef
    summary_msgs = [msg for msg in msgs if msg.meta.tag == "commit-summary"]
    assert len(summary_msgs) == 1
    assert MementoRef.read(summary_msgs[0]) is not None
    assert "first" in summary_msgs[0].to_content_string()
