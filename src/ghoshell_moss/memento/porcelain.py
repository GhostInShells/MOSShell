"""
Memento porcelain — 信封之上的 MOSS 强类型桥 (v2 适配).

契约层 (abc.py) 只认 MomentRecord 信封. 本模块是信封与 MOSS 体系 (Moment / Message)
之间的桥, 属于主权层: Moment <-> MomentRecord 编解码, MementoRef 引用, 窗口渲染.
"""

from __future__ import annotations

import json
from typing import Iterable, Sequence

from ghoshell_moss.core.blueprint.memento import Moment
from ghoshell_moss.memento.abc import (
    BranchWindow,
    CommitNotFoundError,
    CommitView,
    Line,
    Memento,
    MomentRecord,
)
from ghoshell_moss.message import Addition, Message

__all__ = [
    "MOSS_MOMENT_TYPE",
    "MementoRef",
    "moment_to_record",
    "record_to_moment",
    "update_moment",
    "make_merge_message",
    "commit_summary_message",
    "window_messages",
]

MOSS_MOMENT_TYPE = "moss.moment/v1"


class MementoRef(Addition):
    """
    指向 commit 的强类型引用, 挂在 Message.additional 上.

    note_seq 是渲染打戳 — (commit_id, note_seq) 从 memento.notes() 复原当时视图.
    """

    origin: str = ""
    line: str = ""
    commit_id: str = ""
    note_seq: int = 0

    @classmethod
    def keyword(cls) -> str:
        return "memento.ref"


def _ref_of(line: Line, view: CommitView) -> MementoRef:
    ref = line.ref
    return MementoRef(
        origin=ref.origin if ref else "",
        line=line.name,
        commit_id=view.id,
        note_seq=view.note_seq,
    )


# ── Moment codec ──


def moment_to_record(
    moment: Moment, *, threads: Sequence[str] = ()
) -> MomentRecord:
    """Moment -> 信封. payload 用 Moment 标准序列化."""
    payload = json.loads(
        moment.to_json(exclude_perspectives=True, exclude_hint=True)
    )
    return MomentRecord(
        id=moment.id,
        created=moment.created,
        type=MOSS_MOMENT_TYPE,
        payload=payload,
        threads=list(threads),
    )


def record_to_moment(record: MomentRecord) -> Moment:
    """信封 -> Moment."""
    if record.type != MOSS_MOMENT_TYPE:
        raise ValueError(
            f"record {record.id!r} has type {record.type!r}, not {MOSS_MOMENT_TYPE!r}"
        )
    return Moment.model_validate(record.payload)


def update_moment(
    line: Line, moment: Moment, *, threads: Sequence[str] = ()
) -> MomentRecord:
    """便捷入口: 编码 + 写入 staging."""
    record = moment_to_record(moment, threads=threads)
    line.record(record)
    return record


# ── merge ≡ 带引用的 Message ──


def commit_summary_message(line: Line, view: CommitView) -> Message:
    """commit 折叠展示: content = 释义摘要, additional 携带 MementoRef."""
    msg = Message.new(tag="commit-summary").with_content(view.summary())
    _ref_of(line, view).set(msg)
    return msg


def make_merge_message(memento: Memento, commit_id: str) -> Message:
    """
    把 commit 包装成 Message — 孔径一 (输入队列) 的载体.
    :raise CommitNotFoundError:
    """
    detail = memento.show(commit_id)
    # 构造 CommitView 用于 MementoRef
    from ghoshell_moss.memento.abc import CommitNote

    note = detail.notes[-1] if detail.notes else CommitNote(ref=commit_id)
    note_seq = len(detail.notes) - 1 if detail.notes else 0
    from ghoshell_moss.memento.abc import Commit

    view = CommitView(commit=detail.commit, note=note, note_seq=note_seq)
    # 需要 line 信息来构造 MementoRef — 从 commits.jsonl 推断
    msg = Message.new(tag="commit-summary").with_content(view.summary())
    # 简化: 直接从 commit_id + memento.owner 构造 ref
    MementoRef(origin=memento.owner, commit_id=commit_id, note_seq=note_seq).set(msg)
    return msg


# ── 窗口渲染 ──


def window_messages(
    line: Line, window: BranchWindow
) -> Iterable[Message]:
    """BranchWindow -> 模型可读消息序列 (时间升序)."""
    for view in window.summaries:
        yield commit_summary_message(line, view)
    for record in window.details:
        if record.type == MOSS_MOMENT_TYPE:
            yield from record_to_moment(record).as_history_messages()
        else:
            yield Message.new(tag="memento-record").with_content(
                json.dumps(record.payload, ensure_ascii=False)
            )
