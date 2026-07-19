"""
Memento porcelain — 信封之上的 MOSS 强类型层.

契约层 (abc.py) 只认 MomentRecord 信封, 零 payload schema 依赖.
本模块是信封与 MOSS 体系 (Moment / Message) 之间的桥, 属于主权层:
- Moment <-> MomentRecord 编解码 (payload type 'moss.moment/v1')
- MementoRef: commit 引用的强类型 Addition
- merge ≡ 带引用的 Message
- BranchWindow -> 模型可读消息序列

其它 payload 生产者仿照本模块自带 codec 即可, 不需要动契约.
"""

from __future__ import annotations

import json
from typing import Iterable, Sequence

from ghoshell_moss.core.blueprint.memento import Moment
from ghoshell_moss.memento.abc import (
    BranchWindow,
    CommitNotFoundError,
    CommitView,
    MementoBranch,
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
    指向具体 commit 的强类型引用, 挂在 Message.additional 上.

    消费方: MementoRef.read(message) 拿到实例, 需要原文时
    memento.get_branch(branch_id, fork=fork).commit_records(commit_id) 展开.
    note_seq 是渲染打戳 (FORMAT.md §5.2) — 记录生成本引用时读到的释义版本,
    "当时模型看见什么" 由 (commit_id, note_seq) 从 branch.notes() 复原.
    """

    fork: str
    branch_id: str
    commit_id: str
    commit_seq: int
    note_seq: int = 0

    @classmethod
    def keyword(cls) -> str:
        return "memento.ref"


def _ref_of(branch: MementoBranch, view: CommitView) -> MementoRef:
    return MementoRef(
        fork=branch.meta.fork,
        branch_id=branch.meta.branch_id,
        commit_id=view.id,
        commit_seq=view.seq,
        note_seq=view.note_seq,
    )


# ── Moment codec ──


def moment_to_record(moment: Moment, *, threads: Sequence[str] = (), by: str = "") -> MomentRecord:
    """
    Moment -> 信封. payload 用 Moment 自己的标准序列化
    (丢 perspectives 动态快照与 hint, 语义同 Moment.to_json 的默认排除).
    """
    payload = json.loads(moment.to_json(exclude_perspectives=True, exclude_hint=True))
    return MomentRecord(
        id=moment.id,
        created=moment.created,
        type=MOSS_MOMENT_TYPE,
        payload=payload,
        threads=list(threads),
        by=by,
    )


def record_to_moment(record: MomentRecord) -> Moment:
    """信封 -> Moment. type 不匹配说明拿错了 payload, 立刻抛错."""
    if record.type != MOSS_MOMENT_TYPE:
        raise ValueError(f"record {record.id!r} has payload type {record.type!r}, not {MOSS_MOMENT_TYPE!r}")
    return Moment.model_validate(record.payload)


def update_moment(
    branch: MementoBranch, moment: Moment, *, threads: Sequence[str] = (), by: str = ""
) -> MomentRecord:
    """便捷入口: 编码 + 写入 staging. 退化态 (蠢记忆) 的日常写入面."""
    record = moment_to_record(moment, threads=threads, by=by)
    branch.update(record)
    return record


# ── merge ≡ 带引用的 Message ──


def commit_summary_message(branch: MementoBranch, view: CommitView) -> Message:
    """commit 的折叠展示形态: content = 当前释义正文, additional 携带 MementoRef."""
    msg = Message.new(tag="commit-summary").with_content(view.summary())
    _ref_of(branch, view).set(msg)
    return msg


def make_merge_message(branch: MementoBranch, commit_id: str) -> Message:
    """
    把一个 commit 包装成 Message — 孔径一 (输入队列) 的载体.
    旁路 commit -> 主路收带 memento.ref 的 Message -> mindflow 仲裁.
    :raise CommitNotFoundError: commit_id 不在该 branch 历史中.
    """
    view = branch.get_commit(commit_id)
    if view is None:
        raise CommitNotFoundError(f"commit {commit_id!r} not in history of branch {branch.meta.branch_id}")
    return commit_summary_message(branch, view)


# ── 窗口渲染 ──


def window_messages(branch: MementoBranch, window: BranchWindow) -> Iterable[Message]:
    """
    BranchWindow -> 模型可读消息序列 (时间升序):
    折叠区每个 commit 一条 commit-summary (带 MementoRef, 可 `show <commit_id>` 缺页展开),
    明细区按 Moment 历史语义展开; 非 moss.moment payload 降级为 payload JSON 文本.
    """
    for view in window.summaries:
        yield commit_summary_message(branch, view)
    for record in window.details:
        if record.type == MOSS_MOMENT_TYPE:
            yield from record_to_moment(record).as_history_messages()
        else:
            yield Message.new(tag="memento-record").with_content(
                json.dumps(record.payload, ensure_ascii=False)
            )
