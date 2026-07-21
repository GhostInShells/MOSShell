"""
Memento — 轨迹第一公民的认知基建.

契约层: FORMAT.md (磁盘格式) + abc.py (模型/ABC/Protocol). 人类 review 冻结.
主权层: fs_memento (参考实现) / porcelain (MOSS 强类型桥) / witness (git 见证) —
  v2 abc 重写后待重做.
"""

from ghoshell_moss.memento.abc import (
    # trailer
    TRAILER_KIND,
    TRAILER_MEMENTO_REF,
    TRAILER_RESUMES,
    TRAILER_SUSPENDS,
    TRAILER_THREAD,
    join_trailers,
    split_trailers,
    trailer_values,
    # id
    new_commit_id,
    # 数据模型
    BranchRef,
    BranchWindow,
    Commit,
    CommitDetail,
    CommitNote,
    CommitRef,
    CommitView,
    MomentRecord,
    # hook
    MementoHooks,
    NullHooks,
    # Protocol / ABC
    Line,
    Memento,
    # 异常
    CommitNotFoundError,
    EmptyStagingError,
    LineNotFoundError,
    MementoError,
    MomentFrozenError,
    MomentNotInCommitError,
    ReadonlyLineError,
)

from ghoshell_moss.memento.fs_memento import FsMemento, new_filesystem_memento
from ghoshell_moss.memento.porcelain import (
    MOSS_MOMENT_TYPE,
    MementoRef,
    commit_summary_message,
    make_merge_message,
    moment_to_record,
    record_to_moment,
    update_moment,
    window_messages,
)
# ── 主权层 (witness 待重做) ──
# from ghoshell_moss.memento.witness import Witness, ensure_witness_repo, snapshot

__all__ = [
    "MomentRecord",
    "Commit",
    "CommitNote",
    "CommitView",
    "BranchRef",
    "CommitRef",
    "BranchWindow",
    "CommitDetail",
    "MementoHooks",
    "NullHooks",
    "Line",
    "Memento",
    "MementoError",
    "ReadonlyLineError",
    "LineNotFoundError",
    "CommitNotFoundError",
    "MomentFrozenError",
    "MomentNotInCommitError",
    "EmptyStagingError",
    "split_trailers",
    "join_trailers",
    "trailer_values",
    "new_commit_id",
    "TRAILER_THREAD",
    "TRAILER_RESUMES",
    "TRAILER_SUSPENDS",
    "TRAILER_KIND",
    "TRAILER_MEMENTO_REF",
    # fs
    "FsMemento",
    "new_filesystem_memento",
    # porcelain
    "MOSS_MOMENT_TYPE",
    "MementoRef",
    "moment_to_record",
    "record_to_moment",
    "update_moment",
    "make_merge_message",
    "commit_summary_message",
    "window_messages",
]
