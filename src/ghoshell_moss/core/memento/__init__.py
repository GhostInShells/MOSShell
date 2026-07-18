"""
Memento — 轨迹第一公民的认知基建.

契约层: FORMAT.md (磁盘格式) + abc.py (信封模型 ABC). 人类 review 冻结.
主权层: fs_memento (参考实现) / porcelain (MOSS 强类型桥) / witness (git 见证).
"""

from ghoshell_moss.core.memento.abc import (
    TRAILER_KIND,
    TRAILER_MEMENTO_REF,
    TRAILER_RESUMES,
    TRAILER_SUSPENDS,
    TRAILER_THREAD,
    BasePointer,
    BranchMeta,
    BranchNotFoundError,
    BranchWindow,
    Commit,
    CommitKind,
    CommitNote,
    CommitNotFoundError,
    CommitView,
    EmptyStagingError,
    Memento,
    MementoBranch,
    MementoError,
    MementoHooks,
    MomentFrozenError,
    MomentNotInCommitError,
    MomentRecord,
    NullHooks,
    ReadonlyBranchError,
    join_trailers,
    new_branch_id,
    new_commit_id,
    split_trailers,
    trailer_values,
)
from ghoshell_moss.core.memento.fs_memento import (
    FsMemento,
    FsMementoBranch,
    new_filesystem_memento,
)
from ghoshell_moss.core.memento.porcelain import (
    MOSS_MOMENT_TYPE,
    MementoRef,
    commit_summary_message,
    make_merge_message,
    moment_to_record,
    record_to_moment,
    update_moment,
    window_messages,
)
from ghoshell_moss.core.memento.witness import Witness, ensure_witness_repo, snapshot

__all__ = [
    # 契约层
    "MomentRecord",
    "Commit",
    "CommitNote",
    "CommitView",
    "CommitKind",
    "BasePointer",
    "BranchMeta",
    "BranchWindow",
    "MementoHooks",
    "NullHooks",
    "MementoBranch",
    "Memento",
    "MementoError",
    "ReadonlyBranchError",
    "BranchNotFoundError",
    "CommitNotFoundError",
    "MomentFrozenError",
    "MomentNotInCommitError",
    "EmptyStagingError",
    "split_trailers",
    "join_trailers",
    "trailer_values",
    "new_commit_id",
    "new_branch_id",
    "TRAILER_THREAD",
    "TRAILER_RESUMES",
    "TRAILER_SUSPENDS",
    "TRAILER_KIND",
    "TRAILER_MEMENTO_REF",
    # fs 实现
    "FsMementoBranch",
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
    # 见证层
    "ensure_witness_repo",
    "snapshot",
    "Witness",
]
