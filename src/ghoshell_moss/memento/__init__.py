"""Memento — the cognitive trajectory: branches, commits, and the notes that interpret them.

The contract fixes behavior, not storage: any implementation satisfying it is valid.
See ``SPECIFICATION.md`` for the full specification.
"""

from .abcd import (
    COMMIT_MEMENTO_FILE,
    BranchAddr,
    BranchEvent,
    Child,
    CommitRef,
    Confluence,
    Note,
    CommitMessage,
    BranchRef,
    ForkRef,
    BranchMeta,
    CommitView,
    BranchHistory,
    BranchView,
    Branch,
    Memento,
    MementoReadonlyError,
)
from ._fs_memento import (
    FsMemento,
    FsBranch,
    new_local_memento,
    new_local_memento_from_root,
    owner_dir_for,
)
