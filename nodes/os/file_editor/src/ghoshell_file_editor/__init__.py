"""ghoshell_file_editor — the OS-control file editor node.

Axis-1 (this package): pure data structures + a durable store, no IO beyond
the optional append-only log. Axis-2 (protocol) and axis-3 (UI) build on top.
"""

from .structure import (
    Action,
    Anchor,
    Author,
    Effect,
    Kind,
    Reply,
    Seq,
    Thread,
    Version,
    cascade_seqs,
    diff_of,
    effect_of,
    is_mutating,
    result_content,
)
from .store import ThreadStore

__all__ = [
    "Action",
    "Anchor",
    "Author",
    "Effect",
    "Kind",
    "Reply",
    "Seq",
    "Thread",
    "ThreadStore",
    "Version",
    "cascade_seqs",
    "diff_of",
    "effect_of",
    "is_mutating",
    "result_content",
]
