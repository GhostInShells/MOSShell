"""ghoshell_file_editor — the OS-control file editor node.

Axis-1 (this package): pure data structures + a durable store, no IO beyond
the optional append-only log. Axis-2 (protocol) and axis-3 (UI) build on top.
"""

from .structure import (
    BASE,
    Action,
    ActionState,
    Anchor,
    Author,
    Effect,
    Kind,
    Reply,
    Seq,
    Thread,
    Verdict,
    cascade_seqs,
    changes_content,
    compute_effect,
    diff_of,
    effect_of,
    rewind_target,
    tail_content,
)
from .store import ThreadStore

__all__ = [
    "BASE",
    "Action",
    "ActionState",
    "Anchor",
    "Author",
    "Effect",
    "Kind",
    "Reply",
    "Seq",
    "Thread",
    "ThreadStore",
    "Verdict",
    "cascade_seqs",
    "changes_content",
    "compute_effect",
    "diff_of",
    "effect_of",
    "rewind_target",
    "tail_content",
]
