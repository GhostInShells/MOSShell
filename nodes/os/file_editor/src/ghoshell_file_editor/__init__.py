"""ghoshell_file_editor — the OS-control file editor node.

Axis-1 (this package): pure data structures + the store, no IO beyond the draft
working copies. Axis-2 (protocol) and axis-3 (UI) build on top.
"""

from .store import DocStore
from .structure import (
    SIDE_EFFECTS,
    STREAMING_KINDS,
    Action,
    ActionState,
    Author,
    Dialogue,
    Effect,
    Kind,
    Thread,
    ThreadState,
    content_at,
    diff_of,
    effect_of,
    line_count,
    render_source,
    replace_once,
    side_effect,
    slice_region,
)

__all__ = [
    "Action",
    "ActionState",
    "Author",
    "Dialogue",
    "DocStore",
    "Effect",
    "Kind",
    "SIDE_EFFECTS",
    "STREAMING_KINDS",
    "Thread",
    "ThreadState",
    "content_at",
    "diff_of",
    "effect_of",
    "line_count",
    "render_source",
    "replace_once",
    "side_effect",
    "slice_region",
]
