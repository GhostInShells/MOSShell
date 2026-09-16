"""Render-ready projections of store state (axis 2/3 shared).

The human surface and the model-facing channel project the same store; these
functions keep the wire shapes in one place. They are render-friendly — flat
dicts a UI can draw without re-deriving data, not raw object dumps.
"""

from __future__ import annotations

from typing import Any

from .store import ThreadStore


def action_view(thread_id: str, action) -> dict[str, Any]:
    effect = action.effect
    return {
        "thread": thread_id,
        "seq": action.seq.n,
        "kind": action.kind,
        "author": action.author,
        "description": action.description,
        "state": action.state,
        "verdict": action.verdict,
        "verdict_by": action.verdict_by,
        "effect": (
            {"before": effect.before, "after": effect.after, "diff": effect.diff}
            if effect is not None else None
        ),
        "replies": [
            {
                "n": r.n, "author": r.author, "anchor": r.anchor,
                "diff": r.diff, "text": r.text,
            }
            for r in action.replies
        ],
    }


def thread_view(thread) -> dict[str, Any]:
    head = thread.head
    return {
        "id": thread.id,
        "label": thread.label,
        "path": thread.path,
        "motivation": thread.motivation,
        "state": thread.state,
        "head": head.seq.n if head is not None else None,
        "actions": [action_view(thread.id, a) for a in thread.actions.values()],
    }


def snapshot(store: ThreadStore) -> dict[str, Any]:
    """The full state a connecting surface needs to render."""
    return {
        "type": "state",
        "threads": [thread_view(t) for t in store.threads()],
    }
