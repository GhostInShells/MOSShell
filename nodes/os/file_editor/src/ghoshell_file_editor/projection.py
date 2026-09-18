"""Render-ready projections of store state (axes 2 and 3 shared).

The human surface and the model-facing channel project the same store; these
functions keep the wire shapes in one place.

Two sizes of frame, deliberately: an **action** frame carries only identity and
the mechanical effect line — never content. The card is meant to say *what
happened*, not *what it says*; content arrives in a **detail** frame only when
the human clicks a card. That is what keeps a stream of long documents cheap.
"""

from __future__ import annotations

from typing import Any

from .store import DocStore
from .structure import Action, Thread, content_at, line_count, render_source, side_effect

__all__ = ["action_view", "thread_view", "detail_view", "snapshot"]

SNAPSHOT_ACTIONS = 60
"""How many of the most recent actions a connecting surface is handed."""


def _dialogue_view(action: Action) -> list[dict[str, Any]]:
    return [
        {"author": d.author, "text": d.text, "at": d.at} for d in action.dialogue
    ]


def action_view(action: Action) -> dict[str, Any]:
    """A card: identity plus the mechanical effect line. No content."""
    return {
        "n": action.n,
        "kind": action.kind,
        "author": action.author,
        "label": action.label,
        "state": action.state,
        "effect": side_effect(action),
        "at": action.at,
        "dialogue": _dialogue_view(action),
    }


def thread_view(thread: Thread) -> dict[str, Any]:
    """A thread's shape — everything but its text."""
    return {
        "id": thread.id,
        "label": thread.label,
        "path": thread.path,
        "state": thread.state,
        "auto": thread.auto,
        "version": thread.version,
        "lines": line_count(thread.content),
        "chars": len(thread.content),
        "exported_to": thread.exported_to,
    }


def detail_view(thread: Thread, action: Action) -> dict[str, Any]:
    """The three tabs of one card, resolved server-side.

    - ``source`` — the effect tab: markdown for the change this action made
    - ``full`` — the full-text tab: the document as this action left it
    - ``diff`` — the raw unified diff, for a reader who wants the mechanics
    """
    return {
        "thread": thread.id,
        "n": action.n,
        "source": render_source(action),
        "full": content_at(thread, action.n),
        "diff": action.effect.diff if action.effect is not None else "",
        "effect": side_effect(action),
    }


def snapshot(store: DocStore) -> dict[str, Any]:
    """The full state a connecting surface needs to render."""
    actions: list[dict[str, Any]] = []
    for thread in store.threads():
        for action in thread.actions:
            actions.append({"thread": thread.id, **action_view(action)})
    return {
        "type": "snapshot",
        "threads": [thread_view(t) for t in store.threads()],
        "actions": actions[-SNAPSHOT_ACTIONS:],
    }
