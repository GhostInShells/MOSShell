"""Render-ready projections of the window state — the wire shapes, in one place.

The channel and the surface project the same store; these functions keep the frame
shapes together so the two faces cannot drift. A frame carries what changed (a
delta), never the whole state; the whole state travels once, as a ``snapshot`` on
connect.

Two surfaces to keep in step:

* the **stage** — the active group's items, in order, with their resolved layout
  and the focus. Rendered by the ``stage`` frame.
* the **desktop** — items in the pool but in no group, floating. Rendered by the
  ``desktop`` frame.

A single ``state`` frame carries both, because almost every mutation touches
both: arranging pulls items off the desktop, dismissing puts them back.
"""

from __future__ import annotations

from dataclasses import asdict

from ghoshell_moss.types.topics.audio import AudioSampleTopic

from .model import Item, ScreenModel

__all__ = [
    "item_view",
    "snapshot",
    "state_frame",
    "open_frame",
    "close_frame",
    "activate_frame",
    "focus_frame",
    "veil_frame",
    "audio_frame",
    "notice_frame",
    "face_frame",
]


def item_view(item: Item) -> dict:
    return {
        "id": item.id,
        "url": item.url,
        "label": item.label,
        "group": item.group,
        "icon": item.icon,
        # The face travels with the item's identity: the same node speaks the same
        # line in every view, so it is not a property of an arrangement.
        "face_line": item.face_line,
        "face_art": item.face_art,
    }


def _layout_view(model: ScreenModel) -> dict | None:
    layout = model.layout()
    if layout is None:
        return None
    return {
        "family": layout.family,
        "dir": layout.dir,
        "cols": layout.cols,
        "rows": layout.rows,
        "cells": [asdict(c) for c in layout.cells],
    }


def _state(model: ScreenModel) -> dict:
    """The 'render this' set: what is on stage, what floats, what is focused.

    ``groups`` rides along because it is a **place list the human reads**, not a
    model-side arrangement: the sidebar must stay correct when a group appears or
    empties itself, and the group list otherwise travels only once, in the
    ``snapshot`` on connect — a fresh group would be invisible until a reload.
    Names in creation order; membership is derivable from each item's ``group``.
    """
    return {
        "arena": model.arena(),
        "ids": model.arena_items(),
        "layout": _layout_view(model),
        "focus": model.focus(),
        "overlay": model.overlay(),
        "view_by": model.view_by(),
        "desktop": model.desktop_items(),
        "groups": model.groups(),
        # The pool order travels with the state rather than being re-derived on the
        # page: "what is near the eye" is the model's decision (``raise``), and a
        # page-side guess from insertion time would be a second authority over it.
        "order": model.pool_order(),
    }


def state_frame(model: ScreenModel) -> dict:
    """The active arrangement plus the floating desktop — one delta frame."""
    return {"type": "state", **_state(model)}


def snapshot(model: ScreenModel) -> dict:
    """The full state a connecting surface needs to render.

    ``items`` carries every pooled item's identity (url/label/icon and its
    ``group``), so the page can materialize an iframe for a desktop item it has
    never seen before — the ``desktop`` list alone is just ids. Group membership
    comes from those items; the group list itself is in ``groups``.
    """
    return {
        "type": "snapshot",
        **_state(model),
        "items": [item_view(i) for i in model.items()],
    }


def open_frame(item: Item) -> dict:
    return {"type": "open", "item": item_view(item)}


def close_frame(item_id: str) -> dict:
    return {"type": "close", "id": item_id}


def activate_frame(model: ScreenModel, *, by_model: bool) -> dict:
    return {"type": "activate", "by_model": by_model, **_state(model)}


def focus_frame(item_id: str | None) -> dict:
    """The view moved to one item (or left it) — a view event, not a layout one."""
    return {"type": "focus", "id": item_id}


def veil_frame(gesture: str, **fields: object) -> dict:
    return {"type": "veil", "gesture": gesture, **fields}


def audio_frame(sample: AudioSampleTopic) -> dict:
    return {"type": "audio", "sample": sample.model_dump(exclude={"meta"})}


def face_frame(item: Item) -> dict:
    """A tile's own line / picture changed — a delta, not a whole state.

    Faces change on their own clock (a card appears, a stream starts), so they
    get their own frame: re-sending the arrangement to deliver one line would
    make every node's idle chatter repaint the stage.
    """
    return {
        "type": "face",
        "id": item.id,
        "face_line": item.face_line,
        "face_art": item.face_art,
    }


def notice_frame(text: str) -> dict:
    """A line for the human's activity log — service up/down, or the model acting."""
    return {"type": "notice", "text": text}
