"""Render-ready projections of the window state — the wire shapes, in one place.

The channel and the surface project the same store; these functions keep the frame
shapes together so the two faces cannot drift. A frame carries what changed (a
delta), never the whole state; the whole state travels once, as a ``snapshot`` on
connect.

Two surfaces to keep in step:

* the **stage** — the active group's items, in order, with their resolved layout
  and fullscreen. Rendered by the ``stage`` frame.
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
    "fullscreen_frame",
    "veil_frame",
    "audio_frame",
    "notice_frame",
]


def item_view(item: Item) -> dict:
    return {
        "id": item.id,
        "url": item.url,
        "label": item.label,
        "group": item.group,
        "icon": item.icon,
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
    """The 'render this' set: what is on stage, what floats, what is fullscreen."""
    return {
        "arena": model.arena(),
        "ids": model.arena_items(),
        "layout": _layout_view(model),
        "fullscreen": model.fullscreen(),
        "desktop": model.desktop_items(),
    }


def state_frame(model: ScreenModel) -> dict:
    """The active arrangement plus the floating desktop — one delta frame."""
    return {"type": "state", **_state(model)}


def snapshot(model: ScreenModel) -> dict:
    """The full state a connecting surface needs to render.

    ``items`` carries every pooled item's identity (url/label/icon), so the page
    can materialize an iframe for a desktop item it has never seen before — the
    ``desktop`` list alone is just ids.
    """
    groups = [
        {"name": g, "items": [item_view(model.get(i)) for i in model.group_items(g)]}
        for g in model.groups()
    ]
    return {
        "type": "snapshot",
        **_state(model),
        "items": [item_view(i) for i in model.items()],
        "groups": groups,
    }


def open_frame(item: Item) -> dict:
    return {"type": "open", "item": item_view(item)}


def close_frame(item_id: str) -> dict:
    return {"type": "close", "id": item_id}


def activate_frame(model: ScreenModel, *, by_model: bool) -> dict:
    return {"type": "activate", "by_model": by_model, **_state(model)}


def fullscreen_frame(item_id: str | None) -> dict:
    return {"type": "fullscreen", "id": item_id}


def veil_frame(gesture: str, **fields: object) -> dict:
    return {"type": "veil", "gesture": gesture, **fields}


def audio_frame(sample: AudioSampleTopic) -> dict:
    return {"type": "audio", "sample": sample.model_dump(exclude={"meta"})}


def notice_frame(text: str) -> dict:
    """A line for the human's activity log — service up/down, or the model acting."""
    return {"type": "notice", "text": text}
