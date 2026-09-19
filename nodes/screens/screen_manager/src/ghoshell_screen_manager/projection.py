"""Render-ready projections of the window state — the wire shapes, in one place.

The channel and the surface project the same store; these functions keep the frame
shapes together so the two faces cannot drift. A frame carries what changed (a delta),
never the whole state; the whole state travels once, as a ``snapshot`` on connect.
"""

from __future__ import annotations

from dataclasses import asdict

from ghoshell_moss.types.topics.audio import AudioSampleTopic

from .model import Item, ScreenModel

__all__ = [
    "item_view",
    "snapshot",
    "open_frame",
    "close_frame",
    "arrange_frame",
    "activate_frame",
    "fullscreen_frame",
    "veil_frame",
    "audio_frame",
]


def item_view(item: Item) -> dict:
    return {"id": item.id, "url": item.url, "label": item.label, "group": item.group}


def _layout_view(model: ScreenModel) -> dict | None:
    if not model.active_items():
        return None
    layout = model.layout()
    return {
        "family": layout.family,
        "dir": layout.dir,
        "cols": layout.cols,
        "rows": layout.rows,
        "cells": [asdict(c) for c in layout.cells],
    }


def snapshot(model: ScreenModel) -> dict:
    """The full state a connecting surface needs to render."""
    groups = [
        {"name": g, "items": [item_view(model.get(i)) for i in model.group_items(g)]}
        for g in model.groups()
    ]
    return {
        "type": "snapshot",
        "active": model.active(),
        "fullscreen": model.fullscreen(),
        "layout": _layout_view(model),
        "groups": groups,
    }


def open_frame(item: Item) -> dict:
    return {"type": "open", "item": item_view(item)}


def close_frame(item_id: str) -> dict:
    return {"type": "close", "id": item_id}


def _state_view(model: ScreenModel) -> dict:
    """The active group's identity, order and resolved layout — the 'render this' trio."""
    return {
        "group": model.active(),
        "ids": model.active_items(),
        "layout": _layout_view(model),
    }


def arrange_frame(model: ScreenModel) -> dict:
    return {"type": "arrange", **_state_view(model)}


def activate_frame(model: ScreenModel, *, by_model: bool) -> dict:
    return {"type": "activate", "by_model": by_model, **_state_view(model)}


def fullscreen_frame(item_id: str | None) -> dict:
    return {"type": "fullscreen", "id": item_id}


def veil_frame(gesture: str, **fields: object) -> dict:
    return {"type": "veil", "gesture": gesture, **fields}


def audio_frame(sample: AudioSampleTopic) -> dict:
    return {"type": "audio", "sample": sample.model_dump(exclude={"meta"})}
