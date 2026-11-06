"""The 0.2.0 wire shape — how a flat dsh envelope becomes our model, and back.

dsh sends events as one flat dict: `type` / `seq` / `time` beside the payload, with a couple of
optional keys (`ignorable`, `surfaceOp`, `sourceEventSeqs`) that are themselves envelope-level.
Our model splits that into `meta` and `data`, so something has to do the split — and *this* is
that something, for this version.

Why it lives here and not on the model:

- the split is a fact about **this** dsh's wire, not about the idea of a session event. A future
  version that sends something else changes this file and leaves the model alone;
- the model stays a declaration. When the conversion lives inside the class, every model grows a
  second life as a converter, and the layer that is supposed to be pure data starts carrying
  behaviour that a generation has to override;
- the next version will be built by copying this one. Copying a `wire.py` is a smaller, sharper
  act than copying a model and then editing the behaviour out of it.

`model_from_event` / `model_to_event` are the demo of the other half: a typed subclass declares
an `event_type()` discriminator, and the conversion dispatches on it. That is the pattern to
copy when a new event type arrives — declare the class, do not touch this file.

The one exception is `ToolResultEvent`: dsh's payload carries its own `meta` (tool-private
presentation data) which collides with the envelope's `meta` by name. That rename is handled in
the two functions below, which is another argument for keeping conversion outside the classes —
the workaround stays in one place, visible, instead of being an override buried in a subclass.
"""

from __future__ import annotations

from typing import Any, TypeVar

from ghoshell_dsh.blueprint.models.session_events import (
    SessionEvent,
    SessionEventMeta,
    SessionEventModel,
    ToolResultEvent,
)

__all__ = ["event_from_dict", "event_to_dict", "model_from_event", "model_to_event"]

_E = TypeVar("_E", bound=SessionEventModel)


def event_from_dict(d: dict) -> SessionEvent:
    """Build the envelope from dsh's flat dict; `data` is kept exactly as sent."""
    meta = SessionEventMeta(
        type=d.get("type", ""),
        seq=d.get("seq", 0),
        time=d.get("time", 0),
        ignorable=d.get("ignorable"),
        surfaceOp=d.get("surfaceOp"),
        sourceEventSeqs=d.get("sourceEventSeqs"),
    )
    return SessionEvent(meta=meta, data=d.get("data", {}) or {})


def event_to_dict(event: SessionEvent) -> dict:
    """Flatten the envelope back into dsh's shape. The inverse of `event_from_dict`."""
    d: dict[str, Any] = {
        "type": event.meta.type,
        "seq": event.meta.seq,
        "time": event.meta.time,
        "data": event.data,
    }
    if event.meta.ignorable is not None:
        d["ignorable"] = event.meta.ignorable
    if event.meta.surfaceOp is not None:
        d["surfaceOp"] = event.meta.surfaceOp
    if event.meta.sourceEventSeqs is not None:
        d["sourceEventSeqs"] = event.meta.sourceEventSeqs
    return d


def model_from_event(model_cls: type[_E], event: SessionEvent) -> _E | None:
    """Project an envelope onto a typed model, by its `event_type()` discriminator.

    Returns `None` when the names differ — no parsing is attempted on a payload known to be
    something else, which is what keeps an unknown event from raising inside a dispatcher.
    """
    if event.meta.type != model_cls.event_type():
        return None

    if model_cls is ToolResultEvent:
        data = dict(event.data)
        tool_meta = data.pop("meta", None)
        obj = model_cls.model_validate(data)
        obj.tool_meta = tool_meta
        obj.meta = event.meta
        return obj  # type: ignore[return-value]

    obj = model_cls.model_validate(event.data)
    obj.meta = event.meta
    return obj


def model_to_event(model: SessionEventModel) -> SessionEvent:
    """Put a typed payload back into the envelope."""
    data = model.model_dump(exclude={"meta"}, exclude_none=True)
    if isinstance(model, ToolResultEvent):
        data.pop("tool_meta", None)
        if model.tool_meta is not None:
            data["meta"] = model.tool_meta
    return SessionEvent(meta=model.meta, data=data)
