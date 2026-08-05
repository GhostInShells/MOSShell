"""Event handler registry — dispatches typed lifecycle events to registered EventHandlers."""
from __future__ import annotations

from typing import Callable

from ghoshell_moss.host.listener.contracts import (
    EventHandler,
    VoiceLifecycleEvent,
    StreamStateChanged,
    AsrPartial,
    AsrFinal,
    BufferUpdated,
)

Disposer = Callable[[], None]

# Per-event-type dispatch table — one key per concrete event class.
_DISPATCH: dict[type[VoiceLifecycleEvent], str] = {
    StreamStateChanged: "on_stream_state_changed",
    AsrPartial: "on_asr_partial",
    AsrFinal: "on_asr_final",
    BufferUpdated: "on_buffer_updated",
}


class EventBus:
    """Typed event dispatcher — one EventHandler method per event type."""

    def __init__(self) -> None:
        self._handlers: list[EventHandler] = []

    def add(self, handler: EventHandler) -> Disposer:
        self._handlers.append(handler)

        def _remove() -> None:
            self._handlers.remove(handler)

        return _remove

    def dispatch(self, event: VoiceLifecycleEvent) -> None:
        method = _DISPATCH.get(type(event))
        if method is None:
            return
        for h in self._handlers:
            getattr(h, method)(event)
