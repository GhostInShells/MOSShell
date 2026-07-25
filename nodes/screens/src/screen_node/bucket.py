"""EventBucket — thread-safe peek/drain event buffer.

GUI thread pushes events. Matrix thread peeks (non-destructive tail read)
for context_messages, or drains (destructive) via explicit command.

Pattern: g1 listener bucket (peek/drain dual-face, janus.Queue single-point
transfer), applied here to screen interaction events instead of audio.
"""

import threading
import time

import janus

# Max events retained for peek before old ones are trimmed.
_MAX_BUFFER = 200

# Max size of the drain signal queue.
_DRAIN_Q_MAXSIZE = 512


class EventBucket:
    """Thread-safe event buffer with peek (read-only) and drain semantics."""

    def __init__(self):
        self._events: list[dict] = []
        self._lock = threading.Lock()
        self._drain_q: janus.Queue | None = None

    # ---- lifecycle (called from Matrix thread in channel startup) ----------

    def start(self) -> None:
        if self._drain_q is None:
            self._drain_q = janus.Queue(maxsize=_DRAIN_Q_MAXSIZE)

    # ---- push (GUI thread) -------------------------------------------------

    def push(self, event_type: str, **kwargs) -> None:
        """GUI thread: record an event."""
        event = {"type": event_type, "timestamp": time.time(), **kwargs}
        with self._lock:
            self._events.append(event)
            if len(self._events) > _MAX_BUFFER:
                self._events = self._events[-(_MAX_BUFFER // 2):]

        # Signal drain availability (fire-and-forget, don't block GUI thread)
        if self._drain_q is not None:
            try:
                self._drain_q.sync_q.put_nowait(event)
            except janus.SyncQueueFull:
                pass

    # ---- peek (Matrix thread, non-destructive) -----------------------------

    def peek(self, n: int = 10) -> list[dict]:
        """Matrix thread: read tail-N events without consuming them."""
        with self._lock:
            return self._events[-n:]

    # ---- drain (Matrix thread, destructive) --------------------------------

    async def drain(self, timeout: float = 0.05) -> list[dict]:
        """Matrix thread: drain all accumulated events via queue."""
        if self._drain_q is None:
            return []
        drained = []
        while True:
            try:
                event = self._drain_q.async_q.get_nowait()
                drained.append(event)
            except janus.QueueEmpty:
                break
        return drained

    # ---- snapshot -----------------------------------------------------------

    def snapshot(self) -> dict:
        """Thread-safe snapshot for bridge state reads."""
        with self._lock:
            return {
                "event_count": len(self._events),
                "recent": self._events[-5:],
            }
