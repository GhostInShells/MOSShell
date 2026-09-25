"""PushSession — the first-class object the whole node revolves around.

A push is not a command; it is a **session**: a named, addressable thing that is
born (requested), lives (approved and streaming), and can be revoked by either
face. That is the structural fix over the terminal card — a card is a one-shot
proposal, a session is long-lived and keeps an identity until it is stopped.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

__all__ = ["SessionState", "PushSession", "TERMINAL_STATES"]

TERMINAL_STATES = {"live", "stopped", "denied", "failed"}


class SessionState:
    """Session states as plain string constants (wire format is the string).

    ``requested`` is a human-side action (the model called request); ``pending``
    is the model-facing name for the same moment. The two are collapsed into one
    state — ``pending`` — because there is no information in the distinction once
    the request is in the store.
    """

    PENDING = "pending"    # asked for, waiting on a human verdict
    LIVE = "live"          # approved, producer subprocess streaming
    STOPPED = "stopped"    # someone stopped it (either face)
    DENIED = "denied"      # the human refused
    FAILED = "failed"      # approved but the producer died / could not start


@dataclass
class PushSession:
    """One push: who asked, what source, and where it is now.

    ``owner`` distinguishes the two kinds of sessions the store holds: a session
    the ghost requested (``owner="model"``) versus one the human opened directly
    (``owner="human"``). The model may stop the former, never the latter.
    """

    id: int
    source: str
    label: str = ""
    owner: str = "model"
    description: str = ""
    state: str = SessionState.PENDING
    fps: float = 10.0
    max_width: int = 1280
    quality: int = 5
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    settled: Optional[float] = None
    process_index: Optional[int] = None
    failure: str = ""
    last_frame_at: Optional[float] = None

    def touch(self) -> None:
        self.updated = time.time()

    def settle(self, state: str) -> None:
        self.state = state
        self.touch()
        if state in TERMINAL_STATES and self.settled is None:
            self.settled = time.time()

    def view(self) -> dict:
        """The wire shape both faces render. The live stream URL is *not* here —
        it is a property of the surface (the node's own HTTP endpoint), not of a
        session."""
        return {
            "id": self.id,
            "source": self.source,
            "label": self.label,
            "owner": self.owner,
            "description": self.description,
            "state": self.state,
            "fps": self.fps,
            "max_width": self.max_width,
            "quality": self.quality,
            "created": self.created,
            "updated": self.updated,
            "failure": self.failure,
            "has_frame": self.last_frame_at is not None,
        }
