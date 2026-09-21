"""Frame — one exec worth of observation.

The unit is intentionally lightweight: a frame is what the model just did.
The surface shows it, the human sees it. There is no verdict step — the
approval question was collapsed to a single master switch on the store.
"""

from __future__ import annotations

import time
from enum import Enum

from pydantic import BaseModel, Field

__all__ = ["Frame", "FrameKind", "FrameState", "TERMINAL_STATES"]


class FrameKind(str, Enum):
    EXEC = "exec"
    AEXEC = "aexec"


class FrameState(str, Enum):
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


TERMINAL_STATES: frozenset[FrameState] = frozenset({FrameState.DONE, FrameState.ERROR})


class Frame(BaseModel):
    id: int
    kind: FrameKind
    source: str
    state: FrameState = FrameState.RUNNING
    result: str = ""
    created: float = Field(default_factory=time.time)
    updated: float = Field(default_factory=time.time)
    ended: float | None = None

    @property
    def settled(self) -> bool:
        return self.state in TERMINAL_STATES

    def elapsed(self) -> float:
        end = self.ended if self.ended is not None else time.time()
        return round(end - self.created, 3)

    def view(self) -> dict:
        return self.model_dump(mode="json")
