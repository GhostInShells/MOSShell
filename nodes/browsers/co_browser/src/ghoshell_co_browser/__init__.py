"""Co-browser — supervised playwright node (channel + web surface + frame store)."""

from __future__ import annotations

from .channel import build_co_browser_channel
from .frame import Frame, FrameKind, FrameState
from .store import FrameStore
from .surface import CoBrowserSurface

__all__ = [
    "Frame",
    "FrameKind",
    "FrameState",
    "FrameStore",
    "CoBrowserSurface",
    "build_co_browser_channel",
]
