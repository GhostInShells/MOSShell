"""FrameStore — frames plus the master enabled flag.

Both faces (the channel and the web surface) read and write it. There is no
verdict rendezvous anymore: control is coarse-grained — the human flips the
whole channel off, or leaves it on. Fine-grained approval was removed in
favor of pure observation, since audit is what turns the trust question off.
"""

from __future__ import annotations

import time

from .frame import Frame, FrameKind, FrameState, TERMINAL_STATES

__all__ = ["FrameStore"]


_MAX_FRAMES = 200
"""Cap on frames kept in memory. Only settled frames are evicted — a running
frame is never dropped out from under its subprocess."""


class FrameStore:
    def __init__(self) -> None:
        self._frames: dict[int, Frame] = {}
        self._order: list[int] = []
        self._counter = 0
        self._enabled = True

    # -- master switch ------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, value: bool) -> bool:
        self._enabled = bool(value)
        return self._enabled

    # -- frames -------------------------------------------------------------

    def new_frame(self, kind: FrameKind, source: str) -> Frame:
        self._counter += 1
        frame = Frame(id=self._counter, kind=kind, source=source)
        self._frames[frame.id] = frame
        self._order.append(frame.id)
        self._trim()
        return frame

    def get(self, frame_id: int) -> Frame | None:
        return self._frames.get(frame_id)

    def frames(self) -> list[Frame]:
        return [self._frames[i] for i in self._order if i in self._frames]

    def set_state(self, frame_id: int, state: FrameState) -> Frame:
        frame = self._require(frame_id)
        frame.state = state
        frame.updated = time.time()
        if state in TERMINAL_STATES and frame.ended is None:
            frame.ended = frame.updated
        return frame

    def append_result(self, frame_id: int, text: str) -> Frame:
        frame = self._require(frame_id)
        frame.result = (frame.result + text) if frame.result else text
        frame.updated = time.time()
        return frame

    def _require(self, frame_id: int) -> Frame:
        frame = self._frames.get(frame_id)
        if frame is None:
            raise KeyError(f"no frame {frame_id}")
        return frame

    def _trim(self) -> None:
        while len(self._order) > _MAX_FRAMES:
            for i, fid in enumerate(self._order):
                frame = self._frames.get(fid)
                if frame is not None and frame.settled:
                    self._order.pop(i)
                    self._frames.pop(fid, None)
                    break
            else:
                return
