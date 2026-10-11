"""Stand-ins for the surface and the signaler — both injected, so tests need
nothing from a running node."""

from __future__ import annotations


class Recorder:
    """Collects the frames the channel broadcasts and the signals it sends."""

    def __init__(self) -> None:
        self.frames: list[dict] = []
        self.signals: list[object] = []
        self.rail: str = ""

    async def broadcast(self, frame: dict) -> None:
        self.frames.append(frame)

    def rail_state(self) -> str:
        """Stand in for the human-side sidebar state the surface reports."""
        return self.rail

    def __call__(self, signal: object) -> None:
        self.signals.append(signal)

    def types(self) -> list[str]:
        return [f["type"] for f in self.frames]

    def of(self, kind: str) -> list[dict]:
        return [f for f in self.frames if f["type"] == kind]
