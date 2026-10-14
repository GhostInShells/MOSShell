"""Stand-ins for the surface and the signaler — both injected, so tests need
nothing from a running node."""

from __future__ import annotations


class Recorder:
    """Collects the frames the channel broadcasts and the signals it sends."""

    def __init__(self) -> None:
        self.frames: list[dict] = []
        self.signals: list[object] = []
        # the surface's presentation knob — how an open card is shown
        self.layout = "page"
        self.layout_by = "default"

    async def broadcast(self, frame: dict) -> None:
        self.frames.append(frame)

    def layout_state(self) -> tuple[str, str]:
        return self.layout, self.layout_by

    async def set_layout(self, mode: str) -> str:
        self.layout, self.layout_by = mode, "model"
        return mode

    def __call__(self, signal: object) -> None:
        self.signals.append(signal)

    def types(self) -> list[str]:
        return [f["type"] for f in self.frames]

    def of(self, kind: str) -> list[dict]:
        return [f for f in self.frames if f["type"] == kind]


def chunks(parts: list[str]):
    async def gen():
        for part in parts:
            yield part

    return gen()
