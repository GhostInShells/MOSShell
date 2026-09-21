"""Stand-ins for the surface and the signaler — both injected, so tests need
nothing from a running node."""

from __future__ import annotations


class FakeSurface:
    def __init__(self) -> None:
        self.frames: list[dict] = []
        self.url = "http://127.0.0.1:0"
        self.get_state = None
        self.on_action = None

    async def broadcast(self, frame: dict) -> None:
        self.frames.append(frame)

    async def stop(self) -> None:
        pass

    def types(self) -> list[str]:
        return [f["type"] for f in self.frames]

    def of(self, kind: str) -> list[dict]:
        return [f for f in self.frames if f["type"] == kind]


class SignalRecorder:
    def __init__(self) -> None:
        self.signals: list[object] = []

    def __call__(self, signal: object) -> None:
        self.signals.append(signal)
