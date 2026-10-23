"""Stand-ins for the surface and the signaler — both injected, so tests need
nothing from a running node."""

from __future__ import annotations


class Recorder:
    """Collects the frames the channel broadcasts and the signals it sends."""

    def __init__(self) -> None:
        self.frames: list[dict] = []
        self.signals: list[object] = []
        self.rail: str = ""
        self.viewport: str = ""

    async def broadcast(self, frame: dict) -> None:
        self.frames.append(frame)

    def rail_state(self) -> str:
        """Stand in for the human-side sidebar state the surface reports."""
        return self.rail

    def viewport_state(self) -> str:
        """Stand in for the current screen window's size."""
        return self.viewport

    def __call__(self, signal: object) -> None:
        self.signals.append(signal)

    def types(self) -> list[str]:
        return [f["type"] for f in self.frames]

    def of(self, kind: str) -> list[dict]:
        return [f for f in self.frames if f["type"] == kind]


class ThemedSurface(Recorder):
    """A surface that can also read theme files and serve local pages.

    A theme may name the occupant of the desk, and that shape needs exactly two
    capabilities the recorder lacks: reading a theme's json, and turning a
    workspace path into a url the page can load.
    """

    def __init__(self) -> None:
        super().__init__()
        self.specs: dict[str, dict] = {}
        self.served: list[object] = []

    def themes(self) -> list[str]:
        return sorted(self.specs)

    def theme(self, name: str) -> dict | None:
        return self.specs.get(name)

    def add_local(self, path) -> str:
        self.served.append(path)
        return f"/local/{path}"
