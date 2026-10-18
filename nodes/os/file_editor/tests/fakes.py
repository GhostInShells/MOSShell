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

    async def set_cursor(
        self, thread: str, start: int, end: int | None = None, *, by: str = "model"
    ) -> dict:
        lo = max(1, int(start))
        self.cursor = {"thread": thread, "start": lo, "end": max(lo, int(end or lo))}
        self.cursor_by = by
        return self.cursor

    def cursor_state(self) -> tuple[str, int, int, str]:
        cur = getattr(self, "cursor", None) or {"thread": "", "start": 0, "end": 0}
        return (
            cur.get("thread", ""),
            int(cur.get("start", 0)),
            int(cur.get("end", 0)),
            getattr(self, "cursor_by", "default"),
        )

    def view_at_state(self) -> tuple[str, int]:
        return getattr(self, "view_at", ("", 0))

    async def set_view_at(self, thread: str, line: int) -> None:
        self.view_at = (thread, int(line))

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
