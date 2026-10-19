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
        self._cursors: dict = {}
        self.previews: list[dict] = []

    async def broadcast(self, frame: dict) -> None:
        self.frames.append(frame)

    def layout_state(self) -> tuple[str, str]:
        return self.layout, self.layout_by

    async def set_layout(self, mode: str) -> str:
        self.layout, self.layout_by = mode, "model"
        return mode

    async def set_cursor(
        self, thread: str, start: int, end: int | None = None, *, version: int = 0,
        by: str = "model",
    ) -> dict:
        lo = max(1, int(start))
        self._cursors[thread] = {
            "start": lo, "end": max(lo, int(end or lo)), "version": version, "by": by
        }
        return self._cursors[thread]

    def cursors(self) -> dict:
        return self._cursors

    def cursor_of(self, thread: str) -> dict | None:
        return self._cursors.get(thread)

    def view_at_state(self) -> tuple[str, int]:
        return getattr(self, "view_at", ("", 0))

    def selection_state(self) -> dict | None:
        return getattr(self, "selection", None)

    def format_state(self) -> tuple[str, str]:
        return getattr(self, "fmt", ("auto", "default"))

    async def set_format(self, mode: str, by: str = "model") -> str:
        chosen = (mode or "").strip().lower()
        # The real surface accepts every name in renderers/; the fake carries the
        # shipped library, which is what a test can see without a disk.
        known = {"auto", "raw", "tmp", "markdown", "code", "plain"}
        if chosen not in known:
            raise ValueError(f"format must be one of {tuple(sorted(known))}, got {mode!r}")
        self.fmt = (chosen, by)
        return chosen

    async def push_preview(self, thread: str, script: str, label: str = "") -> dict:
        self.previews.append({"thread": thread, "script": script, "label": label})
        return {"seq": len(self.previews), "label": label or "草稿 1", "script": script}

    def drafts(self, thread: str) -> list[dict]:
        return [p for p in self.previews if p["thread"] == thread]

    def draft_threads(self) -> list[str]:
        return sorted({p["thread"] for p in self.previews})

    def current_face(self, thread: str) -> dict:
        return {}

    async def show_draft(self, thread: str, seq: int) -> dict:
        raise ValueError("no drafts in the fake")

    def list_renderers(self) -> list[dict]:
        return [
            {"name": "raw", "ext": [], "description": "带行号的原样文本", "file": "raw.js"},
            {"name": "markdown", "ext": [".md"], "description": "markdown", "file": "markdown.js"},
        ]

    async def set_selection(self, thread: str, start: int, end: int, text: str = "") -> None:
        self.selection = (
            {"thread": thread, "start": int(start), "end": int(end), "text": text[:400]}
            if thread and start > 0 and end >= start
            else None
        )

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
