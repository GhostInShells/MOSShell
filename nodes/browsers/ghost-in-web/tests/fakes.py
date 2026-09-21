"""Shared test doubles."""

from __future__ import annotations


class RecordingDispatcher:
    """Stands in for the WS server. Records actions; returns a scripted result."""

    def __init__(self, result: dict | None = None) -> None:
        self.actions: list[tuple[str, str, object]] = []
        self.said: list[tuple[str, str]] = []
        self._result = result or {"ok": True, "result": "done"}

    async def send_action(self, label: str, action: str, value: object = None) -> dict:
        self.actions.append((label, action, value))
        return dict(self._result)

    async def say(self, label: str, text: str) -> None:
        self.said.append((label, text))
