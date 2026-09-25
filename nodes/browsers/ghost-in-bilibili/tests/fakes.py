"""Test doubles for the transport boundary."""

from __future__ import annotations


class Recorder:
    """Fake Dispatcher — records outbound commands, returns a scripted result."""

    def __init__(self, result: dict | None = None) -> None:
        self.actions: list[tuple[str, str, object | None]] = []
        self.saids: list[tuple[str, str]] = []
        self.result = result or {"ok": True, "result": "ok"}

    async def send_action(self, label, action, value=None):
        self.actions.append((label, action, value))
        return dict(self.result)

    async def say(self, label, text):
        self.saids.append((label, text))
        return None
