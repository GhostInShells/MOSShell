"""Shared state — the single source of truth between Matrix and Reflex UI."""

import time
from typing import Optional

import pydantic
import reflex as rx


class CommandRecord(pydantic.BaseModel):
    """A single command in the desktop activity stream."""

    model_config = {"extra": "forbid"}

    id: str
    channel_path: str = ""
    command_name: str = ""
    summary: str = ""
    status: str = "pending"
    created_at: float = pydantic.Field(default_factory=time.time)
    stale: bool = False
    payload: dict = pydantic.Field(default_factory=dict)
    result: Optional[str] = None
    approval_prompt: Optional[str] = None
    human_reply: Optional[str] = None


class DesktopState(rx.State):
    """Root state for the desktop GUI."""

    commands: list[CommandRecord] = []
    selected_id: str = ""
    show_stale: bool = False

    @rx.var
    def visible_commands(self) -> list[CommandRecord]:
        if self.show_stale:
            return self.commands
        return [c for c in self.commands if not c.stale]

    @rx.var
    def selected_command(self) -> Optional[CommandRecord]:
        for c in self.commands:
            if c.id == self.selected_id:
                return c
        return None

    @rx.event
    def select_command(self, command_id: str):
        self.selected_id = command_id

    @rx.event
    def toggle_stale(self):
        self.show_stale = not self.show_stale
