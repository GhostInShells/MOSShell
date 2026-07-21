"""Shared state — the single source of truth between Matrix and Reflex UI."""

import time
from enum import Enum
from typing import Optional

import reflex as rx


class CommandStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    COMPLETED = "completed"
    ERROR = "error"


class CommandRecord(rx.Base):
    """A single command in the desktop activity stream."""

    id: str
    channel_path: str = ""        # e.g. "desktop.bash", "desktop.file_editor"
    command_name: str = ""        # e.g. "exec", "view", "str_replace"
    summary: str = ""             # one-line description for the sidebar
    status: CommandStatus = CommandStatus.PENDING
    created_at: float = 0.0
    stale: bool = False
    # detail fields — populated as the command progresses
    payload: dict = {}
    result: Optional[str] = None
    approval_prompt: Optional[str] = None
    human_reply: Optional[str] = None

    def __init__(self, **kwargs):
        if "created_at" not in kwargs or kwargs["created_at"] == 0.0:
            kwargs["created_at"] = time.time()
        super().__init__(**kwargs)


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
