"""Desktop GUI state — command stream, approval gates, dialogue."""

import difflib
import time
import uuid
from enum import Enum
from typing import Optional

import pydantic
import reflex as rx


# -- domain enums --

class CommandStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    COMPLETED = "completed"
    ERROR = "error"
    STALE = "stale"


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    STALE = "stale"


# -- domain models --

class CommandInvocation(pydantic.BaseModel):
    """A single command invocation from Ghost through the desktop channel."""

    model_config = {"extra": "forbid"}

    id: str
    channel_path: str       # "desktop.bash" | "desktop.file_editor"
    command_name: str        # "exec" | "view" | "str_replace" | "write" | "create" | "insert" | "undo_edit"
    params: dict = {}
    summary: str = ""
    status: CommandStatus = CommandStatus.PENDING
    created_at: float = pydantic.Field(default_factory=time.time)
    updated_at: float = pydantic.Field(default_factory=time.time)
    result: Optional[str] = None
    result_type: str = ""   # "stdout" | "stderr" | "file_diff" | "file_content" | "mixed"
    diff_opcodes: Optional[list[dict]] = None  # pre-computed diff for str_replace


class ApprovalRequest(pydantic.BaseModel):
    """Approval gate for a command that requires human decision."""

    model_config = {"extra": "forbid"}

    command_id: str
    prompt: str = ""
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: float = pydantic.Field(default_factory=time.time)
    resolved_at: Optional[float] = None
    human_reply: str = ""


class DialogueMessage(pydantic.BaseModel):
    """A single message in the human-Ghost dialogue thread for a command."""

    model_config = {"extra": "forbid"}

    id: str
    command_id: str
    sender: str  # "human" | "ghost"
    message: str
    timestamp: float = pydantic.Field(default_factory=time.time)


# -- Reflex state --

class DesktopState(rx.State):
    """Root state — commands stream, approval gates, dialogue threads."""

    commands: list[CommandInvocation] = []
    approvals: list[ApprovalRequest] = []
    dialogues: list[DialogueMessage] = []

    selected_id: str = ""
    show_stale: bool = False
    approval_reason: str = ""
    dialogue_input: str = ""

    # mock injection form fields (dev)
    mock_exec_cmd: str = "ls -la"
    mock_exec_approval: bool = False
    mock_str_replace_path: str = "/etc/config.yaml"
    mock_view_path: str = "/etc/config.yaml"

    # -- computed --

    @rx.var
    def visible_commands(self) -> list[CommandInvocation]:
        if self.show_stale:
            return sorted(self.commands, key=lambda c: c.created_at, reverse=True)
        return sorted(
            [c for c in self.commands if c.status != CommandStatus.STALE],
            key=lambda c: c.created_at, reverse=True,
        )

    @rx.var
    def selected_command(self) -> Optional[CommandInvocation]:
        for c in self.commands:
            if c.id == self.selected_id:
                return c
        return None

    @rx.var
    def selected_approval(self) -> Optional[ApprovalRequest]:
        for a in self.approvals:
            if a.command_id == self.selected_id:
                return a
        return None

    @rx.var
    def selected_dialogues(self) -> list[DialogueMessage]:
        return [d for d in self.dialogues if d.command_id == self.selected_id]

    @rx.var
    def pending_approval_count(self) -> int:
        return sum(1 for a in self.approvals if a.status == ApprovalStatus.PENDING)

    # -- actions --

    @rx.event
    def select_command(self, command_id: str):
        self.selected_id = command_id

    @rx.event
    def toggle_stale(self, checked: bool):
        self.show_stale = checked

    @rx.event
    def set_approval_reason(self, value: str):
        self.approval_reason = value

    @rx.event
    def set_dialogue_input(self, value: str):
        self.dialogue_input = value

    # mock form handlers
    @rx.event
    def on_mock_exec_cmd_change(self, value: str):
        self.mock_exec_cmd = value

    @rx.event
    def on_mock_exec_approval_change(self, value: bool):
        self.mock_exec_approval = value

    @rx.event
    def on_mock_str_replace_path_change(self, value: str):
        self.mock_str_replace_path = value

    @rx.event
    def on_mock_view_path_change(self, value: str):
        self.mock_view_path = value

    # -- approval actions --

    @rx.event
    def approve(self):
        cmd = self._lookup_command(self.selected_id)
        if cmd is None:
            return
        now = time.time()
        self._update_command(self.selected_id, status=CommandStatus.APPROVED, updated_at=now)
        self._resolve_approval(self.selected_id, ApprovalStatus.APPROVED, now)
        self.approval_reason = ""

    @rx.event
    def reject(self):
        cmd = self._lookup_command(self.selected_id)
        if cmd is None:
            return
        now = time.time()
        self._update_command(self.selected_id, status=CommandStatus.REJECTED, updated_at=now)
        self._resolve_approval(self.selected_id, ApprovalStatus.REJECTED, now)
        self.approval_reason = ""

    @rx.event
    def send_dialogue(self):
        if not self.dialogue_input.strip():
            return
        msg = DialogueMessage(
            id=uuid.uuid4().hex[:12],
            command_id=self.selected_id,
            sender="human",
            message=self.dialogue_input.strip(),
        )
        self.dialogues = self.dialogues + [msg]
        self.dialogue_input = ""

    # -- mock command injection (dev only) --

    @rx.event
    def inject_mock_command(self, channel_path: str, command_name: str, params_str: str, needs_approval: bool):
        import json

        try:
            params = json.loads(params_str) if params_str.strip() else {}
        except json.JSONDecodeError:
            params = {"raw": params_str}

        cmd_id = uuid.uuid4().hex[:12]
        now = time.time()
        summary = self._make_summary(channel_path, command_name, params)

        if needs_approval:
            status = CommandStatus.AWAITING_APPROVAL
            approval = ApprovalRequest(
                command_id=cmd_id,
                prompt=f"Allow Ghost to execute {channel_path}:{command_name}?",
                status=ApprovalStatus.PENDING,
                created_at=now,
            )
            self.approvals = self.approvals + [approval]
        else:
            status = CommandStatus.COMPLETED
            approval = None

        # pre-compute diff for str_replace (server-side, since difflib can't run in render)
        diff = None
        if command_name == "str_replace":
            old_str = params.get("old_str", "")
            new_str = params.get("new_str", "")
            if old_str or new_str:
                diff = self._compute_diff_opcodes(old_str, new_str)

        cmd = CommandInvocation(
            id=cmd_id,
            channel_path=channel_path,
            command_name=command_name,
            params=params,
            summary=summary,
            status=status,
            created_at=now,
            updated_at=now,
            result=self._mock_result(channel_path, command_name, params) if not needs_approval else None,
            result_type=self._mock_result_type(channel_path, command_name) if not needs_approval else "",
            diff_opcodes=diff,
        )
        self.commands = self.commands + [cmd]
        if not self.selected_id:
            self.selected_id = cmd_id

    # -- internal helpers --

    def _lookup_command(self, cmd_id: str) -> Optional[CommandInvocation]:
        for c in self.commands:
            if c.id == cmd_id:
                return c
        return None

    def _update_command(self, cmd_id: str, **kwargs):
        self.commands = [
            (c.model_copy(update=kwargs) if c.id == cmd_id else c)
            for c in self.commands
        ]

    def _resolve_approval(self, cmd_id: str, status: ApprovalStatus, now: float):
        self.approvals = [
            (
                a.model_copy(update={
                    "status": status,
                    "resolved_at": now,
                    "human_reply": self.approval_reason,
                })
                if a.command_id == cmd_id else a
            )
            for a in self.approvals
        ]

    @staticmethod
    def _make_summary(channel_path: str, command_name: str, params: dict) -> str:
        if channel_path == "desktop.bash":
            if command_name == "exec":
                return params.get("text__", "")[:80]
            if command_name == "run":
                return f"[bg] {params.get('text__', '')[:60]}"
        if channel_path == "desktop.file_editor":
            path = params.get("path", "?")
            if command_name == "view":
                return f"View {path}"
            if command_name == "str_replace":
                return f"Edit {path}"
            if command_name == "write":
                return f"Write {path}"
            if command_name == "create":
                return f"Create {path}"
            if command_name == "insert":
                return f"Insert into {path}"
        return f"{channel_path}:{command_name}"

    @staticmethod
    def _mock_result(channel_path: str, command_name: str, params: dict) -> str:
        if channel_path == "desktop.bash" and command_name == "exec":
            return "$ ls -la\ntotal 42\ndrwxr-xr-x  5 user  staff   160 Jul 23 10:00 .\ndrwxr-xr-x 20 user  staff   640 Jul 23 09:00 ..\n-rw-r--r--  1 user  staff  1234 Jul 20 15:30 config.yaml"
        if channel_path == "desktop.file_editor":
            path = params.get("path", "/unknown")
            if command_name == "view":
                return f"  1  # Configuration File\n  2  server:\n  3    host: 0.0.0.0\n  4    port: 8080\n  5  database:\n  6    url: postgresql://localhost:5432/mydb"
            if command_name == "str_replace":
                return "File edited: 2 replacements in 1 section."
            if command_name == "write":
                return f"File written: {path} (1.2KB)"
            if command_name == "create":
                return f"File created: {path}"
        return "(mock result)"

    @staticmethod
    def _mock_result_type(channel_path: str, command_name: str) -> str:
        if channel_path == "desktop.bash":
            return "stdout"
        if command_name == "view":
            return "file_content"
        if command_name == "str_replace":
            return "file_diff"
        if command_name in ("write", "create"):
            return "file_content"
        return "stdout"

    @staticmethod
    def _compute_diff_opcodes(old: str, new: str) -> list[dict]:
        matcher = difflib.SequenceMatcher(None, old, new)
        result = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            old_lines = old[i1:i2].splitlines() or [""]
            new_lines = new[j1:j2].splitlines() or [""]
            result.append({
                "tag": tag,
                "old_text": "\n".join(old_lines),
                "new_text": "\n".join(new_lines),
            })
        return result
