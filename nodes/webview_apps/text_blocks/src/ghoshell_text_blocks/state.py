"""Text Blocks Reflex state — block list, dialogs, edit/submit flow."""

from __future__ import annotations

import difflib
import time
import uuid
from typing import Optional

import reflex as rx

# -- data models --


class BlockData(rx.Base):
    id: int
    title: str = ""
    content: str = ""
    source: str = ""        # "g" | "u"
    status: str = "sealed"  # "streaming" | "sealed" | "error"
    lock: str = ""           # "" | "g" | "u"
    version_count: int = 0
    created_at: float = 0.0


class ActionEntry(rx.Base):
    kind: str = ""
    summary: str = ""
    at: float = 0.0


# -- state --


class TextBlocksState(rx.State):
    blocks: list[BlockData] = []
    actions: list[ActionEntry] = []
    summary: str = "no blocks"
    surface_url: str = "http://127.0.0.1:8765"

    # dialog state
    dialog_mode: str = ""        # "new" | "edit" | "view"
    dialog_block_id: int = 0
    dialog_title: str = ""
    dialog_content: str = ""
    dialog_readonly: bool = False
    dialog_human_note: str = ""

    # streaming counter — increment to signal UI refresh
    _clock: int = 0
    _next_id: int = 1

    # -- computed --

    @rx.var
    def sorted_blocks(self) -> list[BlockData]:
        return sorted(self.blocks, key=lambda b: b.id)

    @rx.var
    def pending_diff_count(self) -> int:
        return len([a for a in self.actions if a.kind == "block_edit"])

    @rx.var
    def dialog_block(self) -> Optional[BlockData]:
        for b in self.blocks:
            if b.id == self.dialog_block_id:
                return b
        return None

    # -- life: model streaming simulation --

    @rx.event
    def tick(self):
        """polling hook — in S2 this drains the real store queue."""
        self._clock += 1

    # -- new block (human) --

    @rx.event
    def open_new_block_dialog(self):
        self.dialog_mode = "new"
        self.dialog_block_id = 0
        self.dialog_title = ""
        self.dialog_content = ""
        self.dialog_readonly = False
        self.dialog_human_note = ""

    @rx.event
    def set_new_title(self, value: str):
        self.dialog_title = value

    @rx.event
    def set_new_content(self, value: str):
        self.dialog_content = value

    @rx.event
    def create_block(self):
        if not self.dialog_content.strip():
            return
        title = self.dialog_title.strip() or _first_sentence(self.dialog_content)
        bid = self._next_id
        self._next_id += 1
        now = time.time()
        block = BlockData(
            id=bid, title=title, content=self.dialog_content.strip(),
            source="u", status="sealed", version_count=1,
            created_at=now,
        )
        self.blocks = self.blocks + [block]
        self._record("block_create", f"u: created #{bid} \"{title}\"")
        self._update_summary()
        self.dialog_mode = ""

    @rx.event
    def cancel_new_block(self):
        self.dialog_mode = ""

    # -- edit block (human) --

    @rx.event
    def open_edit_dialog(self, block_id: int):
        block = self._find(block_id)
        if block is None or block.lock == "g":
            return
        self.dialog_mode = "edit"
        self.dialog_block_id = block_id
        self.dialog_title = block.title
        self.dialog_content = block.content
        self.dialog_readonly = False
        self.dialog_human_note = ""

    @rx.event
    def set_edit_content(self, value: str):
        self.dialog_content = value

    @rx.event
    def set_edit_human_note(self, value: str):
        self.dialog_human_note = value

    @rx.event
    def submit_edit(self):
        block = self._find(self.dialog_block_id)
        if block is None:
            return
        if self.dialog_content == block.content:
            self.dialog_mode = ""
            return
        # compute diff for the action log
        diff_text = _compute_unified_diff(
            block.content, self.dialog_content,
            label=f"block #{block.id}",
        )
        # update block
        self.blocks = [
            (b.model_copy(update={
                "content": self.dialog_content,
                "version_count": b.version_count + 1,
            }) if b.id == block.id else b)
            for b in self.blocks
        ]
        self._record(
            "block_edit",
            f"u: edited #{block.id}\n{diff_text}",
        )
        self._update_summary()
        self.dialog_mode = ""

    @rx.event
    def cancel_edit(self):
        self.dialog_mode = ""

    # -- view block (streaming or read-only) --

    @rx.event
    def open_view_dialog(self, block_id: int):
        block = self._find(block_id)
        if block is None:
            return
        self.dialog_mode = "view"
        self.dialog_block_id = block_id
        self.dialog_title = block.title
        self.dialog_content = block.content
        self.dialog_readonly = True

    @rx.event
    def close_view_dialog(self):
        self.dialog_mode = ""

    # -- model simulation (S1 dev) --

    @rx.event
    def sim_model_create(self):
        """simulate a model creating a streaming block (S1 dev only)."""
        title = f"model draft {self._next_id}"
        bid = self._next_id
        self._next_id += 1
        now = time.time()
        block = BlockData(
            id=bid, title=title, content="model streaming simulation...",
            source="g", status="streaming", lock="g", version_count=1,
            created_at=now,
        )
        self.blocks = self.blocks + [block]
        self._record("block_create", f"g: created #{bid} (streaming)")
        self._update_summary()

    @rx.event
    def sim_model_seal(self, block_id: int):
        """simulate model sealing a block."""
        self.blocks = [
            (b.model_copy(update={"status": "sealed", "lock": ""})
             if b.id == block_id else b)
            for b in self.blocks
        ]
        self._record("block_edit", f"g: sealed #{block_id}")
        self._update_summary()

    # -- helpers --

    def _find(self, block_id: int) -> BlockData | None:
        for b in self.blocks:
            if b.id == block_id:
                return b
        return None

    def _record(self, kind: str, summary: str):
        entry = ActionEntry(kind=kind, summary=summary, at=time.time())
        self.actions = self.actions + [entry]
        if len(self.actions) > 20:
            self.actions = self.actions[-20:]

    def _update_summary(self):
        total = len(self.blocks)
        if total == 0:
            self.summary = "no blocks"
            return
        ids = sorted(b.id for b in self.blocks)
        first, last = ids[0], ids[-1]
        parts = [f"{total} blocks (#{first}..#{last})"]
        streaming = [b.id for b in self.blocks if b.status == "streaming"]
        if streaming:
            parts.append(f"streaming: {', '.join(f'#{b}' for b in streaming)}")
        self.summary = "  |  ".join(parts)


# -- helpers --


def _first_sentence(text: str, max_len: int = 80) -> str:
    for sep in ("。", "\n", ". ", "！", "？"):
        idx = text.find(sep)
        if 0 < idx < max_len:
            return text[:idx + len(sep)].strip()
    return text[:max_len].strip()


def _compute_unified_diff(
    old: str, new: str, label: str = "", context: int = 3,
) -> str:
    a = old.splitlines(keepends=True)
    b = new.splitlines(keepends=True)
    fromfile = f"{label} (before)" if label else "before"
    tofile = f"{label} (after)" if label else "after"
    return "".join(difflib.unified_diff(a, b, fromfile=fromfile, tofile=tofile, n=context))
