"""BlockStore — thread-safe single source of truth for text blocks.

Shared between the channel (Matrix daemon thread) and the Reflex UI
(main thread). All mutation methods acquire ``self._lock``.
"""

from __future__ import annotations

import difflib
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

Source = Literal["g", "u"]  # g = ghost (model), u = user (human)
LockOwner = Literal["g", "u", None]
BlockStatus = Literal["streaming", "sealed", "error"]
ActionKind = Literal[
    "block_create", "block_edit", "block_submit",
    "block_revise", "block_dump", "block_read_file",
]


# -- domain models --


@dataclass
class BlockVersion:
    version: int
    source: Source
    content: str
    created_at: float = field(default_factory=time.time)


@dataclass
class Block:
    id: int
    title: str = ""
    versions: list[BlockVersion] = field(default_factory=list)
    lock: LockOwner = None
    status: BlockStatus = "sealed"
    created_at: float = field(default_factory=time.time)

    @property
    def current(self) -> BlockVersion | None:
        return self.versions[-1] if self.versions else None

    @property
    def content(self) -> str:
        v = self.current
        return v.content if v else ""

    @property
    def source(self) -> Source | None:
        v = self.current
        return v.source if v else None

    @property
    def version_count(self) -> int:
        return len(self.versions)

    @property
    def with_line_numbers(self) -> str:
        lines = self.content.split("\n")
        return "\n".join(f"{i + 1:>6}\t{line}" for i, line in enumerate(lines))

    def new_version(self, source: Source, content: str = "") -> BlockVersion:
        v = BlockVersion(
            version=len(self.versions) + 1,
            source=source,
            content=content,
        )
        self.versions.append(v)
        return v

    def append_to_current(self, chunk: str) -> None:
        if self.current is None:
            self.new_version("g", chunk)
        else:
            self.current.content += chunk

    def replace_lines(
        self, line_no: int, count: int, new_text: str,
    ) -> int:
        lines = self.content.split("\n")
        # line_no is 1-indexed to match display
        idx = line_no - 1
        if idx < 0:
            idx = 0
        if idx >= len(lines):
            lines.append(new_text)
            replaced = 0
        else:
            end = min(idx + count, len(lines))
            replaced = end - idx
            lines[idx:end] = [new_text]
        self.current.content = "\n".join(lines)
        return replaced


@dataclass
class Action:
    kind: ActionKind
    block_id: int | None = None
    summary: str = ""
    at: float = field(default_factory=time.time)


class ActionLog:
    def __init__(self, maxsize: int = 20) -> None:
        self._actions: list[Action] = []
        self._maxsize = maxsize

    def record(self, action: Action) -> None:
        self._actions.append(action)
        if len(self._actions) > self._maxsize:
            self._actions = self._actions[-self._maxsize:]

    def recent(self, n: int = 5) -> list[Action]:
        return list(reversed(self._actions[-n:]))


# -- diff model --


@dataclass
class Diff:
    block_id: int
    unified_diff: str
    anchor_quote: str = ""
    human_note: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    @classmethod
    def compute(
        cls,
        block_id: int,
        old_content: str,
        new_content: str,
        human_note: str = "",
        context_lines: int = 3,
    ) -> Diff:
        a = old_content.splitlines(keepends=True)
        b = new_content.splitlines(keepends=True)
        ud = "".join(difflib.unified_diff(
            a, b,
            fromfile=f"block #{block_id} (before)",
            tofile=f"block #{block_id} (after)",
            n=context_lines,
        ))
        # anchor: first changed line from old content
        anchor = ""
        for line in ud.split("\n"):
            if line.startswith("-") and not line.startswith("---"):
                anchor = line[1:].strip()
                if len(anchor) > 80:
                    anchor = anchor[:77] + "..."
                break
        return cls(
            block_id=block_id,
            unified_diff=ud,
            anchor_quote=anchor,
            human_note=human_note,
        )


# -- BlockStore --


@dataclass
class DumpResult:
    count: int
    path: str


class BlockStore:
    """Thread-safe central state for text blocks."""

    def __init__(self, session_uid: str = "") -> None:
        self._lock = threading.Lock()
        self._blocks: dict[int, Block] = {}
        self._order: list[int] = []
        self._next_id: int = 1
        self._diffs: list[Diff] = []
        self.action_log = ActionLog()
        self.session_uid = session_uid or uuid.uuid4().hex[:8]
        self._tmp_dir: str | None = None
        # callbacks for Reflex integration
        self._on_create: Callable[[int], None] | None = None
        self._on_update: Callable[[int], None] | None = None
        self._on_seal: Callable[[int], None] | None = None

    # -- callbacks --

    def set_on_create(self, cb: Callable[[int], None]) -> None:
        self._on_create = cb

    def set_on_update(self, cb: Callable[[int], None]) -> None:
        self._on_update = cb

    def set_on_seal(self, cb: Callable[[int], None]) -> None:
        self._on_seal = cb

    # -- block CRUD --

    def create(
        self,
        source: Source,
        title: str = "",
        lock: LockOwner = None,
        content: str = "",
    ) -> int:
        with self._lock:
            bid = self._next_id
            self._next_id += 1
            block = Block(id=bid, title=title, lock=lock)
            if lock == "g":
                block.status = "streaming"
            if content:
                block.new_version(source, content)
            self._blocks[bid] = block
            self._order.append(bid)
        self.action_log.record(Action(
            kind="block_create", block_id=bid,
            summary=f"{source}: created #{bid}" + (f' "{title}"' if title else ""),
        ))
        if self._on_create:
            self._on_create(bid)
        return bid

    def acquire_lock(self, block_id: int, owner: LockOwner) -> Block:
        with self._lock:
            block = self._blocks[block_id]
            block.lock = owner
            if owner == "g":
                block.status = "streaming"
            return block

    def release_lock(self, block_id: int) -> None:
        with self._lock:
            block = self._blocks[block_id]
            block.lock = None

    def append_to_current(self, block_id: int, chunk: str) -> None:
        with self._lock:
            block = self._blocks[block_id]
            block.append_to_current(chunk)
        if self._on_update:
            self._on_update(block_id)

    def seal(self, block_id: int) -> Block:
        with self._lock:
            block = self._blocks[block_id]
            block.status = "sealed"
            block.lock = None
        self.action_log.record(Action(
            kind="block_submit", block_id=block_id,
            summary=f"g: sealed #{block_id}",
        ))
        if self._on_seal:
            self._on_seal(block_id)
        return block

    def get(self, block_id: int) -> Block | None:
        with self._lock:
            return self._blocks.get(block_id)

    def get_order(self) -> list[int]:
        with self._lock:
            return list(self._order)

    def snapshot(self) -> list[Block]:
        with self._lock:
            return [self._blocks[bid] for bid in self._order]

    # -- diff bucket --

    def push_diff(self, diff: Diff) -> None:
        with self._lock:
            self._diffs.append(diff)
        self.action_log.record(Action(
            kind="block_edit", block_id=diff.block_id,
            summary=f"u: edited #{diff.block_id} (diff pending)",
        ))

    def peek_diffs(self, limit: int = 5) -> list[Diff]:
        with self._lock:
            return list(self._diffs[-limit:])

    def drain_diffs(self) -> list[Diff]:
        with self._lock:
            drained = list(self._diffs)
            self._diffs.clear()
        return drained

    # -- dump --

    def dump(
        self, path: str = "", ids: list[int] | None = None,
    ) -> DumpResult:
        with self._lock:
            target_ids = ids or self._order
            if not path:
                tmp = self._tmp_dir or f"/tmp/text_blocks_{self.session_uid}"
                Path(tmp).mkdir(parents=True, exist_ok=True)
                self._tmp_dir = tmp
                path = tmp
            out_dir = Path(path)
            if out_dir.suffix:
                # single file — concatenate all blocks
                out_dir.parent.mkdir(parents=True, exist_ok=True)
                with open(out_dir, "w") as f:
                    for bid in target_ids:
                        block = self._blocks[bid]
                        f.write(f"# {block.title or f'Block #{bid}'}\n\n")
                        f.write(block.content)
                        f.write("\n\n")
            else:
                # directory — one file per block
                out_dir.mkdir(parents=True, exist_ok=True)
                for bid in target_ids:
                    block = self._blocks[bid]
                    safe = (block.title or f"block_{bid}").replace("/", "_")
                    fname = f"{bid:03d}_{safe}.md"
                    with open(out_dir / fname, "w") as f:
                        f.write(f"# {block.title or f'Block #{bid}'}\n\n")
                        f.write(block.content)
        self.action_log.record(Action(
            kind="block_dump", summary=f"dumped {len(target_ids)} blocks -> {path or self._tmp_dir}",
        ))
        return DumpResult(count=len(target_ids), path=path or self._tmp_dir)

    # -- summary for context_messages --

    def summary(self) -> str:
        with self._lock:
            total = len(self._order)
            if total == 0:
                return "no blocks"
            first, last = self._order[0], self._order[-1]
            parts = [f"{total} blocks (#{first}..#{last})"]
            streaming = [
                bid for bid in self._order
                if self._blocks[bid].status == "streaming"
            ]
            if streaming:
                parts.append(f"streaming: {', '.join(f'#{b}' for b in streaming)}")
            if self._diffs:
                parts.append(f"{len(self._diffs)} pending diff(s)")
            return "  |  ".join(parts)

    def index(self) -> str:
        with self._lock:
            lines = []
            for bid in self._order:
                b = self._blocks[bid]
                lock_str = f" lock={b.lock}" if b.lock else ""
                title_str = f' "{b.title}"' if b.title else ""
                lines.append(
                    f"#{b.id}  source={b.source or '-'}  "
                    f"status={b.status}  v{b.version_count}{lock_str}{title_str}"
                )
            return "\n".join(lines)
