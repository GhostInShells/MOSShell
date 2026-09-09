"""memento 的极简 filesystem 实现 — 实现 ``abcd.py`` 的 Branch / Memento 契约.

存储只绑定一个本地 path (owner 根目录):

    {owner}/
      branches.jsonl                        # 历史 branch 名单 (append-only)
      {branch_name}.ref.json                # 当前指针 (BranchRef)
      branches/{branch_id}/
        meta.json                           # BranchMeta
        commits.jsonl                       # 正序 append-only CommitRef (权威)
        commit_notes.jsonl                  # 旁路 Note (last-wins, 可丢可重建)
        .lock                               # FileLocker 写门控锁文件

Discipline:
- 只写正序 append / 原子写, 读侧跳过撕裂尾行 (append-crash 残留).
- 同步读 = 缓存快路径; ``a<name>`` = aiofiles 重读磁盘 (IO-costly).
- 写门控 = ``async with branch:`` (FileLocker, fast-fail) + 进程内 asyncio.Lock 互锁.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiofiles
import ulid

from ghoshell_moss.contracts.workspace import FileLocker
from ghoshell_moss.memento.abcd import (
    Branch,
    BranchMeta,
    BranchRef,
    BranchView,
    CommitRef,
    CommitSummary,
    ForkRef,
    Memento,
    Note,
)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_aware(t: datetime) -> datetime:
    if t.tzinfo is None:
        return t.replace(tzinfo=timezone.utc)
    return t


# ── filesystem helpers ─────────────────────────────────────────────────────────


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读 jsonl, 跳过撕裂尾行 (append-crash 残留). 非尾行解析失败则抛错."""
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        return []
    lines = raw.rstrip("\n").split("\n")
    result: list[dict[str, Any]] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            result.append(json.loads(stripped))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                continue  # torn last line, legal
            raise ValueError(f"jsonl parse error at line {i + 1} in {path}") from None
    return result


async def _aread_jsonl(path: Path) -> list[dict[str, Any]]:
    """aiofiles 版 jsonl 读 (IO-costly). 语义同 _read_jsonl."""
    if not path.exists():
        return []
    async with aiofiles.open(path, "r", encoding="utf-8") as f:
        raw = await f.read()
    if not raw.strip():
        return []
    lines = raw.rstrip("\n").split("\n")
    result: list[dict[str, Any]] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            result.append(json.loads(stripped))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                continue
            raise ValueError(f"jsonl parse error at line {i + 1} in {path}") from None
    return result


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """O_APPEND 追加单行 JSON rows. POSIX O_APPEND 保证 <PIPE_BUF 行原子."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    """tmp + fsync + 原子 rename 写单个 JSON 文件."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.")
    try:
        os.write(fd, json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    os.rename(tmp_name, str(path))


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return None
    return json.loads(raw)


# ── FsBranch ───────────────────────────────────────────────────────────────────


class FsBranch(Branch):
    """一个指向目录的 branch 对象."""

    def __init__(self, memento: "FsMemento", branch_id: str, name: str, description: str, created: datetime):
        self._memento = memento
        self._branch_id = branch_id
        self._name = name
        self._description = description
        self._created = created
        self._locker: FileLocker | None = None
        self._commits_cache: list[CommitRef] | None = None
        self._notes_cache: dict[str, Note] | None = None

    # ── 路径 ──
    def _dir(self) -> Path:
        return self._memento.root / "branches" / self._branch_id

    def _commits_path(self) -> Path:
        return self._dir() / "commits.jsonl"

    def _notes_path(self) -> Path:
        return self._dir() / "commit_notes.jsonl"

    def _meta_path(self) -> Path:
        return self._dir() / "meta.json"

    def _lock_path(self) -> Path:
        return self._dir() / ".lock"

    # ── 构造期事实 ──
    @property
    def ref(self) -> BranchRef:
        return BranchRef(name=self._name, description=self._description, branch_id=self._branch_id, created=self._created)

    @property
    def path(self) -> Path:
        return self._dir()

    def meta(self) -> BranchMeta:
        data = _read_json(self._meta_path())
        if data is None:
            raise FileNotFoundError(f"branch meta missing: {self._meta_path()}")
        return BranchMeta(**data)

    # ── 读 (缓存快路径) ──
    def commits(self) -> list[CommitRef]:
        if self._commits_cache is None:
            self._commits_cache = [CommitRef(**row) for row in _read_jsonl(self._commits_path())]
        return list(self._commits_cache)

    def notes(self) -> dict[str, Note]:
        if self._notes_cache is None:
            result: dict[str, Note] = {}
            for row in _read_jsonl(self._notes_path()):
                note = Note(**row)
                result[note.commit_id] = note  # last-wins
            self._notes_cache = result
        return dict(self._notes_cache)

    # ── 读 (aiofiles, IO-costly) ──
    async def acommits(self) -> list[CommitRef]:
        rows = await _aread_jsonl(self._commits_path())
        self._commits_cache = [CommitRef(**row) for row in rows]
        return list(self._commits_cache)

    async def anotes(self) -> dict[str, Note]:
        result: dict[str, Note] = {}
        for row in await _aread_jsonl(self._notes_path()):
            note = Note(**row)
            result[note.commit_id] = note
        self._notes_cache = result
        return dict(result)

    # ── 写 ──
    def commit(
        self,
        *,
        message: str = "",
        metatype: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> CommitRef:
        ref = CommitRef(metatype=metatype, metadata=metadata or {})
        _append_jsonl(self._commits_path(), [ref.model_dump(mode="json")])
        self._commits_cache = None
        if message:
            self.note(ref.id, message)
        return ref

    def note(self, commit_id: str, message: str) -> Note:
        note = Note(commit_id=commit_id, message=message)
        _append_jsonl(self._notes_path(), [note.model_dump(mode="json")])
        self._notes_cache = None
        return note

    def fork(self, name: str, description: str = "") -> "Branch":
        commits = self.commits()
        tip = commits[-1].id if commits else ""
        fork_from = ForkRef(branch_id=self._branch_id, commit_id=tip)
        return self._memento._create_branch(
            name, description, metatype="", metadata=None, fork_from=fork_from
        )

    async def acommit(
        self,
        *,
        message: str = "",
        metatype: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> CommitRef:
        async with self:
            return self.commit(message=message, metatype=metatype, metadata=metadata)

    async def anote(self, commit_id: str, message: str) -> Note:
        async with self:
            return self.note(commit_id, message)

    async def afork(self, name: str, description: str = "") -> "Branch":
        async with self:
            return self.fork(name, description)

    # ── 读侧投影 ──
    def _summary(self, commit: CommitRef, notes: dict[str, Note]) -> CommitSummary:
        message = notes[commit.id].message if commit.id in notes else ""
        return CommitSummary(id=commit.id, message=message)

    def _parent(self) -> "FsBranch | None":
        fork_from = self.meta().fork_from
        if fork_from is None:
            return None
        return self._memento._branch_by_id(fork_from.branch_id)

    def view(self, *, n: int = 10) -> BranchView:
        return self._build_view(self.commits(), self.notes(), n)

    async def aview(self, *, n: int = 10) -> BranchView:
        commits = await self.acommits()
        notes = await self.anotes()
        return self._build_view(commits, notes, n)

    def _build_view(self, commits: list[CommitRef], notes: dict[str, Note], n: int) -> BranchView:
        latest = [self._summary(c, notes) for c in commits[-n:]]
        history = [self._summary(c, notes) for c in commits[:-n]]
        parent = self._parent()
        previous = parent.view(n=n) if parent is not None else None
        tip = commits[-1].id if commits else ""
        return BranchView(
            name=self._name,
            description=self._description,
            branch_id=self._branch_id,
            created=self._created,
            commit_id=tip,
            previous=previous,
            history=history,
            latest=latest,
            commits_total=len(commits),
        )

    # ── 查询 ──
    def query_commits(
        self,
        *,
        from_date: datetime | None = None,
        until_date: datetime | None = None,
    ) -> list[CommitRef]:
        result = self.commits()
        if from_date is not None:
            lo = _ensure_aware(from_date)
            result = [c for c in result if c.created >= lo]
        if until_date is not None:
            hi = _ensure_aware(until_date)
            result = [c for c in result if c.created <= hi]
        return result

    async def aquery_commits(
        self,
        *,
        from_date: datetime | None = None,
        until_date: datetime | None = None,
    ) -> list[CommitRef]:
        result = await self.acommits()
        if from_date is not None:
            lo = _ensure_aware(from_date)
            result = [c for c in result if c.created >= lo]
        if until_date is not None:
            hi = _ensure_aware(until_date)
            result = [c for c in result if c.created <= hi]
        return result

    # ── 写门控 ──
    async def __aenter__(self) -> "FsBranch":
        locker = FileLocker(self._lock_path())
        if not locker.acquire():  # fast-fail, flock 提供进程内+跨进程互斥
            raise BlockingIOError(f"branch '{self._name}' write lock busy")
        self._locker = locker
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._locker is not None:
            self._locker.release()
            self._locker = None


# ── FsMemento ──────────────────────────────────────────────────────────────────


class FsMemento(Memento):
    """owner 级 filesystem memento. root = owner 目录."""

    def __init__(self, root: Path):
        self._root = root

    @property
    def root(self) -> Path:
        return self._root

    def _branches_jsonl(self) -> Path:
        return self._root / "branches.jsonl"

    def _ref_path(self, name: str) -> Path:
        return self._root / f"{name}.ref.json"

    def _branch_dir(self, branch_id: str) -> Path:
        return self._root / "branches" / branch_id

    def create_branch(
        self,
        name: str,
        description: str = "",
        *,
        metatype: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> Branch:
        if self._ref_path(name).exists():
            raise NameError(f"branch '{name}' already exists")
        return self._create_branch(name, description, metatype=metatype, metadata=metadata, fork_from=None)

    def _create_branch(
        self,
        name: str,
        description: str,
        *,
        metatype: str,
        metadata: dict[str, Any] | None,
        fork_from: ForkRef | None,
    ) -> FsBranch:
        """public-internal: create_branch 与 Branch.fork 共享的单一创建路径."""
        branch_id = str(ulid.ULID())
        created = _now_utc()
        meta = BranchMeta(
            branch_id=branch_id,
            name=name,
            description=description,
            metatype=metatype,
            metadata=metadata or {},
            fork_from=fork_from,
            created=created,
        )
        ref = BranchRef(name=name, description=description, branch_id=branch_id, created=created)

        self._branch_dir(branch_id).mkdir(parents=True, exist_ok=True)
        _write_json(self._branch_dir(branch_id) / "meta.json", meta.model_dump(mode="json"))
        _write_json(self._ref_path(name), ref.model_dump(mode="json"))
        _append_jsonl(self._branches_jsonl(), [ref.model_dump(mode="json")])
        return FsBranch(self, branch_id, name, description, created)

    def _branch_by_id(self, branch_id: str) -> FsBranch | None:
        data = _read_json(self._branch_dir(branch_id) / "meta.json")
        if data is None:
            return None
        meta = BranchMeta(**data)
        return FsBranch(self, meta.branch_id, meta.name, meta.description, meta.created)

    def get_branch(self, name: str) -> Branch | None:
        data = _read_json(self._ref_path(name))
        if data is None:
            return None
        ref = BranchRef(**data)
        return FsBranch(self, ref.branch_id, ref.name, ref.description, ref.created)

    def list_branches(self) -> list[BranchRef]:
        result: list[BranchRef] = []
        for p in sorted(self._root.glob("*.ref.json")):
            data = _read_json(p)
            if data is not None:
                result.append(BranchRef(**data))
        return result

    def delete_branch(self, name: str) -> None:
        ref = self._ref_path(name)
        if not ref.exists():
            raise FileNotFoundError(f"branch '{name}' not found")
        ref.unlink()


def new_local_memento(root: Path) -> FsMemento:
    """构建一个 owner 级 filesystem memento."""
    return FsMemento(root)
