"""Memento's minimal filesystem implementation — the Branch / Memento contract in ``abcd.py``.

Storage binds to one local path (the owner root):

    {owner}/
      branches.jsonl                        # branch roster, append-only
      {branch_name}.ref.json                # the current pointer (BranchRef)
      branches/{branch_id}/
        meta.json                           # BranchMeta
        commits.jsonl                       # append-only CommitRef, in order (authoritative)
        commit_notes.jsonl                  # bypass Notes (last-wins, droppable, rebuildable)
        .lock                               # FileLocker write gate
      commits/{YYYY}/{MM}/cmt_{coord}/       # the commit's node space (a convention, lazy)
        MEMENTO.md                          # the node's index file, self-explaining entry

Discipline:
- Writes are ordered appends and atomic replaces; a reader skips a torn trailing line.
- A sync read is the cached fast path; an ``a``-prefixed method re-reads through aiofiles (IO-costly).
- Write gating is ``async with branch:`` (FileLocker, fail fast) plus a process-local asyncio.Lock.
- The node space is an **unmanaged asset area**: memento only reports whether ``MEMENTO.md``
  exists — one walk per branch view, one stat per single read — and never reads or cleans it.
  A directory appears only when something is put there: ``ensure_memento``, or a hook moving a
  finished file in. The read path never creates one.
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
    COMMIT_MEMENTO_FILE,
    Branch,
    BranchMeta,
    BranchRef,
    BranchView,
    CommitRef,
    CommitView,
    ForkRef,
    Memento,
    Note,
)
from ghoshell_moss.memento.hooks import hook_env, hook_paths, render_view_text, run_slot

__all__ = ["FsBranch", 'FsMemento', 'new_local_memento']

_MEMENTO_TEMPLATE = """# MEMENTO — {coord}

This directory belongs to commit `{coord}` (created {created}).

Nothing here is managed: memento does not version it, sync it, or clean it up. What
this place holds, and how it is governed, is yours to write down.

This template text is still here, which means nothing has been written yet. Replace
it with the index of this place — what this commit produced, what a reader should
look at first.
"""


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_aware(t: datetime) -> datetime:
    if t.tzinfo is None:
        return t.replace(tzinfo=timezone.utc)
    return t


# ── filesystem helpers ─────────────────────────────────────────────────────────


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read jsonl, skipping a torn trailing line (append-crash debris). A non-tail parse failure raises."""
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
    """The aiofiles jsonl read (IO-costly). Same semantics as _read_jsonl."""
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
    """Append single-line JSON rows with O_APPEND; POSIX keeps a sub-PIPE_BUF line atomic."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def _write_text(path: Path, text: str) -> None:
    """Write one text file through tmp + fsync + atomic rename (the parent directory is created)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.")
    try:
        os.write(fd, text.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    os.rename(tmp_name, str(path))


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    """Write one JSON file through tmp + fsync + atomic rename."""
    _write_text(path, json.dumps(obj, ensure_ascii=False, separators=(",", ":")))


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return None
    return json.loads(raw)


def _collapse_broken_commits(views: list[CommitView]) -> list[CommitView]:
    """A run of consecutive broken commits (non-empty ``Note.error``) keeps only the first.

    A broken commit still enters the view — the model must perceive it — but a run of broken
    placeholders must not flood it. The fold is one pass in coordinate order.
    """
    collapsed: list[CommitView] = []
    previous_broken = False
    for view in views:
        broken = view.is_broken
        if broken and previous_broken:
            continue
        collapsed.append(view)
        previous_broken = broken
    return collapsed


# ── FsBranch ───────────────────────────────────────────────────────────────────


class FsBranch(Branch):
    """A branch object pointing at one directory."""

    def __init__(self, memento: "FsMemento", branch_id: str, name: str, description: str, created: datetime):
        self._memento = memento
        self._branch_id = branch_id
        self._name = name
        self._description = description
        self._created = created
        self._locker: FileLocker | None = None
        self._commits_cache: list[CommitRef] | None = None
        self._notes_cache: dict[str, Note] | None = None

    # ── paths ──
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

    # ── construction-time facts ──
    @property
    def ref(self) -> BranchRef:
        return BranchRef(name=self._name, description=self._description, branch_id=self._branch_id,
                         created=self._created)

    @property
    def path(self) -> Path:
        return self._dir()

    def meta(self) -> BranchMeta:
        data = _read_json(self._meta_path())
        if data is None:
            raise FileNotFoundError(f"branch meta missing: {self._meta_path()}")
        return BranchMeta(**data)

    @property
    def index(self) -> int:
        return self.meta().index

    # ── reads (cached fast path) ──
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

    # ── reads (aiofiles, IO-costly) ──
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

    # ── writes ──
    def commit(
            self,
            *,
            message: str = "",
            metatype: str = "",
            metadata: dict[str, Any] | None = None,
    ) -> CommitRef:
        ref = CommitRef(seq=len(self.commits()) + 1, metatype=metatype, metadata=metadata or {})
        _append_jsonl(self._commits_path(), [ref.model_dump(mode="json")])
        self._commits_cache = None
        if message:
            self.note(ref.id, message)
        return ref

    def note(self, commit_id: str, message: str, error: str = "") -> Note:
        note = Note(commit_id=commit_id, message=message, error=error)
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

    async def anote(self, commit_id: str, message: str, error: str = "") -> Note:
        async with self:
            return self.note(commit_id, message, error)

    async def afork(self, name: str, description: str = "") -> "Branch":
        async with self:
            return self.fork(name, description)

    def ensure_memento(self, seq: int) -> Path:
        view = self.get_commit(seq)
        if view is None:
            raise KeyError(f"branch '{self._name}' has no commit at seq {seq}")
        path = view.memento_path(self._memento.root)
        if not path.exists():  # never overwrite an existing file — that content is the writer's
            _write_text(path, _MEMENTO_TEMPLATE.format(coord=view.coord, created=view.created.isoformat()))
        return path

    # ── read-side projection ──
    def _commit_view(self, commit: CommitRef, notes: dict[str, Note], branch_index: int) -> CommitView:
        return CommitView(branch_index=branch_index, ref=commit, note=notes.get(commit.id))

    def _node_index(self) -> set[Path]:
        """Every node index file that exists, collected in one walk.

        The cost is the number of directories in the node space, not the number of commits:
        a branch that never wrote a node walks almost nothing, and an anchor that took
        nothing in never leaves a directory to walk into.
        """
        root = self._memento.root / "commits"
        if not root.is_dir():
            return set()
        return set(root.rglob(COMMIT_MEMENTO_FILE))

    def _probe_memento(self, view: CommitView, nodes: set[Path] | None = None) -> CommitView:
        """Fill the node field: one stat for a single read, one collected set for a whole view."""
        path = view.memento_path(self._memento.root)
        if nodes is None:
            view.memento = path if path.is_file() else None
        else:
            view.memento = path if path in nodes else None
        return view

    def _parent(self, meta: BranchMeta) -> "FsBranch | None":
        fork_from = meta.fork_from
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
        meta = self.meta()
        views = _collapse_broken_commits([self._commit_view(c, notes, meta.index) for c in commits])
        nodes = self._node_index()
        for cv in views:
            self._probe_memento(cv, nodes)
        latest = views[-n:]
        history = views[:-n]
        parent = self._parent(meta)
        previous = parent.view(n=n) if parent is not None else None
        tip = commits[-1].id if commits else ""
        return BranchView(
            name=self._name,
            description=self._description,
            branch_id=self._branch_id,
            index=meta.index,
            created=self._created,
            commit_id=tip,
            previous=previous,
            history=history,
            latest=latest,
            commits_total=len(commits),
        )

    def get_commit(self, seq: int) -> CommitView | None:
        if seq < 1:
            return None
        commits = self.commits()
        if seq > len(commits):
            return None
        return self._probe_memento(self._commit_view(commits[seq - 1], self.notes(), self.index))

    async def aget_commit(self, seq: int) -> CommitView | None:
        if seq < 1:
            return None
        commits = await self.acommits()
        if seq > len(commits):
            return None
        notes = await self.anotes()
        return self._probe_memento(self._commit_view(commits[seq - 1], notes, self.index))

    # ── generation with hooks ──
    async def _render_slot(self, slot: str, *, coord: str = "", commit_dir: Path | None = None) -> str:
        """Run this branch's mounted view slot and return what it contributes to the page.

        The slot belongs to memento, not to whoever renders the view: the branch already knows
        its own directory, so it also knows where the hooks are mounted (branch first, then the
        root) and what the protocol promises them. Letting a caller append the output instead
        is how a plan written into the branch directory failed to reach the boot memory.
        """
        roots = [self._dir(), self._memento.root]
        if not hook_paths(roots, slot):
            return ""
        env = hook_env(
            slot,
            root=self._memento.root,
            branch=self._name,
            branch_dir=self._dir(),
            coord=coord,
            commit_dir=commit_dir,
        )
        try:
            results = await run_slot(slot, roots, env_base=env, log_root=self._memento.root)
        except Exception:  # noqa: BLE001 — the read surface must not die because a hook did
            return ""
        return render_view_text(results)

    async def aview_with_hooks(self, *, n: int = 10) -> BranchView:
        """The view an actual reader gets: the projection **plus** what the directory renders.

        Cheap views (``view`` / ``aview``) stay hook-free on purpose — they are called from sync
        paths and inside other projections. This one is the explicit opt-in, and it re-runs the
        slot on every call: a plan that moved a minute ago must not come back stale.
        """
        view = await self.aview(n=n)
        view.ground = await self._render_slot("branch_view")
        return view

    async def aget_commit_with_hooks(self, seq: int) -> CommitView | None:
        """A commit read with its directory's own rendering — same contract as the branch view.

        The slot runs only for a commit whose directory exists — a node index, or a file a hook
        moved in. Nothing there means nothing to render, and the read path never creates the
        directory to find out.
        """
        view = await self.aget_commit(seq)
        if view is None:
            return view
        commit_dir = view.memento_path(self._memento.root).parent
        if not commit_dir.is_dir():
            return view
        view.ground = await self._render_slot("commit_view", coord=view.coord, commit_dir=commit_dir)
        return view

    # ── queries ──
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

    # ── write gating ──
    async def __aenter__(self) -> "FsBranch":
        locker = FileLocker(self._lock_path())
        if not locker.acquire():  # fast fail; flock is the in-process and cross-process mutex
            raise BlockingIOError(f"branch '{self._name}' write lock busy")
        self._locker = locker
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._locker is not None:
            self._locker.release()
            self._locker = None


# ── FsMemento ──────────────────────────────────────────────────────────────────


class FsMemento(Memento):
    """The owner-level filesystem memento. ``root`` is the owner directory."""

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
        """public-internal: the single creation path shared by create_branch and Branch.fork."""
        branch_id = str(ulid.ULID())
        index = len(_read_jsonl(self._branches_jsonl())) + 1
        created = _now_utc()
        meta = BranchMeta(
            branch_id=branch_id,
            index=index,
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

    def get_branch_by_index(self, index: int) -> "FsBranch | None":
        if index < 1:
            return None
        rows = _read_jsonl(self._branches_jsonl())
        if index > len(rows):
            return None
        ref = BranchRef(**rows[index - 1])
        return self._branch_by_id(ref.branch_id)

    def resolve_commit(self, coord: str) -> CommitView | None:
        parts = coord.split("-", 1)
        if len(parts) != 2:
            return None
        try:
            branch_index, seq = int(parts[0]), int(parts[1])
        except ValueError:
            return None
        branch = self.get_branch_by_index(branch_index)
        if branch is None:
            return None
        return branch.get_commit(seq)

    def delete_branch(self, name: str) -> None:
        ref = self._ref_path(name)
        if not ref.exists():
            raise FileNotFoundError(f"branch '{name}' not found")
        ref.unlink()


def new_local_memento(root: Path) -> FsMemento:
    """Build an owner-level filesystem memento."""
    return FsMemento(root)
