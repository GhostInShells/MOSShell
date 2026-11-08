"""Memento's minimal filesystem implementation — the Branch / Memento contract in ``abcd.py``.

Storage binds to one local path (the owner root):

    {owner}/
      branches.jsonl                        # branch roster, append-only
      {branch_name}.ref.json                # the current pointer (BranchRef)
      branches/{branch_id}/
        meta.json                           # BranchMeta
        commits.jsonl                       # append-only CommitRef, in order (authoritative)
        commit_notes.jsonl                  # bypass Notes (last-wins, droppable, rebuildable)
        confluences.jsonl                   # inbox: references received from other branches
        children.jsonl                      # outbox: branches created from this one
        .lock                               # FileLocker write gate
      commits/{YYYY}/{MM}/cmt_{coord}/       # the commit's node space (a convention, lazy)
        MEMENTO.md                          # the node's index file, self-explaining entry

Discipline:
- Writes are ordered appends and atomic replaces; a reader skips a torn trailing line.
- A sync read is the cached fast path; an ``a``-prefixed method re-reads through aiofiles (IO-costly).
- Write gating is ``async with branch:`` (FileLocker, fail fast) — one writer per branch.
  Only read-modify-write operations take it (``commit`` / ``fork``); appends do not.
- The node space is an **unmanaged asset area**: memento only reports whether ``MEMENTO.md``
  exists — one walk per branch view, one stat per single read — and never reads or cleans it.
  A directory appears only when something is put there: ``ensure_memento``, or a hook moving a
  finished file in. The read path never creates one.
"""

from __future__ import annotations

import json
import logging
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
    BranchAddr,
    BranchEvent,
    BranchView,
    Child,
    CommitRef,
    Confluence,
    CommitView,
    ForkRef,
    Memento,
    Note,
)
from ghoshell_moss.memento.hooks import hook_env, hook_paths, render_view_text, run_slot

__all__ = ["FsBranch", 'FsMemento', 'new_local_memento']

logger = logging.getLogger(__name__)

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

    def __init__(
            self,
            memento: "FsMemento",
            branch_id: str,
            name: str,
            description: str,
            created: datetime,
            *,
            index: int | None = None,
            parent_id: str = "",
    ):
        self._memento = memento
        self._branch_id = branch_id
        self._name = name
        self._description = description
        self._created = created
        self._index = index
        self._parent_id = parent_id
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

    def _inbox_path(self) -> Path:
        return self._dir() / "confluences.jsonl"

    def _claimed_path(self) -> Path:
        return self._dir() / ".confluences.jsonl"

    def _children_inbox_path(self) -> Path:
        return self._dir() / "children.jsonl"

    def _claimed_children_path(self) -> Path:
        return self._dir() / ".children.jsonl"

    # ── construction-time facts ──
    @property
    def ref(self) -> BranchRef:
        return BranchRef(name=self._name, description=self._description, branch_id=self._branch_id,
                         parent_id=self._parent_id, created=self._created)

    def address(self) -> BranchAddr:
        """Who this branch is, in the shape a record stores — the input to every event."""
        return BranchAddr(branch_id=self._branch_id, branch_name=self._name, branch_index=self.index)

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
        if self._index is None:  # one meta read per branch object, then memory
            self._index = self.meta().index
        return self._index

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
            recipients: list[BranchAddr] | None = None,
    ) -> CommitRef:
        confluences = self._absorb_confluences()
        children = self._absorb_children()
        ref = CommitRef(
            seq=len(self.commits()) + 1,
            metatype=metatype,
            metadata=metadata or {},
            confluences=confluences,
            children=children,
        )
        _append_jsonl(self._commits_path(), [ref.model_dump(mode="json")])
        self._commits_cache = None
        # The anchor is durable, so the claimed batches now belong to it: drop the files. A
        # crash before this point leaves them behind, and the next commit re-reads and de-dupes.
        for claimed in (self._claimed_path(), self._claimed_children_path()):
            if not claimed.exists():
                continue
            try:
                claimed.unlink()
            except OSError:  # the commit is durable; a leftover file is re-read and de-duped
                logger.warning("could not clear claimed staging file %s", claimed, exc_info=True)
        self._memento._emit("commit", self.address(), coord=f"{self.index}-{ref.seq}")
        if message:
            self.note(ref.id, message, recipients=recipients)
        return ref

    # ── staging: both directions, one shape ──
    @staticmethod
    def _claim(inbox: Path, claimed: Path) -> list[dict[str, Any]]:
        """Take a staging file atomically, and leave it where a crash can still find it.

        ``os.replace`` rather than copy + delete: a same-directory rename has no "two copies"
        and no "no copy" in between. Rows arriving after the claim land in a fresh file and
        belong to the next commit.
        """
        rows: list[dict[str, Any]] = []
        if claimed.exists():  # a previous attempt died before it could absorb or unlink
            rows.extend(_read_jsonl(claimed))
        if inbox.exists():
            os.replace(inbox, claimed)
            rows.extend(_read_jsonl(claimed))
        return rows

    def _absorb_confluences(self) -> list[Confluence]:
        """Turn the claimed inbox rows into this commit's ``confluences``, dropping duplicates.

        De-duplication happens here and only here, keyed on the source coordinate: a delivery
        the sender retried, or a batch re-read after a crash, must not enter the trajectory
        twice.
        """
        incoming = self._claim(self._inbox_path(), self._claimed_path())
        if not incoming:
            return []
        taken: set[tuple[str, int]] = set()
        commits = self.commits()
        if commits:  # crash after append, before unlink: the last anchor already holds them
            for received in commits[-1].confluences:
                taken.add((received.branch.branch_id, received.commit_seq))
        absorbed: list[Confluence] = []
        for row in incoming:
            item = Confluence(**row)
            key = (item.branch.branch_id, item.commit_seq)
            if key in taken:
                continue
            taken.add(key)
            absorbed.append(item)
        return absorbed

    def _absorb_children(self) -> list[Child]:
        """The outgoing mirror: branches created since the last commit, de-duplicated by identity."""
        incoming = self._claim(self._children_inbox_path(), self._claimed_children_path())
        if not incoming:
            return []
        taken: set[str] = set()
        commits = self.commits()
        if commits:
            taken.update(c.branch.branch_id for c in commits[-1].children)
        absorbed: list[Child] = []
        for row in incoming:
            item = Child(**row)
            if item.branch.branch_id in taken:
                continue
            taken.add(item.branch.branch_id)
            absorbed.append(item)
        return absorbed

    def confluences(self) -> list[Confluence]:
        """The inbox: what arrived and has not been absorbed into a commit yet."""
        return [Confluence(**row) for row in _read_jsonl(self._inbox_path())]

    async def aconfluences(self) -> list[Confluence]:
        """IO-costly: re-read the inbox from disk."""
        rows = await _aread_jsonl(self._inbox_path())
        return [Confluence(**row) for row in rows]

    def confluence(self, confluence: Confluence) -> None:
        """Deliver one reference into this branch's inbox — the only cross-branch write."""
        key = (confluence.branch.branch_id, confluence.commit_seq)
        if any((c.branch.branch_id, c.commit_seq) == key for c in self.confluences()):
            return  # already waiting in the inbox
        if any(
            (r.branch.branch_id, r.commit_seq) == key
            for commit in self.commits() for r in commit.confluences
        ):
            return  # already absorbed by an earlier commit
        _append_jsonl(self._inbox_path(), [confluence.model_dump(mode="json")])
        self._memento._emit("confluence", self.address(), coord=confluence.coord)

    def pending_children(self) -> list[Child]:
        """Branches created from this one that the next commit has not absorbed yet."""
        return [Child(**row) for row in _read_jsonl(self._children_inbox_path())]

    async def apending_children(self) -> list[Child]:
        """IO-costly: re-read the pending children from disk."""
        rows = await _aread_jsonl(self._children_inbox_path())
        return [Child(**row) for row in rows]

    def add_child(self, child: Child) -> None:
        """Record a branch created from this one — written by the parent, absorbed at commit."""
        key = child.branch.branch_id
        if any(c.branch.branch_id == key for c in self.pending_children()):
            return
        _append_jsonl(self._children_inbox_path(), [child.model_dump(mode="json")])

    def note(
            self,
            commit_id: str,
            message: str,
            error: str = "",
            *,
            recipients: list[BranchAddr] | None = None,
    ) -> Note:
        # ``recipients`` is the one field that is not last-wins: a rewrite must not re-address
        # a message a summarizer merely restated.
        previous = self.notes().get(commit_id)
        if recipients is None:
            recipients = previous.recipients if previous is not None else []
        note = Note(commit_id=commit_id, message=message, error=error, recipients=list(recipients))
        _append_jsonl(self._notes_path(), [note.model_dump(mode="json")])
        self._notes_cache = None
        # Delivery happens once, on the first note: the address is part of the message, and a
        # rewrite is only a restatement.
        if previous is None and recipients:
            self._deliver(note)
        return note

    def _deliver(self, note: Note) -> None:
        """Write one ``Confluence`` into every recipient's inbox.

        A failure raises: the note is already durable, and the caller can retry — the receiver
        de-duplicates on the source coordinate, so a retry is free. Nothing records a delivery
        receipt: the trajectory holds the declaration only.
        """
        ref = next((c for c in self.commits() if c.id == note.commit_id), None)
        if ref is None:
            raise KeyError(f"cannot deliver note: no commit {note.commit_id} in branch '{self._name}'")
        title = note.message.split("\n", 1)[0]
        source = self.address()
        for recipient in note.recipients:
            target = self._resolve(recipient)
            target.confluence(Confluence(branch=source, commit_seq=ref.seq, title=title))

    def _resolve(self, address: BranchAddr) -> "FsBranch":
        """Find a branch from an address — by position first, by name as a fallback.

        Position is the stable half: the roster is append-only, so an index survives archive.
        The name is only a fallback, and the id is checked whenever it is present — a reused
        name must never silently point at a different branch.
        """
        target: FsBranch | None = None
        if address.branch_index > 0:
            candidate = self._memento.get_branch_by_index(address.branch_index)
            if candidate is not None and (
                not address.branch_id or candidate.ref.branch_id == address.branch_id
            ):
                target = candidate
        if target is None and address.branch_name:
            candidate = self._memento.get_branch(address.branch_name)
            if candidate is not None and (
                not address.branch_id or candidate.ref.branch_id == address.branch_id
            ):
                target = candidate
        if target is None:
            raise NameError(
                f"confluence target not resolvable: name='{address.branch_name}' "
                f"index={address.branch_index} id='{address.branch_id}'"
            )
        return target

    def fork(self, name: str, description: str = "") -> "Branch":
        commits = self.commits()
        tip = commits[-1].id if commits else ""
        fork_from = ForkRef(branch_id=self._branch_id, commit_id=tip)
        branch = self._memento._create_branch(
            name, description, metatype="", metadata=None, fork_from=fork_from
        )
        # Two sides of one relation, each written by its own owner: the child records where it
        # came from (its meta), the parent records that it exists (this row). Nobody writes into
        # the other's trajectory, and the parent's record only becomes history at its next commit.
        self.add_child(Child(branch=branch.address()))
        self._memento._emit("child", self.address())
        return branch

    async def acommit(
            self,
            *,
            message: str = "",
            metatype: str = "",
            metadata: dict[str, Any] | None = None,
            recipients: list[BranchAddr] | None = None,
    ) -> CommitRef:
        async with self:
            return self.commit(
                message=message, metatype=metatype, metadata=metadata, recipients=recipients
            )

    async def anote(
            self,
            commit_id: str,
            message: str,
            error: str = "",
            *,
            recipients: list[BranchAddr] | None = None,
    ) -> Note:
        # No write permission: a note is one appended line, and a duplicated decision is
        # idempotent downstream. Taking the branch lock here would only make a delivery
        # queue behind an unrelated commit.
        return self.note(commit_id, message, error, recipients=recipients)

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
            logger.exception("view slot %r failed on branch %r", slot, self._name)
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
        self._hooks: list[Callable[[BranchEvent], None]] = []
        self._roster_cache: list[dict[str, Any]] | None = None

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
        ref = BranchRef(
            name=name,
            description=description,
            branch_id=branch_id,
            parent_id=fork_from.branch_id if fork_from else "",
            created=created,
        )

        self._branch_dir(branch_id).mkdir(parents=True, exist_ok=True)
        _write_json(self._branch_dir(branch_id) / "meta.json", meta.model_dump(mode="json"))
        _write_json(self._ref_path(name), ref.model_dump(mode="json"))
        _append_jsonl(self._branches_jsonl(), [ref.model_dump(mode="json")])
        self._roster_cache = None
        return FsBranch(self, branch_id, name, description, created, index=index,
                        parent_id=fork_from.branch_id if fork_from else "")

    def _branch_by_id(self, branch_id: str) -> FsBranch | None:
        data = _read_json(self._branch_dir(branch_id) / "meta.json")
        if data is None:
            return None
        meta = BranchMeta(**data)
        return FsBranch(self, meta.branch_id, meta.name, meta.description, meta.created,
                        index=meta.index,
                        parent_id=meta.fork_from.branch_id if meta.fork_from else "")

    def get_branch(self, name: str) -> Branch | None:
        data = _read_json(self._ref_path(name))
        if data is None:
            return None
        ref = BranchRef(**data)
        return FsBranch(self, ref.branch_id, ref.name, ref.description, ref.created,
                        parent_id=ref.parent_id)

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

    def _roster(self) -> list[dict[str, Any]]:
        """``branches.jsonl`` in memory. One load, then pure lookups until the next write."""
        if self._roster_cache is None:
            self._roster_cache = _read_jsonl(self._branches_jsonl())
        return self._roster_cache

    def children_of(self, name: str) -> list[BranchAddr]:
        """The parent-child tree, read off branch data — never off a trajectory.

        A roster row carries ``parent_id`` (written once, at creation), so the tree is a filter
        over one cached list: no meta read, no commit scan. Complete from the moment a child
        exists — including children whose parent never commits again. What it cannot say is
        *when in the trajectory* the child appeared; that is ``CommitRef.children``.
        """
        parent = self.get_branch(name)
        if parent is None:
            raise FileNotFoundError(f"branch '{name}' not found")
        parent_id = parent.ref.branch_id
        return [
            BranchAddr(branch_id=row["branch_id"], branch_name=row["name"], branch_index=index)
            for index, row in enumerate(self._roster(), start=1)
            if (row.get("parent_id") or "") == parent_id
        ]

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

    def archive_branch(self, name: str) -> list[BranchAddr]:
        """Archive this branch and its subtree, deepest first. The pointer goes; the data stays."""
        branch = self.get_branch(name)
        if branch is None:
            raise FileNotFoundError(f"branch '{name}' not found")

        children: dict[str, list[BranchAddr]] = {}
        for index, row in enumerate(self._roster(), start=1):
            parent_id = row.get("parent_id") or ""
            if parent_id:
                children.setdefault(parent_id, []).append(
                    BranchAddr(branch_id=row["branch_id"], branch_name=row["name"], branch_index=index)
                )

        order: list[BranchAddr] = []

        def walk(address: BranchAddr) -> None:
            for child in children.get(address.branch_id, []):
                walk(child)
            order.append(address)

        walk(branch.address())

        archived: list[BranchAddr] = []
        for address in order:
            ref = self._ref_path(address.branch_name)
            if not ref.exists():
                # Already archived, e.g. a child archived on its own before its parent. Not a
                # failure and not a silent omission: the return value is "what this call
                # archived", so a branch that was already out of the active set is not in it.
                continue
            ref.unlink()
            archived.append(address)
            self._emit("archived", address)
        self._roster_cache = None
        return archived

    def on_event(self, callback: Callable[[BranchEvent], None]) -> Callable[[], None]:
        self._hooks.append(callback)

        def _off() -> None:
            try:
                self._hooks.remove(callback)
            except ValueError:
                pass

        return _off

    def _emit(self, kind: str, branch: BranchAddr, coord: str = "") -> None:
        """Tell whoever is watching. A failing subscriber never turns a write into a failure."""
        if not self._hooks:
            return
        event = BranchEvent(kind=kind, branch=branch, coord=coord)
        for callback in list(self._hooks):
            try:
                callback(event)
            except Exception:  # noqa: BLE001 — the write already happened
                logger.exception("branch event hook failed: %s", kind)


def new_local_memento(root: Path) -> FsMemento:
    """Build an owner-level filesystem memento."""
    return FsMemento(root)
