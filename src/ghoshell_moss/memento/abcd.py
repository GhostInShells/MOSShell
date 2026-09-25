"""memento — a lean cognitive-trajectory index (RC contract).

This file is the contract layer (abstract classes + data models). It is far leaner
than v3 (abc.py) by design:

- A commit is **an anchor that can restore one session** and carries no moment. The
  moment bytes live in the agent session; memento only holds the key in ``metadata``
  (conventionally ``session_id`` + ``tail``) pointing at them. Restoration belongs to
  the consumer (dolores) — memento stores, it does not interpret.
- A branch is a directory: ``meta.json`` / ``commits.jsonl`` (append-only, in order,
  authoritative) / ``commit_notes.jsonl`` (side-channel summary, safe to lose and
  rebuild).
- A fork is a **reference** (read the parent's and the child's ``commits.jsonl`` and
  splice them into one continuous trajectory), never a copy.
- **Coordinate = ``{branch_index}-{commit_seq}``** (e.g. ``27-1027``): the branch
  index is the creation ordinal in the owner's ``branches.jsonl``, the commit seq is
  the row ordinal in the branch's ``commits.jsonl``. Both are frozen at production
  time and both files are append-only, so a coordinate is permanently stable. Seeing
  and citing both use the coordinate; ``id`` (ULID) is global identity only and never
  enters a view.
- **Single source of truth for the message**: a commit carries no message; the message
  lives only in a Note (one home), last-wins. ``commit(message=...)`` is convenience
  sugar = anchor + a seeded Note, but on disk the message still lands only in the Note
  row. title = first line of the message, body = the rest (matching git ``-m``).
- **Node space (convention, not mechanism)**: every commit may own a directory derived
  from a convention, holding its own materials; the index file ``MEMENTO.md`` is that
  node's self-explaining entry point. memento only computes the address
  (``CommitView.memento_path``) and observes whether it exists (``CommitView.memento``);
  it **never creates, reads, or cleans** content — content belongs to whoever writes
  it, an unmanaged asset area whose versioning is that writer's call.
  ``Branch.ensure_memento`` is the writer-side explicit get-or-create (directory +
  index template) and never happens on a read path.
- Dropped in this version: confluence / segment as a first-class citizen / per-commit
  directory / staging / dual-source message.

memento binds to a single local path and is not strongly tied to a storage backend; the
contract surface does not own any storage implementation. The operation surface is
sketched with ABCs, the data models are concrete pydantic. The implementation layer is
the model's sovereignty; the contract layer is human-reviewed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import AwareDatetime, BaseModel, Field
import ulid

__all__ = [
    "COMMIT_MEMENTO_FILE",
    "CommitRef",
    "Note",
    "BranchRef",
    "ForkRef",
    "BranchMeta",
    "CommitView",
    "BranchView",
    "Branch",
    "Memento",
]

COMMIT_MEMENTO_FILE = "MEMENTO.md"
"""File name of a commit node's index — part of the on-disk convention.

A directory containing this file is a memento node. The name is unique to memento on
purpose: ``README.md`` exists in every repository on earth, so a model grepping
downward would drown in noise, while every hit on this name is a real node. See
``CommitView.memento_path`` for where it lives.
"""


def _unique_id() -> str:
    return str(ulid.ULID())


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


class CommitRef(BaseModel):
    """One commit anchor — one row of ``commits.jsonl``. **A bare anchor, no message.**

    Carries no moment. ``metadata`` holds "the keys sufficient to restore one session";
    the concrete convention is defined by the consumer (e.g. ``{'ref': ..., 'prev_turn':
    ...}``) and memento only gets/sets it, never parses it. This commit's message lives
    in the corresponding Note — the single source of truth.

    ``seq`` is the in-branch commit ordinal (1-based), frozen at commit time — the
    second half of the coordinate.
    """

    id: str = Field(default_factory=_unique_id)
    seq: int = Field(
        default=0,
        description="In-branch commit ordinal (1-based), frozen at commit time — second half of the coordinate.",
    )
    metatype: str = Field(default="", description="Type of commit produced, e.g. 'session'.")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Opaque extension, conventionally defined by the consumer. memento only gets/sets, never parses.",
    )
    created: AwareDatetime = Field(default_factory=_now_utc)


class Note(BaseModel):
    """A commit's one and only summary — one row of ``commit_notes.jsonl``.

    A side channel: the producer (who is also the consumer) writes and reads it on its
    own schedule; producing it slowly is fine, and strict ordering is not required.
    A later write for the same ``commit_id`` overrides the earlier one (last-wins). The
    message here is the commit's only message; title = first line (truncated), body =
    the rest. Length is guaranteed by the producer (admission control is the producing
    side's job).

    A non-empty ``error`` marks this message as a **broken placeholder**: the terminal
    state written when side-channel production failed fatally (readonly, never retried).
    The commit still enters a view as a message (the model must perceive that a broken
    commit appeared in its summary), but folding keeps only the first of consecutive
    broken commits (see ``BranchView``).
    """

    commit_id: str = Field(...)
    message: str = Field(default="", description="The commit's summary message (matching git -m).")
    error: str = Field(
        default="",
        description="Broken-placeholder marker; non-empty means the message is not a real summary (terminal state, never retried).",
    )


class BranchRef(BaseModel):
    """Points at a branch's current pointer — the content of ``{branch_name}.ref.json``."""

    name: str = Field(...)
    description: str = Field(default="")
    branch_id: str = Field(...)
    created: AwareDatetime = Field(default_factory=_now_utc)


class ForkRef(BaseModel):
    """A fork reference — locates the parent branch plus the anchor commit.

    ``branch_id`` locates the parent's ``commits.jsonl`` (under per-branch storage a
    commit_id alone cannot locate a file). ``commit_id`` is the anchor in the parent
    where the fork happened. The parent's name is recoverable from ``branches.jsonl``
    and is therefore not duplicated here. Deliberately narrow: no name / description /
    created.
    """

    branch_id: str = Field(description="Parent branch_id (a reference, not a copy).")
    commit_id: str = Field(description="The anchor commit id in the parent where this branch forked.")


class BranchMeta(BaseModel):
    """Branch metadata written at creation — ``branches/{branch_id}/meta.json``."""

    branch_id: str = Field(...)
    index: int = Field(
        default=0,
        description="Branch ordinal within the owner (1-based), frozen at creation — first half of the coordinate.",
    )
    name: str = Field(...)
    description: str = Field(default="")
    metatype: str = Field(default="", description="Type of the branch when produced.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Extensible metadata.")
    fork_from: ForkRef | None = Field(
        default=None,
        description="Fork reference (parent pointer + anchor commit). None for a root branch.",
    )
    created: AwareDatetime = Field(default_factory=_now_utc)


class CommitView(BaseModel):
    """The read-side unit of one commit — ``CommitRef`` + its ``Note`` + the coordinate.

    ``BranchView`` holds it and ``Branch`` exposes it. ``ref`` carries every anchor fact
    (created / metatype / metadata); ``note`` is the home of the message and is None when
    no Note exists. ``coord`` = ``{branch_index}-{ref.seq}`` — the address models and
    humans use to see and cite a commit (``id`` is a ULID, expensive in tokens, and never
    enters a view). title = first line of the message (truncated), body = the rest;
    truncation for rendering is the consumer's job — this model only does deterministic
    derivation.

    ``memento`` is the absolute path of this commit's node index file (``MEMENTO.md``)
    **when that file exists**, and None otherwise. It is the only observation surface of
    the node space: memento reports that a node is there, never what is inside it.
    """

    branch_index: int = Field(description="First half of the coordinate: branch ordinal within the owner (1-based).")
    ref: CommitRef = Field(description="The commit anchor — created / metatype / metadata all live here.")
    note: Note | None = Field(default=None, description="The home of the commit's message; None when no Note exists.")
    memento: Path | None = Field(
        default=None,
        description="Absolute path of the node index file (MEMENTO.md) when it exists; None = this commit has no node yet.",
    )

    @property
    def coord(self) -> str:
        """The coordinate ``{branch_index}-{commit_seq}`` as a string (e.g. ``27-1027``)."""
        return f"{self.branch_index}-{self.ref.seq}"

    @property
    def seq(self) -> int:
        return self.ref.seq

    @property
    def created(self) -> AwareDatetime:
        return self.ref.created

    @property
    def message(self) -> str:
        return self.note.message if self.note is not None else ""

    @property
    def title(self) -> str:
        return self.message.split("\n", 1)[0]

    @property
    def body(self) -> str:
        lines = self.message.split("\n")
        return "\n".join(lines[1:])

    @property
    def error(self) -> str:
        """Broken-placeholder marker (empty = a real summary, or not produced yet)."""
        return self.note.error if self.note is not None else ""

    @property
    def is_broken(self) -> bool:
        """Whether this commit's message is a broken placeholder (terminal side-channel failure)."""
        return bool(self.error)

    def memento_path(self, root: Path) -> Path:
        """Absolute path of this commit's node index file — pure convention, exists or not.

        Shape: ``{root}/commits/{YYYY}/{MM}/cmt_{branch_index}-{seq}/MEMENTO.md``

        - ``{YYYY}/{MM}`` comes from ``created`` in **UTC**; ``created`` is immutable
          after write, so the address is permanently stable. Any rewrite of ``created``
          would move the node — that is why it must never happen.
        - ``cmt_{coord}`` is memento's internal unique address (the coordinate). The
          prefix keeps a bare ``27-1027`` from reading like a date on disk.
        - This method is the single definition of that layout: the implementation and
          every consumer call it, nobody re-derives the shape.
        """
        utc = self.created.astimezone(timezone.utc)
        return (
            root / "commits" / f"{utc.year:04d}" / f"{utc.month:02d}"
            / f"cmt_{self.branch_index}-{self.ref.seq}" / COMMIT_MEMENTO_FILE
        )


class BranchView(BaseModel):
    """A branch's budgeted context view — the projection of a read branch.

    Folding policy (near = detailed, far = coarse, deterministic): ``latest`` gives detail
    for the most recent N commits; ``history`` folds earlier commits into summaries (Note
    preferred); ``previous`` is the compacted recap of the forked parent.

    Broken commits (non-empty ``Note.error``) still enter ``history`` / ``latest`` so the
    model perceives them, but **consecutive** broken commits keep only the first (a run of
    broken placeholders must not flood the view).

    Budget guarantee: the producer guarantees Note message length (admission); the render
    action is issued by the consumer. memento does not enforce a budget, it only projects.
    """

    name: str
    description: str
    branch_id: str
    index: int = Field(description="Branch ordinal within the owner (1-based) — first half of the coordinate.")
    created: AwareDatetime
    commit_id: str = Field(description="Id of the most recent commit, for talking to this commit.")
    previous: "BranchView | None" = Field(default=None, description="Forked parent branch (reference).")
    history: list[CommitView] = Field(default_factory=list, description="Earlier commits, folded summaries.")
    latest: list[CommitView] = Field(default_factory=list, description="Most recent commits, detail.")
    commits_total: int = Field(default=0)


class Branch(ABC):
    """A branch object pointing at a directory: meta.json / commits.jsonl / commit_notes.jsonl.

    Reads come in two tiers whose meaning is "do we go back to disk / do we take write
    permission":

    - A sync read is the cached fast path; ``a<name>`` is a real disk read (aiofiles) and
      its docstring is marked **"IO-costly"**. The explicit ``a*`` tier exists to give the
      caller an offload point, so it is not forced into ``asyncio.to_thread`` with a
      hand-rolled task it will forget to await.
    - ``async with branch:`` is **write-permission gating**: take the branch's write
      permission or fail fast. This is a non-enforcing, cooperative probe, not a security
      or permission boundary. The default implementation is a process-level file lock and
      **does not enforce per-file admission constraints** (an out-of-band writer that edits
      files directly cannot be stopped — that is a closed storage system's job).
    - The write operations ``acommit`` / ``afork`` / ``anote`` declare that they hold the
      coroutine lock, and they are **mutually exclusive** (only one writer on a branch at
      a time).
    """

    @property
    @abstractmethod
    def ref(self) -> BranchRef:
        """The current pointer (name / description / branch_id / created). Available at construction."""

    @property
    @abstractmethod
    def path(self) -> Path:
        """The branch's own directory path."""

    @property
    @abstractmethod
    def index(self) -> int:
        """Branch ordinal within the owner (1-based, frozen at creation) — first half of the coordinate."""

    @abstractmethod
    def meta(self) -> BranchMeta:
        """Branch metadata (metatype / metadata / fork_from). Available at construction (root_path
        suffices); never async."""

    @abstractmethod
    def commits(self) -> list[CommitRef]:
        """``commits.jsonl`` in order (cached fast path). See ``acommits`` for the IO-costly read."""

    @abstractmethod
    async def acommits(self) -> list[CommitRef]:
        """IO-costly: re-read ``commits.jsonl`` from disk with aiofiles."""

    @abstractmethod
    def notes(self) -> dict[str, Note]:
        """commit_id -> Note, later writes overriding earlier ones (cached fast path)."""

    @abstractmethod
    async def anotes(self) -> dict[str, Note]:
        """IO-costly: re-read ``commit_notes.jsonl``."""

    @abstractmethod
    def get_commit(self, seq: int) -> CommitView | None:
        """Take a CommitView by in-branch seq (cached fast path); out of range (including < 1) returns None.

        The node observation (``CommitView.memento``) is probed here: it is a dynamic fact.
        """

    @abstractmethod
    async def aget_commit(self, seq: int) -> CommitView | None:
        """IO-costly: re-read from disk and take a CommitView by seq."""

    @abstractmethod
    def commit(
        self,
        *,
        message: str = "",
        metatype: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> CommitRef:
        """Write one commit anchor to ``commits.jsonl`` (append-only). A bare anchor, no message.

        Convenience sugar: when ``message`` is non-empty, a Note is seeded internally (the
        message still lands only in the Note — single source of truth). The default
        implementation appends atomically with O_APPEND; a single writer takes this fast
        path. The message may be filled in later (``note``).
        """

    @abstractmethod
    async def acommit(
        self,
        *,
        message: str = "",
        metatype: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> CommitRef:
        """Locked write, mutually exclusive with ``afork``: commit while holding the lock."""

    @abstractmethod
    def note(self, commit_id: str, message: str, error: str = "") -> Note:
        """Step 2 (side channel, may be late): write the one and only summary message for an existing
        commit_id, last-wins.

        Called by a background task or the producer outside the critical path, a little later;
        length is guaranteed by the producer (admission). title is derived from the first line.
        A non-empty ``error`` is the terminal broken-placeholder state (side-channel fatal
        failure): the commit still enters the view but is never retried.
        """

    @abstractmethod
    async def anote(self, commit_id: str, message: str, error: str = "") -> Note:
        """Locked write, mutually exclusive with ``afork``: note while holding the lock."""

    @abstractmethod
    def view(self, *, n: int = 10) -> BranchView:
        """The most recent n commits in detail + earlier ones folded; includes the forked parent's
        recap.

        Not a pure cache fast path: the detail window probes each commit's node index
        (``CommitView.memento``) because the node space is dynamic by definition.
        """

    @abstractmethod
    async def aview(self, *, n: int = 10) -> BranchView:
        """IO-costly: aggregate multiple files into a view; the most expensive read."""

    @abstractmethod
    def query_commits(
        self,
        *,
        from_date: datetime | None = None,
        until_date: datetime | None = None,
    ) -> list[CommitRef]:
        """Filter this branch's own ``commits.jsonl`` by created-time range. No full-text search."""

    @abstractmethod
    async def aquery_commits(
        self,
        *,
        from_date: datetime | None = None,
        until_date: datetime | None = None,
    ) -> list[CommitRef]:
        """IO-costly: as above, but re-read from disk."""

    @abstractmethod
    def fork(self, name: str, description: str = "") -> "Branch":
        """Fork a new branch (reference the parent, never copy). The new branch's ``commits.jsonl``
        starts empty and fork_from points back at the parent commit; the read side splices the
        parent's and the child's files into one continuous trajectory. Single-writer sync fast
        path."""

    @abstractmethod
    async def afork(self, name: str, description: str = "") -> "Branch":
        """Locked write, mutually exclusive with ``acommit``: create directory + meta + ref, a
        multi-file write, while holding the lock."""

    @abstractmethod
    def ensure_memento(self, seq: int) -> Path:
        """Get or create this commit's node: create the directory and seed the index template;
        return the index file's absolute path.

        Idempotent: an existing directory is left alone, and an existing index file is **never
        overwritten** (that is the writer's content). A seq that does not exist raises KeyError.
        This is the writer-side interface; readers use the ``CommitView.memento`` field. memento
        seeds only the template — everything after that is the writer's.
        """

    @abstractmethod
    async def __aenter__(self) -> "Branch":
        """Write-permission gating: take the branch's write permission or fail fast. A
        non-enforcing cooperative convention; a process-level file lock by default."""

    @abstractmethod
    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """Release write permission."""


class Memento(ABC):
    """The owner-level entry point. One owner = one directory.

    Under ``owner/``: ``branches.jsonl`` (the historical branch roster), ``{branch_name}.ref.json``
    (the current pointer), ``branches/{branch_id}/`` (each branch's directory), and
    ``commits/{YYYY}/{MM}/cmt_{coord}/`` (the commit node space, created only by writers).
    """

    @property
    @abstractmethod
    def root(self) -> Path:
        """The memento root directory (bound to a single local path; not strongly tied to a storage
        backend)."""

    @abstractmethod
    def create_branch(
        self,
        name: str,
        description: str = "",
        *,
        metatype: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> Branch:
        """Create a new branch. The name must not exist, otherwise NameError."""

    @abstractmethod
    def get_branch(self, name: str) -> Branch | None:
        """Take a branch by name; None when it does not exist."""

    @abstractmethod
    def list_branches(self) -> list[BranchRef]:
        """The active branch list (glob of ref.json)."""

    @abstractmethod
    def get_branch_by_index(self, index: int) -> Branch | None:
        """Take a branch by its ordinal within the owner; None when it does not exist."""

    @abstractmethod
    def resolve_commit(self, coord: str) -> CommitView | None:
        """Resolve a coordinate (e.g. ``"27-1027"``) back into a CommitView; malformed or missing
        returns None."""

    @abstractmethod
    def delete_branch(self, name: str) -> None:
        """Delete a name pointer (remove ref.json); the branch directory and its commits survive."""
