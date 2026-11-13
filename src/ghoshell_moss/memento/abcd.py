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
- **References go both ways, and the two halves are not symmetric.** A commit view has an
  outgoing half — the addresses the message is sent to, written once on the Note
  (``recipients``), because an address is part of the message. And an incoming half — the
  references this branch had received before it made this commit, absorbed from the branch's
  inbox into the anchor (``CommitRef.confluences``), carrying the source coordinate and the
  title and nothing else. The incoming half exists for **folded history**: a rebuild must be
  able to see what was asked of it without reading every commit and without walking into
  another branch.
- Dropped in this version: segment as a first-class citizen / per-commit directory /
  a staging directory / dual-source message.

memento binds to a single local path and is not strongly tied to a storage backend; the
contract surface does not own any storage implementation. The operation surface is
sketched with ABCs, the data models are concrete pydantic. The implementation layer is
the model's sovereignty; the contract layer is human-reviewed.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator
import ulid

__all__ = [
    "COMMIT_MEMENTO_FILE",
    "CommitRef",
    "BranchAddr",
    "Child",
    "Confluence",
    "Note",
    "CommitMessage",
    "BranchRef",
    "BranchEvent",
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
    """The record's instant — UTC, one axis for every row.

    The stored value is an absolute point; which clock a reader sees is a render-time
    question (the body's OS zone), never something the record freezes.
    """
    return datetime.now(timezone.utc)


class BranchAddr(BaseModel):
    """A branch address — the half of a reference that says *who*.

    Its boundary is one thing only: a branch. Never a commit, never an owner, never a URL.

    Three fields do three different jobs, and they must not take each other's place:

    - ``branch_id`` is the identity. **Every resolution starts here.**
    - ``branch_index`` is the position — the first half of the coordinate, so a reference can
      be rendered as ``{index}-{seq}`` without touching the disk.
    - ``branch_name`` is for the eye. Names rot (a deleted name can be given to a new branch),
      so a name must never be used to resolve a relationship — only to show one.
    """

    branch_id: str = Field(description="Stable identity (ULID). The only field a relationship resolves on.")
    branch_name: str = Field(description="Human-readable name. Display only — never a resolution key.")
    branch_index: int = Field(
        default=0,
        description="Branch ordinal within the owner (1-based) — first half of the coordinate; 0 = not filled.",
    )


class BranchEvent(BaseModel):
    """Something happened at a branch address — told to whoever is watching.

    Thin on purpose: it carries the address, never the content. A subscriber that needs the
    content reads it from that address; memento does not interpret what it announced.

    Kinds: ``commit`` (an anchor became durable, ``coord`` filled), ``confluence`` (a
    reference landed in this branch's inbox), ``child`` (a branch was created from this one),
    ``archived`` (this branch left the active set).
    """

    kind: str = Field(description="commit | confluence | child | archived")
    branch: BranchAddr = Field(description="Where it happened.")
    coord: str = Field(default="", description="Commit coordinate when the event is about one.")
    created: AwareDatetime = Field(default_factory=_now_utc)


class Confluence(BaseModel):
    """One message that arrived from another branch before this commit — the *what* half.

    It exists for the folded history, not for the live session. A rebuild must be able to see
    what was asked of it without reading every commit and without walking into another branch,
    so the title is copied here even though the full message lives at the origin.

    Four semantics, and none of them may be dropped:

    1. it came from a branch — carried by the type itself;
    2. that branch's name — ``branch.branch_name``, for display and for business logic;
    3. its real identity and position — ``branch.branch_id`` and ``commit_seq``;
    4. what it said — ``title``, the only copied content.

    How this is rendered is the renderer's business; the fields are the contract.
    """

    branch: BranchAddr = Field(description="Which branch it came from.")
    commit_seq: int = Field(
        description="That branch's seq for the commit that sent it — the address half only an incoming reference has.",
    )
    title: str = Field(description="The only copied content: the first line of the source note.")
    created: AwareDatetime = Field(default_factory=_now_utc, description="When this reference was delivered.")

    @property
    def coord(self) -> str:
        """The source commit's coordinate (``{index}-{seq}``); empty when the index was never filled."""
        if self.branch.branch_index <= 0:
            return ""
        return f"{self.branch.branch_index}-{self.commit_seq}"


class Child(BaseModel):
    """A branch created from this one — the *outgoing* half of the same exchange.

    Mirror of ``Confluence``, and the opposite in one respect only: its author is this branch
    itself. Nothing has to notify us of what we did, so there is no delivery — the row is
    appended when the branch is created and absorbed by the next commit, exactly like an
    incoming reference. That is what makes the creation show up in history at a position,
    instead of living only in the child's meta.

    A ``Child`` says *that a branch was made from here and when*, never *why*. Whether it was
    a delegation, a split of the same line, or anything else, is the consumer's vocabulary —
    memento keeps one operation and one record.
    """

    branch: BranchAddr = Field(description="The created branch's address.")
    created: AwareDatetime = Field(default_factory=_now_utc, description="When it was created.")


class CommitRef(BaseModel):
    """One commit anchor — one row of ``commits.jsonl``. **An anchor, no message.**

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
    children: list[Child] = Field(
        default_factory=list,
        description=(
            "Branches created from this commit's branch before it was written, absorbed from "
            "the branch's ``children.jsonl`` at commit time. Says that and when, never why."
        ),
    )
    confluences: list[Confluence] = Field(
        default_factory=list,
        description=(
            "What this commit received from other branches: everything the branch's inbox held "
            "when this anchor was written, absorbed at commit time. Only the source coordinate "
            "and the title travel; the rest is read at the origin when it is actually needed."
        ),
    )
    rejected: list[str] = Field(
        default_factory=list,
        description=(
            "Staged rows this commit could not absorb. The commit still lands — a branch must "
            "never need file-level repair to move again — and the raw lines are kept in the "
            "branch's ``*.rejected.jsonl``. Empty on a healthy commit."
        ),
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
    recipients: list[BranchAddr] = Field(
        default_factory=list,
        description=(
            "Who this message is sent to: one address per receiving branch — the outgoing half "
            "of a commit view. Written once, with the note; a later rewrite of ``message`` must "
            "keep it (pass ``recipients=None`` to keep the existing value, ``[]`` to declare "
            "that this note is sent to nobody). Delivery is a separate act; this list is only "
            "the declaration."
        ),
    )


_COMMIT_OP_LINE = re.compile(
    r"^(?P<op>[A-Za-z][A-Za-z0-9_-]*)"
    r"(?:\((?P<target>[^()]*)\))?"
    r"\s*[:：]\s*(?P<content>.*)$"
)
"""The one op-line grammar — shared by stored messages and the case agent's reply parser.

Only the first line is ever matched. ``op`` is an ASCII identifier, ``target`` may not
contain parentheses, and the colon may be ASCII or full-width.
"""

_OP_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")


class CommitMessage(BaseModel):
    """A commit's message as a protocol value — the typed reading of a Note's message.

    The first line may be an op line, ``op(target): content``; everything after it is
    ``body``. Prose is a legal message: ``op`` is empty and ``content`` is the whole first
    line. ``Note`` remains the storage row; this value object is how its message is read
    and written.

    Reads are tolerant: :meth:`parse` never raises unless ``strict=True``; a first line
    that does not match the grammar is prose. ``op`` is a **candidate reading, not a
    guarantee** — a title like ``D11：...`` parses as op ``D11``; a consumer must validate
    the word against its own vocabulary. Construction is strict: these validators are the
    one place a protocol message can be built.

    ``to_str`` / ``str`` is the whole write path: the storage API keeps taking strings.
    """

    model_config = ConfigDict(frozen=True)

    op: str = Field(default="", description="The op word; empty when the message is prose.")
    target: str = Field(default="", description="The parenthesized target; empty when there is none.")
    content: str = Field(
        default="",
        description="The human title after ``op(target):``; the whole first line when op is empty.",
    )
    body: str = Field(
        default="",
        description="Everything after the first line; the separator blank line is stripped.",
    )

    @field_validator("op")
    @classmethod
    def _op_is_an_identifier(cls, value: str) -> str:
        value = value.strip()
        if value and _OP_WORD.fullmatch(value) is None:
            raise ValueError(f"op must be an ASCII identifier, got {value!r}")
        return value

    @field_validator("target")
    @classmethod
    def _target_has_no_parentheses(cls, value: str) -> str:
        value = value.strip()
        if "(" in value or ")" in value:
            raise ValueError(f"target cannot contain parentheses, got {value!r}")
        return value

    @field_validator("content")
    @classmethod
    def _content_is_one_line(cls, value: str) -> str:
        value = value.strip()
        if "\n" in value:
            raise ValueError("content is the first line only; put the rest in body")
        return value

    @field_validator("body")
    @classmethod
    def _body_is_the_remainder(cls, value: str) -> str:
        return value.strip("\n")

    @model_validator(mode="after")
    def _target_belongs_to_an_op(self) -> CommitMessage:
        if self.target and not self.op:
            raise ValueError("a target belongs to an op; pass an op or drop the target")
        return self

    @property
    def headline(self) -> str:
        """``op(target): content`` on one line — the full first line of the message."""
        if not self.op:
            return self.content
        head = f"{self.op}({self.target})" if self.target else self.op
        return f"{head}: {self.content}" if self.content else head

    def to_str(self) -> str:
        """The message as it is stored: the headline, then a blank line and the body."""
        line = self.headline
        return f"{line}\n\n{self.body}" if self.body else line

    def __str__(self) -> str:
        return self.to_str()

    @classmethod
    def parse(cls, message: str, *, strict: bool = False) -> CommitMessage:
        """Read a message; tolerant by default, strict only when asked.

        ``strict=True`` raises when the first line is not an op line — for writers and
        tests, never for readers of existing notes.
        """
        first, _, rest = message.partition("\n")
        first = first.strip()
        match = _COMMIT_OP_LINE.match(first)
        if match is None:
            if strict:
                raise ValueError(f"a commit message must start with an op line, got {first!r}")
            return cls(content=first, body=rest)
        return cls(
            op=match.group("op"),
            target=(match.group("target") or "").strip(),
            content=match.group("content").strip(),
            body=rest,
        )


class BranchRef(BaseModel):
    """Points at a branch's current pointer — the content of ``{branch_name}.ref.json``."""

    name: str = Field(...)
    description: str = Field(default="")
    branch_id: str = Field(...)
    parent_id: str = Field(
        default="",
        description=(
            "Branch this one was created from; empty for a root. Written once, with the row, so "
            "the parent-child tree is derivable from the roster alone — no meta read, no "
            "trajectory scan."
        ),
    )
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

    ``memento`` is the absolute path of this commit's node index file (``MEMENTO.md``) when
    that file exists, and None otherwise. Every read fills it — a branch view collects the
    node space in one walk, a single commit read stats one path — so a node is observable
    wherever it sits, not only inside the detail window. It is the only observation surface
    of the node space: memento reports that a node is there, never what is inside it.
    """

    branch_index: int = Field(description="First half of the coordinate: branch ordinal within the owner (1-based).")
    ref: CommitRef = Field(description="The commit anchor — created / metatype / metadata all live here.")
    note: Note | None = Field(default=None, description="The home of the commit's message; None when no Note exists.")
    memento: Path | None = Field(
        default=None,
        description="Absolute path of the node index file (MEMENTO.md) when it exists; None = this commit has no node.",
    )
    ground: str = Field(
        default="",
        description=(
            "What this commit's directory says about itself: the ``commit_view`` slot's stdout, "
            "filled by the generation that runs the hooks. Empty means the view was generated "
            "without them — not that the directory is empty."
        ),
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

    def message_view(self) -> CommitMessage:
        """The message as a protocol value — a pure parse, never cached or stored.

        ``message`` stays the raw string and the single authority; this method is the
        read-side reading, so a view holds no derived state and two calls parse twice.
        """
        return CommitMessage.parse(self.message)

    @property
    def title(self) -> str:
        return self.message_view().headline

    @property
    def body(self) -> str:
        return self.message_view().body

    @property
    def recipients(self) -> list[BranchAddr]:
        """Who this commit's message is sent to — read off its note; empty while there is no note.

        The outgoing half of a commit view. A commit whose message has not been written yet has
        no recipients, because the address is part of the message.
        """
        return self.note.recipients if self.note is not None else []

    @property
    def rejected(self) -> list[str]:
        """What this commit could not absorb — visible on the anchor, not buried in a log."""
        return self.ref.rejected

    @property
    def children(self) -> list[Child]:
        """Branches created from this one before this commit — the other outgoing half."""
        return self.ref.children

    @property
    def confluences(self) -> list[Confluence]:
        """What arrived from other branches before this commit — the incoming half of the view.

        Read off the anchor, so a rebuild sees it without touching another branch.
        """
        return self.ref.confluences

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
    preferred); ``previous`` locates the forked parent and ``fork_coord`` says *where* on it
    this branch left.

    ``previous`` is a **pointer, not a copy**: the parent arrives with its identity and its
    size and with no commits of its own. Rendering a parent's history into a child's view was
    never read by anyone and cost a recursive rebuild per level; whoever wants the parent's
    content reads it by coordinate.

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
    previous: "BranchView | None" = Field(default=None, description="Forked parent branch (identity + size only).")
    fork_coord: str = Field(
        default="",
        description=(
            "Coordinate on ``previous`` where this branch forked (``{index}-{seq}``); empty for "
            "a root branch or when the anchor commit is no longer resolvable."
        ),
    )
    history: list[CommitView] = Field(default_factory=list, description="Earlier commits, folded summaries.")
    latest: list[CommitView] = Field(default_factory=list, description="Most recent commits, detail.")
    commits_total: int = Field(default=0)
    ground: str = Field(
        default="",
        description=(
            "What this branch's directory says about itself: the ``branch_view`` slot's stdout, "
            "filled by the generation that runs the hooks (the plan, the open tasks, whatever the "
            "directory chooses to render). Empty means the view was generated without hooks."
        ),
    )


class Branch(ABC):
    """A branch object pointing at a directory: meta.json / commits.jsonl / commit_notes.jsonl /
    confluences.jsonl (the inbox of references received from other branches).

    Reads come in two tiers whose meaning is "do we go back to disk":

    - A sync read is the cached fast path; ``a<name>`` is a real disk read (aiofiles) and
      its docstring is marked **"IO-costly"**. The explicit ``a*`` tier exists to give the
      caller an offload point, so it is not forced into ``asyncio.to_thread`` with a
      hand-rolled task it will forget to await.
    - ``async with branch:`` is **write permission**: one writer at a time, taken or failed
      fast. How that is achieved is the implementation's business — a file lock, an
      in-memory lock, or nothing where a single writer is guaranteed. It is a cooperative
      convention, not a security boundary, and it **does not enforce per-file admission
      constraints** (an out-of-band writer editing files directly cannot be stopped — that
      is a closed storage system's job).
    - Only read-modify-write operations need it: ``commit`` (``seq`` is derived) and
      ``fork``. ``note`` and anything purely appended do not — an append is already atomic,
      and a duplicate is idempotent downstream.
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
            recipients: list[BranchAddr] | None = None,
    ) -> CommitRef:
        """Write one commit anchor to ``commits.jsonl`` (append-only). An anchor, no message.

        This is the branch's only read-modify-write: ``seq`` is derived from what is already
        there. That — not the append — is the single reason an implementation may need one
        writer at a time. Every other write here is a plain append.

        Convenience sugar: when ``message`` is non-empty, a Note is seeded internally (the
        message still lands only in the Note — single source of truth). The message may be
        filled in later through ``note``.

        ``recipients`` is only meaningful together with ``message``: an address belongs to a
        message, and there is no message without a note.
        """

    @abstractmethod
    async def acommit(
            self,
            *,
            message: str = "",
            metatype: str = "",
            metadata: dict[str, Any] | None = None,
            recipients: list[BranchAddr] | None = None,
    ) -> CommitRef:
        """The same commit, taken under the branch's write permission.

        The gate is an implementation concern, not part of this contract's meaning: a file
        lock, an in-memory lock, or nothing at all are all valid answers to "one writer at a
        time". What the contract promises is the one-writer property — never the mechanism.
        """

    @abstractmethod
    def note(
            self,
            commit_id: str,
            message: str,
            error: str = "",
            *,
            recipients: list[BranchAddr] | None = None,
    ) -> Note:
        """Step 2 (side channel, may be late): write the one and only summary message for an existing
        commit_id, last-wins.

        Called by a background task or the producer outside the critical path, a little later;
        length is guaranteed by the producer (admission). title is derived from the first line.
        A non-empty ``error`` is the terminal broken-placeholder state (side-channel fatal
        failure): the commit still enters the view but is never retried.

        ``recipients`` is the one field that is **not** last-wins. On the first note it is
        stored as given (``None`` = send to nobody, same as ``[]``); on a rewrite ``None``
        keeps the stored value and a non-empty list replaces it. Otherwise a summarizer
        restating a message would silently re-address it.

        Delivery happens here, once, on the first note: each recipient's inbox gets one
        ``Confluence`` line. It raises on failure — the note is already durable, and a retry
        is safe because the receiver de-duplicates on the source coordinate. What is *not*
        recorded anywhere is a delivery receipt: the trajectory holds the declaration only.
        """

    @abstractmethod
    async def anote(
            self,
            commit_id: str,
            message: str,
            error: str = "",
            *,
            recipients: list[BranchAddr] | None = None,
    ) -> Note:
        """Async variant of ``note``.

        No write permission is needed: a note is one appended line, and the only decision it
        makes (first write or rewrite) is idempotent downstream.
        """

    # -- inbox: what other branches sent here ---------------------------------- #

    @abstractmethod
    def confluences(self) -> list[Confluence]:
        """The inbox: references that arrived and have not been absorbed into a commit yet.

        A row leaves the inbox by being absorbed, not by being deleted on read — the file is
        claimed (renamed) at commit time. A non-empty inbox is therefore a queryable fact
        ("something arrived and this branch has not stepped since"), not a queue.
        """

    @abstractmethod
    async def aconfluences(self) -> list[Confluence]:
        """IO-costly: re-read the inbox from disk."""

    @abstractmethod
    def pending_children(self) -> list[Child]:
        """Branches created from this one that the next commit has not absorbed yet.

        **Not** the answer to "which branches are mine" — that question is answered by the
        roster and each child's ``fork_from``, and it holds whether or not this branch ever
        commits again (``Memento.children_of``). This is only the gap between creating a child
        and recording that creation in the trajectory.
        """

    @abstractmethod
    async def apending_children(self) -> list[Child]:
        """IO-costly: re-read the pending children from disk."""

    @abstractmethod
    def add_child(self, child: Child) -> None:
        """Record that a branch was created from this one — written by the parent itself.

        No notification and no delivery: the author is the branch, so the row is simply
        appended and the next commit absorbs it.
        """

    @abstractmethod
    def confluence(self, confluence: Confluence) -> None:
        """Deliver one reference into this branch's inbox — the only cross-branch write.

        The caller is the *source* branch, at the moment its note landed. The row must stay
        small (a single O_APPEND line is atomic only below ``PIPE_BUF``), and no write
        permission is taken: the receiver's gate belongs to its own writer, and a delivery
        must not queue behind an unrelated commit. Duplicates are dropped on the source
        coordinate.
        """

    @abstractmethod
    def view(self, *, n: int = 10, until: int | None = None) -> BranchView:
        """The most recent n commits in detail + earlier ones folded; the forked parent as a pointer.

        ``until`` renders the trajectory **as it stood at a commit**: everything with
        ``seq < until``. That is how memory is re-rendered at compaction — the cut commit's
        span stays verbatim in the surface, so the snapshot must stop just before it rather
        than summarise it twice. ``until=None`` = the whole line.

        Not a pure cache fast path: the detail window probes each commit's node index
        (``CommitView.memento``) because the node space is dynamic by definition.
        """

    @abstractmethod
    async def aview(self, *, n: int = 10, until: int | None = None) -> BranchView:
        """IO-costly: aggregate multiple files into a view; the most expensive read."""

    @abstractmethod
    async def aview_with_hooks(self, *, n: int = 10, until: int | None = None) -> BranchView:
        """``aview`` plus the directory's own rendering, in ``BranchView.ground``.

        The explicit opt-in to the ``branch_view`` slot: ``view`` / ``aview`` stay hook-free
        (cheap, sync-callable, used inside other projections), and this one runs the mounted
        hooks on every call — a view is a snapshot of a moving directory, so a cached hook
        output would be the wrong thing to hand out.
        """

    @abstractmethod
    async def aget_commit_with_hooks(self, seq: int) -> CommitView | None:
        """``aget_commit`` plus the ``commit_view`` slot's stdout in ``CommitView.ground``."""

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
    def children_of(self, name: str) -> list[BranchAddr]:
        """Every branch created from this one — **derived**, not read off a trajectory.

        The roster gives all branches and each child's ``meta.fork_from`` gives its parent, so
        the answer is a join over branch data: complete from the moment a child exists, and
        independent of whether the parent ever commits again. What it cannot say is *when in
        the trajectory* the child appeared — that is ``CommitRef.children``, and it only exists
        once the parent has absorbed the staged ``Child`` row.
        """

    @abstractmethod
    def resolve_commit(self, coord: str) -> CommitView | None:
        """Resolve a coordinate (e.g. ``"27-1027"``) back into a CommitView; malformed or missing
        returns None."""

    @abstractmethod
    def archive_branch(self, name: str) -> list[BranchAddr]:
        """Archive this branch and its whole subtree — **deepest first**.

        Archiving is not deletion: it removes the name pointer (``{name}.ref.json``) and
        nothing else. The directory, the meta and every commit survive, and the roster keeps
        the row, so the branch stays resolvable by index and can be restored from meta later.
        Children are archived before their parent, so a crash midway leaves a parent that still
        exists and can account for what happened, rather than live fragments of a vanished one.

        Returns what was archived, in the order it was archived. memento cannot know whether a
        branch is in use — stopping live work is the caller's business.
        """

    @abstractmethod
    def on_event(self, callback: Callable[[BranchEvent], None]) -> Callable[[], None]:
        """Subscribe to branch events; returns an unsubscribe.

        Fired after the write is durable, synchronously. A subscriber that raises is the
        consumer's bug: it is logged and never turns a completed write into a failure.
        """
