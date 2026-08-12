"""
Memento contract — the public API surface for the cognitive-trajectory system.

Memento organises the "dynamic and static" of a thinking entity's context:
static (completed anchors) live in commits; dynamic (in-progress thought)
lives in branch workspaces. The core motivation is parallel thinking +
history traceability and readability — compaction is a consumer, not the
purpose.

Design invariants (self-contained, no external references needed):

- **Immutable members, open interpretation**: commit members (moments) are
  frozen on creation. Interpretation (title / body / threads) is appended,
  last-wins, with full version history always addressable.
- **Opaque payload**: Memento never parses payload. The ``type`` field is a
  discriminator that consumers use to select a codec. Memento transparently
  stores and returns it.
- **Incarnation from commit, never from staging**: staging has no stable
  identifier; nothing can refer to it as a birth point. Every new line of
  thought starts from a frozen commit anchor.
- **Single-parent chain**: each commit has exactly one parent (or none for
  the root). A reference confluent (one branch submitting its ref to another)
  is recorded as a separate associative event (see ConfluentRecord) — the
  commit parent chain is never altered.
- **Branch identity is stable; name is a movable pointer**: every branch
  has a ``branch_identifier`` (``brn_`` ULID) that never changes. The human-
  readable name is a head file pointing to it — rename, reset, and抢占
  (name takeover) are all name-level operations that do not touch the
  branch workspace or its commit trail.

Storage-layer separation:
  The pydantic models in this module define the CONSUMER-FACING API.
  Storage row types (with jsonl discriminators like the ``t`` field) live
  in ``_storage.py`` — they are private to the implementation layer.
  The two layers share field shapes where the contract mandates it, but
  evolve independently. Changing a storage format must not force a
  consumer API change, and vice versa. This separation fixes the root
  defect of the previous implementation, where ``MomentRecord`` served
  double duty as API envelope and disk row.

Disk format: ``FORMAT.md`` (same directory). Design lineage is preserved
in the feature discuss directory — this module is self-explaining.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Iterable, Literal, Protocol, Sequence, runtime_checkable

from pydantic import AwareDatetime, BaseModel, Field
from ulid import ULID

__all__ = [
    # trailer
    "TRAILER_THREAD",
    "TRAILER_RESUMES",
    "TRAILER_SUSPENDS",
    "TRAILER_KIND",
    "TRAILER_MEMENTO_REF",
    "split_trailers",
    "join_trailers",
    "trailer_values",
    # id generators
    "new_commit_id",
    "new_branch_id",
    "new_moment_id",
    # data models
    "MomentRecord",
    "Commit",
    "CommitNote",
    "CommitView",
    "BranchRef",
    "CommitRef",
    "BranchWindow",
    "CommitDetail",
    "BranchMeta",
    "CheckoutRecord",
    "ConfluentRecord",
    # hook
    "MementoHooks",
    "NullHooks",
    # Protocol / ABC
    "Line",
    "Memento",
    # exceptions
    "MementoError",
    "ReadonlyLineError",
    "LineNotFoundError",
    "BranchNotFoundError",
    "CommitNotFoundError",
    "MomentFrozenError",
    "MomentNotInCommitError",
    "EmptyStagingError",
]

# ── ID generators ──────────────────────────────────────────────────────────────

_COMMIT_ID_PREFIX = "cmt_"
_BRANCH_ID_PREFIX = "brn_"
_MOMENT_ID_PREFIX = "mmt_"


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def new_commit_id() -> str:
    """Generate a commit id: ``cmt_`` + ULID. Prefix guarantees grep reversibility."""
    return f"{_COMMIT_ID_PREFIX}{ULID()}"


def new_branch_id() -> str:
    """Generate a branch uid: ``brn_`` + ULID. Stable identity — survives rename, reset, and抢占."""
    return f"{_BRANCH_ID_PREFIX}{ULID()}"


def new_moment_id() -> str:
    """Generate a moment id: ``mmt_`` + ULID."""
    return f"{_MOMENT_ID_PREFIX}{ULID()}"


# ── trailer tools (FORMAT.md §6) ─────────────────────────────────────────────

TRAILER_THREAD = "Thread"
TRAILER_RESUMES = "Resumes"
TRAILER_SUSPENDS = "Suspends"
TRAILER_KIND = "Kind"
TRAILER_MEMENTO_REF = "Memento-Ref"

_TRAILER_RE = re.compile(r"^([A-Za-z][A-Za-z0-9-]*): .+$")


def split_trailers(body: str) -> tuple[str, list[tuple[str, str]]]:
    """Split body into (text, [(key, value)...]). Trailer block = contiguous Key: Value lines at end."""
    if not body:
        return "", []
    lines = body.split("\n")
    split_at = len(lines)
    for i in range(len(lines) - 1, -1, -1):
        if _TRAILER_RE.match(lines[i]):
            split_at = i
        else:
            break
    text_lines = lines[:split_at]
    trailer_lines = lines[split_at:]
    if text_lines and text_lines[-1] == "":
        text_lines.pop()
    trailers: list[tuple[str, str]] = []
    for line in trailer_lines:
        m = _TRAILER_RE.match(line)
        if m:
            key = line[: m.end(1)]
            value = line[m.end(1) + 2 :]
            trailers.append((key, value))
    return ("\n".join(text_lines), trailers)


def join_trailers(text: str, trailers: Iterable[tuple[str, str]]) -> str:
    """Assemble body = text + blank line + trailer block."""
    lines = text.split("\n") if text else []
    for k, v in trailers:
        lines.append(f"{k}: {v}")
    return "\n".join(lines)


def trailer_values(trailers: Iterable[tuple[str, str]], key: str) -> list[str]:
    """Return all values for a given trailer key, in appearance order."""
    return [v for k, v in trailers if k == key]


# ── Data models ────────────────────────────────────────────────────────────────


class MomentRecord(BaseModel):
    """Moment envelope. Payload is opaque to memento — stored and returned as-is.

    Mutability: same id in staging overwrites (last-wins). Once frozen into a
    commit directory, writing the same id to staging raises MomentFrozenError.
    Threads are interpretation — they can be updated via annotate_moment even
    after freezing.
    """

    id: str = Field(
        description="Moment id (mmt_ prefix). Unique within the branch's "
                    "staging and commit space."
    )
    created: AwareDatetime = Field(default_factory=_now_utc)
    type: str = Field(
        description="Payload schema identifier (e.g. 'pydantic_ai.messages/v2')."
    )
    content: str = Field(
        default="",
        description="Plain-text projection of this moment. Populated by the "
                    "recording agent / runner. Enables structural views (CLI "
                    "window, commit show) to render human-readable output "
                    "without parsing opaque payload. May be empty for moments "
                    "that have no natural text representation.",
    )
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Opaque payload. Memento never parses this.",
    )
    threads: list[str] = Field(
        default_factory=list,
        description="Thread tags. Write-time annotation; can be updated "
                    "after freezing via annotate_moment.",
    )


class Commit(BaseModel):
    """Frozen cognitive anchor. Members are immutable — changing them invalidates
    all descendant branch parent chains."""

    id: str = Field(default_factory=new_commit_id, description="Stable identifier (cmt_ ULID).")
    created: AwareDatetime = Field(default_factory=_now_utc)


class CommitNote(BaseModel):
    """Commit interpretation. Append-only multi-version, last-wins.

    title: one-line summary for window rendering and search.
    body: text + trailer block, full-replacement semantics.
    """

    ref: str = Field(description="Commit id being interpreted.")
    title: str = Field(default="", description="One-line summary.")
    body: str = Field(default="", description="Text + trailer block.")
    ts: AwareDatetime = Field(default_factory=_now_utc, description="Display/diagnostic timestamp.")
    by: str = Field(default="", description="Interpretation author.")

    def text(self) -> str:
        """Body text without trailers."""
        return split_trailers(self.body)[0]

    def trailers(self) -> list[tuple[str, str]]:
        return split_trailers(self.body)[1]

    def threads(self) -> list[str]:
        return trailer_values(self.trailers(), TRAILER_THREAD)


class CommitView(BaseModel):
    """Commit + current interpretation (last-wins). note_seq is a rendering stamp."""

    commit: Commit
    note: CommitNote = Field(description="Current (latest) interpretation.")
    note_seq: int = Field(description="Interpretation version, 0-based.")

    @property
    def id(self) -> str:
        return self.commit.id

    def summary(self) -> str:
        return self.note.title or self.note.text()


class BranchRef(BaseModel):
    """Pointer to a commit, with optional moment-level slice.

    When ``moment_id`` is None the entire commit is referenced. When set,
    only moments from the first up to and including ``moment_id`` are included
    (inclusive prefix slice). An empty prefix is impossible at the type level.

    ``origin`` is the owner who produced the target commit. When equal to the
    current owner it is typically left as the default (empty string).
    """

    origin: str = Field(
        default="",
        description="Owner who produced the target commit. Empty = current owner.",
    )
    commit_id: str = Field(description="Target commit id (cmt_ prefix).")
    moment_id: str | None = Field(
        default=None,
        description="Slice cutoff moment id (inclusive). None = entire commit.",
    )


class CommitRef(BaseModel):
    """A row's worth of commits.jsonl — owner-level append-only timeline.

    Row order = physical append order. ULID timestamp (via commit_id) is a
    secondary sort, not the authority.
    """

    commit_id: str
    branch: str = Field(description="Branch uid (brn_ prefix) at freeze time.")
    parent: BranchRef | None = Field(default=None, description="Parent commit. None for root.")
    ts: AwareDatetime = Field(default_factory=_now_utc, description="Freeze timestamp.")
    kind: Literal["semantic", "mechanical"] = Field(default="semantic")


class BranchWindow(BaseModel):
    """Fast-path window render: folded summaries + recent detail frames."""

    summaries: list[CommitView] = Field(description="Folded zone: last M commit interpretations.")
    details: list[MomentRecord] = Field(description="Detail zone: recent N frames (including staging).")


class CommitDetail(BaseModel):
    """Full return from show <commit_id>."""

    commit: Commit
    moments: list[MomentRecord] = Field(description="Frozen member frames, in commit order.")
    notes: list[CommitNote] = Field(description="All interpretation versions, in append order.")


class BranchMeta(BaseModel):
    """Branch identity and lifecycle. uid is stable; name is a movable pointer.

    This is the API projection of the branch index (branches.jsonl).
    """

    uid: str = Field(
        default_factory=new_branch_id,
        description="Stable branch identity (brn_ ULID). Never changes.",
    )
    name: str = Field(
        default="main",
        description="Current head name. May differ from a previous name "
                    "after rename or抢占 (name takeover).",
    )
    status: Literal["active", "frozen", "abandoned"] = Field(
        default="active",
        description="Branch lifecycle: active (in use) / frozen (completed, "
                    "read-only) / abandoned (discarded, kept for traceability).",
    )
    fork_ref: BranchRef = Field(
        ...,
        description="Checkout origin — the commit this branch was created from.",
    )
    created: AwareDatetime = Field(
        default_factory=_now_utc,
    )
    updated: AwareDatetime = Field(
        default_factory=_now_utc,
    )


class CheckoutRecord(BaseModel):
    """A fork event — recorded when a branch is created from a commit anchor.

    This is the API projection of checkouts.jsonl.
    """

    branch_uid: str = Field(
        ...,
        description="The newly created branch.",
    )
    from_ref: BranchRef = Field(
        ...,
        description="The anchor commit (and optionally moment) forked from.",
    )
    owner: str = Field(
        ...,
    )
    created: AwareDatetime = Field(
        default_factory=_now_utc,
    )


class ConfluentRecord(BaseModel):
    """A reference-confluent event — one branch submits its reference to another.

    The recipient branch's commit parent chain is NOT altered; the confluent
    is an associative event stored in a separate append-only log. This is
    "提交引用而非内容" (submit the reference, not the content) — it eliminates
    the entire conflict-resolution problem domain.

    Metaphor: a tributary stream flowing into the main river (融汇). The
    tributary's own path is unchanged; the river records that it received
    the tributary's waters at this point.
    """

    from_branch_uid: str = Field(...)
    from_owner: str = Field(...)
    to_branch_uid: str = Field(...)
    to_owner: str = Field(...)
    kind: Literal["reference"] = Field(default="reference")
    created: AwareDatetime = Field(default_factory=_now_utc)


# ── Hook protocol ──────────────────────────────────────────────────────────────


@runtime_checkable
class MementoHooks(Protocol):
    """Fire-and-forget event callbacks. Errors must not affect the core write path."""

    def on_record_staged(self, line: str, record: MomentRecord) -> None:
        """Called after a moment is appended to staging."""
        ...

    def on_commit(self, line: str, view: CommitView) -> None:
        """Called after commit() completes. view carries the initial interpretation."""
        ...

    def on_reinterpreted(self, commit_id: str, view: CommitView) -> None:
        """Called after annotate() appends a new interpretation."""
        ...

    def on_line_created(self, name: str, from_ref: BranchRef | None) -> None:
        """Called after create_line() creates a new branch."""
        ...

    def on_line_deleted(self, name: str) -> None:
        """Called after delete_line() removes a head name. The branch workspace survives."""
        ...

    def on_branch_checkout(self, branch_identifier: str, from_ref: BranchRef) -> None:
        """Called after a branch is created (fork event). v3 new."""
        ...


class NullHooks:
    """All-noop default implementation."""

    def on_record_staged(self, line: str, record: MomentRecord) -> None:
        pass

    def on_commit(self, line: str, view: CommitView) -> None:
        pass

    def on_reinterpreted(self, commit_id: str, view: CommitView) -> None:
        pass

    def on_line_created(self, name: str, from_ref: BranchRef | None) -> None:
        pass

    def on_line_deleted(self, name: str) -> None:
        pass

    def on_branch_checkout(self, branch_identifier: str, from_ref: BranchRef) -> None:
        pass


# ── Exceptions ─────────────────────────────────────────────────────────────────


class MementoError(Exception):
    """Base exception for the memento system."""


class ReadonlyLineError(MementoError):
    """Write operation called on a read-only line handle."""


class LineNotFoundError(MementoError):
    """Line name does not exist."""


class BranchNotFoundError(MementoError):
    """Branch uid does not exist in the owner's branch index."""


class CommitNotFoundError(MementoError):
    """Commit id does not exist."""


class MomentFrozenError(MementoError):
    """Moment id has been frozen into a commit — staging write rejected."""


class MomentNotInCommitError(MementoError):
    """BranchRef.moment_id is not in the target commit's member set."""


class EmptyStagingError(MementoError):
    """Staging is empty — commit rejected."""


# ── Line (branch handle, Protocol) ─────────────────────────────────────────────


@runtime_checkable
class Line(Protocol):
    """Handle bound to one branch (timeline).

    A branch = stable uid + movable name + BranchRef + staging.
    Obtain via ``memento.get_line(identifier)`` where identifier is a uid
    or a head name. Cross-owner: ``get_line(uid, origin=other)`` returns
    a read-only handle.
    """

    @property
    def branch_identifier(self) -> str:
        """Stable branch uid (brn_ prefix). Never changes across rename or reset."""
        ...

    @property
    def name(self) -> str:
        """Current head name. May change — use uid for stable identity."""
        ...

    @property
    def ref(self) -> BranchRef | None:
        """Current ref pointer. None = root line, never committed."""
        ...

    @property
    def readonly(self) -> bool:
        """True when this is a cross-owner read-only handle."""
        ...

    # ── Write ──

    def record(self, record: MomentRecord) -> None:
        """Append a moment to staging. Same id overwrites (last-wins).

        :raise ReadonlyLineError:
        :raise MomentFrozenError: the id is already frozen in a commit directory.
        """
        ...

    def commit(
        self,
        text: str = "",
        *,
        kind: Literal["semantic", "mechanical"] = "semantic",
        threads: Sequence[str] = (),
        resumes: Sequence[str] = (),
        suspends: Sequence[str] = (),
        extra_trailers: Sequence[tuple[str, str]] = (),
        boundary_moment_id: str | None = None,
        by: str = "",
    ) -> CommitView:
        """Freeze staging → new commit.

        :param kind: 'semantic' (agent self-declared) or 'mechanical' (rule-triggered).
        :param boundary_moment_id: freeze only the prefix of staging up to and
            including this moment id (inclusive). Remaining moments stay in staging.
            None = freeze all.
        :param resumes: thread commit ids being resumed.
        :param suspends: thread names being suspended.
        :raise ReadonlyLineError:
        :raise EmptyStagingError:
        """
        ...

    # ── Read ──

    def staging(self) -> list[MomentRecord]:
        """Unfrozen moments, in first-write order."""
        ...

    def log(self) -> list[CommitView]:
        """Branch history along the parent chain."""
        ...

    def window(self, *, detail_n: int = 10, summary_m: int = -1) -> BranchWindow:
        """Sliding window fast path. detail_n = recent N frames (including staging).
        summary_m = interpretation summaries before the detail zone, -1 = all.
        """
        ...


# ── Memento (owner facade, ABC) ────────────────────────────────────────────────


class Memento(ABC):
    """Owner facade. One instance is bound to one owner.

    Cross-owner read-only: ``get_line(uid, origin=other)`` returns a read-only
    handle. Cross-owner write does not exist — a new thinking space = a new
    Memento instance for a new owner.

    Degenerate form: single line + auto-commit usage only touches
    ``get_line("main")`` + record/commit/staging/log — fork vocabulary
    never appears.
    """

    @property
    @abstractmethod
    def owner(self) -> str:
        """Owner identity."""
        ...

    # ── Branch management ──

    @abstractmethod
    def create_line(
        self,
        name: str,
        *,
        from_ref: BranchRef | None = None,
        overlay: dict[str, Any] | None = None,
    ) -> Line:
        """Create a new branch.

        Generates a stable branch uid (brn_ ULID), creates the workspace
        directory, writes the head file (name → uid), and appends a row to
        branches.jsonl.

        :param name: initial head name for this branch. Can be changed later
            via rename or抢占 (name takeover).
        :param from_ref: checkout origin. None = root line (no predecessor,
            first commit starts this owner's history).
        :param overlay: incarnation injection. Only meaningful when from_ref
            crosses an owner boundary. Landed in owner meta.json, immutable
            after creation.
        :return: write handle for the new branch.
        """
        ...

    @abstractmethod
    def get_line(self, identifier: str, *, origin: str | None = None) -> Line:
        """Get a line handle.

        ``identifier`` is first resolved as a branch uid (brn_ prefix);
        if not found, resolved as a head name. ``origin`` != self.owner
        returns a read-only cross-owner handle.

        :raise BranchNotFoundError:
        :raise LineNotFoundError: (for name-based lookup)
        """
        ...

    @abstractmethod
    def list_lines(self) -> list[str]:
        """Active branch head names. Equivalent to globbing the heads/ directory."""
        ...

    @abstractmethod
    def list_all_branches(self) -> list[BranchMeta]:
        """All branches (including abandoned and frozen). Reads branches.jsonl.

        This is the full-search API. The degenerate path (single line, active
        only) uses ``list_lines()``.
        """
        ...

    @abstractmethod
    def delete_line(self, name: str) -> None:
        """Remove a head name. The branch workspace and all commits survive.

        This is name-level deletion, not branch deletion. The uid workspace
        and its commit trail remain intact and are still reachable via
        ``list_all_branches()`` and ``get_line(uid)``.

        :raise LineNotFoundError:
        """
        ...

    # ── Commit read & interpretation ──

    @abstractmethod
    def show(self, commit_id: str) -> CommitDetail:
        """Expand a commit: all frozen members + all interpretation versions.

        :raise CommitNotFoundError:
        """
        ...

    @abstractmethod
    def notes(self, commit_id: str) -> list[CommitNote]:
        """All interpretation versions for a commit, in append order.

        :raise CommitNotFoundError:
        """
        ...

    @abstractmethod
    def annotate(
        self, commit_id: str, title: str = "", body: str = "", *, by: str = ""
    ) -> CommitView:
        """Aperture two: append a commit interpretation. Full-replacement
        semantics; prior versions are always addressable via ``notes()``.

        :raise CommitNotFoundError:
        """
        ...

    @abstractmethod
    def annotate_moment(
        self, commit_id: str, moment_id: str, threads: Sequence[str], *, by: str = ""
    ) -> None:
        """Moment-level interpretation — replace threads wholesale. Legal after
        freezing (threads are interpretation, not members).

        :raise CommitNotFoundError:
        :raise MomentNotInCommitError:
        """
        ...

    # ── Owner-level queries ──

    @abstractmethod
    def log(self) -> list[CommitRef]:
        """commits.jsonl physical timeline — all lines' commits in append order."""
        ...

    @abstractmethod
    def commit_space(self, commit_id: str) -> str:
        """Absolute path to the commit's autonomous directory. Resolved at
        runtime; never persisted in memento structures.

        :raise CommitNotFoundError:
        """
        ...

    @abstractmethod
    def checkouts(self) -> list[CheckoutRecord]:
        """Fork events from checkouts.jsonl, in append order.

        Forward direction: branches created by this owner from any anchor.
        """
        ...

    @abstractmethod
    def confluences(self) -> list[ConfluentRecord]:
        """Reference-confluent events from confluents.jsonl, in append order.

        Forward direction: confluents received by this owner's branches.
        """
        ...
