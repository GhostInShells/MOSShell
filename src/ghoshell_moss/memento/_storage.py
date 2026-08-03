"""
Memento storage row types — FORMAT v3 disk-format data structures.

Discipline (2026-08-03, 5th contract reopen):
- No magic values: all type discriminators, status labels, and kind constants
  are module-level names. Inline string literals are prohibited.
- No silent failures: storage write paths raise typed exceptions or log at
  warning/error level. Swallowing data-loss with try/except is prohibited.
- No short abbreviations: field names and variable names are self-describing.
  The exception is the ``t`` field (row type discriminator) and ``id`` /
  ``type`` fields — these match the JSON key convention established by
  the contract (FORMAT.md) and are the same names used throughout the codebase.
- Key technical decisions are documented as inline comments at the relevant
  code site. No cross-references to FEATURE.md or other discussion files.
- This module defines storage schemas only — the private row types used for
  jsonl serialization. API models live in ``abc.py``. The two layers share
  field shapes where the contract mandates it, but evolve independently.
  Changing a row type must not force a consumer-facing API change, and vice
  versa. The projection between them is ``fs_memento.py``'s responsibility.

Storage layout (FORMAT v3)::

    {owner}/
      meta.json
      branches.jsonl          # BranchMetaRow — owner-level branch index
      heads/{name}            # plain-text: branch_uid
      ws/{branch_uid}/
        ref                   # JSON: BranchRef
        staging.jsonl         # StagingRow (t:"moment")
        status.json           # BranchStatusRow
      commits/{Y-m}/cmt_{ULID}/
        meta.json             # CommitMetaRow
        moments.jsonl         # FrozenMomentRow (t:"moment")
        notes.jsonl           # CommitNoteRow | MomentNoteRow
      commits.jsonl           # CommitRefRow — owner-level timeline
      checkouts.jsonl         # CheckoutRow — fork events
      confluents.jsonl        # ConfluentRow — reference-confluence events
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field
from ulid import ULID

from ghoshell_common.contracts import LoggerItf

# ── Logger ────────────────────────────────────────────────────────────────────

_STORAGE_LOGGER_NAME = "moss.memento.storage"


def new_storage_logger(name: str | None = None) -> LoggerItf:
    """Create a logger for a storage component. Accepts an optional sub-name."""
    full_name = f"{_STORAGE_LOGGER_NAME}.{name}" if name else _STORAGE_LOGGER_NAME
    return logging.getLogger(full_name)


# ── Type discriminators (jsonl "t" field values, no magic strings) ─────────────

ROW_TYPE_MOMENT = "moment"
ROW_TYPE_COMMIT = "commit"
ROW_TYPE_COMMIT_NOTE = "commit_note"
ROW_TYPE_MOMENT_NOTE = "moment_note"
ROW_TYPE_COMMIT_REF = "commit_ref"
ROW_TYPE_BRANCH_META = "branch_meta"
ROW_TYPE_CHECKOUT = "checkout"
ROW_TYPE_CONFLUENT = "confluent"

# ── Commit kind constants ──────────────────────────────────────────────────────

COMMIT_KIND_SEMANTIC = "semantic"
COMMIT_KIND_MECHANICAL = "mechanical"

# ── Branch lifecycle status constants ──────────────────────────────────────────

BRANCH_STATUS_ACTIVE = "active"
BRANCH_STATUS_FROZEN = "frozen"
BRANCH_STATUS_ABANDONED = "abandoned"

# ── Confluent kind constants ───────────────────────────────────────────────────

CONFLUENT_KIND_REFERENCE = "reference"

# ── ID prefixes ────────────────────────────────────────────────────────────────

COMMIT_ID_PREFIX = "cmt_"
BRANCH_ID_PREFIX = "brn_"
MOMENT_ID_PREFIX = "mmt_"

# ── ID generators ──────────────────────────────────────────────────────────────


def new_commit_identifier() -> str:
    """Generate a commit id: ``cmt_`` + ULID."""
    return f"{COMMIT_ID_PREFIX}{ULID()}"


def new_branch_identifier() -> str:
    """Generate a branch uid: ``brn_`` + ULID."""
    return f"{BRANCH_ID_PREFIX}{ULID()}"


def new_moment_identifier() -> str:
    """Generate a moment id: ``mmt_`` + ULID."""
    return f"{MOMENT_ID_PREFIX}{ULID()}"


def _now_utc() -> datetime:
    """UTC wall-clock timestamp. Logical ordering is enforced by append order."""
    return datetime.now(timezone.utc)


# ── Shared ref model (storage-side representation of BranchRef) ────────────────


class BranchRefFields(BaseModel):
    """Storage-side ref fields. Same shape as the API-side ``BranchRef`` in ``abc.py``.

    Defined here so that storage row types can embed the ref shape without
    importing from the API layer. The implementation layer projects between
    this and ``abc.BranchRef``.
    """

    origin: str = Field(
        default="",
        description="Owner who produced the target commit. Empty = current owner.",
    )
    commit_id: str = Field(
        ...,
        description="Target commit id (cmt_ prefix).",
    )
    moment_id: str | None = Field(
        default=None,
        description="Slice cutoff moment id (inclusive). None = entire commit.",
    )


# ── Row types (storage-layer pydantic models) ──────────────────────────────────


class StagingRow(BaseModel):
    """A moment row in staging.jsonl — unfrozen, overwritable by id (last-wins)."""

    t: Literal["moment"] = Field(
        default=ROW_TYPE_MOMENT,
        description="Row type discriminator for jsonl streaming parsers.",
    )
    id: str = Field(
        ...,
        description="Moment id (mmt_ prefix). Unique within the branch's staging "
                    "and commit space.",
    )
    created: datetime = Field(
        default_factory=_now_utc,
    )
    type: str = Field(
        ...,
        description="Payload schema identifier (e.g. 'pydantic_ai.messages/v2'). "
                    "Used by consumers to select a codec.",
    )
    content: str = Field(
        default="",
        description="Plain-text projection of the moment. v3 new: enables "
                    "structural views (CLI window, commit show) to render "
                    "human-readable output without parsing opaque payload.",
    )
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Opaque payload. Memento never parses this.",
    )
    threads: list[str] = Field(
        default_factory=list,
        description="Thread tags. Write-time annotation; can be updated "
                    "via moment_note after freezing.",
    )


class FrozenMomentRow(StagingRow):
    """A moment row in moments.jsonl inside a frozen commit directory.

    Shares the same field structure as ``StagingRow``. The semantic difference
    (frozen / overwriteable) is physical (directory location), not schema-level.
    A separate class preserves the ability to diverge the schemas if needed.
    """


class CommitMetaRow(BaseModel):
    """meta.json in a commit directory — immutable after creation."""

    commit_id: str = Field(...)
    created: datetime = Field(default_factory=_now_utc)
    parent: BranchRefFields | None = Field(
        default=None,
        description="Single-parent anchor. None only for the root commit "
                    "of an owner. ancestry is frozen at commit time.",
    )
    kind: str = Field(
        default=COMMIT_KIND_SEMANTIC,
        description="semantic (agent self-declared) or mechanical (rule-triggered).",
    )


class CommitRefRow(BaseModel):
    """A row in commits.jsonl — owner-level append-only timeline.

    Logical ordering is physical append order (POSIX O_APPEND).
    ULID timestamp (via commit_id) provides a secondary sort.
    """

    t: Literal["commit_ref"] = Field(
        default=ROW_TYPE_COMMIT_REF,
    )
    commit_id: str = Field(...)
    branch_uid: str = Field(
        ...,
        description="Branch uid (brn_ prefix) that produced this commit. "
                    "v3: was 'branch' (name string), now stable uid.",
    )
    parent: BranchRefFields | None = Field(default=None)
    ts: datetime = Field(default_factory=_now_utc)
    kind: str = Field(default=COMMIT_KIND_SEMANTIC)


class CommitNoteRow(BaseModel):
    """A commit-level note row in notes.jsonl — last-wins by commit_id."""

    t: Literal["commit_note"] = Field(default=ROW_TYPE_COMMIT_NOTE)
    ref: str = Field(..., description="Commit id being interpreted.")
    title: str = Field(default="")
    body: str = Field(default="")
    ts: datetime = Field(default_factory=_now_utc)
    by: str = Field(default="")


class MomentNoteRow(BaseModel):
    """A moment-level annotation row in notes.jsonl — last-wins by moment id."""

    t: Literal["moment_note"] = Field(default=ROW_TYPE_MOMENT_NOTE)
    ref: str = Field(..., description="Moment id being annotated.")
    threads: list[str] = Field(default_factory=list)
    ts: datetime = Field(default_factory=_now_utc)
    by: str = Field(default="")


class BranchMetaRow(BaseModel):
    """A row in branches.jsonl — owner-level branch index, append-only.

    Appended on branch creation and on every status change. Members are
    indexed by uid (stable across name changes). The ``name`` field records
    the head name at the time of append; name history is reconstructed from
    the append log.
    """

    t: Literal["branch_meta"] = Field(default=ROW_TYPE_BRANCH_META)
    uid: str = Field(
        ...,
        description="Stable branch identity (brn_ prefix). Never changes.",
    )
    name: str = Field(
        ...,
        description="Head name at the time this row was appended. "
                    "May differ from the current head name after rename or抢占.",
    )
    status: str = Field(
        default=BRANCH_STATUS_ACTIVE,
        description="Branch lifecycle: active | frozen | abandoned.",
    )
    fork_ref: BranchRefFields = Field(
        ...,
        description="Checkout origin — the commit (and optionally moment) "
                    "this branch was created from.",
    )
    created: datetime = Field(default_factory=_now_utc)
    updated: datetime = Field(default_factory=_now_utc)


class CheckoutRow(BaseModel):
    """A row in checkouts.jsonl — fork event record.

    Appended by the DERIVING side (the owner who creates the new branch).
    No cross-owner coordination is needed; this is a local append.
    """

    t: Literal["checkout"] = Field(default=ROW_TYPE_CHECKOUT)
    branch_uid: str = Field(
        ...,
        description="The newly created branch (brn_ prefix).",
    )
    from_ref: BranchRefFields = Field(
        ...,
        description="The commit (and optionally moment) this branch forks from.",
    )
    owner: str = Field(
        ...,
        description="Local owner who performed the checkout.",
    )
    created: datetime = Field(default_factory=_now_utc)


class ConfluentRow(BaseModel):
    """A row in confluents.jsonl — reference-confluence event record.

    Records that one branch submitted its reference to another branch
    (one stream flowing into another — 融汇). This is a reference confluent:
    the recipient branch's commit parent chain is NOT altered. The confluent
    event is stored in this separate append-only log.

    Appended by the RECEIVING side (the owner of the target branch).
    """

    t: Literal["confluent"] = Field(default=ROW_TYPE_CONFLUENT)
    from_branch_uid: str = Field(...)
    from_owner: str = Field(...)
    to_branch_uid: str = Field(...)
    to_owner: str = Field(...)
    kind: str = Field(default=CONFLUENT_KIND_REFERENCE)
    created: datetime = Field(default_factory=_now_utc)


class BranchStatusRow(BaseModel):
    """status.json in a branch workspace directory.

    Carries the branch's current lifecycle status and a free-form task
    description. This file is in-place overwrite (not append-only) — it
    represents the dynamic workspace state.
    """

    status: str = Field(
        default=BRANCH_STATUS_ACTIVE,
        description="Lifecycle: active | frozen | abandoned.",
    )
    title: str = Field(
        default="",
        description="Human-readable branch title. One line.",
    )
    description: str = Field(
        default="",
        description="Free-form task description or current plan summary. "
                    "Structural complement to PLAN.md — machine-readable status, "
                    "while PLAN.md is the human-authored task document.",
    )
    updated: datetime = Field(default_factory=_now_utc)
