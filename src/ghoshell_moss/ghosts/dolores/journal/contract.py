"""Journal contract — the ghost's second-order memory of its own days.

The journal is not a log. A log records what happened; the journal is the ghost's own
compression of what happened, written by the ghost, for the ghost. Three levels — day,
month, year — each a higher compression of the one below, written together so the
narrative reads downward without a gap.

**First order and second order.** The mechanism supplies the first order: where the
documents live, how one is shaped, which status a day is in, how many characters each
level may spend, when a write is due. The ghost supplies the second order: what is worth
recording, the prose, the summary, and each month's own thinking (its ``AGENTS.md``).
This contract belongs to the first order and stops there — no method here writes prose,
and no implementation may grow one.

**Files are the truth.** The models below are read-side projections of documents on
disk, not a second store. Every read re-parses and every write re-reads, so there is no
in-memory state to drift from the files. Unknown frontmatter keys are preserved and
re-emitted: a future version may add fields this one does not know, and reading an old
file must never silently drop them.

**The conventions are on the surface.** The path methods and the frontmatter pair below
are the default design made visible: a reader sees where documents live and how they
serialize without opening the implementation. They are defaults, not promises — the file
implementation may lay the files out differently, and the contract's own method surface
never carries a path. Declaring them here is what stops the next writer from inventing a
second convention.

**The clock is external.** Nothing here schedules itself. A trigger (boot repair, a
compaction boundary, the final sleep) reads ``specification`` and the day's data, composes
the prompt on its own side, and hands it to a writer. That composition is deliberately
not part of this contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, datetime
from pathlib import Path
from typing import ClassVar, Literal

import frontmatter
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field
from typing_extensions import Self

from ghoshell_moss.contracts.configs import ConfigType

__all__ = [
    "BudgetLevel",
    "BudgetReport",
    "Daily",
    "DateView",
    "DocStatus",
    "DoloresJournal",
    "JournalConfig",
    "JournalDoc",
    "JournalStatus",
    "Monthly",
    "Yearly",
]

DocStatus = Literal["pending", "writing", "closed"]
"""
Where one day stands.

``pending``: the day exists — its first anchor opened it — and no writer has worked on it.
``writing``: a writer is mid-flight; a crashed writer is indistinguishable from a slow one,
so the repair path treats it as unfinished. ``closed``: written, rolled up to month and
year, and checked against the budgets.
"""

BudgetLevel = Literal["year", "month", "day"]
"""The three levels, each with its own budget and its own document."""


class JournalConfig(ConfigType):
    """The journal's first-order knobs.

    Stored in the ghost's own config store, so a ghost can tune its budgets without
    touching the ghost-wide assembly. Budgets are **reports, not gates**: the renderer
    emits them as numbers and ratios, and the writer is asked to stay within them, but
    nothing here rejects a document for being long — an aesthetic bound enforced by code
    stops being an aesthetic.
    """

    model_config = ConfigDict(extra="ignore")

    directory: str = Field(
        default="journal",
        description=(
            "the journal root, relative to the ghost home. The default layout below it is "
            "declared on the document classes; an implementation may lay the files out its own way."
        ),
    )
    prompt_key: str = Field(
        default="journal.spec",
        description=(
            "prompter key of the first-order prompt — motivation, layout, status machine, "
            "the view shape. Generated from the prompt registry, never stored in a stub document."
        ),
    )
    tz: str = Field(
        default="",
        description=(
            "the day boundary: an IANA name or an offset like `+08:00`; empty = this machine's "
            "clock. A journal day is a *local* calendar day; each document's `created` is a UTC "
            "instant, so a backfilled day still says when it was actually written."
        ),
    )
    month_details: int = Field(
        default=2,
        ge=1,
        description=(
            "how many recent months the view carries in full — the current month and the one "
            "before it, by default. The detail band sits at the newest end of the month window."
        ),
    )
    month_summaries: int = Field(
        default=3,
        ge=0,
        description=(
            "N in the month band's N + details: how many months before the detail band are "
            "carried as summaries only, newest first."
        ),
    )
    day_summaries: int = Field(
        default=7,
        ge=0,
        description=(
            "M in the day band's M + 1: how many days before the last closed day are carried "
            "as summaries. The last closed day itself is the +1 and is drawn in full."
        ),
    )
    include_today: bool = Field(
        default=True,
        description=(
            "carry the current day as its own element when it is not the last closed day. It "
            "tells the reader where `now` sits; off when the face should stop at the last "
            "closed day."
        ),
    )
    backfill_days: int = Field(
        default=3,
        ge=0,
        description=(
            "the repair window: how far back a day that was opened and never closed is still "
            "reconstructed. Older gaps are left as history — the journal does not rewrite a past "
            "it did not live."
        ),
    )
    day_chars: int = Field(default=2000, ge=0, description="the character budget of one day's body.")
    month_chars: int = Field(default=4000, ge=0, description="the character budget of one month's body.")
    year_chars: int = Field(default=3000, ge=0, description="the character budget of one year's body.")
    summary_chars: int = Field(
        default=120,
        ge=0,
        description=(
            "the character budget of a summary, at every level. One number: a summary is a "
            "head, not a second body."
        ),
    )

    @classmethod
    def conf_name(cls) -> str:
        return "journal"


class JournalDoc(BaseModel):
    """One period's document: frontmatter + body, and nothing else.

    ``extra='allow'`` is the forward-compatibility rule made explicit: a key this version
    does not know is kept and written back unchanged. ``created`` is the UTC instant the
    document was first materialized — not the period it describes. A backfilled day is
    written later than it happened, and the two facts must stay separable.

    The frontmatter pair lives here, not in the implementation, so a writer never invents
    a second serialization: ``to_post`` / ``from_post`` are the only shape a document
    takes between a model and a file, whichever code path touches it.
    """

    model_config = ConfigDict(extra="allow")

    FORMAT_VERSION: ClassVar[int] = 1
    """The document shape this version writes. A reader accepts any version it can parse —
    the number tells a future writer which shape produced the file; the tolerance comes from
    unknown keys surviving, not from the number."""

    format_version: int = Field(
        default=FORMAT_VERSION,
        description="the document shape this was written with; unknown keys survive regardless.",
    )
    summary: str = Field(default="", description="the bounded head of this period, one line.")
    content: str = Field(default="", description="the body, in the period's own voice.")
    created: AwareDatetime | None = Field(
        default=None,
        description=(
            "when the document was first materialized, in UTC. None = the file did not say; "
            "the reader does not invent time."
        ),
    )

    @classmethod
    def from_post(cls, post: frontmatter.Post) -> Self:
        """Read a document back from a frontmatter post: metadata is the fields, body is ``content``."""
        data = dict(post.metadata)
        data["content"] = post.content
        return cls(**data)

    def to_post(self) -> frontmatter.Post:
        """The document as a frontmatter post. ``content`` is the body, everything else is metadata.

        ``frontmatter.Post(content, **metadata)`` — the metadata dict is splatted, not passed
        as a keyword: passing it whole would nest every field under a literal ``metadata`` key.
        """
        data = self.model_dump(exclude_none=True)
        content = data.pop("content", "") or ""
        return frontmatter.Post(content, **data)

    @classmethod
    def from_text(cls, text: str) -> Self:
        """Parse a document from its text — the half an async reader needs after ``await read``."""
        return cls.from_post(frontmatter.loads(text))

    def to_text(self) -> str:
        """Serialize the document to text — the half an async writer needs before ``await write``."""
        return frontmatter.dumps(self.to_post())

    @classmethod
    def load(cls, file: Path) -> Self:
        """Parse a document file — the synchronous read half of the file convention."""
        return cls.from_text(file.read_text(encoding="utf-8"))

    def dump(self, file: Path) -> None:
        """Write a document file — the synchronous write half. Parents are created; nothing is merged."""
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(self.to_text(), encoding="utf-8")


class Daily(JournalDoc):
    """One local calendar day. The only level with a state machine.

    A day does not exist until something anchors it: the first ``record_commit`` of the
    day creates the document. Absence is the natural state — a body is not always present,
    and a day it did not live should not be invented.
    """

    FILENAME: ClassVar[str] = "daily.md"
    """The day's document — one directory per day, one file per level."""

    COMMITS_FILENAME: ClassVar[str] = ".commits.jsonl"
    """The day's anchors, one JSON record per line in arrival order. It is what makes "this
    day happened" checkable, what a backfill reads to find the day's last span, and what a
    debugger reads to see when each anchor landed."""

    day: date = Field(description="the local day this document belongs to.")
    status: DocStatus = Field(
        default="pending",
        description="pending → writing → closed; see DocStatus.",
    )

    @classmethod
    def directory(cls, root: Path, day: date) -> Path:
        """``<root>/Y2026/M11/D18`` — the same tree at every level."""
        return root.joinpath(f"Y{day.year:04d}", f"M{day.month:02d}", f"D{day.day:02d}")

    @classmethod
    def filepath(cls, root: Path, day: date) -> Path:
        return cls.directory(root, day).joinpath(cls.FILENAME)

    @classmethod
    def commits_filepath(cls, root: Path, day: date) -> Path:
        return cls.directory(root, day).joinpath(cls.COMMITS_FILENAME)


class Monthly(JournalDoc):
    """One month — the events, and how they changed."""

    FILENAME: ClassVar[str] = "monthly.md"
    """The month's document."""

    AGENTS_FILENAME: ClassVar[str] = "AGENTS.md"
    """The month's own thinking — written by the ghost, never by the mechanism. Its
    presence is the convention; its content is the second order and no code reads it."""

    year: int = Field(description="the calendar year.")
    month: int = Field(ge=1, le=12, description="the calendar month, 1-12.")

    @classmethod
    def directory(cls, root: Path, year: int, month: int) -> Path:
        return root.joinpath(f"Y{year:04d}", f"M{month:02d}")

    @classmethod
    def filepath(cls, root: Path, year: int, month: int) -> Path:
        return cls.directory(root, year, month).joinpath(cls.FILENAME)

    @classmethod
    def agents_filepath(cls, root: Path, year: int, month: int) -> Path:
        return cls.directory(root, year, month).joinpath(cls.AGENTS_FILENAME)


class Yearly(JournalDoc):
    """One year — the compressed biography."""

    FILENAME: ClassVar[str] = "yearly.md"
    """The year's document."""

    year: int = Field(description="the calendar year.")

    @classmethod
    def directory(cls, root: Path, year: int) -> Path:
        return root.joinpath(f"Y{year:04d}")

    @classmethod
    def filepath(cls, root: Path, year: int) -> Path:
        return cls.directory(root, year).joinpath(cls.FILENAME)


class DateView(BaseModel):
    """What the journal looks like *now*, read through one date.

    Not a historical snapshot: the same date rendered later may differ, because the
    documents it reads have changed. The three lists are windows, not archives — every
    level is bounded, and anything older is a file away rather than a token away.
    """

    at: date = Field(description="the coordinate the view was computed against.")
    years: list[Yearly] = Field(
        default_factory=list,
        description="every year, summaries only — the lifetime, flat.",
    )
    months: list[Monthly] = Field(
        default_factory=list,
        description=(
            "the month window, oldest first: `month_details` + `month_summaries` months. The "
            "newest `month_details` are drawn in full; the rest as summaries."
        ),
    )
    days: list[Daily] = Field(
        default_factory=list,
        description=(
            "the `day_summaries` days before `last_day`, oldest first — summaries only. "
            "`last_day` itself is not repeated here: it is the +1 of the day band."
        ),
    )
    last_day: Daily | None = Field(
        default=None,
        description="the most recent closed day at or before `at`; its body is drawn in full.",
    )
    today: Daily | None = Field(
        default=None,
        description="the document for `at`, if one exists. None = no anchor opened the day.",
    )


class BudgetReport(BaseModel):
    """One level's budget, measured. A signal for the writer, never a gate."""

    level: BudgetLevel
    subject: str = Field(description="which document — `2026`, `2026-11`, `2026-11-18`.")
    limit: int = Field(description="the configured character budget.")
    current: int = Field(description="the body's current character count.")
    ratio: float = Field(description="current / limit; > 1 means the writer should have cut, and the report says so.")


class JournalStatus(BaseModel):
    """What the writer needs to know before it writes: where the day stands, and what it costs."""

    at: date = Field(description="the day this status was computed for.")
    today: DocStatus | None = Field(
        default=None,
        description=(
            "the document for `at`; None = no anchor opened the day. Absence is natural — "
            "a body is not always present, so a day it did not live has no document."
        ),
    )
    budgets: list[BudgetReport] = Field(
        default_factory=list,
        description=(
            "one report per level for the period containing `at`: the cost of the face the "
            "writer is about to change."
        ),
    )
    problems: list[str] = Field(
        default_factory=list,
        description=(
            "documents that exist but could not be read or parsed, as `<path>: <reason>`. "
            "Empty = the journal is healthy. The face never carries these: they belong to the "
            "repair path, not to the self the model reads."
        ),
    )


class DoloresJournal(ABC):
    """The first-order surface of the journal.

    Read side: ``view`` / ``render`` / ``status`` / ``commits`` — everything a caller needs
    to consult the journal. Event side: ``record_commit`` — the one moment the mechanism
    acts without a writer, and the moment that opens a day.

    None of these methods write prose. The writer is a ghost; this contract only tells it
    what to read and where its own judgment begins.
    """

    @abstractmethod
    def specification(self) -> str:
        """The first-order prompt: motivation, layout, status machine, budgets, view shape.

        Rendered from the prompt registry (``JournalConfig.prompt_key``), not stored in the
        journal directory: a stub document is data, and a prompt is not data.
        """

    @abstractmethod
    async def view(self, at: date | None = None) -> DateView:
        """Compute the view anchored at ``at`` (today when None), from the files as they are now."""

    @abstractmethod
    async def render(self, at: date | None = None) -> str:
        """Render the view as the journal block — one text, ready to be wrapped by its caller.

        Density follows position, not element type: every year is drawn as a body, truncated
        at its budget; the newest ``month_details`` months are drawn in full and the older
        ones as summaries; the ``day_summaries`` days are summaries; the last closed day is a
        body; ``today``, when it is not the last closed day, is a summary with its status.
        Budgets never appear here — they are the writer's report, not the reader's face.
        """

    @abstractmethod
    async def status(self, at: date | None = None) -> JournalStatus:
        """Where the day stands and what each level costs."""

    @abstractmethod
    async def commits(self, at: date | None = None) -> list[str]:
        """The anchors attached to a day, oldest first — the material a writer reconstructs from."""

    @abstractmethod
    async def export(self) -> str:
        """Write the journal's root ``AGENTS.md`` from the spec and the configured limits.

        Idempotent and overwritten on every call: the page is a projection of the spec, not a
        document the ghost edits. The limits are appended at the tail — one source, the config —
        so the writer reads concrete numbers without the spec text carrying a second copy of them.
        Returns the path written.
        """

    @abstractmethod
    async def record_commit(self, coord: str, at: datetime | None = None) -> None:
        """Attach a memento coordinate to the day it belongs to, opening the day on first contact.

        Opening means: the day's document exists (``pending``) and the coordinate is in its
        ``.commits``. Idempotent — one coordinate appears once, and an existing document is
        read, never rewritten. Coordinates carry their branch, so a day shared by the ego and
        its emanations holds all of their anchors in one list.
        """
