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

``pending``: the day exists — the mechanism opened it at boot, at a day boundary, or by its
first anchor — and no writer has worked on it.
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
        default=15,
        ge=0,
        description=(
            "how many recent days the day band carries: the most recent days that have a "
            "document, ending at the newest such day at or before `at`. A day still being "
            "written is part of the recent life — anchoring the band on the last closed day "
            "would hide every day after it. A day with no prose yet is still drawn, with its "
            "date and status. 0 turns the band off."
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
    month_chars: int = Field(default=3000, ge=0, description="the character budget of one month's body.")
    year_chars: int = Field(default=2000, ge=0, description="the character budget of one year's body.")
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

    **The period is the path's.** ``day`` / ``year`` / ``month`` are declared on the model
    so every instance carries its period, but the frontmatter never holds them: ``to_post``
    leaves them out, and ``load`` / ``period_of`` read them back from the document's
    location — the same split ``MossMeta`` makes for its ``file`` field. The path is the
    index; the frontmatter carries only what the path cannot say.

    The serialization lives here, not in the implementation, so a writer never invents a
    second one: ``to_post`` / ``from_post`` are the only shape a document takes between a
    model and a file, whichever code path touches it.
    """

    model_config = ConfigDict(extra="allow")

    FORMAT_VERSION: ClassVar[int] = 1
    """The document shape this version writes. A reader accepts any version it can parse —
    the number tells a future writer which shape produced the file; the tolerance comes from
    unknown keys surviving, not from the number."""

    PERIOD_FIELDS: ClassVar[tuple[str, ...]] = ()
    """The fields the document's location supplies (see ``period_of``). They never appear
    in the frontmatter, so a document written by an older version — or by hand — is still
    readable: the directory says which period it is."""

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
    def period_of(cls, file: Path) -> dict[str, object]:
        """The period fields the document's location supplies — the inverse of ``directory``.

        The base class supplies none; each level parses its own layout. A path this level
        does not recognize raises ``ValueError``: a document out of place has no period.
        """
        return {}

    @classmethod
    def from_post(cls, post: frontmatter.Post, **period: object) -> Self:
        """Read a document back from a frontmatter post: metadata is the fields, body is ``content``.

        ``period`` supplies the fields the frontmatter does not carry. The location wins over
        a stale copy in the metadata: a document sits where it sits.
        """
        data = {**post.metadata, **period, "content": post.content}
        return cls(**data)

    def to_post(self) -> frontmatter.Post:
        """The document as a frontmatter post. ``content`` is the body, everything else is metadata.

        The period fields are left out entirely — the path already carries them, and a second
        copy would be a second truth.

        ``frontmatter.Post(content, **metadata)`` — the metadata dict is splatted, not passed
        as a keyword: passing it whole would nest every field under a literal ``metadata`` key.
        """
        data = self.model_dump(exclude={"content", *self.PERIOD_FIELDS}, exclude_none=True)
        return frontmatter.Post(self.content, **data)

    @classmethod
    def from_text(cls, text: str, **period: object) -> Self:
        """Parse a document from its text — the half an async reader needs after ``await read``.

        ``period`` is the reader's knowledge of where the text came from; see ``from_post``.
        """
        return cls.from_post(frontmatter.loads(text), **period)

    def to_text(self) -> str:
        """Serialize the document to text — the half an async writer needs before ``await write``."""
        return frontmatter.dumps(self.to_post())

    @classmethod
    def load(cls, file: Path) -> Self:
        """Parse a document file — the period is read from the file's location."""
        return cls.from_text(file.read_text(encoding="utf-8"), **cls.period_of(file))

    def dump(self, file: Path) -> None:
        """Write a document file — the synchronous write half. Parents are created; nothing is merged."""
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(self.to_text(), encoding="utf-8")


class Daily(JournalDoc):
    """One local calendar day. The only level with a state machine.

    The mechanism opens a day: at boot, at the day boundary, or when its first anchor is
    recorded. Existence is a calendar fact, not an anchor fact; the day is drawn as
    ``pending`` until a writer gives it prose. A day the ghost did not live is not
    invented — ``establish`` only opens the day it is asked for.
    """

    FILENAME: ClassVar[str] = "daily.md"
    """The day's document — one directory per day, one file per level."""

    COMMITS_FILENAME: ClassVar[str] = ".commits.jsonl"
    """The day's anchors, one JSON record per line in arrival order. It is what makes "this
    day happened" checkable, what a backfill reads to find the day's last span, and what a
    debugger reads to see when each anchor landed."""

    PERIOD_FIELDS: ClassVar[tuple[str, ...]] = ("day",)

    day: date = Field(description="the local day this document belongs to, supplied by its directory.")
    status: DocStatus = Field(
        default="pending",
        description="pending → writing → closed; see DocStatus.",
    )

    @classmethod
    def period_of(cls, file: Path) -> dict[str, object]:
        """``.../Y2026/M11/D18/daily.md`` → ``{'day': date(2026, 11, 18)}``."""
        try:
            return {"day": date(
                int(file.parent.parent.parent.name[1:]),
                int(file.parent.parent.name[1:]),
                int(file.parent.name[1:]),
            )}
        except (ValueError, IndexError) as exc:
            raise ValueError(f"daily: not a day document path: {file}") from exc

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

    PERIOD_FIELDS: ClassVar[tuple[str, ...]] = ("year", "month")

    year: int = Field(description="the calendar year, supplied by the directory.")
    month: int = Field(ge=1, le=12, description="the calendar month, supplied by the directory, 1-12.")

    @classmethod
    def period_of(cls, file: Path) -> dict[str, object]:
        """``.../Y2026/M11/monthly.md`` → ``{'year': 2026, 'month': 11}``."""
        try:
            return {"year": int(file.parent.parent.name[1:]), "month": int(file.parent.name[1:])}
        except (ValueError, IndexError) as exc:
            raise ValueError(f"monthly: not a month document path: {file}") from exc

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

    AGENTS_FILENAME: ClassVar[str] = "AGENTS.md"
    """The year's own thinking — written by the ghost, never by the mechanism. It sits at the
    year's root and sets the general style; the month's adds what that month needs."""

    PERIOD_FIELDS: ClassVar[tuple[str, ...]] = ("year",)

    year: int = Field(description="the calendar year, supplied by the directory.")

    @classmethod
    def period_of(cls, file: Path) -> dict[str, object]:
        """``.../Y2026/yearly.md`` → ``{'year': 2026}``."""
        try:
            return {"year": int(file.parent.name[1:])}
        except (ValueError, IndexError) as exc:
            raise ValueError(f"yearly: not a year document path: {file}") from exc

    @classmethod
    def directory(cls, root: Path, year: int) -> Path:
        return root.joinpath(f"Y{year:04d}")

    @classmethod
    def filepath(cls, root: Path, year: int) -> Path:
        return cls.directory(root, year).joinpath(cls.FILENAME)

    @classmethod
    def agents_filepath(cls, root: Path, year: int) -> Path:
        return cls.directory(root, year).joinpath(cls.AGENTS_FILENAME)


class DateView(BaseModel):
    """What the journal looks like *now*, read through one date.

    Not a historical snapshot: the same date rendered later may differ, because the
    documents it reads have changed. Months and days are windows, not archives — anything
    older is a file away rather than a token away; the year list is the lifetime, flat.
    """

    at: date = Field(description="the coordinate the view was computed against.")
    years: list[Yearly] = Field(
        default_factory=list,
        description="every year, drawn as a body and truncated at its budget — the lifetime, flat.",
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
            "the day band, oldest first: up to `day_summaries` recent days that have a "
            "document, ending at the newest such day at or before `at`. Open days are part of "
            "it — `closed` decides depth, not existence."
        ),
    )
    last_day: Daily | None = Field(
        default=None,
        description=(
            "the newest closed day inside the day band; its body is drawn in full, in place. "
            "None = no day in the band has closed yet — the band is summaries only."
        ),
    )
    today: Daily | None = Field(
        default=None,
        description=(
            "the document for `at`, if one exists. Drawn in the band when it carries a "
            "summary; an open day without prose is drawn as its own pending row, with the "
            "anchor count from `today_anchors`. None = the day has not been established."
        ),
    )
    today_anchors: list[str] = Field(
        default_factory=list,
        description=(
            "the anchors attached to `today`, oldest first — what makes an open day readable "
            "before its prose exists."
        ),
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
            "the day's status; None = the day has no document. The mechanism establishes the "
            "current day at boot and at the day boundary, so absence here is a pre-hook state "
            "or a removed document, not the normal shape of a lived day."
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

    Read side: ``view`` / ``render`` / ``status`` / ``commits`` / ``today`` — everything a
    caller needs to consult the journal. Event side: ``establish`` (open the day's documents)
    and ``record_commit`` (attach an anchor to its day). Those are the mechanism's two writes;
    both are idempotent and neither writes prose.

    The writer is a ghost; this contract only tells it what to read and where its own
    judgment begins.
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
        ones as summaries; the day band carries the ``day_summaries`` most recent days that
        have a summary, and the newest closed day inside it is drawn in full, in place.
        ``closed`` decides depth, not existence — a day still being written is part of the
        recent life. A row without text is not drawn; ``status`` marks only the days that
        have not closed. Budgets never appear here — they are the writer's report, not the
        reader's face.
        """

    @abstractmethod
    async def status(self, at: date | None = None) -> JournalStatus:
        """Where the day stands and what each level costs."""

    @abstractmethod
    async def commits(self, at: date | None = None) -> list[str]:
        """The anchors attached to a day, oldest first — the material a writer reconstructs from."""

    @abstractmethod
    def today(self) -> date:
        """The journal's current calendar day, in its configured clock."""

    @abstractmethod
    async def establish(self, at: date | datetime | None = None) -> date:
        """Open the day's documents — year, month, day — creating only what is missing.

        Idempotent. ``None`` = the journal's current day. Returns the day opened. This is
        the mechanism's calendar write: it never writes prose; an opened day is ``pending``
        and the renderer draws it even before the writer fills it.
        """

    @abstractmethod
    async def export(self) -> str:
        """Write the journal's root ``AGENTS.md`` from the spec and the configured limits.

        Idempotent and overwritten on every call: the page is a projection of the spec, not a
        document the ghost edits. The limits are appended at the tail — one source, the config —
        so the writer reads concrete numbers without the spec text carrying a second copy of them.
        Returns the path written.
        """

    @abstractmethod
    async def record_commit(
            self,
            coord: str,
            at: datetime | None = None,
            *,
            owner: str = "",
            branch: str = "",
    ) -> None:
        """Attach a memento coordinate to the day it belongs to.

        The day's documents are opened if missing (``pending``), then one row is appended to
        ``.commits``. Idempotent — one coordinate appears once, and an existing document is
        read, never rewritten. Coordinates carry their branch, so a day shared by the ego and
        its emanations holds all of their anchors in one list.

        A row carries the anchor's instant in the journal's clock (an offset, not UTC), the
        coordinate, and the ``owner`` and ``branch`` names it resolves against — a coordinate
        is only unique within its owner.
        """
