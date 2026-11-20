"""FileJournal — the filesystem-first journal.

The documents are the truth; this module is only their reader plus the writes that are
not prose (opening a day's calendar containers, and appending its anchors). It keeps no
state between calls: every method re-reads the tree, so a restart, a hand edit, or a
second reader cannot make it lie.

File content is read and written through ``aiofiles``: the shell's event loop also carries
voice, GUI and the scheduler, and a journal read on a network-mounted home must not stall
them. Directory stat and listing stay synchronous — a readdir is cheap, and the memento
filesystem implementation draws the same line. A window's documents are loaded
concurrently, so a view costs one round of IO rather than one per file.

Layout — the default declared on the document classes in ``contract.py``, under
``<home>/<config.directory>``::

    <root>/Y2026/yearly.md
    <root>/Y2026/AGENTS.md               # ghost-written; this module never touches it
    <root>/Y2026/M11/monthly.md
    <root>/Y2026/M11/AGENTS.md          # ghost-written; this module never touches it
    <root>/Y2026/M11/D18/daily.md
    <root>/Y2026/M11/D18/.commits.jsonl

A broken document is skipped and reported in ``status.problems``, never raised: one bad
file must not take the memory surface down with it. A missing document is not an error at
all — absence is the natural state of a day the body did not live.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, Iterator
from datetime import date, datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import TypeVar
from zoneinfo import ZoneInfo

import aiofiles

from .contract import (
    BudgetLevel,
    BudgetReport,
    Daily,
    DateView,
    DoloresJournal,
    JournalConfig,
    JournalDoc,
    JournalStatus,
    Monthly,
    Yearly,
)

__all__ = ["FileJournal"]

DOC = TypeVar("DOC", bound=JournalDoc)

_EXPORT_HEADER = (
    "<!-- generated from the journal spec (journal.spec); local edits are overwritten on export -->"
)


def _zone(tz: str | None) -> tzinfo:
    """``tz`` → tzinfo. Empty = the machine's clock; ``+08:00`` = a fixed offset; else IANA.

    The same rule the memento channel uses for anchoring: a day boundary and a rendered
    date must read the same clock, or a 19:30 entry lands in the next day without an error.
    """
    text = (tz or "").strip()
    if not text:
        return datetime.now().astimezone().tzinfo or timezone.utc
    if text[0] in "+-":
        sign = 1 if text[0] == "+" else -1
        hours, _, minutes = text[1:].partition(":")
        return timezone(sign * timedelta(hours=int(hours), minutes=int(minutes or 0)))
    return ZoneInfo(text)


def _escape(text: str) -> str:
    """Escape the three characters that would read as markup.

    The journal is prose; a day that quotes a tag must not open one.
    """
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _months_back(day: date, count: int) -> Iterator[tuple[int, int]]:
    year, month = day.year, day.month
    for _ in range(count):
        yield year, month
        if month == 1:
            year, month = year - 1, 12
        else:
            month -= 1


class FileJournal(DoloresJournal):
    """The default journal: one tree of markdown documents, read fresh on every call.

    Three inputs are enough to run it — a ghost home, a config, and a logger. The prompt is
    the only optional one: the read side and "open a day" work without it, and
    ``specification`` says so plainly when the seam has not wired a prompt registry yet.
    """

    def __init__(
            self,
            home: Path,
            config: JournalConfig | None = None,
            *,
            logger: logging.Logger | None = None,
            prompt: Callable[[str], str] | None = None,
    ) -> None:
        """
        :param home: the ghost home; ``config.directory`` names the journal root under it,
            so the layout lives in one place (the config) instead of at every call site.
        :param config: the first-order knobs; defaults are used when omitted.
        :param logger: where skipped documents and opened days are reported.
        :param prompt: renders the first-order prompt by key (``PromptManager.render``);
            optional, because only ``specification`` needs it.
        """
        self._config = config or JournalConfig()
        self._home = Path(home)
        self._root = self._home / self._config.directory
        self._logger = logger or logging.getLogger("dolores.journal")
        self._prompt = prompt
        # Resolve the clock once: a bad tz fails at construction, not from inside view().
        self._tz = _zone(self._config.tz)

    @property
    def home(self) -> Path:
        return self._home

    @property
    def root(self) -> Path:
        """The journal root — ``home / config.directory``."""
        return self._root

    @property
    def config(self) -> JournalConfig:
        return self._config

    # ── read side ────────────────────────────────────────────────────

    async def view(self, at: date | None = None) -> DateView:
        day = at or self._day_of(None)
        problems: list[str] = []
        years, months, days, last_day, today = await self._collect(day, problems)
        today_anchors: list[str] = []
        if today is not None:
            today_anchors = await self._aread_coords(
                Daily.commits_filepath(self._root, day), problems,
            )
        self._report(problems)
        return DateView(
            at=day,
            years=years,
            months=months,
            days=days,
            last_day=last_day,
            today=today,
            today_anchors=today_anchors,
        )

    def today(self) -> date:
        """The journal's current calendar day, in its configured clock."""
        return self._day_of(None)

    async def render(self, at: date | None = None) -> str:
        view = await self.view(at)
        attrs = f'at="{view.at.isoformat()}"'
        rows: list[str] = []
        for year in view.years:
            text = self._clip(year.content or year.summary, self._config.year_chars)
            if text:
                rows.append(self._row("year", f'year="{year.year}"', text))
        for index, month in enumerate(view.months):
            detail = index >= len(view.months) - self._config.month_details
            text = (
                self._clip(month.content or month.summary, self._config.month_chars)
                if detail else
                self._clip(month.summary, self._config.summary_chars)
            )
            if text:
                rows.append(self._row("month", f'month="{month.year:04d}-{month.month:02d}"', text))
        # The band is chronological; the one deep read sits where its day sits. `closed` is
        # the settled state and needs no mark — an open day says where it stands.
        full_day = view.last_day.day if view.last_day is not None else None
        for entry in view.days:
            if entry.day == full_day:
                rows.append(self._row(
                    "last_day", f'date="{entry.day.isoformat()}"',
                    self._clip(entry.content or entry.summary, self._config.day_chars),
                ))
            elif entry.summary.strip():
                status = "" if entry.status == "closed" else f' status="{entry.status}"'
                rows.append(self._row(
                    "day", f'date="{entry.day.isoformat()}"{status}',
                    self._clip(entry.summary, self._config.summary_chars),
                ))
            else:
                # An established day is part of the face even before it has prose: the date
                # and the status are the readable minimum. The current day also carries its
                # anchor count, which is what the mechanism already knows about it.
                text = ""
                if view.today is not None and entry.day == view.today.day and view.today_anchors:
                    count = len(view.today_anchors)
                    text = f"{count} anchor" if count == 1 else f"{count} anchors"
                status = "" if entry.status == "closed" else f' status="{entry.status}"'
                rows.append(self._row(
                    "day", f'date="{entry.day.isoformat()}"{status}',
                    self._clip(text, self._config.summary_chars),
                ))
        # An established day is part of the face even before it has prose: the date and the
        # status are the readable minimum, and the anchor count says what the day already
        # holds. The writer's summary, once written, takes the row over.
        drawn = {entry.day for entry in view.days}
        if view.today is not None and view.today.day not in drawn:
            today = view.today
            if today.summary.strip():
                text = today.summary
            elif view.today_anchors:
                count = len(view.today_anchors)
                text = f"{count} anchor" if count == 1 else f"{count} anchors"
            else:
                text = ""
            status = "" if today.status == "closed" else f' status="{today.status}"'
            rows.append(self._row(
                "day", f'date="{today.day.isoformat()}"{status}',
                self._clip(text, self._config.summary_chars),
            ))
        if not rows:
            return f"<journal {attrs}/>"
        return "\n".join([f"<journal {attrs}>", *rows, "</journal>"])

    async def status(self, at: date | None = None) -> JournalStatus:
        day = at or self._day_of(None)
        problems: list[str] = []
        # Scan the same window the view draws: a broken document anywhere in it is missing
        # from the face, so it belongs in the repair path's report, not only in a log line.
        years, months, _days, _last_day, today = await self._collect(day, problems)
        monthly = next((entry for entry in months if entry.year == day.year and entry.month == day.month), None)
        yearly = next((entry for entry in years if entry.year == day.year), None)
        # Read the anchor log too: a day with an unreadable .commits must say so here, or the
        # repair path would treat "no anchors" and "cannot read anchors" as the same thing.
        await self._aread_coords(Daily.commits_filepath(self._root, day), problems)
        self._report(problems)
        return JournalStatus(
            at=day,
            today=today.status if today is not None else None,
            budgets=[
                self._budget("day", day.isoformat(), self._config.day_chars, today.content if today else ""),
                self._budget(
                    "month", f"{day.year:04d}-{day.month:02d}", self._config.month_chars,
                    monthly.content if monthly else "",
                ),
                self._budget("year", f"{day.year:04d}", self._config.year_chars, yearly.content if yearly else ""),
            ],
            problems=problems,
        )

    async def commits(self, at: date | None = None) -> list[str]:
        day = at or self._day_of(None)
        problems: list[str] = []
        coords = await self._aread_coords(Daily.commits_filepath(self._root, day), problems)
        self._report(problems)
        return coords

    # ── event side ───────────────────────────────────────────────────

    async def establish(self, at: date | datetime | None = None) -> date:
        """Open the year, month and day documents, creating only what is missing.

        Idempotent and prose-free: an opened day is ``pending``. ``None`` = the journal's
        current day; a ``date`` is taken as the day itself; a ``datetime`` is read in the
        journal's clock. Returns the day opened.
        """
        if isinstance(at, datetime):
            day = self._day_of(at)
        elif isinstance(at, date):
            day = at
        else:
            day = self._day_of(None)
        created = datetime.now(timezone.utc)
        await self._ensure_doc(
            Yearly.filepath(self._root, day.year),
            Yearly(year=day.year, created=created),
        )
        await self._ensure_doc(
            Monthly.filepath(self._root, day.year, day.month),
            Monthly(year=day.year, month=day.month, created=created),
        )
        opened = await self._ensure_doc(
            Daily.filepath(self._root, day),
            Daily(day=day, created=created),
        )
        if opened:
            self._logger.info("journal: opened day %s", day.isoformat())
        return day

    async def _ensure_doc(self, path: Path, document: JournalDoc) -> bool:
        """Create a document only when it is absent. True = this call wrote it."""
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            async with aiofiles.open(path, "x", encoding="utf-8") as stream:
                await stream.write(document.to_text())
        except FileExistsError:
            return False
        return True

    async def record_commit(
            self,
            coord: str,
            at: datetime | None = None,
            *,
            owner: str = "",
            branch: str = "",
    ) -> None:
        coord = coord.strip()
        if not coord:
            raise ValueError("journal: an anchor coordinate cannot be empty")
        day = self._day_of(at)
        await self.establish(day)
        commits = Daily.commits_filepath(self._root, day)
        commits.parent.mkdir(parents=True, exist_ok=True)
        # Append-only: the read side de-duplicates, so a racing duplicate is silence rather
        # than a reason to read the whole log before every anchor. The timestamp is the
        # anchor's own instant in the journal's clock: a reader sees when it happened where it
        # happened, not a UTC number it has to translate.
        moment = at or datetime.now(timezone.utc)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=self._tz)
        record = {
            "at": moment.astimezone(self._tz).isoformat(),
            "coord": coord,
            "owner": owner,
            "branch": branch,
        }
        async with aiofiles.open(commits, "a", encoding="utf-8") as stream:
            await stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ── prompt ───────────────────────────────────────────────────────

    def specification(self) -> str:
        if self._prompt is None:
            raise RuntimeError(
                "journal specification is not wired: pass prompt=<callable taking the key> "
                "(the seam supplies PromptManager.render)"
            )
        return self._prompt(self._config.prompt_key)

    async def export(self) -> str:
        """Write ``journal/AGENTS.md`` — the spec plus the configured limits at the tail."""
        spec = self.specification()
        path = self._root / "AGENTS.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        text = f"{_EXPORT_HEADER}\n\n{spec.rstrip()}\n\n{self._limits_text()}\n"
        async with aiofiles.open(path, "w", encoding="utf-8") as stream:
            await stream.write(text)
        return str(path)

    def _limits_text(self) -> str:
        return (
            "## Limits (generated)\n"
            f"- day: {self._config.day_chars} characters\n"
            f"- month: {self._config.month_chars} characters\n"
            f"- year: {self._config.year_chars} characters\n"
            f"- summary: {self._config.summary_chars} characters\n"
        )

    # ── internals ────────────────────────────────────────────────────

    async def _collect(
            self, day: date, problems: list[str],
    ) -> tuple[list[Yearly], list[Monthly], list[Daily], Daily | None, Daily | None]:
        """Load the window in one round: years, months and today concurrently."""
        years, months, today = await asyncio.gather(
            self._years(problems),
            self._months(day, problems),
            self._aload(Daily, Daily.filepath(self._root, day), problems),
        )
        days = await self._recent_days(day, problems)
        last_day = next((entry for entry in reversed(days) if entry.status == "closed"), None)
        return years, months, days, last_day, today

    def _day_of(self, at: datetime | None) -> date:
        """The journal day of ``at`` (now when None) in the configured clock."""
        if at is None:
            return datetime.now(self._tz).date()
        if at.tzinfo is None:
            at = at.replace(tzinfo=self._tz)
        return at.astimezone(self._tz).date()

    async def _aload(self, kind: type[DOC], path: Path, problems: list[str]) -> DOC | None:
        try:
            async with aiofiles.open(path, encoding="utf-8") as stream:
                text = await stream.read()
        except FileNotFoundError:
            return None
        except OSError as exc:
            self._problem(problems, path, exc)
            return None
        try:
            return kind.from_text(text, **kind.period_of(path))
        except Exception as exc:  # one broken document must not blank the view
            self._problem(problems, path, exc)
            return None

    async def _years(self, problems: list[str]) -> list[Yearly]:
        paths = sorted(self._root.glob("Y*/yearly.md"))
        if not paths:
            return []
        docs = await asyncio.gather(*(self._aload(Yearly, path, problems) for path in paths))
        return [doc for doc in docs if doc is not None]

    async def _months(self, day: date, problems: list[str]) -> list[Monthly]:
        count = self._config.month_details + self._config.month_summaries
        pairs = list(reversed(list(_months_back(day, count))))
        if not pairs:
            return []
        docs = await asyncio.gather(
            *(self._aload(Monthly, Monthly.filepath(self._root, year, month), problems) for year, month in pairs)
        )
        return [doc for doc in docs if doc is not None]

    async def _recent_days(self, day: date, problems: list[str]) -> list[Daily]:
        """The day band: the most recent ``day_summaries`` days that have a document.

        Anchored on the newest day with a document, not on the last closed day: a day still
        being written is part of the recent life, and anchoring on the last close would hide
        every day after it. A day with no prose yet is part of the band too — its date and
        status are readable, and hiding it would make an established day vanish. ``closed``
        decides depth (see ``render``), not existence.
        """
        count = self._config.day_summaries
        if count <= 0:
            return []
        found: list[Daily] = []
        for entry in self._day_candidates(day):
            if len(found) >= count:
                break
            document = await self._aload(Daily, Daily.filepath(self._root, entry), problems)
            if document is not None:
                found.append(document)
        found.reverse()
        return found

    def _day_candidates(self, at: date) -> list[date]:
        """Every day with a document, newest first, bounded by ``at``.

        One readdir instead of walking back day by day: the journal may have been silent for
        weeks, and the band still walks back from the newest day with something to say.
        """
        found: list[date] = []
        for path in self._root.glob("Y*/M*/D*/daily.md"):
            try:
                entry = Daily.period_of(path).get("day")
            except ValueError:
                continue
            if isinstance(entry, date) and entry <= at:
                found.append(entry)
        return sorted(found, reverse=True)

    async def _aread_coords(self, path: Path, problems: list[str]) -> list[str]:
        try:
            async with aiofiles.open(path, encoding="utf-8") as stream:
                text = await stream.read()
        except FileNotFoundError:
            return []
        except OSError as exc:
            self._problem(problems, path, exc)
            return []
        # One JSON record per line; a bad line is reported, not fatal. De-dupe on read: a
        # duplicate coordinate is silence, not a second visit to the span.
        seen: set[str] = set()
        coords: list[str] = []
        for number, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                record = None
            coord = record.get("coord") if isinstance(record, dict) else None
            if not isinstance(coord, str) or not coord:
                self._problem(problems, path, ValueError(f"line {number}: unreadable anchor record"))
                continue
            if coord in seen:
                continue
            seen.add(coord)
            coords.append(coord)
        return coords

    def _budget(self, level: BudgetLevel, subject: str, limit: int, body: str) -> BudgetReport:
        current = len(body)
        return BudgetReport(
            level=level,
            subject=subject,
            limit=limit,
            current=current,
            ratio=(current / limit) if limit > 0 else 0.0,
        )

    def _row(self, name: str, attrs: str, text: str) -> str:
        text = text.strip()
        if not text:
            return f"  <{name} {attrs}/>"
        return f"  <{name} {attrs}>{_escape(text)}</{name}>"

    def _clip(self, text: str, limit: int) -> str:
        """Cut a body or summary at its budget. A limit of 0 or less means "no cut"."""
        text = (text or "").strip()
        if limit > 0 and len(text) > limit:
            return text[:limit].rstrip() + "…"
        return text

    def _problem(self, problems: list[str], path: Path, exc: Exception) -> None:
        problems.append(f"{path}: {exc}")

    def _report(self, problems: list[str]) -> None:
        for entry in problems:
            self._logger.error("journal: %s", entry)
