"""The journal organ — the contract's models, the filesystem implementation, and the export.

Answers: how a day is opened, what the view window selects, what the face draws, how a broken
document is reported, and what the root ``AGENTS.md`` carries. The mechanism-side wiring
(triggers, memory) is not here yet; these tests pin the read/write surface it will stand on.
"""

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from ghoshell_moss.ghosts.dolores.journal.contract import (
    Daily,
    JournalConfig,
    Monthly,
    Yearly,
)
from ghoshell_moss.ghosts.dolores.journal.journal_file import FileJournal
from ghoshell_moss.ghosts.dolores.mindset.seeds import JOURNAL_SPEC_SEED

AT = date(2026, 11, 19)
TZ = timezone(timedelta(hours=8))


def _journal(home: Path, **overrides) -> FileJournal:
    return FileJournal(home, JournalConfig(**overrides))


def _closed(root: Path, day: date, *, summary: str = "", content: str = "") -> None:
    Daily(
        day=day,
        status="closed",
        summary=summary or f"day {day.day}",
        content=content or f"body {day.day}",
    ).dump(Daily.filepath(root, day))


class TestJournalDocs:
    def test_frontmatter_roundtrip_keeps_unknown_keys(self, tmp_path):
        doc = Daily(day=AT, status="closed", summary="head", content="body", mood="tired")
        path = Daily.filepath(tmp_path, AT)
        doc.dump(path)

        loaded = Daily.load(path)
        assert loaded == doc
        assert loaded.mood == "tired"

    def test_default_paths(self, tmp_path):
        assert Daily.filepath(tmp_path, AT) == tmp_path / "Y2026/M11/D19/daily.md"
        assert Daily.commits_filepath(tmp_path, AT) == tmp_path / "Y2026/M11/D19/.commits.jsonl"
        assert Monthly.filepath(tmp_path, 2026, 11) == tmp_path / "Y2026/M11/monthly.md"
        assert Monthly.agents_filepath(tmp_path, 2026, 11) == tmp_path / "Y2026/M11/AGENTS.md"
        assert Yearly.filepath(tmp_path, 2026) == tmp_path / "Y2026/yearly.md"
        assert Yearly.agents_filepath(tmp_path, 2026) == tmp_path / "Y2026/AGENTS.md"

    def test_location_supplies_the_period(self, tmp_path):
        # A document written before the period fields existed (or by hand): only the
        # directory says which day it is. It must stay readable.
        path = Daily.filepath(tmp_path, AT)
        path.parent.mkdir(parents=True)
        path.write_text("---\nsummary: legacy\nstatus: closed\n---\n\nlegacy body\n", encoding="utf-8")

        loaded = Daily.load(path)
        assert loaded.day == AT
        assert loaded.status == "closed"
        assert loaded.content.strip() == "legacy body"

    def test_period_is_not_written_into_the_frontmatter(self, tmp_path):
        path = Daily.filepath(tmp_path, AT)
        Daily(day=AT, status="closed", summary="head", content="body").dump(path)

        _, header, _ = path.read_text(encoding="utf-8").split("---", 2)
        assert "day:" not in header


class TestJournalRecord:
    @pytest.mark.asyncio
    async def test_record_commit_opens_the_day_and_dedupes(self, tmp_path):
        journal = _journal(tmp_path)
        await journal.record_commit("1-784", datetime(2026, 11, 19, 10, tzinfo=TZ))
        await journal.record_commit("1-784", datetime(2026, 11, 19, 11, tzinfo=TZ))
        await journal.record_commit("2-1", datetime(2026, 11, 19, 12, tzinfo=TZ))

        assert Daily.load(Daily.filepath(journal.root, AT)).status == "pending"
        assert await journal.commits(AT) == ["1-784", "2-1"]

    @pytest.mark.asyncio
    async def test_empty_coordinate_is_refused(self, tmp_path):
        with pytest.raises(ValueError):
            await _journal(tmp_path).record_commit("   ")

    @pytest.mark.asyncio
    async def test_bad_anchor_line_is_a_problem_not_a_failure(self, tmp_path):
        journal = _journal(tmp_path)
        await journal.record_commit("1-784", datetime(2026, 11, 19, 10, tzinfo=TZ))
        path = Daily.commits_filepath(journal.root, AT)
        path.write_text(path.read_text(encoding="utf-8") + "not-json\n", encoding="utf-8")

        assert await journal.commits(AT) == ["1-784"]
        assert (await journal.status(AT)).problems


class TestJournalEstablish:
    @pytest.mark.asyncio
    async def test_establish_opens_the_calendar_containers(self, tmp_path):
        journal = _journal(tmp_path)

        assert await journal.establish(AT) == AT
        assert Yearly.filepath(journal.root, 2026).exists()
        assert Monthly.filepath(journal.root, 2026, 11).exists()
        daily = Daily.load(Daily.filepath(journal.root, AT))
        assert daily.status == "pending"
        assert daily.summary == ""

    @pytest.mark.asyncio
    async def test_establish_is_idempotent_and_never_overwrites(self, tmp_path):
        journal = _journal(tmp_path)
        await journal.establish(AT)
        path = Daily.filepath(journal.root, AT)
        Daily(day=AT, status="closed", summary="head", content="body").dump(path)

        await journal.establish(AT)

        loaded = Daily.load(path)
        assert loaded.status == "closed"
        assert loaded.content == "body"

    @pytest.mark.asyncio
    async def test_record_commit_opens_month_and_year(self, tmp_path):
        journal = _journal(tmp_path)
        at = datetime(2026, 12, 1, 0, 5, tzinfo=TZ)

        await journal.record_commit("1-1", at)

        assert Yearly.filepath(journal.root, 2026).exists()
        assert Monthly.filepath(journal.root, 2026, 12).exists()
        assert Daily.filepath(journal.root, date(2026, 12, 1)).exists()
        assert await journal.commits(date(2026, 12, 1)) == ["1-1"]


class TestJournalView:
    @pytest.mark.asyncio
    async def test_window_density_and_truncation(self, tmp_path):
        journal = _journal(
            tmp_path,
            month_details=2,
            month_summaries=1,
            day_summaries=3,
            year_chars=20,
            summary_chars=30,
        )
        root = journal.root
        Yearly(year=2026, summary="year head", content="Y" * 60).dump(Yearly.filepath(root, 2026))
        for month in (9, 10, 11):
            Monthly(year=2026, month=month, summary=f"m{month}", content=f"M{month}").dump(
                Monthly.filepath(root, 2026, month)
            )
        for day in range(10, 19):
            _closed(root, date(2026, 11, day))
        await journal.record_commit("1-784", datetime(2026, 11, 19, 10, tzinfo=TZ))

        view = await journal.view(AT)
        assert [entry.month for entry in view.months] == [9, 10, 11]
        assert [entry.day.isoformat() for entry in view.days] == [
            "2026-11-17", "2026-11-18", "2026-11-19",
        ]
        assert view.last_day is not None
        assert view.last_day.day == date(2026, 11, 18)
        assert view.today is not None
        assert view.today.status == "pending"
        assert view.today_anchors == ["1-784"]

        text = await journal.render(AT)
        assert "YYYYYYYYYYYYYYYYYYYY…" in text
        assert '<month month="2026-09">m9</month>' in text
        assert '<month month="2026-10">M10</month>' in text
        assert '<day date="2026-11-17">day 17</day>' in text
        assert '<last_day date="2026-11-18">body 18</last_day>' in text
        assert '<day date="2026-11-19" status="pending">1 anchor</day>' in text
        assert "<today" not in text

    @pytest.mark.asyncio
    async def test_pending_day_without_prose_is_drawn(self, tmp_path):
        journal = _journal(tmp_path)
        Daily(day=date(2026, 11, 18), status="pending").dump(
            Daily.filepath(journal.root, date(2026, 11, 18))
        )
        await journal.establish(AT)

        view = await journal.view(AT)
        assert [entry.day.isoformat() for entry in view.days] == [
            "2026-11-18", "2026-11-19",
        ]

        text = await journal.render(AT)
        assert '<day date="2026-11-18" status="pending"/>' in text
        assert '<day date="2026-11-19" status="pending"/>' in text

    @pytest.mark.asyncio
    async def test_today_anchors_are_read_for_the_pending_row(self, tmp_path):
        journal = _journal(tmp_path)
        await journal.record_commit("1-784", datetime(2026, 11, 19, 10, tzinfo=TZ))

        view = await journal.view(AT)
        assert view.today_anchors == ["1-784"]

        text = await journal.render(AT)
        assert '<day date="2026-11-19" status="pending">1 anchor</day>' in text

    @pytest.mark.asyncio
    async def test_open_tail_is_not_hidden(self, tmp_path):
        journal = _journal(tmp_path, day_summaries=3)
        root = journal.root
        _closed(root, date(2026, 11, 16))
        _closed(root, date(2026, 11, 17))
        Daily(
            day=date(2026, 11, 18), status="writing", summary="day 18", content="body 18",
        ).dump(Daily.filepath(root, date(2026, 11, 18)))

        view = await journal.view(AT)
        assert [entry.day.isoformat() for entry in view.days] == [
            "2026-11-16", "2026-11-17", "2026-11-18",
        ]
        assert view.last_day is not None
        assert view.last_day.day == date(2026, 11, 17)
        assert view.today is None

        text = await journal.render(AT)
        assert '<day date="2026-11-18" status="writing">day 18</day>' in text
        assert '<last_day date="2026-11-17">body 17</last_day>' in text

    @pytest.mark.asyncio
    async def test_rows_without_text_are_not_drawn(self, tmp_path):
        journal = _journal(tmp_path)
        Monthly(year=2026, month=10, summary="", content="").dump(
            Monthly.filepath(journal.root, 2026, 10)
        )
        _closed(journal.root, date(2026, 11, 18))

        text = await journal.render(AT)
        assert "<month" not in text
        assert '<last_day date="2026-11-18">body 18</last_day>' in text

    @pytest.mark.asyncio
    async def test_closed_today_is_the_last_day_not_a_today_element(self, tmp_path):
        journal = _journal(tmp_path)
        await journal.record_commit("1-784", datetime(2026, 11, 19, 10, tzinfo=TZ))
        _closed(journal.root, AT, summary="today head", content="today body")

        text = await journal.render(AT)
        assert '<last_day date="2026-11-19">today body</last_day>' in text
        assert "<today" not in text

    @pytest.mark.asyncio
    async def test_empty_journal_renders_self_closing(self, tmp_path):
        assert await _journal(tmp_path).render(AT) == '<journal at="2026-11-19"/>'

    @pytest.mark.asyncio
    async def test_content_is_escaped(self, tmp_path):
        journal = _journal(tmp_path)
        _closed(journal.root, date(2026, 11, 18), summary="a <memory> day", content="yesterday body")
        _closed(journal.root, AT, summary="today head", content="a <memory> body")

        text = await journal.render(AT)
        assert "a &lt;memory&gt; day" in text
        assert "a &lt;memory&gt; body" in text


class TestJournalStatus:
    @pytest.mark.asyncio
    async def test_budgets_and_broken_document(self, tmp_path):
        journal = _journal(tmp_path, day_chars=100)
        Daily.filepath(journal.root, AT).parent.mkdir(parents=True, exist_ok=True)
        Daily.filepath(journal.root, AT).write_text("---\nbad: [\n---\n", encoding="utf-8")

        status = await journal.status(AT)
        assert status.today is None
        assert status.problems
        assert "daily.md" in status.problems[0]
        day_budget = next(report for report in status.budgets if report.level == "day")
        assert day_budget.limit == 100
        assert day_budget.current == 0


class TestJournalChannel:
    def test_journal_is_a_gated_virtual_child_of_ghost(self, tmp_path):
        from ghoshell_moss.ghosts.dolores.channel import build_dolores_channel

        ghost = build_dolores_channel(journal=_journal(tmp_path))
        assert ghost.gate() is True
        children = ghost.build.get_virtual_children()
        assert "journal" in children
        assert set(children["journal"].build.own_commands()) == {"render", "status", "commits"}


class TestJournalExport:
    def test_specification_renders_the_seed(self, tmp_path):
        journal = FileJournal(tmp_path, JournalConfig(), prompt=lambda key: JOURNAL_SPEC_SEED)
        assert journal.specification() == JOURNAL_SPEC_SEED

    @pytest.mark.asyncio
    async def test_export_writes_spec_and_limits(self, tmp_path):
        journal = FileJournal(
            tmp_path, JournalConfig(day_chars=1234), prompt=lambda key: JOURNAL_SPEC_SEED,
        )
        path = Path(await journal.export())
        text = path.read_text(encoding="utf-8")

        assert path == journal.root / "AGENTS.md"
        assert "generated from the journal spec" in text
        assert "# Journal" in text
        assert "- day: 1234 characters" in text

    @pytest.mark.asyncio
    async def test_export_without_a_prompt_reader_fails_loudly(self, tmp_path):
        with pytest.raises(RuntimeError):
            await _journal(tmp_path).export()
