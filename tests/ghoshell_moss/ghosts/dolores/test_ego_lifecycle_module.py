"""The ego lifecycle module surface: registration is inert, start is the one activation point."""

from datetime import date, datetime, timezone

import pytest

from ghoshell_moss.ghosts.dolores.mindset.ego_lifecycle import CommitEvent, ThinkingStartEvent
from ghoshell_moss.ghosts.dolores.mindset.ego_lifecycle_bus import EgoLifecycleBus
from ghoshell_moss.ghosts.dolores.mindset.ego_lifecycle_module import EgoLifecycleRegistry
from ghoshell_moss.ghosts.dolores.mindset.journal_module import (
    EgoLifecycleRegistryProvider,
    JournalModule,
    JournalModuleProvider,
)


class _Module:
    def __init__(self, *, fail_enter: bool = False, fail_listen: bool = False) -> None:
        self.entered = 0
        self.listened = 0
        self.exited = 0
        self.fail_enter = fail_enter
        self.fail_listen = fail_listen

    async def __aenter__(self):
        self.entered += 1
        if self.fail_enter:
            raise RuntimeError("enter failed")
        return self

    async def listen(self, lifecycle) -> None:
        self.listened += 1
        if self.fail_listen:
            raise RuntimeError("listen failed")

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.exited += 1


class _Container:
    def __init__(self, journal) -> None:
        self._journal = journal

    def force_fetch(self, contract):
        return self._journal


class _FakeJournal:
    def __init__(self) -> None:
        self.current = date(2026, 11, 18)
        self.established: list[date] = []
        self.recorded: list[tuple[str, datetime | None]] = []

    def today(self) -> date:
        return self.current

    async def establish(self, at: date | datetime | None = None) -> date:
        day = self.current if at is None else at
        self.established.append(day)
        return day

    async def record_commit(
            self, coord: str, at: datetime | None = None, *, owner: str = "", branch: str = "",
    ) -> None:
        self.recorded.append((coord, at))


class TestEgoLifecycleRegistry:
    @pytest.mark.asyncio
    async def test_registration_is_inert_until_start(self):
        registry = EgoLifecycleRegistry()
        module = _Module()
        registry.register("h", module)

        assert (module.entered, module.listened, module.exited) == (0, 0, 0)

        await registry.start(object())
        assert (module.entered, module.listened) == (1, 1)

        await registry.stop()
        assert module.exited == 1

    @pytest.mark.asyncio
    async def test_duplicate_names_are_rejected(self):
        registry = EgoLifecycleRegistry()
        registry.register("h", _Module())

        with pytest.raises(ValueError):
            registry.register("h", _Module())

    @pytest.mark.asyncio
    async def test_register_after_start_is_rejected(self):
        registry = EgoLifecycleRegistry()
        await registry.start(object())

        with pytest.raises(RuntimeError):
            registry.register("late", _Module())

    @pytest.mark.asyncio
    async def test_start_is_idempotent(self):
        registry = EgoLifecycleRegistry()
        module = _Module()
        registry.register("h", module)

        await registry.start(object())
        await registry.start(object())

        assert module.entered == 1

    @pytest.mark.asyncio
    async def test_non_critical_failure_is_isolated(self):
        registry = EgoLifecycleRegistry()
        broken = _Module(fail_enter=True)
        healthy = _Module()
        registry.register("broken", broken)
        registry.register("healthy", healthy)

        await registry.start(object())

        assert broken.entered == 1
        assert healthy.entered == 1
        assert healthy.listened == 1

    @pytest.mark.asyncio
    async def test_critical_failure_stops_started_hooks_and_raises(self):
        registry = EgoLifecycleRegistry()
        first = _Module()
        critical = _Module(fail_enter=True)
        registry.register("first", first)
        registry.register("critical", critical, critical=True)

        with pytest.raises(RuntimeError):
            await registry.start(object())

        assert first.exited == 1

    @pytest.mark.asyncio
    async def test_stop_before_start_is_a_noop(self):
        registry = EgoLifecycleRegistry()
        module = _Module()
        registry.register("h", module)

        await registry.stop()

        assert module.entered == 0
        assert module.exited == 0


class TestJournalModule:
    @pytest.mark.asyncio
    async def test_start_opens_today_and_midnight_check_opens_the_next_day(self):
        journal = _FakeJournal()
        module = JournalModule(_Container(journal))
        bus = EgoLifecycleBus()
        registry = EgoLifecycleRegistry()
        registry.register("journal", module)

        await registry.start(bus)
        assert journal.established == [date(2026, 11, 18)]

        journal.current = date(2026, 11, 19)
        await bus.fire_thinking_start(ThinkingStartEvent(seq=1))
        assert journal.established[-1] == date(2026, 11, 19)

        await registry.stop()

    @pytest.mark.asyncio
    async def test_commit_event_is_recorded_with_its_own_instant(self):
        journal = _FakeJournal()
        module = JournalModule(_Container(journal))
        bus = EgoLifecycleBus()
        registry = EgoLifecycleRegistry()
        registry.register("journal", module)
        await registry.start(bus)
        at = datetime(2026, 11, 18, 23, 59, tzinfo=timezone.utc)

        bus.fire_commit(CommitEvent(commit_id="c1", coord="1-7", at=at))
        await bus.wait_all_done()

        assert journal.recorded == [("1-7", at)]
        await registry.stop()

    @pytest.mark.asyncio
    async def test_stop_unregisters_the_hooks_listeners(self):
        journal = _FakeJournal()
        module = JournalModule(_Container(journal))
        bus = EgoLifecycleBus()
        registry = EgoLifecycleRegistry()
        registry.register("journal", module)
        await registry.start(bus)
        await registry.stop()

        bus.fire_commit(CommitEvent(
            commit_id="c1", coord="1-7", at=datetime(2026, 11, 18, 23, 59, tzinfo=timezone.utc),
        ))
        await bus.wait_all_done()

        assert journal.recorded == []


class TestJournalModuleProvider:
    @pytest.mark.asyncio
    async def test_provider_bootstrap_registers_and_start_activates(self, tmp_path):
        from ghoshell_container.containers import Container

        from ghoshell_moss.ghosts.dolores.journal.contract import Daily, DoloresJournal, JournalConfig
        from ghoshell_moss.ghosts.dolores.journal.journal_file import FileJournal

        container = Container(name="test")
        container.register(EgoLifecycleRegistryProvider())
        container.register(JournalModuleProvider())
        container.bootstrap()
        journal = FileJournal(tmp_path, JournalConfig(tz="+08:00"))
        container.set(DoloresJournal, journal)

        registry = container.force_fetch(EgoLifecycleRegistry)
        bus = EgoLifecycleBus()
        await registry.start(bus)
        try:
            assert Daily.filepath(journal.root, journal.today()).exists()
        finally:
            await registry.stop()
