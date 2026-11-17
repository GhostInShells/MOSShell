"""SchedulerNucleus — the plan store and the "time came due → signal" mapping.

What is under test:

- records round-trip through YAML, and a malformed file costs one record, not the set;
- firings expand from the rule plus the watermark, lead offsets included;
- a stopped process catches up at most one firing per plan, and the message says so;
- the four tiers map to the four framework signals — the scheduler never fires an impulse;
- a spent one-shot archives itself; a recurring plan keeps its rule and advances;
- ``clear()`` drops the cache, never the files;
- ``chat`` enters the channel surface only when a recaller is bound.
"""

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from ghoshell_moss.core.blueprint.mindflow import Signal
from ghoshell_moss.core.mindflow.scheduler import (
    AnchorRef,
    FilesystemTimerStore,
    Recurrence,
    SchedulerNucleus,
    SchedulerRecaller,
    TaskClass,
    TimerTask,
    encode_anchor,
    parse_duration,
    parse_offset,
)

TZ = timezone(timedelta(hours=8))


def at(hour: int, minute: int = 0, day: int = 18) -> datetime:
    return datetime(2026, 11, day, hour, minute, tzinfo=TZ)


@pytest.fixture
def store(tmp_path):
    return FilesystemTimerStore(tmp_path / "scheduler")


class FakeRecaller(SchedulerRecaller):
    """A recaller that owns a made-up metatype — enough to prove the seam."""

    def __init__(self) -> None:
        self.asked: list[str] = []

    def metatype(self) -> str:
        return "test-session"

    def stamp(self):
        return encode_anchor("test-session", turn=7)

    def describe(self, anchor: AnchorRef) -> str:
        return "written at turn 7"

    async def chat(self, anchor: AnchorRef, prompt: str) -> str:
        self.asked.append(prompt)
        return f"answer to {prompt}"


def make_nucleus(store, **kwargs) -> SchedulerNucleus:
    nucleus = SchedulerNucleus(store, clock=lambda: at(12), **kwargs)
    return nucleus


def arm(nucleus: SchedulerNucleus, **kwargs) -> str:
    defaults = {"kind": "once", "text": "check the release", "tier": "notify",
                "offsets": None, "now": at(12), "due": at(13)}
    defaults.update(kwargs)
    return nucleus._arm(**defaults)


# ── parsing and shape ────────────────────────────────────────────


def test_duration_units():
    assert parse_duration("90") == 90
    assert parse_duration("5m") == 300
    assert parse_duration("2h") == 7200
    assert parse_duration("1d") == 86400
    assert parse_duration("1w") == 604800
    with pytest.raises(ValueError):
        parse_duration("soon")


def test_offsets_are_leads_not_forecasts():
    assert parse_offset("at") == 0
    assert parse_offset("-10m") == -600
    with pytest.raises(ValueError):
        parse_offset("+10m")


def test_record_shape_must_match_kind():
    with pytest.raises(ValueError):
        TimerTask(id="sch-000001", kind="once", text="x", created=at(12))
    with pytest.raises(ValueError):
        TimerTask(id="sch-000001", kind="every", text="x", created=at(12), due=at(13))
    with pytest.raises(ValueError):
        TimerTask(id="sch-000001", kind="once", text="x", created=at(12),
                  due=at(13), rule=Recurrence(every=60, start=at(12)))


def test_bad_offset_is_refused_at_creation_not_at_fire_time():
    with pytest.raises(ValueError):
        TimerTask(id="sch-000001", kind="once", text="x", created=at(12),
                  due=at(13), notify=["+5m"])


# ── firing arithmetic ────────────────────────────────────────────


def test_lead_offsets_expand_around_the_due_instant():
    task = TimerTask(id="sch-000001", kind="once", text="x", created=at(12),
                     due=at(13), notify=["-10m", "at"])
    assert [i.strftime("%H:%M") for i in task.firings_between(at(12), at(14))] == ["12:50", "13:00"]


def test_watermark_starts_at_creation_so_a_lead_born_too_late_never_fires():
    task = TimerTask(id="sch-000001", kind="once", text="x", created=at(12, 55),
                     due=at(13), notify=["-10m", "at"])
    # The -10m lead (12:50) predates the plan; only the due instant is still ahead.
    assert [i.strftime("%H:%M") for i in task.due_firings(at(12, 56))] == []


def test_recurrence_expands_on_a_fixed_grid_bounded_by_until():
    rule = Recurrence(every=3600, start=at(9), until=at(12))
    task = TimerTask(id="sch-000002", kind="every", text="x", created=at(8), rule=rule)
    assert [o.strftime("%H:%M") for o in task.occurrences_between(at(9), at(13))] == [
        "09:00", "10:00", "11:00", "12:00",
    ]


def test_next_instant_is_strictly_after_now():
    task = TimerTask(id="sch-000001", kind="once", text="x", created=at(12), due=at(13))
    assert task.next_instant_after(at(12, 59)) == at(13)
    assert task.next_instant_after(at(13)) is None


# ── store ────────────────────────────────────────────────────────


def test_store_round_trip_and_yaml_shape(store):
    task = TimerTask(id="sch-000001", kind="once", text="line one\nline two",
                     created=at(12), due=at(13), notify=["-10m", "at"],
                     task_class=TaskClass.KNOCK, anchor=encode_anchor("x", turn=3))
    store.save(task)
    loaded = store.load()
    assert len(loaded) == 1
    assert loaded[0].text == "line one\nline two"
    assert loaded[0].task_class is TaskClass.KNOCK
    assert loaded[0].notify == ["-10m", "at"]
    raw = (store.root / "tasks" / "sch-000001.yml").read_text(encoding="utf-8")
    assert "class: knock" in raw


def test_ids_are_monotonic_and_survive_a_restart(store):
    assert (store.allocate_id(), store.allocate_id()) == ("sch-000001", "sch-000002")
    # A record landed but the counter did not (a crash between the two writes). The next
    # process must not hand out an id that is already in use.
    store.save(TimerTask(id="sch-000009", kind="once", text="x", created=at(12), due=at(13)))
    reopened = FilesystemTimerStore(store.root)
    assert reopened.allocate_id() == "sch-000010"


def test_a_broken_record_costs_only_itself(store):
    store.save(TimerTask(id="sch-000001", kind="once", text="x", created=at(12), due=at(13)))
    (store.root / "tasks" / "sch-000002.yml").write_text("not: [a, record", encoding="utf-8")
    loaded = store.load()
    assert [task.id for task in loaded] == ["sch-000001"]


# ── nucleus: firing ──────────────────────────────────────────────


@pytest.mark.parametrize(("tier", "expected"), [
    ("aside", "aside"),
    ("knock", "knock"),
    ("notify", "notify"),
    ("interrupt", "interrupt"),
])
def test_each_tier_maps_to_its_signal(store, tier, expected):
    nucleus = make_nucleus(store)
    emitted: list[Signal] = []
    nucleus.with_bus(emitted.append, lambda impulse: None)
    arm(nucleus, tier=tier, due=at(13))

    nucleus.pump(at(12, 59))
    assert emitted == []

    nucleus.pump(at(13))
    assert [signal.name for signal in emitted] == [expected]


def test_the_scheduler_never_fires_an_impulse(store):
    nucleus = make_nucleus(store)
    impulses: list[object] = []
    nucleus.with_bus(lambda signal: None, impulses.append)
    arm(nucleus, due=at(13))
    nucleus.pump(at(13))
    assert impulses == []
    assert nucleus.peek() is None


def test_a_spent_one_shot_archives_itself(store):
    nucleus = make_nucleus(store)
    nucleus.with_bus(lambda signal: None, lambda impulse: None)
    arm(nucleus, due=at(13))
    nucleus.pump(at(13))

    assert list((store.root / "tasks").glob("*.yml")) == []
    assert len(store.load_archived()) == 1


def test_a_recurring_plan_advances_and_never_repeats_the_same_instant(store):
    nucleus = make_nucleus(store)
    emitted: list[Signal] = []
    nucleus.with_bus(emitted.append, lambda impulse: None)
    rule = Recurrence(every=3600, start=at(13))
    arm(nucleus, kind="every", rule=rule, due=None)

    nucleus.pump(at(13))
    nucleus.pump(at(13))
    assert len(emitted) == 1

    nucleus.pump(at(14))
    assert len(emitted) == 2


def test_a_graceful_stop_catches_up_once_and_says_it_was_late(store):
    nucleus = make_nucleus(store)
    emitted: list[Signal] = []
    nucleus.with_bus(emitted.append, lambda impulse: None)
    rule = Recurrence(every=3600, start=at(13))
    arm(nucleus, kind="every", rule=rule, due=None)

    # Three hours pass with the process asleep; one emission, not four. The latest
    # occurrence is the one delivered, and it is late by the remainder.
    nucleus.pump(at(16, 20))
    assert len(emitted) == 1
    text = emitted[0].messages[0].to_content_string()
    assert "20m late" in text
    assert "3 earlier firing(s) skipped" in text


def test_late_is_not_reported_for_an_on_time_tick(store):
    nucleus = make_nucleus(store)
    emitted: list[Signal] = []
    nucleus.with_bus(emitted.append, lambda impulse: None)
    arm(nucleus, due=at(13))

    nucleus.pump(at(13, 0))
    text = emitted[0].messages[0].to_content_string()
    assert "late" not in text
    assert "sch-000001" in text


def test_nothing_fires_before_the_mindflow_is_running(store):
    nucleus = SchedulerNucleus(store, clock=lambda: at(13), ready=lambda: False)
    emitted: list[Signal] = []
    nucleus.with_bus(emitted.append, lambda impulse: None)
    arm(nucleus, due=at(13))

    nucleus.pump(at(13))
    assert emitted == []
    # The plan is still armed: a bus that was not ready must not consume it.
    assert len(store.load()) == 1


def test_clear_keeps_the_files(store):
    nucleus = make_nucleus(store)
    arm(nucleus, due=at(13))
    nucleus.clear()
    assert (store.root / "tasks" / "sch-000001.yml").exists()
    nucleus.pump(at(13))
    assert (store.root / "tasks" / "sch-000001.yml").exists()


# ── channel surface ──────────────────────────────────────────────


def test_chat_is_unavailable_without_a_recaller(store):
    # The command exists as a definition but reports itself unavailable; the facade never
    # shows an unavailable command to the model.
    channel = make_nucleus(store).as_channel()
    commands = channel.build.own_commands()
    assert commands["chat"].is_available() is False
    assert commands["at"].is_available() is True


def test_chat_becomes_available_when_a_recaller_is_bound(store):
    channel = make_nucleus(store, recaller=FakeRecaller()).as_channel()
    assert channel.build.own_commands()["chat"].is_available() is True


@pytest.mark.asyncio
async def test_arming_through_the_channel_writes_a_record_and_answers(store):
    recaller = FakeRecaller()
    nucleus = make_nucleus(store, recaller=recaller)
    channel = nucleus.as_channel()
    receipt = await channel.build.own_commands()["in"](seconds=3600, text="check the release")

    assert "armed sch-000001" in receipt
    tasks = store.load()
    assert len(tasks) == 1
    assert tasks[0].anchor.metatype == "test-session"
    assert tasks[0].due == at(13)


@pytest.mark.asyncio
async def test_list_renders_the_window_as_json(store):
    nucleus = make_nucleus(store)
    channel = nucleus.as_channel()
    await channel.build.own_commands()["at"](when=at(13).isoformat(), text="看发布")

    import json
    payload = json.loads(await channel.build.own_commands()["list"]())
    assert payload["window"][0] == at(12).isoformat()
    assert [task["id"] for task in payload["tasks"]] == ["sch-000001"]
    assert payload["tasks"][0]["firings"] == [at(13).isoformat()]


@pytest.mark.asyncio
async def test_chat_returns_a_receipt_and_delivers_the_answer_later(store, monkeypatch):
    """The channel does not hold the queue: receipt now, answer as a mind delivery."""
    from ghoshell_moss.core.blueprint.channel_builder import CommandUtil

    recaller = FakeRecaller()
    nucleus = make_nucleus(store, recaller=recaller)
    channel = nucleus.as_channel()
    await channel.build.own_commands()["at"](when=at(13).isoformat(), text="x")
    sent = []
    monkeypatch.setattr(CommandUtil, "send_signal", lambda signal: sent.append(signal))
    async with channel.bootstrap() as runtime:
        receipt = await runtime.execute_command("chat", args=("sch-000001", "why"))
        for _ in range(100):
            if sent:
                break
            await asyncio.sleep(0.01)

    assert receipt.startswith("[scheduler] asking sch-000001")
    assert recaller.asked == ["why"]
    assert sent[0].name == "mind/delivery"
    assert sent[0].messages[0].to_content_string() == "answer to why"
