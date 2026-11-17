"""SchedulerNucleus — the ghost's own time plan, as a sensing unit.

The nucleus has exactly one job: when a plan's instant comes due, name the signal tier
that plan asked for and broadcast it. It never arbitrates. That is deliberate — the four
tiers (``aside`` / ``knock`` / ``notify`` / ``interrupt``) already carry the policy, each
enforced by its own nucleus, and re-implementing the ladder here would give the system a
second truth.

Three consequences worth stating up front:

- ``peek`` / ``attended`` / ``suppress`` stay empty. The scheduler produces no impulse of
  its own; it hands signals to the bus and lets the matching nucleus compete.
- ``clear()`` never touches the disk. ``clear`` is the emergency reset hammer; wiping a
  ghost's plans with it would delete state the ghost cannot rebuild. The in-memory cache
  drops and reloads instead.
- The tick loop waits for ``mindflow.is_running()``. ``add_signal`` ignores signals while
  the bus is down, so firing at boot before the mindflow starts would mark a plan as
  delivered and lose it.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Iterable
from datetime import datetime, timedelta
from pathlib import Path

from ghoshell_container import IoCContainer
from typing_extensions import Self

from ghoshell_moss.contracts.logger import LoggerItf, get_moss_logger
from ghoshell_moss.core.blueprint.channel_builder import Channel, CommandUtil, new_channel
from ghoshell_moss.core.blueprint.environment import Environment
from ghoshell_moss.core.blueprint.mindflow import (
    Impulse,
    Mindflow,
    Nucleus,
    NucleusMeta,
    Signal,
    SignalMeta,
    SignalName,
)
from ghoshell_moss.core.mindflow.aside_nucleus import new_aside_signal
from ghoshell_moss.core.mindflow.interrupt_nucleus import new_interrupt_signal
from ghoshell_moss.core.mindflow.knock_nucleus import new_knock_signal
from ghoshell_moss.core.mindflow.mind_delivery_nucleus import new_mind_delivery_signal
from ghoshell_moss.core.mindflow.notify_nucleus import new_notify_signal

from .models import (
    AnchorRef,
    Recurrence,
    TaskClass,
    TimerTask,
    human_delta,
    parse_duration,
    parse_time,
)
from .recaller import SchedulerRecaller
from .store import FilesystemTimerStore, TimerStore

__all__ = [
    "ScheduledPlanError",
    "SchedulerNucleus",
    "SchedulerNucleusMeta",
    "scheduler_root",
]

#: Where one ghost's plans live. Kept under the workspace runtime tree so the files are
#: machine state, not repository content; the stub ships the ``.gitignore`` that keeps it
#: out of version control.
_RUNTIME_DIR = "runtime/scheduler"


def scheduler_root(env: Environment) -> Path:
    """``<workspace>/runtime/scheduler/ghost-{name}`` — one store per ghost."""
    return env.workspace_path.joinpath(_RUNTIME_DIR, f"ghost-{env.ghost_name}")


class ScheduledPlanError(ValueError):
    """The caller asked for a plan that cannot exist (bad time, bad tier, bad window)."""


def _local_now() -> datetime:
    return datetime.now().astimezone()


def _parse_tier(value: str) -> TaskClass:
    raw = str(value or TaskClass.NOTIFY.value).strip().lower()
    try:
        return TaskClass(raw)
    except ValueError:
        allowed = ", ".join(item.value for item in TaskClass)
        raise ScheduledPlanError(f"unknown tier {value!r}; use one of: {allowed}") from None


class SchedulerNucleus(Nucleus):
    """Turns "the clock reached this instant" into one of the framework's signals."""

    NAME = "scheduler"
    DESCRIPTION = "the ghost's own time plan — timers that come due as signals"

    def __init__(
            self,
            store: TimerStore,
            *,
            recaller: SchedulerRecaller | None = None,
            clock: Callable[[], datetime] | None = None,
            logger: LoggerItf | None = None,
            ready: Callable[[], bool] | None = None,
            tick_seconds: float = 15.0,
    ) -> None:
        self._store = store
        self._recaller = recaller
        self._clock = clock or _local_now
        self._logger = logger or get_moss_logger()
        self._ready = ready or (lambda: True)
        self._tick_seconds = tick_seconds
        self._tasks: dict[str, TimerTask] = {}
        self._channel: Channel | None = None
        self._broadcast: Callable[[Signal], None] | None = None
        self._running = False
        self._loaded = False
        self._tick_task: asyncio.Task | None = None

    # ── Nucleus protocol ─────────────────────────────────────────

    def name(self) -> str:
        return self.NAME

    def description(self) -> str:
        return self.DESCRIPTION

    def status(self) -> str:
        """The plan set as a warm fact — never the clock.

        The next firing time is operational state: it changes every minute for a short
        recurrence, and a notice that tracks it turns the warm slot into hot data. The
        next instant is read on demand through ``list``; the firing itself carries it.
        """
        if not self._loaded:
            return "loading"
        if not self._tasks:
            return "no plans"
        return f"{len(self._tasks)} plan(s) armed"

    def signals(self) -> list[SignalName]:
        # The scheduler is a producer only. Listening to nothing is the design, not an
        # omission: a timer has no input besides the clock.
        return []

    def add_signal(self, signal: Signal) -> None:
        """No-op — see ``signals``."""

    def with_bus(
            self,
            signal_broadcast: Callable[[Signal], None],
            fire_impulse: Callable[[Impulse], None],
    ) -> None:
        # Only the broadcast side is used: firing an impulse would mean choosing priority
        # and mode here, duplicating what the four signal nuclei already decide.
        self._broadcast = signal_broadcast

    def peek(self, no_stale: bool = True) -> Impulse | None:
        return None

    def suppress(self, suppress_by: Impulse, suppressed: Impulse | None = None) -> None:
        """No-op — the scheduler never competes for attention."""

    def attended(self, impulse: Impulse) -> None:
        """No-op — the scheduler never produces an impulse."""

    def clear(self) -> None:
        """Drop the in-memory cache only; the files remain the truth."""
        self._tasks.clear()
        self._loaded = False

    def is_running(self) -> bool:
        return self._running

    async def __aenter__(self) -> Self:
        self._running = True
        self._reload()
        self._tick_task = asyncio.create_task(self._tick_loop(), name="scheduler-tick")
        self._logger.info(
            "scheduler: %d plan(s) armed at %s", len(self._tasks), self._store.root,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self._running = False
        if self._tick_task is not None:
            self._tick_task.cancel()
            try:
                await self._tick_task
            except asyncio.CancelledError:
                pass
            self._tick_task = None

    def as_channel(self) -> Channel | None:
        if self._channel is None:
            self._channel = self._build_channel()
        return self._channel

    # ── the clock ────────────────────────────────────────────────

    def _reload(self) -> None:
        self._tasks = {task.id: task for task in self._store.load()}
        self._loaded = True

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self._reload()

    async def _tick_loop(self) -> None:
        while self._running:
            try:
                self.pump()
            except asyncio.CancelledError:
                raise
            except Exception:
                self._logger.exception("scheduler: pump failed; the plan set is unchanged")
            await asyncio.sleep(self._sleep_seconds())

    def _sleep_seconds(self) -> float:
        if not self._loaded or not self._tasks:
            return self._tick_seconds
        now = self._clock()
        upcoming = [
            instant
            for instant in (task.next_instant_after(now) for task in self._tasks.values())
            if instant is not None
        ]
        if not upcoming:
            return self._tick_seconds
        return max(0.2, min(self._tick_seconds, (min(upcoming) - now).total_seconds()))

    def pump(self, now: datetime | None = None) -> list[TimerTask]:
        """Emit every plan whose instant has come due. Returns the plans that fired.

        The tick loop calls this on its own schedule; passing ``now`` is the test seam —
        the clock is data, so a test advances it instead of sleeping through it.
        """
        if self._broadcast is None or not self._ready():
            return []
        self._ensure_loaded()
        now = now or self._clock()
        fired: list[TimerTask] = []
        for task in list(self._tasks.values()):
            due = task.due_firings(now)
            if not due:
                continue
            instant = due[-1]
            missed = len(due) - 1
            if task.kind == "every":
                # at-most-once: persist the watermark *before* the signal can exist, so a
                # crash between the two skips one tick instead of doubling a recurring act.
                task.last_fire_at = instant
                task.fires += 1
                self._store.save(task)
                self._emit(task, instant, now, missed)
            else:
                # at-least-once: a crash between the emission and the write re-fires the
                # reminder on the next boot — a duplicate reminder is cheaper than a loss.
                self._emit(task, instant, now, missed)
                task.last_fire_at = instant
                task.fires += 1
                if instant >= task.final_instant():
                    self._store.archive(task)
                    self._tasks.pop(task.id, None)
                else:
                    self._store.save(task)
            fired.append(task)
        return fired

    def _emit(self, task: TimerTask, instant: datetime, now: datetime, missed: int) -> None:
        assert self._broadcast is not None
        late = (now - instant).total_seconds()
        text = self._render(task, instant, late, missed)
        signal = self._new_signal(task.task_class, text, task)
        self._logger.info(
            "scheduler: firing %s [%s] at %s%s",
            task.id,
            task.task_class.value,
            instant.isoformat(),
            f" (late {human_delta(late)})" if late >= 60 else "",
        )
        self._broadcast(signal)

    def _render(self, task: TimerTask, instant: datetime, late: float, missed: int) -> str:
        stamp = instant.astimezone().strftime("%Y-%m-%d %H:%M")
        header = f"[plan] {task.id} · {stamp}"
        if late >= 60:
            header += f" · {human_delta(late)} late"
        if missed:
            header += f" · {missed} earlier firing(s) skipped"
        lines = [header, task.text.strip()]
        if self._recaller is not None:
            try:
                where = self._recaller.describe(task.anchor)
            except Exception:
                self._logger.exception("scheduler: recaller could not describe %s", task.id)
                where = ""
            if where:
                lines.append(where)
            lines.append(f'that self is still reachable: mindflow.scheduler:chat id="{task.id}"')
        return "\n".join(lines)

    @staticmethod
    def _new_signal(tier: TaskClass, text: str, task: TimerTask) -> Signal:
        description = f"{task.id}: {task.text.strip().splitlines()[0][:80]}"
        if tier is TaskClass.ASIDE:
            return new_aside_signal(text, description=description)
        if tier is TaskClass.KNOCK:
            return new_knock_signal(text, description=description)
        if tier is TaskClass.INTERRUPT:
            return new_interrupt_signal(text, description=description)
        return new_notify_signal(text, description=description)

    # ── arming ───────────────────────────────────────────────────

    def _stamp_anchor(self):
        if self._recaller is None:
            return None
        try:
            return self._recaller.stamp()
        except Exception:
            self._logger.exception("scheduler: recaller could not stamp the current position")
            return None

    def _arm(
            self,
            *,
            kind: str,
            text: str,
            tier: str,
            offsets: list[str] | None,
            now: datetime,
            due: datetime | None = None,
            rule: Recurrence | None = None,
    ) -> str:
        self._ensure_loaded()
        if not str(text).strip():
            return "error: the plan needs a text — it is the line your future self reads"
        try:
            task_class = _parse_tier(tier)
        except ScheduledPlanError as error:
            return f"error: {error}"
        notify = list(offsets) if offsets else ["at"]
        task = TimerTask(
            id=self._store.allocate_id(),
            kind=kind,
            text=str(text),
            task_class=task_class,
            notify=notify,
            due=due,
            rule=rule,
            created=now,
            anchor=self._stamp_anchor() or AnchorRef(),
        )
        self._store.save(task)
        self._tasks[task.id] = task
        return self._receipt(task, now)

    def _receipt(self, task: TimerTask, now: datetime) -> str:
        upcoming = task.next_instant_after(now)
        when = "no future firing"
        if upcoming is not None:
            when = (
                f"{upcoming.astimezone():%Y-%m-%d %H:%M} "
                f"(in {human_delta((upcoming - now).total_seconds())})"
            )
        lines = [
            f"armed {task.id} · {when} · tier {task.task_class.value} · "
            f"notify {', '.join(task.notify)}",
        ]
        context = self._window_context(task, now)
        if context:
            lines.append(context)
        return "\n".join(lines)

    def _window_context(self, task: TimerTask, now: datetime) -> str:
        """Say what else is already planned that day — the decision is being made now."""
        anchor_time = task.due or (task.rule.start if task.rule else now)
        day_start = anchor_time.astimezone().replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)
        instants = sorted(
            instant
            for other in self._tasks.values()
            if other.id != task.id
            for instant in other.firings_between(day_start, day_end)
        )
        if not instants:
            return ""
        return f"that day: {len(instants)} other firing(s), first at {instants[0]:%H:%M}"

    # ── the read surface and the channel ─────────────────────────

    def _window(self, start: str, end: str, now: datetime) -> tuple[datetime, datetime]:
        low = parse_time(start, now=now) if start else now
        high = parse_time(end, now=now) if end else low + timedelta(days=1)
        if high <= low:
            raise ScheduledPlanError("end must be after start")
        return low, high

    def _build_channel(self) -> Channel:
        chan = new_channel(name=self.NAME, description=self.DESCRIPTION)

        @chan.build.instruction
        async def _instruction() -> str:
            return (
                "## Scheduler (your own time plan)\n"
                "A plan is a line to a future self: it fires at the time you choose and the\n"
                "message carries a pointer back to whoever wrote it. The files under\n"
                f"`{self._store.root}` are the truth — one YAML per plan; `list` renders them\n"
                "as one JSON.\n\n"
                "Tiers say how hard a plan may push: `aside` never interrupts; `knock` is lost\n"
                "if you are busy; `notify` is never lost (it lands in the next frame when it\n"
                "cannot take attention); `interrupt` stops everything.\n"
            )

        @chan.build.notice
        def _notice() -> str:
            return self.status() or "no plans"

        @chan.build.command(name="at")
        async def _at(when: str, text: str, tier: str = "notify",
                      notify: list[str] | None = None) -> str:
            """Plan one reminder at an absolute time.

            :param when: ISO time, e.g. `2026-11-18T19:30:00+08:00`; without an offset it
                is read in the machine's zone.
            :param text: the line your future self will read — what to do, not what happened.
            :param tier: how hard it may push — `aside` (never interrupt) / `knock` (lost if
                busy) / `notify` (never lost, default) / `interrupt` (stop everything).
            :param notify: lead offsets relative to the due instant, e.g. `["-10m", "at"]`;
                `at` is the instant itself. Default `["at"]`.
            """
            now = self._clock()
            try:
                due = parse_time(when, now=now)
            except ValueError as error:
                return f"error: {error}"
            return self._arm(kind="once", text=text, tier=tier, offsets=notify, now=now, due=due)

        @chan.build.command(name="in")
        async def _in(seconds: float, text: str, tier: str = "notify",
                      notify: list[str] | None = None) -> str:
            """Plan one reminder after a delay — the shortcut for "in N minutes".

            :param seconds: delay in seconds (float, so 90.5 is legal).
            :param text: the line your future self will read.
            :param tier: `aside` / `knock` / `notify` (default) / `interrupt`.
            :param notify: lead offsets relative to the due instant; default `["at"]`.
            """
            now = self._clock()
            due = now + timedelta(seconds=float(seconds))
            return self._arm(kind="once", text=text, tier=tier, offsets=notify, now=now, due=due)

        @chan.build.command(name="every")
        async def _every(interval: str, text: str, start: str = "", until: str = "",
                         tier: str = "notify", notify: list[str] | None = None) -> str:
            """Plan a repeating reminder on a fixed rate.

            :param interval: how often, e.g. `30m`, `2h`, `1d`, `1w` (plain number = seconds).
            :param text: the line your future self will read each time.
            :param start: first occurrence, ISO time; empty = now.
            :param until: last allowed occurrence, ISO time; empty = forever.
            :param tier: `aside` / `knock` / `notify` (default) / `interrupt`.
            :param notify: lead offsets relative to each occurrence; default `["at"]`.
            """
            now = self._clock()
            try:
                every = parse_duration(interval)
                first = parse_time(start, now=now) if start else now
                last = parse_time(until, now=now) if until else None
            except ValueError as error:
                return f"error: {error}"
            try:
                rule = Recurrence(every=every, start=first, until=last)
            except ValueError as error:
                return f"error: {error}"
            return self._arm(kind="every", text=text, tier=tier, offsets=notify, now=now, rule=rule)

        @chan.build.command(name="list")
        async def _list(start: str = "", end: str = "", all: bool = False,
                        include_fired: bool = False) -> str:
            """Read the plan set as one JSON — the whole window, occurrences expanded.

            :param start: window start, ISO; empty = now.
            :param end: window end, ISO; empty = start + 24h.
            :param all: ignore the window and return every armed plan with its next firing.
            :param include_fired: also return recently finished one-shot plans.
            """
            self._ensure_loaded()
            now = self._clock()
            try:
                low, high = self._window(start, end, now)
            except (ValueError, ScheduledPlanError) as error:
                return json.dumps({"error": str(error)}, ensure_ascii=False)
            tasks = list(self._tasks.values())
            if not all:
                tasks = [t for t in tasks if t.firings_between(low, high)]
            payload = {
                "now": now.isoformat(),
                "window": [low.isoformat(), high.isoformat()],
                "tasks": [task.view((low, high)) for task in tasks],
            }
            if include_fired:
                payload["fired"] = [
                    task.view((low, high)) for task in self._store.load_archived(limit=50)
                ]
            return json.dumps(payload, ensure_ascii=False, indent=2)

        @chan.build.command(name="cancel")
        async def _cancel(task_id: str) -> str:
            """Remove one plan; it will never fire again.

            :param task_id: the id from `list`, e.g. `sch-000017`.
            """
            self._ensure_loaded()
            task = self._tasks.get(task_id)
            if task is None:
                return f"no plan {task_id} in the armed set"
            self._store.delete(task_id)
            self._tasks.pop(task_id, None)
            return f"cancelled {task_id}"

        @chan.build.command(name="snooze")
        async def _snooze(task_id: str, seconds: float) -> str:
            """Push a one-shot plan later without losing its identity.

            :param task_id: the id from `list`.
            :param seconds: how much later; negative moves it earlier.
            """
            self._ensure_loaded()
            task = self._tasks.get(task_id)
            if task is None:
                return f"no plan {task_id} in the armed set"
            if task.kind != "once" or task.due is None:
                return "error: snooze applies to one-shot plans; cancel and re-arm a rule"
            now = self._clock()
            task.due = task.due + timedelta(seconds=float(seconds))
            # Only instants after *now* may fire: the lead the ghost already saw must not
            # come back just because the due instant moved.
            task.last_fire_at = now
            self._store.save(task)
            return self._receipt(task, now)

        @chan.build.command(name="reload")
        async def _reload_from_disk() -> str:
            """Re-read the plan files from disk — use after editing one by hand."""
            self._reload()
            return f"{len(self._tasks)} plan(s) armed"

        @chan.build.command(name="chat", available=lambda: self._recaller is not None,
                            always_observe=True)
        async def _chat(task_id: str, prompt: str) -> str:
            """Talk with the self that wrote a plan — the answer arrives later.

            Use it when a plan fires and you no longer know why you wrote it. The other
            side sees only what had happened when the plan was written. This returns a
            receipt at once, and the answer arrives as a delivery of its own, so nothing
            holds the turn.

            :param task_id: the id from `list` (archived one-shots work too).
            :param prompt: what you want to ask that self.
            """
            task = self._find(task_id)
            if task is None:
                return f"no plan {task_id}"
            if self._recaller is None:
                return "error: no recaller is bound, so there is no way back to that self"

            async def _answer() -> Signal:
                # The delivery is the only way back once the receipt is out: a failure must
                # come back as content, not as an exception nobody will ever read.
                try:
                    text = await self._recaller.chat(task.anchor, prompt)
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    self._logger.exception("scheduler: chat with %s failed", task_id)
                    return new_mind_delivery_signal(
                        f"[scheduler] chat with {task_id} failed: {error}",
                        kind="scheduler.chat",
                        description=f"chat with {task_id} failed",
                    )
                return new_mind_delivery_signal(
                    text,
                    kind="scheduler.chat",
                    description=f"answer from {task_id}",
                )

            await CommandUtil.create_signal_task(closure=_answer)
            return f"[scheduler] asking {task_id} — the answer will arrive as a signal"

        return chan

    def _find(self, task_id: str) -> TimerTask | None:
        self._ensure_loaded()
        task = self._tasks.get(task_id)
        if task is not None:
            return task
        for archived in self._store.load_archived(limit=200):
            if archived.id == task_id:
                return archived
        return None


class SchedulerNucleusMeta(NucleusMeta):
    """Openbox registration — the scheduler ships with every ghost by default."""

    def name(self) -> str:
        return SchedulerNucleus.NAME

    def description(self) -> str:
        return SchedulerNucleus.DESCRIPTION

    def signals(self) -> Iterable[type[SignalMeta]]:
        return ()

    def factory(self, container: IoCContainer) -> Nucleus:
        env = container.force_fetch(Environment)
        logger = container.get(LoggerItf)
        store = FilesystemTimerStore(scheduler_root(env), logger=logger)
        recaller = container.get(SchedulerRecaller)
        mindflow = container.get(Mindflow)

        def ready() -> bool:
            # A due plan at boot must not be consumed before the bus can take it.
            return mindflow is None or mindflow.is_running()

        return SchedulerNucleus(
            store,
            recaller=recaller,
            logger=logger,
            ready=ready,
        )
