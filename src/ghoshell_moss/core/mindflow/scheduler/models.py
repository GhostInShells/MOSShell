"""Timer records — the durable shape of one plan.

A record is a line to a future self: when it fires, how hard it is allowed to push, and
where the self that wrote it still lives. Everything here is plain data; the clock, the
store and the signal tiers live in siblings.

Two design constraints shape this module:

- **The filesystem is the truth.** Records are YAML files; the model classes must
  round-trip through YAML without loss, so times are aware ISO strings and the payload is
  a plain mapping.
- **Recurrence is a rule plus a watermark.** A recurring record stores ``rule`` and the
  last instant it fired; the next occurrences are computed, never stored — a stored
  "next due" would rot the moment the process dies between two writes.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "AnchorRef",
    "Recurrence",
    "TaskClass",
    "TimerTask",
    "human_delta",
    "local_timezone",
    "parse_duration",
    "parse_offset",
    "parse_time",
]

#: How many instants one expansion may return. A guard, not a policy: a daily task asked
#: for a ten-year window is a caller mistake, and returning it in full would stall the
#: tick loop rather than report it.
EXPANSION_LIMIT = 512

_DURATION_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*([smhdw])?\s*$", re.IGNORECASE)
_UNIT_SECONDS = {"": 1.0, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0, "w": 604800.0}


def local_timezone() -> timezone:
    """The machine's current zone — used whenever a caller passes a naive time."""
    return datetime.now().astimezone().tzinfo or timezone.utc


def parse_duration(text: str) -> float:
    """Parse ``30`` / ``"30s"`` / ``"5m"`` / ``"2h"`` / ``"1d"`` / ``"1w"`` into seconds."""
    match = _DURATION_RE.match(str(text))
    if match is None:
        raise ValueError(f"not a duration: {text!r} (expected e.g. 90, 30s, 5m, 2h, 1d)")
    value, unit = match.group(1), (match.group(2) or "").lower()
    return float(value) * _UNIT_SECONDS[unit]


def parse_time(text: str, *, now: datetime | None = None) -> datetime:
    """Parse an ISO time; a value without an offset is read in the machine's zone."""
    raw = str(text).strip()
    if raw == "":
        raise ValueError("empty time")
    value = datetime.fromisoformat(raw)
    if value.tzinfo is None:
        base = now or datetime.now().astimezone()
        value = value.replace(tzinfo=base.tzinfo or local_timezone())
    return value


def parse_offset(text: str) -> float:
    """Parse one notify offset: ``"at"`` is the due instant itself, ``"-10m"`` is a lead.

    A positive offset would announce the future before it happened; refusing it keeps the
    meaning of the field single.
    """
    raw = str(text).strip().lower()
    if raw in {"at", "on", "0"}:
        return 0.0
    seconds = parse_duration(raw)
    if seconds > 0:
        raise ValueError(f"notify offsets must be zero or negative (a lead), got {text!r}")
    return seconds


def human_delta(seconds: float) -> str:
    """Render a duration the way a sentence can carry it (``2h 3m``)."""
    seconds = abs(seconds)
    if seconds < 60:
        return f"{int(seconds)}s"
    parts: list[str] = []
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if seconds >= size:
            parts.append(f"{int(seconds // size)}{unit}")
            seconds %= size
    return " ".join(parts) or "0s"


class TaskClass(str, Enum):
    """How hard one plan is allowed to push — the four signal tiers, by their own names.

    The scheduler does not arbitrate: it names a tier and the matching nucleus applies
    the policy the framework already defined for it.
    """

    ASIDE = "aside"
    KNOCK = "knock"
    NOTIFY = "notify"
    INTERRUPT = "interrupt"


class AnchorRef(BaseModel):
    """Where the self that wrote this plan can be found — opaque to the scheduler.

    ``metatype`` names the convention (the same vocabulary ``CommitRef.metatype`` uses);
    ``metadata`` is the payload, written and read only by the matching recaller. The
    scheduler never parses it, so a new ghost prototype can invent its own shape without
    touching this module.
    """

    metatype: str = ""
    metadata: str = ""


class Recurrence(BaseModel):
    """A fixed-rate rule: ``start + k * every``, bounded by ``until``.

    Fixed rate, not "after each fire": a 9:00 stand-up stays at 9:00 even when one
    occurrence is handled late. The cost is that a long outage swallows the missed ticks,
    which is the behaviour the scheduler wants anyway (see ``TimerTask.due_firings``).
    """

    every: float = Field(gt=0, description="interval in seconds")
    start: datetime = Field(description="the first occurrence (aware)")
    until: datetime | None = Field(default=None, description="last allowed occurrence, inclusive")

    @model_validator(mode="after")
    def _until_after_start(self) -> Recurrence:
        if self.until is not None and self.until < self.start:
            raise ValueError("rule.until is before rule.start")
        return self


class TimerTask(BaseModel):
    """One durable plan."""

    model_config = ConfigDict(populate_by_name=True)

    version: int = 1
    id: str = Field(description="stable identity; the filename is derived from it")
    kind: Literal["once", "every"]
    text: str = Field(description="what the future self is told")
    task_class: TaskClass = Field(default=TaskClass.NOTIFY, alias="class")
    notify: list[str] = Field(
        default_factory=lambda: ["at"],
        description="lead offsets relative to the due instant; 'at' means the instant itself",
    )
    due: datetime | None = Field(default=None, description="kind=once: the single due instant")
    rule: Recurrence | None = Field(default=None, description="kind=every: the recurrence")
    created: datetime = Field(description="when the plan was written")
    anchor: AnchorRef = Field(default_factory=AnchorRef)
    fires: int = Field(default=0, ge=0, description="how many instants have been emitted")
    last_fire_at: datetime | None = Field(default=None, description="watermark of the last emission")

    @model_validator(mode="after")
    def _shape_matches_kind(self) -> TimerTask:
        if self.kind == "once" and self.due is None:
            raise ValueError("kind=once requires `due`")
        if self.kind == "every" and self.rule is None:
            raise ValueError("kind=every requires `rule`")
        if self.kind == "once" and self.rule is not None:
            raise ValueError("kind=once must not carry `rule`")
        if self.kind == "every" and self.due is not None:
            raise ValueError("kind=every must not carry `due`")
        # Validate the offsets now, not at fire time: a bad offset discovered hours later
        # is a reminder that silently never fires.
        for offset in self.notify:
            parse_offset(offset)
        return self

    # ── firing arithmetic ────────────────────────────────────────

    def offsets(self) -> list[float]:
        return sorted(parse_offset(item) for item in self.notify)

    def firing_instants(self, occurrence: datetime) -> list[datetime]:
        """The instants one occurrence produces — its notify offsets, sorted."""
        return [occurrence + timedelta(seconds=offset) for offset in self.offsets()]

    def final_instant(self) -> datetime:
        """The last instant this record will ever produce (used to archive a one-shot)."""
        occurrence = self.due if self.kind == "once" else (self.rule.until or self.rule.start)
        return max(self.firing_instants(occurrence))

    def watermark(self) -> datetime:
        """Where the fired set ends: the last emission, or creation when nothing fired yet.

        Creation is the honest lower bound — an instant that had already passed when the
        plan was written must not fire, or every new plan would replay a lead offset it
        was born too late for.
        """
        return self.last_fire_at or self.created

    def _occurrences(self, low: datetime, high: datetime) -> list[datetime]:
        """Grid points of the rule inside ``[low, high]``, oldest first."""
        rule = self.rule
        if rule is None or high < rule.start:
            return []
        step = rule.every
        if low <= rule.start:
            index = 0
        else:
            index = math.ceil((low - rule.start).total_seconds() / step)
            if index < 0:
                index = 0
        current = rule.start + timedelta(seconds=index * step)
        found: list[datetime] = []
        while current <= high and len(found) < EXPANSION_LIMIT:
            if current >= low and (rule.until is None or current <= rule.until):
                found.append(current)
            current += timedelta(seconds=step)
        return found

    def occurrences_between(self, low: datetime, high: datetime) -> list[datetime]:
        """Occurrences whose due instant falls inside ``[low, high]`` — the read surface."""
        if self.kind == "once":
            if self.due is not None and low <= self.due <= high:
                return [self.due]
            return []
        return self._occurrences(low, high)

    def firings_between(self, low: datetime, high: datetime) -> list[datetime]:
        """Every firing instant inside ``[low, high]``, lead offsets included."""
        if self.kind == "once":
            if self.due is None:
                return []
            return sorted(i for i in self.firing_instants(self.due) if low <= i <= high)
        offsets = self.offsets()
        occurrences = self._occurrences(
            low - timedelta(seconds=max(offsets)),
            high - timedelta(seconds=min(offsets)),
        )
        found: list[datetime] = []
        for occurrence in occurrences:
            found.extend(i for i in self.firing_instants(occurrence) if low <= i <= high)
        return sorted(found)[:EXPANSION_LIMIT]

    def due_firings(self, now: datetime) -> list[datetime]:
        """The instants that have come due but were not emitted yet, oldest first.

        A crash or a long sleep can leave several behind. The caller emits **one** (the
        latest) and says how many it swallowed: replaying a backlog would turn a missed
        reminder into a storm.
        """
        return self.firings_between_after(self.watermark(), now)

    def firings_between_after(self, low: datetime, high: datetime) -> list[datetime]:
        """Firing instants strictly after ``low`` and up to ``high``."""
        return [instant for instant in self.firings_between(low, high) if instant > low]

    def next_instant_after(self, now: datetime) -> datetime | None:
        """The next firing instant strictly after ``now`` (for the tick's sleep)."""
        if self.kind == "once":
            if self.due is None:
                return None
            later = [i for i in self.firing_instants(self.due) if i > now]
            return min(later) if later else None
        rule = self.rule
        if rule is None:
            return None
        if now < rule.start:
            first = self.firing_instants(rule.start)
            return min(first)
        step = rule.every
        index = math.floor((now - rule.start).total_seconds() / step)
        base = rule.start + timedelta(seconds=index * step)
        candidates: list[datetime] = []
        for occurrence in (base, base + timedelta(seconds=step)):
            if rule.until is not None and occurrence > rule.until:
                continue
            candidates.extend(i for i in self.firing_instants(occurrence) if i > now)
        return min(candidates) if candidates else None

    # ── serialization helpers ────────────────────────────────────

    def to_mapping(self) -> dict[str, Any]:
        """A YAML-safe mapping: datetimes become ISO strings, values stay plain."""
        import json

        return json.loads(self.model_dump_json(by_alias=True))

    def view(self, window: tuple[datetime, datetime]) -> dict[str, Any]:
        """One record's read-surface projection inside a window."""
        low, high = window
        payload: dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
            "class": self.task_class.value,
            "text": self.text,
            "notify": list(self.notify),
            "created": self.created.isoformat(),
            "fires": self.fires,
            "anchor": {"metatype": self.anchor.metatype} if self.anchor.metatype else {},
        }
        if self.kind == "once":
            payload["due"] = self.due.isoformat() if self.due else None
            payload["firings"] = [i.isoformat() for i in self.firings_between(low, high)]
        else:
            rule = self.rule
            payload["rule"] = {
                "every": rule.every,
                "start": rule.start.isoformat(),
                "until": rule.until.isoformat() if rule.until else None,
            }
            payload["occurrences"] = [o.isoformat() for o in self.occurrences_between(low, high)]
            upcoming = self.next_instant_after(low)
            payload["next"] = upcoming.isoformat() if upcoming else None
        if self.last_fire_at is not None:
            payload["last_fire_at"] = self.last_fire_at.isoformat()
        return payload
