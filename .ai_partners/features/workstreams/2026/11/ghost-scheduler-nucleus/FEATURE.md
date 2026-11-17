---
created: 2026-11-18
depends: []
description: 'A nucleus that turns "time came due" into signals: durable timer records,
  a filesystem store, recurring rules, span queries, and a recall path back to the
  self who wrote the plan.'
milestone: null
priority: P1
status: in-progress
status_note: nucleus, store and recall seam implemented; real-machine run pending
  a restart
title: Ghost Scheduler Nucleus — 时间规划作为一种感知
updated: '2026-11-18'
---

# Ghost Scheduler Nucleus

> Naming: `scheduler`, not `timer`. A timer is one kind of task; the workstream also
> carries recurrence, span queries and recall, and `timer` would have to be re-explained
> later.

## Motivation

The ghost plans across hours and days, but nothing holds the plan. `mindflow.idle-knock`
covers exactly one case — "remind me after N seconds of quiet" — and is cancelled the
moment a real signal arrives. Anything longer lives in the model's context until a
compaction eats it.

Scheduling is also the one capability where **forgetting is the point**: the ghost writes
a line to its future self and is then free to drop it. That only works if the line is
durable, carries its own trigger, and — when it fires — points back at the self that
wrote it, so the future can ask "why did I plan this?" instead of guessing.

## The model

### A record is a line to a future self

```yaml
version: 1
id: sch-000017
kind: once                # once | every
text: |
  19:30 看发布
class: notify             # aside | knock | notify | interrupt
notify: ["-10m", "at"]    # lead offsets, relative to the due instant
due: 2026-11-18T19:30:00+08:00
rule: {every: 1d, start: ..., until: null}    # kind=every
created: ...
anchor:                   # where the self that wrote this plan still lives
  metatype: dsh-session
  metadata: '{"session": "...", "turn": 812}'
fires: 0
last_fire_at: null      # watermark: the last instant emitted
```

### Firing = a signal, never an impulse

The scheduler does not arbitrate. It picks one of the four signal tiers the framework
already defines, and the matching nucleus applies the established policy:

| `class` | signal | semantics |
|---|---|---|
| `aside` | `aside` | never interrupts; buffered until the ghost looks |
| `knock` | `knock` | lost if busy — the caller keeps it |
| `notify` (default) | `notify` | never lost: on a lost challenge the messages land in the next frame's percepts |
| `interrupt` | `interrupt` | FATAL; drop everything, start nothing |

This is why the scheduler is a Nucleus at all: `with_bus` gives it `signal_broadcast`,
which is `mindflow.add_signal`, which routes into those nuclei. `peek`/`attended`/
`suppress` stay empty — the scheduler never competes directly.

### Storage: the filesystem is the truth

```
.moss/runtime/scheduler/ghost-{name}/
  .gitignore          # *, !.gitignore, !README.md
  README.md
  state.yml           # {version, next_id}
  tasks/sch-000017.yml
  fired/2026/11/      # one-shots, kept 30 days
```

- **One file per task**, filename = id. Atomic write (`tmp` → `fsync` → `os.replace`);
  `*.tmp` swept at boot. A corrupt file costs one task, never the set.
- **Flat directory** — no time buckets: a recurring task's due instant changes, and a
  bucket would force a move per fire, breaking id → path.
- **Ids are monotonic and never reused**; `next_id` is healed at load by
  `max(state.next_id, max(existing) + 1)`.
- **Crash semantics differ by kind, on purpose**: `once` is at-least-once (emit, then
  persist the watermark and archive — a crash in between re-fires, and a duplicate
  reminder is cheap); `every` is at-most-once (persist the watermark *before*
  emitting — a skipped tick is cheaper than a doubled workflow). No `state` field:
  the watermark alone decides what already happened.
- **Late firings are delivered once**, not replayed: if several instants came due while
  the process was down, the latest is emitted and the text says how many were missed.

### Read surface: one JSON, self-explaining

`list(start, end)` returns the occurrences inside the window, each with the rule that
produced it:

```json
{"now": "...", "window": ["...", "..."],
 "tasks": [
   {"id": "sch-000017", "kind": "once", "class": "notify", "text": "19:30 看发布",
    "firings": ["2026-11-18T19:20:00+08:00", "2026-11-18T19:30:00+08:00"],
    "anchor": {"metatype": "dsh-session"}},
   {"id": "sch-000018", "kind": "every", "class": "knock", "text": "每天 9 点站会",
    "rule": {"every": "1d", "start": "...", "until": null},
    "occurrences": ["2026-11-18T09:00:00+08:00"], "next": "..."}
 ]}
```

A time span is a **query window**, not a scheduling primitive: no free/busy, no conflict
detection in v1. The "what else is that day" question is answered in the arming receipt,
where it can still change the decision.

### Recall: a sub-service, not a dependency

```python
class SchedulerRecaller(ABC):
    def metatype(self) -> str: ...
    def describe(self, anchor: AnchorRef) -> str: ...
    async def chat(self, anchor: AnchorRef, prompt: str) -> str: ...
```

- Unbound in IoC ⇒ the `chat` command **disappears from the channel surface** and the
  impulse text drops the pointer line. The scheduler itself never imports memento or dsh.
- `anchor.metatype` shares the vocabulary of `CommitRef.metatype`; the payload is opaque
  to the scheduler and belongs to the recaller's prototype.
- Dolores binds `DshSchedulerRecaller`, which needs only `DshConnection` (attach with
  `follow=False`, anchor on the turn) — not memento.

## Key Decisions

1. **Signal, not impulse.** Priority numbers are the wrong layer; `aside / knock /
   notify / interrupt` already encode "how hard is this allowed to push".
2. **`NOTICE` is the default attention tier** (thirdgerb's correction): an impulse at
   NOTICE does not preempt a NOTICE attention, so "urgent" cannot be expressed by
   nudging the number. `notify` is reliable because of its loss path, not its priority.
3. **Default unmounted notice**: the due impulse is the prompt; a periodic "today has
   plans" line has no decision to change. The arming receipt carries the day's context
   instead. `as_channel()` still appears as a `mindflow` virtual child, so the model can
   open it when it needs to.
4. **Turn anchoring via a live ego state**, not via memento: an `EgoState` object is
   published in IoC and updated at session create / turn end. `DoloresEgoContext` stays
   static — its contract says "read once at construction".
5. **Workspace-level runtime dir** (thirdgerb): `.moss/runtime/scheduler/ghost-{name}/`,
   defended by a stub `.gitignore` (`*`, `!.gitignore`, `!README.md`).
6. **Openbox registration**: the nucleus is globally available via
   `ghoshell_moss/matrix/openbox/nuclei.py`; per-project manifests inherit it and may
   override.

## Open questions

- **Expressive power of `every`**: duration + start/until today. Weekday rules would
  need an `rrule`-like field; deliberately deferred.
- **Single writer**: two live processes on the same ghost home could double-fire. A lock
  file modeled on the memento owner gate is the obvious next step.
- **`notify.next`**: today a reminder never forces a re-look (`next=False`); if
  reminders prove too easy to defer, this becomes a per-task flag.
- **Node rendering**: the record tree is meant to be read by a webview node; nothing in
  v1 blocks it, but the JSON shape should be frozen before a node is built on it.

## Exploration paths

- Considered `blocking=False` for the channel — wrong: it frees the queue but the task
  stays scope-bound (see issue #141).
- Considered firing impulses directly with a chosen `mode`; rejected because it would
  duplicate the arbitration policy that four nuclei already implement.

## Methods

- Time is injectable (`clock: Callable[[], datetime]`) so tests never sleep on the wall.
- Occurrence expansion is bounded (limit per call) and computed from the rule + a
  watermark (`last_fire_at`), so the record needs no "next due" field that could rot.

## Implementation Notes

- The tick loop waits for `mindflow.is_running()` before emitting: a due task at boot
  would otherwise be dropped by `add_signal`'s not-running guard.
- `clear()` never deletes durable tasks — it drops the in-memory cache, which is
  reloaded on the next pump. The files are the truth.
- `pump(now=...)` is the test seam and the tick's own heartbeat: the clock is data, so
  tests advance it rather than sleeping through it.
- A boot-time heal archives a one-shot whose watermark already passed its last instant
  — a crash between "watermark written" and "archived" must not leave a zombie plan.
- The recaller is looked up through IoC at nucleus construction (mindflow wiring runs
  after the ghost enters, so `DshConnection` and `EgoState` are already published).

## Status (2026-11-18)

Implemented: `core/mindflow/scheduler/` (models, store, recaller, nucleus), openbox
registration, the workspace stub (`runtime/scheduler/` with its `.gitignore`),
`EgoState` + `DshSchedulerRecaller` in dolores, and two test files
(`test_scheduler.py`, `test_scheduler_recall.py`).

Not yet: a real-machine run (needs a restart — the running process has no `scheduler`
nucleus), a node view, and the single-writer lock.