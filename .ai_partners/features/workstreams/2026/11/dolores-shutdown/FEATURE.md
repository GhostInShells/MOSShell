---
title: Dolores Shutdown — the graceful exit as a lifecycle
status: in-progress
priority: P1
created: 2026-11-20
updated: 2026-11-20
depends: [dolores-concurrent-mind, ghost-cognitive-continuity]
milestone:
description: >-
  A ghost-initiated shutdown that finishes the turn, holds a cancellable grace window, runs
  the dreaming phase in parallel, lets modules prepare or veto, then quiesces and closes the
  matrix — one lifecycle, no serial ego work, no orphan dsh.
status_note: >-
  D20: design settled in the shutdown/dreamers discussion; implementation starts after the
  emanations runtime and the journal writer migration. This workstream owns the shutdown
  phases; dreamers are a separate dolores-dreamers workstream.
---

# Dolores Shutdown — the graceful exit as a lifecycle

> The shutdown sequence is the ghost's last act, not a host kill. It has one job: finish the
> turn, let the domain work happen in parallel, let every module prepare or veto, and only
> then close the matrix. The ego is an observer in the dreaming phase, never the writer.

## Motivation

Today a shutdown is either the host's process exit or a hand-written
`exit: graceful shutdown` anchor. Both are serial ego work: the ego writes the exit note,
the ego decides what to close, and a hung close can leave the dsh process behind. This
workstream makes shutdown a lifecycle:

- the ghost calls `ghost.shutdown`; the command sets a request and returns;
- the turn ends; the ego checks the request internally (no new lifecycle `TurnEndEvent`);
- a grace window gives the human/ghost a chance to regret;
- `on_sleep` runs the domain work as parallel dreamers; the ego is static and observes;
- `on_shutdown` runs the modules' prepare/veto;
- quiesce (stop mindflow, drain, module `__aexit__`) and close (`matrix.close`) follow;
- a forced exit leaves a durable process record; the next boot reaps the dsh.

The journal is the first proof: its prose should not be the ego's serial work. The same
shape carries the final commit note and the superego governance.

## Design Index

- Key design documents: none yet; the D20 shutdown/dreamers discussion is in memento
  `1-797..1-798`.
- Key discussion records: none copied; the memento anchors are the record.

## Key Decisions

| # | Decision | By | What it rules out |
|---|---|---|---|
| D1 | The command is `ghost.shutdown`; `__main__.sleep` stays the blocking primitive | he + me | renaming the primitive, or overloading it |
| D2 | The sequence is request → grace → `on_sleep` → `on_shutdown` → quiesce → close | he + me | a host kill as the only exit |
| D3 | `on_sleep` is the dreaming phase: domain work as parallel dreamers; the ego is static and observes | he | the ego writing the journal/note/governance serially |
| D4 | `on_shutdown` is the event prepare phase: modules flush and may veto; exceptions/timeouts are logged, not vetoes; no veto after commit | me (after him) | a broken handler wedging the exit, or a late veto leaving a half-shutdown |
| D5 | Module `__aexit__` and registry stop are system quiesce, not events; `matrix.close` is the irreversible close | me | treating lifetime teardown as a vetoable callback |
| D6 | The ego perceives the dreamers through a `<dreamers>` block (ghost channel notice + moment context) and an instruction rule "observe, don't duplicate" | he | the ego re-doing the dreamers' work, "why isn't it me" anxiety |
| D7 | Dreamers are a separate `dolores-dreamers` workstream: file-defined writable emanations with their own `dreamers/<name>` branches; this workstream owns only the phase | me (after him) | stuffing the emanation definition strategy into shutdown |
| D8 | A forced exit leaves a durable external-process record (pid/pgid/start time/argv); the next boot reaps it | he | orphan dsh processes after SIGKILL |
| D9 | Budgets: 30s start delay, 120s default prepare budget (hard cap 3600s), 30s per-handler; the 1h is a ceiling, not a default | me | a default one-hour shutdown |

## Exploration paths

- **"Make the whole shutdown irreversible."** Rejected: the OS analogy holds only after the
  commit point. Prepare must stay vetoable; the irreversible part starts at quiesce.
- **"Run the dream in the same run."** Rejected: the grace window needs the mindflow idle so a
  cancel can arrive; a new high-priority thinking is cleaner than blocking the run.
- **"Two phases are enough."** Rejected: prepare (event, vetoable), quiesce (system, stop
  mindflow/drain/`__aexit__`), close (matrix) are three different contracts.
- **"The old exit hook is enough."** It was a file-based process hook, removed with that
  mechanism; the new surface is the ego lifecycle + the `on_commit` event.

## Methods

- The turn/end decision is **internal runtime logic**, not a lifecycle event: `DoloresRun`'s
  settle path checks the shutdown state; the ego does not publish a `TurnEndEvent`.
- The dreamer signal is a high-priority self-wake (`Priority.WARNING`/`ERROR`); the dream
  thinking's attention is raised so normal signals lose. `FATAL` remains the emergency stop.
- Veto is a control path: `ShutdownVote(veto=True, reason=...)`, not an exception. The signal
  to the ghost carries the handler and reason.
- `matrix.close` is synchronous; call it in a thread with a timeout; the forced exit path is
  `os._exit(0)` after logging, and the process record is the safety net.

## Implementation Notes

- The existing `EgoLifecycle`/`EgoLifecycleRegistry` is the substrate; add `on_sleep` /
  `on_shutdown` phases and the `ShutdownEvent` / `ShutdownVote` result types.
- The module registry's `stop` runs the modules' `__aexit__` in reverse; it is the quiesce
  phase, not a veto path.
- The journal module is the first consumer of the lifecycle; the journal writer migration is
  in `ghost-cognitive-continuity` + `dolores-dreamers`.
- The dsh process record: write before spawn, remove after reap; use start time + argv to
  avoid PID reuse; use the process group.

## Open

1. Grace cancel: explicit `ghost.shutdown(cancel=True)` only, or any human input?
2. Prepare budget defaults: 120s + 3600s hard cap, per-handler 30s — confirm.
3. Veto signal priority/payload: `ERROR` + handler/reason — confirm.
4. `matrix.close` / host exit coordination and the forced-exit threshold.
5. Dreamer definition/branch strategy — owned by `dolores-dreamers`.
6. Superego dreamer scope: read-only trajectory or ground mutation.
