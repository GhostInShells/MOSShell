---
title: Ideas Mechanism
status: draft
# priority: importance within the current stage (iteration cycle) — not development urgency
priority: P2
created: 2026-10-12
updated: 2026-10-12
depends: []
milestone:
description: >-
  A lightweight capture layer for surplus creativity — an off-context pressure-release
  mechanism, and a future observable (timeline / bubble) surface.
---

# Ideas Mechanism

> Use `moss features set-status ideas-mechanism <status> -m "note"` to update state.
> See [TOPOLOGY.md](TOPOLOGY.md) for directory layout and [README.md](README.md) for the full convention.

## Motivation

The project regularly produces more ideas than the current stage can absorb. Today the
only home for an idea is a new workstream — but creating a `FEATURE.md` is a heavy,
structured act, and most ideas do not deserve to become workstreams. Ideas that have no
cheap home keep re-entering working context, crowding out the work actually in progress.

The mechanism this workstream builds is a **capture layer**: a zero-friction place to put
an idea down so it can leave the current context. Writing it down is the whole action —
the cognitive pressure is released because the idea is now durably held somewhere,
not because it was executed. Structuring, prioritizing, and promoting happen later, in a
separate rhythm, decided by a calmer reading of the same record.

This is deliberately *lower* than a workstream. It exists so that most ideas can live and
die without ever becoming one. A workstream is the promotion path for the few that
survive; it is not the entry point.

`moss features` already models workstreams (draft → in-progress → completed), stages, and
surface. What is missing is the tier *below* draft: raw, unscheduled, unprioritized
capture — and a way to view it over time.

## Why now

The project has reached a stage where capability generation outpaces consumption. As the
Ghost becomes a live, continuously-running entity, the rate at which ideas surface will
keep rising. Building the tier below `draft` now keeps that rise from turning into queue
growth at the `draft` tier — which is where it currently lands, and where it hurts.

## Design Index

- Key design documents: `design/`
- Key discussion records: `discuss/`

## Key Decisions

<!-- Record each meaningful design choice. This is what the next model incarnation reads first. -->

### D1 — It is a tier below workstream, not a feature of workstreams

The ideas mechanism is **independent and lighter** than the features system. The
relationship is one-way: **triage promotes an idea into a features `draft`; ideas never
depend on or attach to the features structure.** Attaching ideas as a sub-capability of
`features` would let the workstream structure (status, priority, stage) leak downward
into the capture layer and reintroduce the very friction the mechanism exists to remove.
The failure is expensive to reverse once dozens of entries exist under the wrong shape.

### D2 — Capture is zero-structure; structure is applied at triage

At capture time: date + free text, nothing else — no tags, no priority, no template. Any
structure at capture time turns recording into a task, and a task has friction, and
friction makes capture get bypassed. Tags, grouping, and status are applied later, at a
fixed triage rhythm, by a calmer reading.

### D3 — Admission delay is part of the design

An idea captured in a high-energy state is triaged only after a minimum delay (on the
order of a week). This mirrors the project's existing discipline of waiting until there
is enough of a record to judge, rather than acting on the impulse of the moment. The
delay is not suppression — the idea is durably held the moment it is written; it simply
waits for a calmer reading before it is promoted. Three outcomes at triage: promote to a
features `draft`, fold into an existing workstream, or explicitly retire. Retired ideas
are kept, not deleted — the record over time is itself the artifact.

### D4 — Storage is an append-only event log, projected from the source text

Each idea has an editable text as its primary artifact; every edit appends a new
revision rather than mutating in place, so the full history is time-traceable. The index
is a per-period (monthly) `jsonl`; the source documents live in a per-period directory.
The `jsonl` is the **projection** of the source text — the source is authoritative, the
index is derived. Do not let the index schema define the content.

### D5 — No rename

An idea's identity is stable; it is not renamed. This keeps the history and the
timeline addressable.

### D6 — The observable surface is part of v1's intent, not a later add-on

The intended observable form is a **timeline of bubbles** — a view where ideas can be
seen surfacing one by one over time. This is a first-class part of the mechanism's
purpose (an idea that has left working context is still *visible* to a third party),
not decoration. It is, however, an independently valuable and independently scoped
piece: it should be built on the webview-app node pattern (artifacts / calendar /
decision_tree), it depends on the capture layer having real data first, and the
"surfacing" motion is where the time sink lives — the static layout is cheap, the
*feel* is open-ended and has no completion criterion. Guard it accordingly.

## Exploration paths

<!-- Dead ends hit, pivots made, lessons learned. -->

- **Deferred: node-ification / live-stream visualization as part of capture.** The
  capture layer must exist before a visualization of it has anything to show; building
  the view first would produce an empty container. Ordering follows the project's
  existing dependency-unlock discipline: capture first, view later.

## Methods

<!-- Non-obvious implementation patterns. -->

- **Directory-discovery convention**, consistent with the project's `GROUND.md` /
  workstreams pattern: the mechanism is discovered by convention inside an expected
  directory, so a personal instance (e.g. a `thirdgerb`-scoped ideas store under a
  Ghost's ground) and a project-level instance can share the same implementation without
  a schema change. Whether this is a personal capability or a project capability is
  decided by usage, not up front.

## Implementation Notes

<!-- Gotchas, non-obvious behaviors, reasons for rejecting simpler alternatives. -->

- A plain text file per period (e.g. `.ai_partners/ideas/2026-10.md`) is a sufficient
  v0. It already implies monthly slicing and lets usage decide the eventual shape.

## Open Questions

- Personal-scoped instance vs project-scoped instance: resolve by a week of real usage
  rather than by design.
- Whether the bubble timeline ships inside this workstream or as its own follow-up —
  decide at triage, once real entries exist.
