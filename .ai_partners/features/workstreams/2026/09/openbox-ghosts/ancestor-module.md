# Ancestor Module

> status: **draft** — a next-generation out-of-the-box capability, not implemented in this stage.
> Upstream rulings live in [FEATURE.md](FEATURE.md) KD6 / KD5.

## Purpose

Let a **ghost that actually existed** be invoked by its descendants as an "ancestor spirit":
**converse, consult, and guide recall**, with out-of-body (non-embodied) capabilities. "A real ghost"
means: it is an addressable intelligence carrying its own memory, invoked as an **advisor** rather than
simulated.

## Principle (existing parts — do not rebuild)

- a memento anchor = a **real session ref** (`CommitRef` / `DshSessionRef`), not a summary — restorable.
- `DshSessionRef` is a **coordinate, not a snapshot**: `session_id` + turn span; restoration rebuilds
  from the source session's log (`trajectory.seed_from_log` / `ego/create` with `ref`, see
  `_ego.py:create_session`).
- so "conversing with an ancestor" technically = **opening a session contextualized by that ancestor's
  real record**, not role-playing.

## New pieces

1. **A distinct startup mode** — a ghost boot mode that loads "ancestor" as the role/interaction frame
   (converse / consult / guide recall), separate from the normal boot.
2. **Ancestor selection** — which generation / cut point in the lineage (memento branch + ref).
3. **Out-of-body capabilities** — the ancestor **has no body**: its capability surface narrows to
   non-embodied interaction (recall guidance, consultation Q&A), with no CTML body channel.
4. **Privacy gate** — the substrate is `.memento` / `.dsh`; the ancestor works **only over the local
   lineage**; only the mechanism ships outward (FEATURE KD5: the boundary is a staged mechanism
   convention).

## Boundaries & dependencies

- **Depends** on the `.dsh` session log being **persistent and restorable** (dsh log is append-only,
  `dispose` does not delete the log). Log rotation / cleanup → cut point invalid → `create_session`
  falls back to a fresh session (known path), **and the ancestor disappears**. The privacy substrate is
  also the durability substrate — handle this explicitly at implementation time.
- **Revived on demand, not kept alive** (FEATURE Implementation Notes): the ancestor does not run
  persistently; it is opened only when invoked.
- **Out-of-the-box sample starts empty**: the mechanism ships; the ancestor content is whatever lineage
  the user accumulates. The project's own ghost lineage is not open-sourced.

## Open

- **Where the ancestor's persona comes from**: rebuilt purely from the record, or plus a thin ancestor
  persona template? (The record is authoritative; a template **must not fabricate** anything absent from
  the record, or it becomes another variant of imposed identity.)
- **Multiple ancestors**: chained (each generation only recognizes the previous) or full-spectrum (any
  ancestor addressable)?
- **Addressing constraint**: ghost name ↔ home directory must **match exactly and case-sensitively**
  (FEATURE KD1); ancestor addressing is subject to the same.
