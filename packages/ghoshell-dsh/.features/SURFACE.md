---
description: >-
  ghoshell-dsh's L3 deliverables checklist — the package's outward face. Pointers only; no duplicated content.
version: 1
updated: 2026-11-06
---

# SURFACE

`ghoshell-dsh` is the compatibility package for dsh (DeepSeek Harness). Upstream dsh
iterates destructively — down to the patch digit — so this package carries **one distribution
with one directory per dsh generation** (`vX_Y_Z`). Each generation holds the protocol, the
plugin and the stubs that speak to that one dsh version; generations coexist, so a ghost can
be launched against more than one of them and still read its own history.

It serves MOSShell without coupling to it. What it owes its consumer is the capability and
the boundary, never an import: process, protocol, data. The two maintenance targets follow
from that: **keep up with dsh's breaking versions**, each breakage contained inside its own
generation directory, and **ship the companion plugin patch** that adds capability to a
running dsh without touching dsh's own source.

## Checklist

- **Positioning and dependency rule** → `GROUND.md` — what the package is for, and the
  one-way boundary (mechanism layers only, never `ghoshell-moss`).
- **Iteration governance** → `.features/` (this tree) and `AGENTS.md` — the package owns its
  own workstreams, stages, regressions and surface instead of borrowing the repository's.
- **Supported generations** → `src/ghoshell_dsh/` — one self-contained compatibility unit per
  dsh version, currently `v0_2_0/` only. Nothing leaks across generations by import, so an
  upstream break is contained rather than an outage.
- **Public entry points (v0.2.0)** → `src/ghoshell_dsh/v0_2_0/__init__.py` — `DshLauncher`
  and `DshConnection` (process and protocol), `DSH_VERSION` (the dsh version this generation
  is aligned to, a declaration rather than a runtime gate).
- **What we know about dsh** → `.vendors/dsh-research/` — conclusions in, source ignored,
  one tree per `v<ver>+<sha>`; what a conclusion owes a reader is stated in its `README.md`.
- **Proof the protocol holds** → `tests/ghoshell_dsh/` — launcher, session and trajectory
  tests. The channel, message-mapper and runtime tests left with their modules: adapting dsh
  into MOSS is MOSS's work, and it stays there.
- **Surface objects for a consumer** → `src/ghoshell_dsh/v0_2_0/surfaces.py` — the typed
  session/connection views a consumer drives, with no opinion about who the consumer is.
- **Distribution metadata** → `pyproject.toml` — `ghoshell-dsh 0.1.0.dev0`, a workspace
  member whose only dependency is `ghoshell-processes`. Nothing in `src/` imports
  `ghoshell-moss`; the boundary is now a fact rather than an intention. Built and verified
  locally, not published until MOSShell verifies the integration.
- **Plugin patch (v0.2.0)** → (promise) `plugin/` inside the generation: one JS bundle that
  adds capability to a running dsh without patching it, exposing — **anchor call**: a
  zero-side-effect LLM call on a session anchor, assembled from the live session, no turn
  persisted, usage returned to the caller; and **session-ref read**: a session read by
  coordinate, messages derived from events, nothing written. Both are evidenced today only
  as probes in the MOSShell playground, with no home in this tree.

## Regression on the Surface

At each stage close: every pointer resolves, every declared capability exists. What fails is
fixed, or moved out of the promise — in the same commit as the stage close.
