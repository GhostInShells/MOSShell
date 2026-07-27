---
status: planning
period: 2026-07 ~ 2026-08
delivery: v0.1.0
---

# Stage: v0.1.0 — Applications, Dogfooding, Dolores

## Background

Beta1 closed with every core concept in concrete form: CTML, channels,
matrix cell governance, mindflow, ghost runtime, desktop, ground, memento.
The infrastructure phase is over.

Phase 2 (May 7 – Jul 20) proved the L0→L1 shift: models now handle task
execution and feature-level planning, freeing human bandwidth to L2
(structure design + collaboration governance). Commit data: 5x monthly
velocity, 70% "dev:" → 78% "feat/docs/refactor/fix", 30% of commits
co-authored by 10 AI models. Markdown output grew 4.7x — the increase was
design discussion, not documentation.

What the system has not yet proven is itself in sustained application use.
The May–June dogfooding round tested concepts; this stage tests the whole.

## Motivation

A framework whose only user is its own development is not yet a project.
This stage turns MOSS from validated infrastructure into a working system
with real applications and a full-featured resident Ghost. It is
deliberately an application sprint — the goal is to discover what the
infrastructure is actually like to live in, and let that discovery drive
v0.1.0's final shape.

This intent is worth stating explicitly because it is not inferable from
code: a model surveying the repo mid-stage should read this as a sprint
context, not as a codebase asking for defensive reinforcement.

MOSS is not a monolith. It is an n-ghost × m-mode architecture where each
capability can demonstrate independently — not "done when everything is
wired," but "each integration shows immediately." This changes the
parallel development calculus.

## Goals

- [ ] P0 chain closes: node-migration → momento-mori → ghost-ground →
      desktop-gui. The structural spine for Dolores.
- [ ] Dolores assembles from cognitive infrastructure (memento + ground)
      and perceptual infrastructure (desktop-gui + screen-node) into a
      minimal complete Ghost in daily use.
- [ ] G1 development restarts — the physical body thread resumes.
- [ ] Voice input pipeline completes — audio-first streaming perception
      with CTML control reaches demonstrable form.
- [ ] Release v0.1.0 — the project formalizes, leaving beta.

## Plan

The dependency chain, bottom-up:

**Layer 0 — Beta1 cleanup (prerequisite).** Dependency governance,
CLI entry point renaming, root directory cleanup. Designs settled in
beta1 closure notes, execution pending.

**Layer 1 — Structural foundation.** node-migration (.moss_ws/apps →
nodes/). 22 old apps into the six-category taxonomy. Blocks all
node-based work below.

**Layer 2 — Cognitive foundation.** momento-mori (trajectory infrastructure)
→ ghost-ground (cognitive field) → moss-project-ground (self-dogfood).
Memento CLI agent validates the boundary. This is Dolores's memory spine.

**Layer 3 — Perceptual foundation.** desktop-gui (shared perception space),
screen-node (compositor body), text-blocks (shared text carrier),
mindflow-channel (reflexive perception control).

**Layer 4 — Dolores assembly.** Wire ghost-prototype-dolores from Layers
2+3. Ghost runtime safemode as safety gate. This is the integration
target, not a standalone feature.

**Layer 5 — Release.** Close or explicitly defer remaining in-progress
features. Tag v0.1.0.

G1 and voice input run as parallel threads outside this chain, governed
by their own features.

This is a trajectory, not a schedule. The layers describe dependency order,
not a Gantt chart. Some lines will defer to v0.2.0 — see Design Backfill
Protocol below.

## Rhythm

Not Phase 2's burst mode. Sustained, paced parallel development. Human at
L2 governance, models at L1 execution. Some tasks enter dogfooding loop
(model-driven iteration with minimal human intervention). Progress is
measured by independently demonstrable capabilities, not commit count.

## High-Risk Areas

- **Voice input system.** realtime-voice-interaction-logos +
  voice-input-state-machine. Audio pipeline, state machine, CTML
  integration. Design locked, implementation risk high — audio stacks
  have platform-specific failure modes that design documents can't
  anticipate.
- **G1 development.** unitree-g1-integration marked completed Jul 6 but
  needs restart as a new feature. Physical hardware iteration has
  fundamentally different cadence from software — cannot be parallelized
  or accelerated by adding models.
- **Dolores shape.** What "minimal complete Ghost" means is a product
  judgment call held by the human engineer, not defined here. The
  engineering coordinates are Layers 2–4; the product shape is a
  separate line of work outside this staging document.

## Design Backfill Protocol

When a model executing a design-locked feature hits a gap or contradiction:

1. Record the gap in FEATURE.md status_note. Flag the specific decision
   that doesn't hold.
2. Continue on other lines — do not block the parallel set on one gap.
3. Human resolves at next L2 governance window. If resolution would take
   more than one session, defer the feature to v0.2.0.

This is intentionally lossy — some features WILL be deferred — to protect
overall sprint trajectory from collapsing into serial design discussion.

## Acceptance

- P0 chain closed: memento + ground + desktop-gui functional
- Dolores runs as a Ghost with persistent identity (memento),
  working memory (ground), and a perceivable face (desktop/screen)
- G1 development has an active feature and demonstrable progress
- Voice input pipeline demonstrable
- v0.1.0 tagged; project status changes from beta to formal
- Remaining in-progress features are either closed or explicitly deferred
  with a v0.2.0 milestone in their FEATURE.md

Product shape ("what it looks like") is not an acceptance criterion here.
This document carries engineering delivery coordinates.

## Explicitly Deferred to v0.2.0

Design-locked or draft, intentionally not scoped for v0.1.0. FEATURE.md
files carry complete design rationale; implementation waits for the
foundation to stabilize.

- cognitive-anchor
- channel-meta-dyn-static
- logos-expansion
- ghost-tui-refinement
- feishu-channel-integration
- reflex-layout-design

## Associated Workstreams

Layer 1 — Structural:
- feature: node-migration

Layer 2 — Cognitive:
- feature: momento-mori
- feature: memento-cli-and-agent
- feature: ghost-ground
- feature: moss-project-ground

Layer 3 — Perceptual:
- feature: desktop-gui
- feature: screen-node
- feature: text-blocks
- feature: mindflow-channel
- feature: matrix-resources

Layer 4 — Assembly:
- feature: ghost-prototype-dolores
- feature: ghost-runtime-safemode

Parallel threads:
- feature: realtime-voice-interaction-logos
- feature: voice-input-state-machine
- (g1-restart — feature to be created)

Infrastructure carry-over from beta1:
- feature: stage-tracking-convention
- feature: interactive-shell-channel
- feature: moshi

## Milestones

Operational log in milestones/:

(planned — to be created as reached)

## Retrospective

(not yet closed)
