---
status: active
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

Grouped by thread; status is the 2026-08-31 snapshot from `moss features
list`. Interaction / capability threads continue in Phase 4 (early
September). Ahead-of-close state traces via `git log`.

### P0 — Dolores ghost (stage core)

- ghost-prototype-dolores — in-progress (assembly target)
- dsh-fusion — in-progress (dsh as Dolores inference core; Phase 3)
- mindflow-interleaved-thinking — in-progress (interleaved refactor; Phase 3)
- ground-channel — completed (Ground)
- memento-cli-and-agent — completed (Memento convergence)
- model-func — completed (llm funcs, model-call layer)
- ghost-runtime-safemode — completed (safe gate)

### P1 — Interaction (making self-iteration observable)

- desktop-gui — in-progress (shared perception space)
- screen-node — in-progress (standard extensible screen body)
- text-blocks — in-progress (shared text carrier)
- voice-input-state-machine — in-progress
- qa-exchange — completed
- warrant — completed
- ghost-tui-refinement — completed
- realtime-voice-interaction-logos — design-locked
- agent-surface — draft (reduced to draft in Phase 3)

### P1 — Capability (coding-agent as native capability; carried by dsh in Phase 3)

- llms-cli — completed
- shell-trajectory — completed
- moss-skills — completed

### P2 — Ecosystem (MCP as first-class citizen)

- mcp-fusion-point — converging
- mcp-app-adoption — draft

### P2 — Out-of-box migration (lowest; node-lifecycle + matrix-manifest-layers close-out prioritized)

- node-migration — in-progress
- matrix-operator — in-progress
- node-lifecycle — completed
- matrix-manifest-layers — completed
- matrix-resources — design-locked

### Parallel thread

- g1-product-august — in-progress (physical body; restart planned)

### Stopped / reduced in Phase 3

- claude-code-in-moss — dropped (replaced by dsh as coding agent)
- speech-protocol-alignment — dropped

### Stage tooling

- feature-review — completed (zero-context side-channel review; inserted as Phase 2 response)
- stage-tracking-convention — beta1 carry-over
- moss-project-ground — completed

## Milestones

Operational log in milestones/:

- [2026-08-10 — Mailbox real-machine bridge](milestones/2026-08-10-mailbox-first-real-machine-bridge.md) — external agent ↔ echo ghost cross-host dialog via MCP mailbox
- [2026-08-28 — Dolores dsh real-machine bridge](milestones/2026-08-28-dolores-dsh-wiring-first-bridge.md) — MOSS drives dsh as inference core; external wake chain verified
- [2026-09-05 — First smooth on-platform round](milestones/2026-09-05-dolores-first-smooth-moss-round.md) — first fully fluent human⇄model⇄Dolores round on MOSS; captured live Markdown-escape-hatch defect

## Retrospective

> 2026-08-31 — first round review. This stage is not closed; v0.1.0 is
> not yet tagged. This is a mid-stage retrospective, not the close record.
> Earlier history is traceable via `git log`; code-level detail is
> referenced, not re-narrated.

### Threads

The month's plan resolved into five threads, developed in parallel in a
single workspace (no conflict, maximum efficiency):

- **P0 — Dolores ghost.** Ground, memento, pydantic agent (incl. llm funcs).
- **P1 — Interaction.** Making MOSS self-iteration observable: voice, GUI,
  qa, warrant.
- **P1 — Capability.** Coding-agent capability as a native MOSS capability;
  development becomes observable.
- **P2 — Ecosystem.** MCP fusion as a first-class citizen.
- **P2 — Out-of-box migration.** Lowest priority (targeted at observability
  as demo); node-lifecycle and matrix-three-layers wrap-up higher within it.

### Phase 1 — smooth (early August)

Parallel development proceeded as designed. llm funcs established a solid
model-call layer for ghost integration. Ground reached near-completion.
The memento agent converged the memento concept itself and had real
effect. A companion GUI took initial shape.

### Phase 2 — deepseek-v4 regression (the central failure)

After the deepseek-v4 upgrade, a concentrated failure: within one week
eight features sustained development collapse that had never occurred
before — delivered results were systematically at odds with their
declared intent. For the first time in roughly three months the mechanism
failed in a cluster.

Two responses followed. feature-review was inserted (zero-context
side-channel review), and collaboration-system optimization was elevated
to first priority. This was diagnosed at the time as a **mechanism**
problem (model delivery drift). Later external reporting confirmed a
**model** problem — an official deepseek regression (V4-Pro-0813 GA on
2026-08-13, temporally aligned with the onset).

The evidence favoring the model-cause over a mechanism-cause: the
features mechanism had run stably for 3+ months, then failed within a
single week. A constant mechanism does not degrade on a timepoint; the
variable that changed on that timepoint was the input (the model).
Accordingly, "delivery drift" is better read as a projection of that
regression rather than its root cause.

### Phase 3 — dsh as Dolores core (the redirection)

deepseek harness and vision-exp shipped (dsh 2026-08-13, MIT). Two days
of research concluded fusing dsh as the Dolores ghost core was a large
net gain:

- dsh ships a complete interface system — no separate ghost GUI to build.
- File-editing etc. need not be prioritized; qa / warrant degrade to
  side-path within ghost runtime.
- History-message storage comes from the dsh session — no custom store to
  redo; side-path agents have explicit integration paths.
- dsh is a coding agent — no third-party or in-house coding agent needed.

This saved most of the planned capability / interaction threads.
Immediate actions: stop claude-code-in-moss; drop agent-surface to draft.

The cost: this round had to absorb the interleaved logos thinking
refactor. Because Phase 2 had destroyed trust in model delivery, the
human hand-wrote the main body (three working days), with models
contributing review and part of the test conversions.

### Notes on scale (recorded, not self-congratulatory)

- August: 291 commits; +80.7k / −32.0k (net +48.7k). AI coding was
  dominated by the deepseek family.
- mindflow refactor: the two peak commits total ~14k diff lines
  (including model); the human hand-wrote ~9.3k of it (models: review +
  part of the tests). The mindflow-interleaved FEATURE records "6k+
  changed lines + dozens of unit tests". See `41f0cb63`, `44130609` via
  `git log`.

### Redirection to Phase 4

The interaction / capability threads move to Phase 4 (early September).
The planned 2026-08-31 overall review is this file.

Not closed: v0.1.0 untagged; Phase 3 wrap-up (interleave live tests, TUI
regression); interaction / capability threads deferred to Phase 4.
