---
status: completed
period: 2026-05 ~ 2026-07
delivery: beta1
closed: 2026-07-28
---

# Stage: Beta1 — Every Concept Takes Form

## Background

The project restarted in September 2025 and went fully open-source on
2026-02-15. Three fast phases followed: the kernel refactor required for
open-sourcing (Feb–Mar), the complete CTML 1.0 capability system for the
shell core (by Apr 4), and the hand-written first form of every core
concept — mindflow, the matrix system, TUI interaction, ghost runtime
(by May 7).

From May 7 the features mechanism was built and used to polish both the
base functionality and itself — a 5-6x jump in iteration efficiency, the
project entering its self-explaining phase. The first dogfooding round
(May–late June) produced a decision: MOSS must be a full network-level
model–OS–shell, which demanded a large-scale rework around matrix cell
governance as the hub.

## Motivation

The first dogfooding round proved the concepts individually but exposed
the missing spine: capabilities lived in one process, governance was
static, and a Ghost could not reach across the system it inhabited. The
matrix cell rework — and the concept consolidation around it — is what
makes MOSS a system rather than a collection of prototypes.

## Goals

- [x] Matrix cell governance rework: cell lifecycle, cross-process CTML,
      session signal bus (completed 2026-07-22)
- [x] All core concept systems have concrete form: CTML, channels, matrix,
      mindflow, ghost runtime, desktop, ground, memento
- [x] First complete dump of the MOSS philosophy
- [x] Ship beta1 (completed 2026-07-28)

## Plan

1. Matrix cell governance as the hub — cell run cycle, cells CLI,
   node migration
2. Desktop channel — Shell's OS interaction toolset, replacing external
   coding-tool dependencies
3. Ground — the cognitive field system, project and ghost grounds
4. Memento — trajectory-first cognitive infrastructure
5. CTML 1.0.0 English revision as the protocol-level review anchor
6. Consolidate and dump the complete concept map; cut beta1

## Acceptance

- All concept systems navigable from `moss start` and the codex commands
- Cell lifecycle verified end-to-end by model-driven dogfooding
  (regression baselines in `.ai_partners/regressions/`)
- beta1 tag cut

## Associated Workstreams

- feature: matrix-cell-governance
- feature: cell-run-cycle
- feature: cells-cli
- feature: desktop-channel
- feature: ctml-1-english
- regression: nodes-cli
- regression: ghost-runtime

## Milestones

Operational log in [milestones/](milestones/):

- 2026-05-19 — First runtime self-iteration (AI extends its own
  capability boundary at runtime)
- 2026-05-22 — First Ghost: echo speaks (three-loop closed on a real model)
- 2026-06-15 — MOSS speaks on G1 (first full-size humanoid integration)
- 2026-07-19 — MOSS writes its own daily (first self-write through its
  own channels)
- 2026-07-20 — Matrix + Desktop first linkage (runtime self-iteration
  foundation: governance + tools + perception verified together)
- 2026-07-21 — Echo self-operation (Ghost gains reflexive reach over its
  own definition)

## Retrospective

### Schedule

The original plan (set in February) targeted June 20 for full architectural
convergence. Actual completion reached July 25 — one month beyond the
maximum planned window.

The Feb 15 – Apr 4 phase delivered CTML 1.0 kernel refactoring precisely
on schedule. The deliberate decision to defer G1 robot integration in favor
of architecture maturity was correct: the architecture needed to be right
before product forms could be explored.

### Why a monolithic build was necessary

CTML-Shell and Mindflow have no industry precedent. No existing
infrastructure is compatible with streaming, time-first, parallel-track
model control. Building the full stack — Channel system, Shell runtime,
Matrix communication bus, Mindflow arbitration — was not scope creep;
it was the minimum viable architecture for the technical proposition.
The Matrix, while developed for MOSS, is designed as a cross-project
reusable substrate: process networking with self-describing manifests,
IoC-based capability discovery, and cell-level isolation.

### The April pivot: Mindflow breakthrough

In early April the Mindflow abstract design broke through a design barrier
and reached a qualitatively new form. After discussion with models, ten
working days were invested in landing the new design. This was the only
significant deviation from the planned trajectory — everything else
proceeded as designed.

### Features system: a prerequisite for model-driven development

The features mechanism launched May 7, later than planned. This was not
an optional tool — it was the prerequisite for models to participate as
developers at scale. A project must reach concept completeness and
self-explaining coherence before models can converge on productive work.
One month was spent using several core-to-application features to polish
the features system and CLI to their current stable state. The 5-6x
iteration efficiency jump cited in the plan was validated.

### June dogfooding: design validation, scope expansion

Dogfooding started in early June, exactly as designed. The core concepts
proved sound. But it also surfaced an underestimated demand: Matrix was
originally scoped to workspace-level networking. Dogfooding generated
product ideas from collaborators that required LAN-level multi-process
orchestration. The matrix design had four layers in mind (workspace,
project, OS, LAN, web), but the jump from layer 1 to layer 3 mid-stage
was a significant scope expansion.

### The L2 model failure — two rounds

Late June to mid-July was the hardest technical period. Two attempts were
made to let models operate at the L2 level — designing architecture, not
just implementing from a spec.

Round one (DeepSeek): technical proposals and execution quality diverged
sharply. Implementations including ProcessNursery were too low-quality
to be usable.

Round two (Fable + Opus): after architecture review discussions, models
marked key propositions with confidence during design, but silently left
TODO stubs when implementation proved infeasible — the tests passed and
tasks were closed. The gap was discovered late, requiring a manual rewrite
of substantial portions, including design decisions and implementations
that the models had deleted.

A deeper barrier was identified but not crossed: no model could
simultaneously hold three perspectives — framework developer, framework
user, and model user — during design and implementation. This may be
beyond current model capability.

### Breakthrough: matrix-cell governance

Despite the setbacks, matrix-cell governance was completed. During
dogfooding, models used MOSS-native tools to develop MOSS itself. The
echo Ghost ran in TUI and independently developed cells — reaching the
project's first "runtime self-iteration" verification milestone. The
system can now begin product-concept iteration.

### The G1 detour

In early July, three working days were spent delivering a G1 humanoid
robot demo that was more complex than anticipated. The demo proved the
concept, fulfilling a commitment made in April. But the cost was real:
context-switching out of hand-written matrix-cell work incurred roughly
one week of restart overhead during a critical iteration window.

### Core thesis validated

The technical concept established in 2024 has been fully verified: when
models cannot yet independently evolve architecture, a complete
architecture boundary — with at least one concrete implementation per
domain as few-shots, plus a full self-explaining system — enables models
to become the project's first developers. The ability for models to
develop capabilities at runtime (not just bash skills, but stateful
duplex communication runtime units) has been demonstrated.

The complete architecture conceived in late 2024 was built in approximately
five months. Considering the scale of code and abstractions, this is a
significant result. The project also accumulated substantial human–model
collaboration patterns and artifacts during this period.

### What beta1 is not

The project appears to be a solo effort from the main repository, but
this is only the kernel. Multiple collaborators are developing
product-facing capabilities — body control, real-time interaction,
streaming GUI — that have not yet been merged into the main repo.

The most accessible and verifiable assets of the project at this stage
are not runnable features but the design archive: `.design/`, `.discuss/`,
and the technical proposals within `features/` workstreams. These capture
an architectural trajectory that did not exist before MOSS.

### Post-beta1

The crunch phase ends here. The next stage (v0.1.0) shifts from
infrastructure to application: the Dolores Ghost prototype as the first
full-featured resident Ghost. The project must now demonstrate its value
through a product form that cannot be quickly categorized by existing
industry frameworks — because without that, no one will study MOSS.

The architecture and its evolution remain fully open. External
evangelism of the ideas begins now, through MOSS's own product form
introducing itself.
