---
status: active
period: 2026-05 ~ 2026-07
delivery: beta1
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
- [ ] Ship beta1 (target: 2026-07-26)

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

(not yet closed)
