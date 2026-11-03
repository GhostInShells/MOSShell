---
title: Mindflow Visualize Timeline — 认知拓扑的时间轴
status: draft
# priority: importance within the current stage (iteration cycle) — not development urgency
priority: P2
created: 2026-11-04
updated: 2026-11-04
depends: [dolores-concurrent-mind]
milestone:
description: >-
  One ref space, many cognitive graphs (plan / attention / priority / emotion / entity /
  knowledge), and a timeline that plays them back — so a human can read the mind's process
  on the topology and collaborate on it.
---

# Mindflow Visualize Timeline

Status: **draft.** A seed kept alive, not a plan; not on the critical path.

## Motivation

The concurrent mind (see `dolores-concurrent-mind`) already produces one graph — the plan.
The same address space can carry more. `commit ref → session ref → turn ref` are stable
coordinates; any cognitive dimension that can be sampled or event-sourced over them becomes a
layer: plan, attention, priority, emotion, entity, knowledge.

If the layers share the coordinate space, a timeline can fold them: at any ref, the
projection of every layer is "what the mind looked like then"; playing the refs back is a
playback of the thinking process. A human reading the topology does not need the transcript
to see where the mind went, where it stalled, and what it dropped — which is what
collaboration on a thought actually needs.

## The model

- **One coordinate space**: `branch → commit → session(turn)`. The ref is the join key and
  never moves.
- **N layers**: each a graph whose nodes/edges carry refs.
  - event-sourced (replayable): plan, entity, knowledge;
  - sampled at turn/commit boundaries: attention, priority, emotion;
  - human annotations are a layer of their own — the same ops, attributed.
- **Timeline = fold over refs**: the composite at a ref is the projection of every layer;
  playback steps the refs.
- **Human surface**: point at a node or a ref ("this branch", "the attention at this ref");
  the annotation is an op in the same log, so the human and the model edit one graph.

## Constraints

1. **Layers are projections, not truths.** Each layer folds from a log (event-sourced or
   sampled); otherwise N layers become N truths.
2. **Coordinates must be stable.** Commit coords are frozen; turn coords are stable within a
   session; cross-session joins use the session id.

## Open questions

- Which layers first? A guess: plan + attention, then entity / knowledge; emotion and
  priority need a sampling policy before they mean anything.
- Density: commits are sparse; the timeline needs turns (or steps) to be watchable.
- Cognitive load: show 2–3 layers by default, the rest on demand.

## Why draft

It is kept because the idea is a real capability of the same architecture, and a future
"cool node" could implement it quickly once the plan graph exists. Dropping it would lose
the model, not just the UI.
