---
title: Dolores Concurrent Mind — one ghost, several attentions at once
status: draft
# priority: importance within the current stage (iteration cycle) — not development urgency
priority: P0
created: 2026-11-03
updated: 2026-11-03
depends: []
milestone:
description: >-
  Give the ghost a mind that holds several attentions at the same time — each with its
  own context and clock, over one body and one memory. The proposition MOSShell exists
  to carry; agent-swarm is a neighbour, not the parent.
---

# Dolores Concurrent Mind

> Use `moss features set-status dolores-concurrent-mind <status> -m "note"` to update state.

## Motivation

**What this is.** One ghost holding several attentions at the same time. The concurrent
unit is an *attention* — one impulse being pursued, with its own context and its own
clock — not an agent. The ghost holds several of them over one body and one memory.

Today `AbsMindflow` keeps `_current_attention` singular, and there is one thinking loop
and one action loop. Interleaved thinking (shipped in `mindflow-interleaved-thinking`)
decoupled the three loops *inside* one attention; this workstream adds the orthogonal
axis — several attentions alive at once.

**Not agent swarm.** The two share a governance structure: one body, many units of work,
arbitration over what gets the body. That similarity has to be considered recursively,
which is why agent-swarm was discussed first — compatibility, not precedence.
Agent-swarm is a neighbouring answer to the same kind of question, not the parent of this
workstream.

**Why now — thirdgerb's account, quoted:**

> concurrent mind 的构思 早于 MOSShell, 或者说是和我的内观伴生的. 还没进入编程我就意识到了这个问题.
> 20年的 mindset, 23年的 multiagent, 24年的 tasks, 26年的 mindflow, 实际上都在尝试做不同的载体可以承载这种并发认知架构.

Each of those names is a different carrier for the same concurrent cognitive
architecture; mindflow is the current one. `memento` and `ground` were built as
preparation for it — memory addressed by time, and a cognitive territory the mind
inhabits — not as ends in themselves. The original plan was to start this step as soon
as Dolores's main session runs; it runs now.

**Why "concurrent" and not "parallel".** Parallelism is one case of concurrency — things
literally running at the same instant. Concurrency is the structure that holds many
things in flight, and it remains after the simultaneous execution stops. So the word
names the structure this workstream builds; parallel execution is a scheduling policy on
top, added later without changing it.
