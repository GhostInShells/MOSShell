---
created: 2026-11-03
depends: []
description: Give the ghost a mind that holds several attentions at the same time
  — each with its own context and clock, over one body and one memory. The proposition
  MOSShell exists to carry; agent-swarm is a neighbour, not the parent.
milestone: null
priority: P0
status: in-progress
status_note: clone implementation plan complete; dependency graph has no undetermined
  nodes
title: Dolores Concurrent Mind — one ghost, several attentions at once
updated: '2026-11-04'
---

# Dolores Concurrent Mind

> Use `moss features set-status dolores-concurrent-mind <status> -m "note"` to update state.

**This file is the takeover entry.** If you are picking this up with no memory of the
conversation that produced it: read `Where it stands` and `Next action` first, then
`Key Decisions` before proposing anything. `discuss/r1-notes.md` is the *evidence* —
open it only when you need to know how a decision was reached.

**Attribution rule.** Every row below is tagged: `他` = thirdgerb stated it (his words are
in r1); `我推` = moss's inference, **not ratified** — feel free to attack it; `待定` = we
both know it is unresolved. A takeover reader must not treat `我推` as settled.

## Where it stands (2026-11-04)

- **Breadth: done.** One loop, five stations, one admission funnel, three offload
  conditions, offload vs checkout separated.
- **Depth: 1 of 7 seams.** Only seam ① (exit / return) has mechanism conclusions.
- **Method abstraction: not done.** There is one seam's worth of conclusions, not a method.
- **Nothing is built.** No code, no plan change, no todo change.
- Do not read this as a plan. It is a discussion that has found its shape.

## Next action

Take **seam ⑥ (epoch / transfer)**. Open questions: checkout produces *two successors of
one epoch*; the one-trajectory rule requires that only one be absorbed — so what is the
return obligation (merge or abandon), and what happens to an orphaned continuation?

Produce: the contract, including who owns the successor's identity, and what the orphan's
commits mean to the parent. Bring it back to thirdgerb before writing anything else down.

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
inhabits — not as ends in themselves.

**Why "concurrent" and not "parallel".** Parallelism is one case of concurrency — things
literally running at the same instant. Concurrency is the structure that holds many
things in flight, and it remains after the simultaneous execution stops. So the word
names the structure this workstream builds; parallel execution is a scheduling policy on
top, added later without changing it.

## Key Decisions

| # | Decision | By | What it rules out |
|---|---|---|---|
| D1 | the concurrent unit is an **attention**, not an agent | 他 | reusing swarm as the mental model |
| D2 | concurrent ⊃ parallel; the structure must hold on a single thread | 我推 | "prove it by running two at once" |
| D3 | fan-out trigger is **end of thinking**, not before interaction | 他 | dispatch-before-you-act (composing directions costs thinking) |
| D4 | fan-out is **marking, not authoring** — the questions are already formed | 我推 | rich task briefs as the fan-out unit |
| D5 | bypass families 2/3/4 collapse into **one superego**; the thinking bypass stays separate | 他 | three parallel tracks |
| D6 | admission is a funnel whose **product is silence** → needs a discard ledger, a canary, shadow escalation | 我推 | "filters that just drop things" |
| D7 | the 3rd offload condition is **process discardable**, not "compressible" | 我推 (after 他) | declaring coding an un-offloadable task |
| D8 | offloadable = **facts**; not offloadable = **direction**. swarm = fact layer, concurrent mind = direction layer | 他 | "offload the whole feature and review a summary" |
| D9 | judgement must be **isomorphic**, acceptance must be **independent** | 我推 | bypasses grading each other |
| D10 | the accounting unit is **attention**, not tokens; the metric must sit at an interior optimum | 他 | "fewer interruptions is better" |
| D11 | **record before intelligence**: full indexed `signal → impulse` log with rule version, shadows, and replay | 我推 | building stations before the bench |

## Open Questions — the seven seams

| seam | state |
|---|---|
| ① exit / return | **dug** — three mechanism conclusions (r1 §4) |
| ② admission / escalation | open |
| ③ record / improvement | open |
| ④ facts / direction | open |
| ⑤ judgement / acceptance | open |
| ⑥ epoch / transfer | **next** |
| ⑦ silence / injection | open |

Two more, above the seams: (a) does the station table need a sixth station —
**long-horizon projection** (his trees run 12–18 months)? (b) method abstraction is not
done.

## Not In Scope

- agent-swarm compatibility — the next layer; the boundary is facts vs direction.
- economics (budgets, batches, idle accounting) — no data until seam ③.
- surface (plan view, diff, veto) — governance will want it, but it is not the proposition.
- pre-modeling's value — parked in `pre-modeling.md` (cost certain, value uncertain).
- the attention-metric formula — parked until a replay bench plus ~1000 decision samples;
  before that, every coefficient in it would be invented.

## Vocabulary — load-bearing, do not re-invent

- **station（工位）** — a position on the loop, not a feature. Five: surface, history,
  prediction, research, evaluation.
- **seam（缝）** — a boundary where two subsystems meet and something must be decided.
  Depth work is done seam by seam; the seven are listed above.
- **admission（准入）** — the input-side funnel; six levels, cheapest first; every level
  may **abstain** (`我不判`), or errors become silent.
- **discard ledger（丢弃账）** — dropped signals are recorded; without it "I never got it"
  and "I ignored it" are the same sentence.
- **canary** — a test signal that must always escalate. Silence is many causes with one
  appearance; observation cannot separate them, so the only diagnosis is **injection**.
- **shadow escalation（影子升格）** — 1% of dropped signals escalate anyway, purely to
  measure the false-negative rate.
- **offload** — hand a task out; returns `{declaration, change}`; the process is discardable.
- **checkout** — move the main path's own epoch into a branch and back; the process *is*
  the product.
- **three-level return（三级回程）** — summary (constant cost) → follow-up (one turn) →
  checkout (full). Always enter at the cheapest.
- **the invariant** — the main path's context increment is independent of the return's
  volume. If this breaks, concurrency's economics go negative.

## Design Index

- `discuss/r1-notes.md` — round-1 evidence: 17 exchanges with quotes and collisions,
  seven seams, seam ① conclusions, and **the list of 13 mechanisms discussed but not yet
  derived** (`## 待推演的机制清单`) — read that list first.
- `design/vm-plan.md` — the VM plan, sections 0–13. **Its §11 is construction correctness,
  not acceptance**; rename it to *build gate* next time the file is touched.
- `design/artifacts/` — the seven plan diagrams.
- `bypass-commit.md` — the first functional mechanism; problem established, design not settled.
- `pre-modeling.md` — parked proposal (reflection / perspective / prediction as one lens).
- `todo.md` — construction tasks only; action items, no propositions.

## Materials

- `src/ghoshell_moss/core/mindflow/_mindflow.py`, `_attention.py` — today's singular attention
- `src/ghoshell_moss/ghosts/dolores/nucleus.py`, `_runtime.py`
- `src/ghoshell_moss/ghosts/dolores/_ego_memento.py` — the bypass note (verbatim seed, no session log)
- `moss features review` — the zero-context review loop that makes D8 checkable in practice

## Exploration paths

- The loop: **history → prediction → research → evaluation → surface → (back to history)**.
  Its product is not knowledge; it is a cheaper next interaction.
- Any station's improvement should be scored on the replay bench, never in production.
- The five stations run at different frequencies — so it is a pipeline with a scheduler,
  not a ring; the scheduler is the one concurrent mind already needs.