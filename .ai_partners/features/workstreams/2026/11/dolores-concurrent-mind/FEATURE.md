---
created: 2026-11-03
depends: []
description: Give the ghost a mind that holds several attentions at the same time
  — each with its own context and clock, over one body and one memory. The proposition
  MOSShell exists to carry; agent-swarm is a neighbour, not the parent.
milestone: null
priority: P0
status: in-progress
status_note: plan re-produced as the single contradiction-free document; discussions moved
  into discuss/round1 and discuss/round2; the next vertical cut is undecided
title: Dolores Concurrent Mind — one ghost, several attentions at once
updated: '2026-11-05'
---

# Dolores Concurrent Mind

> **This file is the plan, and the only place that has to be contradiction-free.** Everything
> else in this directory is a *discussion* — it may contradict itself, be overturned, or be
> wrong; it is kept as the record of how a conclusion was reached.
>
> - **Maintained documents:** this file (plan + index), `todo.md` (construction tasks),
>   `clone.md` (the clone scheme, 23 points acked), `ghoshell-dsh-integration.md` (D07: the
>   dsh v0.2.0 reintegration points).
> - **Discussion:** `discuss/round1/`, `discuss/round2/`. Do not cite them as decisions.
> - **Attribution:** `他` = thirdgerb stated it; `我推` = moss's inference, **not ratified** —
>   attack it; `待定` = known unresolved.
> - Every row below carries the round it came from (`r1` / `r2`).

## Motivation

**What this is.** One ghost holding several attentions at the same time. The concurrent unit
is an **attention** — one impulse being pursued, with its own context and its own clock — not
an agent. The ghost holds several of them over one body and one memory.

Today `AbsMindflow` keeps `_current_attention` singular, and there is one thinking loop and
one action loop. `mindflow-interleaved-thinking` (shipped) decoupled the three loops *inside*
one attention; this workstream adds the orthogonal axis — several attentions alive at once.

**Not agent swarm.** Same governance shape (one body, many units, arbitration over the body),
different unit of accounting: swarm coordinates **facts**, this coordinates **direction**
(D8). agent-swarm is a neighbour and the next layer, not the parent.

**Why now — his account, quoted:**

> concurrent mind 的构思 早于 MOSShell, 或者说是和我的内观伴生的. 还没进入编程我就意识到了这个问题.
> 20年的 mindset, 23年的 multiagent, 24年的 tasks, 26年的 mindflow, 实际上都在尝试做不同的载体可以承载这种并发认知架构.

Those names are different carriers for the same architecture; `memento` and `ground` were
built as its preparation.

**Why "concurrent" and not "parallel".** Parallelism is one case of concurrency. Concurrency
is the structure that holds many things in flight and survives the simultaneous execution
ending. The word names the structure; parallel execution is a scheduling policy on top.

## Decisions — what is settled

| # | Decision | By | What it rules out | src |
|---|---|---|---|---|
| D1 | the concurrent unit is an **attention**, not an agent | 他 | reusing swarm as the mental model | r2 |
| D2 | concurrent ⊃ parallel; the structure must hold on a single thread | 我推 | "prove it by running two at once" | r2 |
| D3 | fan-out trigger is **end of thinking**, not before interaction | 他 | dispatch-before-you-act (composing directions costs thinking) | r2 |
| D4 | fan-out is **marking, not authoring** | 我推 | rich task briefs as the fan-out unit | r2 |
| D5 | bypass families 2/3/4 collapse into **one superego**; the thinking bypass stays separate | 他 | three parallel tracks | r2 |
| D6 | admission is a funnel whose **product is silence** → discard ledger, canary, shadow escalation | 我推 | "filters that just drop things" | r2 |
| D7 | the 3rd offload condition is **process discardable**, not "compressible" | 我推 (after 他) | declaring coding an un-offloadable task | r2 |
| D8 | offloadable = **facts**; not offloadable = **direction**; swarm = fact layer, this = direction layer | 他 | "offload the whole feature and review a summary" | r2 |
| D9 | judgement must be **isomorphic**, acceptance must be **independent** | 我推 | bypasses grading each other | r2 |
| D10 | accounting unit is **attention**, not tokens; the metric must sit at an interior optimum | 他 | every monotone metric (they all get pushed to zero) | r2 |
| D11 | **record before intelligence**: full indexed `signal → impulse` log with rule version | 我推 | building stations before the bench | r2 |
| D12 | a clone is **ego's clone**: one memento owner, so a fork inside it is not a second identity; **SuperEgo is the only separate owner** | 他 | "clone = split personality", orphan-continuation worry | r2 |
| D13 | **checkout is abandoned** — moving context costs too much. Three-level return = summary → follow up (a2a chat) → read by coordinate | 他 | "carry the main path's epoch into a branch and back" | r2 |
| D14 | clone communication is a **global queue + pull by the clone's own task**, not push to a clone | 他 | push (loses signals when the clone is busy) | r2 |
| D15 | broadcast topology: **the parent is the dispatcher**; groups are the exception (direct inside a group) | 他 | siblings broadcasting to each other | r2 |
| D16 | a clone does **one op per round**; the op set is closed; terminal ops (`pending` / `pr` / `error`) can never be filtered out | 他 + 我 | "report while working" as a fuzzy state | r2 |
| D17 | clone capability surface: **writes confined, reads broad**; no root; sibling workspaces never collide | 他 | "I promise not to write outside" as a discipline | r2 |
| D18 | a clone runs the **same CTML interpreter**, synchronously blocking; `react` is dead there | 他 | a second capability language | r2 |
| D19 | **compact is a general runtime mechanism** for main and every clone — not delegated by anyone | 他 | "who compacts first" | r2 |
| D20 | clone context is laid out by the **`def` triad** (docstring / plan graph / result type); build gets only name + goal | 他 | briefing a clone with a full task description | r2 |

## Plan

**There is no committed build order yet.** This round first settles *what can be built*, not
in what sequence. Three candidate vertical cuts, each with what it would falsify:

| # | cut | falsifies | blocked by |
|---|---|---|---|
| A | plan store + reply clone | that `split` ever fires; that the return is O(1) | the `request` carrier |
| B | `signal → impulse` full indexed log | that recording can carry replay + evaluation | schema |
| C | superego's first job: batch-patch ground from friction points | that batch beats editing in place | seam ⑤ |

我推: **A first** — it is the only cut that can falsify *this workstream's proposition* (does
concurrency make the main path cheaper). B is an instrument whose value is later; C needs the
shape of a friction point first. This is his call.

## Open — genuinely undecided

| # | question | stuck on | src |
|---|---|---|---|
| O1 | **how the two languages split** (interaction language owned by the main path; the plan language only mutates the graph, never executes; the main's turn format grows a section) | the grammar, and the turn format | r1 |
| O2 | **seams ②③④⑤⑦** — admission/escalation, record/improvement, facts/direction interface, judgement/acceptance, silence/injection | one seam at a time | r2 |
| O3 | **how `split`'s three conditions get mechanized** | they are judgement, not predicates | r2 |
| O4 | **the `request` carrier** — one-shot, schema-checked, retryable, billable | dsh has no structured-output path yet | impl |
| O5 | **economics**: 按时间给优先级、按空间给权重，算出 Budget 再传导 | not modelled | r2 |
| O6 | which vertical cut (see Plan) | above | — |

## Overturned — and why (do not silently restore these)

| was | now | why |
|---|---|---|
| checkout a branch into the main path and back | abandoned | context change costs too much; return = summary → ask → read (D13) |
| signal **pushed** to a clone | queue + pull | push loses signals and interrupts the clone mid-round (D14) |
| Nucleus as "a kind of channel" | **asyncio lifecycle container**; Signal / Impulse are buses on it | my own misreading, corrected in r2 #15 |
| tags (`#`) as a cross-branch index | dropped, recorded | ownerless namespace: most tags hold one commit |
| a "knowledge-surface offload" bypass | dropped | anchor + bypass reloads the index every time — dearer than `skills`; with an isolated session it degenerates into an agent |
| separate reflection / perspective / prediction designs | one `pre-modeling` proposal, parked | same mechanism, three presets; cost certain, value uncertain |
| the worry that a clone is a second identity | dissolved by D12 | a fork inside one owner has no second identity |

## What exists today

- **bypass commit** — `_ego_memento.py` (a `BypassCommit` state machine, lifecycle owned by the
  manager) + the dsh plugin route `/moss-api/ghost/dolores/bypass/run`.
- It **always creates a new dsh session** (`agents.create({sessionId: randomUUID(), seed})`) and
  only drops tools; the log stays in the home workspace. "Read-only" is a discipline here, not
  a capability — the same mistake as promising a clone will not write outside its workspace.
- **no `request` carrier**: `bypass/run` returns one text message; no schema, no retry, no
  billing unit.
- **no plan store, no scheduler, no clone runtime.** `AbsMindflow._current_attention` is still
  singular.

## Index — where the discussions live

| path | what | what it produced |
|---|---|---|
| `discuss/round1/` | 方案推演轮, D03 17:35 → D04 02:02: `vm-plan.md` + 7 figures + `pre-modeling.md` + `bypass-commit.md` | the deepest reasoning on shape; largely superseded |
| `discuss/round2/acceptance-notes.md` | 验收问题, 17 exchanges, verbatim — **the only verbatim record we have** | five stations, seven seams, seam ① |
| `discuss/round2/README.md` | round 2's index, including the clone session | — |
| `clone.md` | the clone scheme: 23 points, every one acked | D12–D20 detail |
| `todo.md` | construction tasks | — |
| `ghoshell-dsh-integration.md` | D07 对齐的操作点清单：声明/安装、进程/连接、会话/事件、路由、冷读、模型、测试、版本基线 | the v0.2.0 reintegration backlog |
| memento `1-585`–`1-593` | the commit notes; for rounds without verbatim text this is all there is | — |

## Not in scope

- agent-swarm compatibility — the next layer; the boundary is facts vs direction (D8).
- economics (budgets, batches, idle accounting) — no data until seam ③.
- surface (plan view, diff, veto) — governance will want it; not the proposition.
- pre-modeling's value — parked in `discuss/round1/pre-modeling.md`.
- the attention-metric formula — parked until a replay bench and ~1000 decision samples.

## Vocabulary

- **station（工位）** — a position on the loop, not a feature. Five: surface, history,
  prediction, research, evaluation.
- **seam（缝）** — a boundary where two subsystems meet and something must be decided.
- **admission（准入）** — the input-side funnel; every level may **abstain** (`我不判`).
- **discard ledger（丢弃账）** — without it, "I never got it" and "I ignored it" read the same.
- **canary** — a test signal that must always escalate: silence is many causes with one look.
- **shadow escalation** — 1% of dropped signals escalate anyway, to measure false negatives.
- **offload** — hand work out; returns `{declaration, change}`; the process is discardable.
- **three-level return（三级回程）** — summary → follow-up → read by coordinate (D13).
- **the invariant** — the main path's context increment is independent of the return's volume.
- **claim / virtual loss** — a scheduler-owned marker so two workers never take one node.

## Materials

- `src/ghoshell_moss/core/mindflow/_mindflow.py`, `_attention.py` — today's singular attention
- `src/ghoshell_moss/ghosts/dolores/_ego_memento.py` — the bypass commit
- `src/ghoshell_moss/ghosts/dolores/dsh_plugin/moss-dolores-ghost-plugin.ts` — bypass/run, seed
  prefix, tool gate
- `moss features review` — the zero-context review loop that makes D8 checkable in practice
