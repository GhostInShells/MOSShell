---
created: 2026-11-03
depends: []
description: Give the ghost a mind that holds several attentions at the same time
  — each with its own context and clock, over one body and one memory. The proposition
  MOSShell exists to carry; agent-swarm is a neighbour, not the parent.
milestone: null
priority: P0
status: in-progress
status_note: >-
  plan is contradiction-free; the prompt manifest + commit-prompt bench (cut D) is recorded
  as the superego's evaluation substrate — side-effect-free and buildable first; its trigger
  and scoring are deferred. D12 night closed the bypass debt (read-only sidecar, Ω archive,
  effort/cache measurements) and settled the memory surface + prompt slot direction; the
  build order for tonight is memory surface → ground slot → prompt slots → staged search.
title: Dolores Concurrent Mind — one ghost, several attentions at once
updated: '2026-11-12'
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
| D21 | the bypass is a **read-only sidecar** — defined by the boundary it sits outside, not by which tools it lacks | 他 (after 我推 got it wrong) | "id ≠ ego ⇒ disable everything" — it swallowed dsh's own subagents | D12 |
| D22 | **exactly one live Ψ**: the previous ego is retitled `Ω · start–end` and archived on boot; archived ≠ deleted | 他 | the session list as an append-only pile | D12 |
| D23 | the bypass runs `low`; **`off` is a different prefix family**, so "save thinking" pays the whole uncached prefix | 他 + measurement | `off` as a cheap effort | D12 |
| D24 | memory is a **snapshot**: written once at epoch, holding only the current branch | 他 | a live window over several branches in `<memory>` | D12 |
| D25 | **ground outranks memory**: law belongs in the system prompt; memory carries only the trajectory | 他 | ground duplicated in memory and in notices | D12 |
| D26 | the system prompt is **slot-ized**, synced to the plugin before ego opens, replaced only at session creation | 他 | re-rendering the prompt mid-session (dirty logs, cold cache) | D12 |
| D27 | retrieval is staged — **L0/L1 in program, L2 an `LLMFunc`**; a result page never re-enters the context | 他 | "search returns everything and lets the model re-read" | D12 |

## Plan

**There is no committed build order yet.** This round first settles *what can be built*, not
in what sequence. Four candidate vertical cuts; D is a substrate, not a competing cut:

| # | cut | falsifies | blocked by |
|---|---|---|---|
| A | plan store + reply clone | that `split` ever fires; that the return is O(1) | the `request` carrier |
| B | `signal → impulse` full indexed log | that recording can carry replay + evaluation | schema |
| C | superego's first job: batch-patch ground from friction points | that batch beats editing in place | seam ⑤ |
| D | prompt manifest + commit-prompt bench | that a prompt is evaluable from an anchor; that manifest/files/versions are enough for RSI | superego motivation & scoring (deferred); the registry itself is buildable now |

我推: **A first** — it is the only cut that can falsify *this workstream's proposition* (does
concurrency make the main path cheaper). B is an instrument whose value is later; C needs the
shape of a friction point first. D is not on that axis: it is the superego's evaluation
substrate, side-effect-free on the attention runtime, and can land before A/B/C. The trigger
and the score are its hard half, recorded below. This is his call.

## Prompt manifest & commit-prompt bench (2026-11-12, cut D)

> Recorded before the code wave. This is the superego's evaluation substrate, not a new
> attention mechanism; the registry itself has no side effect on the attention runtime.

### Why

RSI needs an evaluation loop, and an evaluation loop needs prompts that are **addressable,
editable, versioned and reflectable**. Today the mechanism prompts (`memento.note`,
`memento.chat`, the exit record) live as code constants; there is no manifest, no version,
no cheap runtime experiment. The first value point is the commit prompt: given a commit ref,
run a candidate note prompt against the same context and score it.

The motivation cannot come from the ego — the ego is saturated by interaction with the world.
It belongs to the superego (D5/D12), as a trigger-class task. The registry is the substrate;
the trigger and the score are the hard half.

### Decisions (this round)

- **Layout**: `.dolores/prompts/` (prompt versions) + `.dolores/configs/` (manifest + active
  pointers). Prompt versions and the manifest are tracked in git; scratch/drafts are ignored.
  `.gitignore` carries a short directory convention, but the contract is `PromptManager`.
- **Key / path bijection**: manifest key is dotted (`a.b.c`, dsh-compatible); the directory is
  hyphenated (`a-b-c/`); the version is the file stem (`<version>.md`). Segments are
  `[A-Za-z0-9_]` with no hyphens inside, so `.` ↔ `-` is one-to-one.
- **Names**: `PromptSpec` (manifest entry), `PromptVersion` (one file), `PromptsConfig`
  (specs + active pointers), `PromptManager` (service). Errors: `UnknownPromptError`,
  `PromptVersionNotFoundError`, `SlotMismatchError`, `PromptPathError`.
- **Surface**: `specs / get / render / versions / create_version / activate / reset`. Paths are
  computed by the manager; no raw writable path is exposed. Key/version validation and path
  containment live there; the channel is a thin shell.
- **Config**: `materialize=True` with a code-default `PromptsConfig`; stable specs are code
  constants, prompt bodies are stub seeds (never YAML). Runtime `save` + `activate` change the
  active pointer; `create_version` never activates.
- **Ghost in IoC**: `host/ghost_runtime.py` now sets `Ghost`, `GhostMeta` and the concrete
  ghost type before `__aenter__`; the concrete ghost exposes `home()` / `home_meta_dir()`, and
  the store/prompt providers derive their roots from it instead of a hidden environment lookup.
  The store and prompt manager are IoC singletons — the ghost does not hold them.
  `DoloresMeta.contracts()` lists both, and boot instantiates every declared contract after the
  ghost is registered, so a broken manifest fails startup. Writes (config saves, prompt
  versions) go through a same-directory temp file + `os.replace`.
- **Evaluation unit**: `commit ref → (session, turn, seq) → fork/seed → replace the key
  component → one round`. A session log is never spliced — a spliced log is a trace that never
  happened. For the commit prompt the run is one off-record round at the ref's `end_seq`
  (`chat_off_record`), so no real fork is needed; the note task and the bench now share that
  primitive.
- **Superego**: the benchmark's motivation is a superego trigger task (first trigger deferred);
  the score is an independent judge (D9); promotion is an explicit superego action, never
  automatic. The action surface is a gated `ghost.meta.prompts` channel.
- **Harness boundary**: exposing the dsh launcher/connection in IoC is fine; the bench gets a
  narrow `AnchorRunner` (`run_at(anchor, messages) -> reply`), with the dsh implementation
  behind an adapter. The registry, the records and the channel verbs stay harness-neutral.

### Implementation (2026-11-12)

- `MINDHOOK` is a global constant; `DefaultPrompterKeys` (in `mindset/prompter.py`) holds the
  stable prompt keys. `PromptSpec.slots` is `dict[name, description]`; `seed` materializes
  the default version when a key has none.
- The system prompt has a Concurrent Mind slot after the output-protocol notice: parallel
  thinking / functional bypass / CTML and tools may change or be disabled / tasks have their
  own prompt / `<mindhook>` is the generic container. It names no unbuilt mechanism.
- The memento channel instruction embeds the active `memento.note` rules; the manual commit
  and the bypass see the same rules. The bypass message is a `<mindhook>` container with the
  connection point and the prior note only.
- The note task is one `chat_off_record` round at the commit's end (`readonly`, effort off,
  max tokens). Retries live in the task (`note_retry_timeout` / `note_backoff_base` in
  `EgoMementoConfig`); model-unavailable exhaustion leaves the note empty and the named
  notice is the record. Content failures write the terminal error note. `backfill` /
  `resume_tail` are retired — the task is the only writer.
- `memento.chat` uses a `<mindhook hook="memento.chat">` container carrying the
  `memento.chat` preamble and the question.

### Deferred (not this round)

- The superego trigger condition (manual / every N commits / broken note / idle) — pick one
  when the bench has a consumer.
- Multi-turn fork (`AnchorRunner` beyond one round).
- System prompter migration — the `_prompts.py` assembly is not refactored yet.
- Auto-promote, automatic scoring loop, prompt UI, watchdog.

### What it falsifies

- That a prompt can be evaluated from an anchor without a new session paradigm.
- That manifest + files + versions + active pointer is enough for RSI (the first bench will
  tell).
- That the superego can produce the motivation and an independent judge can score it
  (deferred; the honest unknown).

## D12 night — read-only sidecar, the Ω archive, the memory surface (2026-11-12)

Three mechanism waves landed after cut D, plus a design round on what `<memory>` is for. None
of them adds a cut; all of them change a substrate.

### 1. The bypass is a read-only sidecar

The old shape was negative where it mattered: "id ≠ ego ⇒ bypass ⇒ disable every tool, one
turn, collapse the previous turn". It swallowed dsh's own subagents the moment they appeared,
and its stated reason ("not safe to let it act") was never the real one.

> 写权限不是会话属性，是"谁为这次写负责"的属性。 — 他

So the bypass is named by the boundary it sits outside, not by what it cannot do:

- `sandbox/mode: read-only` + `approval/policy: never`, re-appended **every step** (both are
  last-wins log events; a human flipping the session writable in the UI gets pushed back);
- the four-sentence instruction is injected **once per session** (process-local `Set`); a
  second injection only stacks the same paragraph in history;
- the tool surface is a **deny list**, not "all disabled": the eight CTML/lifecycle tools that
  go through `awaitToolResult` would *hang* in a bypass (nobody answers for a non-primary ego),
  which is worse than a refusal. `moss_channel_facade` stays;
- `collapseTurn` / `bypassTurns` are gone (a 0.1.5 workaround for a missing reject semantic);
  `appendReplace` stays — `compact` still uses it.

An archived ego is a legitimate read-only 應身: the archive plus dsh's per-step lock already
make it inert. Commit `5404862b`.

### 2. The Ω archive — exactly one live Ψ

On boot the previous ego is retitled `Ω · 26-11-12 16:27–16:56` (same day compacts to
`HH:MM–HH:MM`, cross-day keeps both ends) and archived; the newest K are retitled, the rest
archived as-is. First real run: 213 sessions closed — 10 retitled, 203 as-is. Commit `1ae421b3`.

Two facts worth keeping:

- **The archive closes an identity, not the content.** An `Ω` session can be resumed as a
  bypass on the same id; his last experiment conversed with one exactly that way (turns 11–12
  appended to the same log, the four-sentence instruction first). He ran it once from his side
  and closed the line: 以后不再主动还原 Ω 会话。
- **A session log is ground truth.** `session.v4.jsonl.zstd` is zstd, not encrypted — `zstd -dc`
  plus `data.message.content[]` (`reasoning` / `text` parts) is the whole tool, and it is how
  the last exchange was recovered.

### 3. What a bypass round actually costs

| observation | number | what it means |
|---|---|---|
| new process, first note | `cache_read=6,912 / 52,482` | only the static head is reused |
| same process, next note | `77,952 / 79,067` (98.6%) | one family, live |
| five-effort probe at one anchor | `inherit`/`max`/`low`/`high` → 64,640; `off` → 46,336 | `off` is the only different family |
| 6m26s idle, then a turn | `64,640 / 121k`, next step `125,952` | prefix TTL ≈ 5 min |
| ego's first request after restart | `11,136 / 40,099` (27.8%), then 99.5% | first-frame cold is not a bypass defect |

- **`off` is not "free thinking", it is another prefix family** — it drops the reasoning block,
  so the prompt is shorter and nothing before it matches. Any "save thinking" call pays the
  whole uncached prefix; `low` is the floor of "fast *and* hitting".
- The two budgets moved out of the prompt into config: `note_max_tokens: 4096` (request
  ceiling) and `note_body_tokens: 1500` (what the model is told). Prompt states format, config
  states scale.
- The main path now logs `kind=ego turn=N step=M` per step — without it every conclusion above
  was inferred from the outside.
- One discipline earned the hard way: **mechanism conclusions need two runs.** Single samples
  misled us three times (`readonly`, `off`, `high`).

### 4. The memory surface is a snapshot

Direction settled this round, not yet implemented:

- `<memory>` holds **exactly one** container: `<memento owner="ego" branch="main">`. Other
  branches exist in the memento channel's notice, never in memory.
- **ground does not go into memory** — its priority is higher. The waking order is
  我是谁 → 最近发生了什么 → 我这几天在干嘛, i.e. existence (system prompt) → memory (memento)
  → status (channel notices).
- Coordinates carry **no owner prefix**; the container states the owner, once.
- Commit stamps become minute precision, local time: `at="D12 16:26"`. Today's `created` is a
  UTC ISO string — eight hours off every other stamp in the same prompt.
- Confluence / child rows exist in the contract and are absent from the renderer; the branch
  opening has no reserved place for the forked parent (`BranchView.previous` is computed and
  then dropped).

### 5. Retrieval is multi-stage, and it is a function

> 如果要做二级召回, 三级召回, 我们应该要定义 LLMFunc 去做. 比如 grep 拿到的是 0级, 返回结果给
> 程序筛一遍内容, 再让 agent 挑 topk 等等. 否则搜索结果在上下文里反复跳两三次, 都是可以程序化
> 的硬开销。 — 他

L0 grep (program, zero tokens) → L1 score / filter / cut (program, zero tokens) → L2 rank
(`LLMFunc`, one round) only when candidates exceed a threshold or the query is semantic. The
model sees one page; a refinement is a new call, never the same batch re-entering.

### 6. Tonight's order (his call)

1. settle this file's debt and commit it (done in this edit);
2. the `memento` surface inside `<memory>` — container, stamps, confluences, parent slot, sizes;
3. ground channel vs. prompt: split them apart and send one request; **ground moves into the
   system prompt**;
4. split the system prompt into slots — and first verify whether dsh's prompt slots can be
   changed by an op later (O7). If they can, the sync design changes; if they cannot, the
   manifest-at-boot design stands.

After these four, concurrent thinking starts for real.

## Open — genuinely undecided

| # | question | stuck on | src |
|---|---|---|---|
| O1 | **how the two languages split** (interaction language owned by the main path; the plan language only mutates the graph, never executes; the main's turn format grows a section) | the grammar, and the turn format | r1 |
| O2 | **seams ②③④⑤⑦** — admission/escalation, record/improvement, facts/direction interface, judgement/acceptance, silence/injection | one seam at a time | r2 |
| O3 | **how `split`'s three conditions get mechanized** | they are judgement, not predicates | r2 |
| O4 | **the `request` carrier** — one-shot, schema-checked, retryable, billable | dsh has no structured-output path yet | impl |
| O5 | **economics**: 按时间给优先级、按空间给权重，算出 Budget 再传导 | not modelled | r2 |
| O6 | which vertical cut (see Plan) | above | — |
| O7 | can a dsh system-prompt **slot be modified by an op after the session starts** — and should we ever? | unverified; needs a read of dsh's `systemPrompt` plugin API before we design the sync | D12 |

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

- **commit note task** — `_ego_memento.py`'s `BypassCommit` state machine owns one one-shot
  `asyncio.Task` per commit; the task runs `chat_off_record` at the commit's `end_seq` with a
  `<mindhook>` container (`readonly`, `low`, max `note_max_tokens`). The old dsh plugin
  `bypass/run` route is no longer used by the note task.
- **the bypass sidecar** (plugin) — non-primary sessions are read-only + never-ask, the
  four-sentence instruction is injected once per process, and the eight `awaitToolResult` tools
  are denied. Every other session kind (dsh subagents, UI forks) is left alone.
- **the Ω archive** — `DoloresEgo.close_previous_sessions()` runs at boot and closes every
  previous ego under the project cwd; the session roster comes from `session/list`, retitle and
  archive are separate `try`s (v3 logs refuse the rename but still archive).
- **usage logging** — `kind=ego turn=N step=M` per step and `kind=note commit=…` per bypass
  round, both with `input / cache_read / output / reasoning`.
- The note's static rules live in the memento channel instruction, read from the active
  `memento.note` prompt; the container carries only the connection point and the prior note.
  `memento.chat` uses the same container with its own preamble.
- **no `request` carrier** for the bypass: `chat_off_record` returns one `ChatReply` (text +
  usage); no schema, no billing unit beyond that.
- **prompt assets are file-based** — `mindset/prompter.py`, the manifest in `.dolores/configs`,
  version files under `.dolores/prompts`, and the seeds in `mindset/seeds.py`. The note/chat
  paths consume the registry.
- **Ghost is in the IoC** (`host/ghost_runtime.py`): `Ghost` / `GhostMeta` / the concrete type
  are set before `__aenter__`; `home()` / `home_meta_dir()` are implemented.
  `DoloresMeta.contracts()` lists the config store + prompt manager, which boot instantiates.
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
