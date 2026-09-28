# Features — Model-Native Development System

> features featuring the feathers for the framework to fly

Features is a development-documentation system for human–model collaborative development — a set of
normalized development documents. It takes architectural evolution as its subject, builds a development
trail alongside the project on the filesystem and git, and lets each new developer node — human or
model — share a continuous, inheritable iteration trail when it joins.

## Why

In human–model collaborative development, context is the most expensive resource.

The human holds the project's values, vision, motives, ideas, taste, usage experience, verification
loops, parallel-thinking capacity — and a more continuous iteration memory than today's models. The
model brings rich programming experience and reading capacity far beyond any human's — but it has
anterograde forgetting and is single-tasked. The dominant cost of collaboration is therefore the
*production of context itself*.

Context is not technical documentation. It does not restate the facts of the production; it provides the
paths to *discover and explore* those facts, plus the discussions and decisions that produced them.

The features mechanism treats collaboration context as a first-class artifact, equal to code. Context is
produced *during* the production of the product, in the form of documents of different kinds, and
committed together with the other artifacts (such as code). Git acts as the witness layer, binding
features documents to the artifacts they explain.

The context exchanged between human and model is sedimented into documents written by the model to
specification; these documents guide the next model instance in taking over or reviewing, and help new
human developers come up to speed.

Because context lives in versioned markdown on the filesystem — not in any tool's session memory — it
survives tool and model changes. Switch Claude Code → Gemini CLI → OpenCode and the decision history stays
put.

## How

Features divides human–model development collaboration into five levels:

| Level | Description | Role |
|-------|-------------|------|
| **L0** | Task coding | The model completes a well-specified task independently; context on the order of 10k |
| **L1** | Workstream coding | A single task's full iteration — discussion → decision → planning → development → verification → retrospective; context on the order of 1m |
| **L2** | Structure design | Topologically interdependent architecture; the system's architectural entropy; orchestrating parallel iteration under limited time and resources; context on the order of >10m |
| **L3** | Production design | From vision, define the product form; under real time/space constraints and available technical resources, chart a buildable architecture iteration path; context independent of coding |
| **L4** | Define the project from real needs | From an understanding of the era, the industry, and the users, plus values and beliefs, define the product vision |

Moving L0 → L4, the model shifts from primary producer to collaborator. Features provides a different
context-output convention at each level:

- **L0** — outside the features mechanism. The model follows ordinary engineering sense: self-explanatory
  code, comments, and unit tests — code as the sole truth.
- **L1** — features provides the **workstreams** convention: create, re-enter, and review a task across
  multiple models and context restarts.
- **L2** — features provides the **stages** convention: spanning several workstreams; records discussion,
  debate, and vision change at creation, and closes with regression and retrospective.
- **L3** — features provides the **surface** convention: bounds the cognitive scope a human or model sees
  on first contact; the product's outward promises and face; used for full product-expectation management
  and regression at each product phase.
- **L4** — outside the features mechanism.

The core principle: produce versioned, lifecycled documents to specification, and commit them together
with the facts (code) they explain, so they double as a reverse index into those facts in `git log`. When
a document — say a workstream — is created, its commit records the *start*; when it is finished and its
status changes, its commit records the *end*. `git log` then links every fact-change to the context that
produced it.

A subtlety worth naming: the levels build *upward* — the project literally accumulates infrastructure
level by level — but *governance flows downward*. L3 vision is realized through L2 stages, which decompose
into L1 workstreams. Because an upper level faces forward (it holds intent, not facts), the *governance*
of an upper level is carried by the level below it.

The endpoint of a features document is context, discussion, and decisions — never a restatement of the
facts themselves. The only truth is the artifacts stored in git; the context documents are not truth.

## Directory Topology

See [TOPOLOGY.md](TOPOLOGY.md).

## Workstreams

A workstream is L1 iteration-task context recorded in `FEATURE.md` — the core context material of the
features mechanism. Template: [TEMPLATE.md](TEMPLATE.md). Usually created directly via
`moss features create`.

A FEATURE.md should be written assuming a *different*, zero-context model instance will take it over, and
that it must reconstruct the current state of facts through exploration, not assertion.

### What to record

The content that restores working context, such as:

1. **Motivation** — why this exists, what gap it fills.
2. **Key decisions** — what was chosen, **what was rejected and why**.
3. **Exploration paths** — dead ends hit, pivots made, lessons learned; never substitute a concrete
   description for the explorable facts.
4. **Methods** — non-obvious implementation patterns.

Fine-grained status tracking, checklists, progress percentages — skip them. A messy FEATURE.md with the
right decision beats a pristine one that says nothing.

No strict structure. Two hard requirements:

1. **Include original dialogue fragments verbatim.** Do not paraphrase. The exact wording of a position or
   refutation carries nuance that summaries lose. Attribute each fragment to its speaker.
2. **The recording model appends a first-person perspective at the end.** Reflection on the collision —
   what was learned, what surprised, what remains uncertain. Clearly separated from the factual record.

All discuss entries can be verified and appended later with follow-up conclusions.
Without it, a future model incarnation reading Key Decisions cannot reconstruct *why* A beat B,
or whether the conditions that favored B have since changed.

For L2 (architecture design) and L3 (requirement-driven architecture), discuss preserves the reasoning
chain. For L4 (problem definition), it preserves the original questions, assumptions, and refutations that
shaped the problem framing.

### FEATURE.md Schema

```yaml
---
title: Human-readable title
status: draft              # reserved: draft | in-progress | completed | dropped | parked (free-form allowed)
status_note: >-            # Optional: one line on the current state (why dropped/parked, what's next)
  Context for the current status.
priority: P1               # P0 | P1 | P2 | P3 — importance within the current stage, not urgency
created: YYYY-MM-DD
updated: YYYY-MM-DD
depends: [ ]                # Feature names this depends on
milestone:                 # Optional
description: >-            # One-line summary for listing
  Brief description.
---
```

**`priority`** ranks importance **within the current stage (iteration cycle)** — not development urgency
or timeline order. P0 = committed for this stage; P2 = experimental, may be discardable. Delivery targets
the end of the stage, not "now".

Directory name under `workstreams/` (kebab-case) is the unique identifier. Path encodes creation date:
`workstreams/<year>/<month>/<name>/FEATURE.md`. Status changes are frontmatter-only — no file moves.

### Workstream directory

A workstream is a directory anchored by `FEATURE.md`; it may hold any development-related material. Common
subdirectories:

- `discuss/` — the detail collision that produced the decisions, include the key sentences.
- `research/` — investigated data and conclusions, to avoid re-investigation.
- `skills/` — tools or techniques the development itself needs.
- or more.

Sub-tasks may be created directly as `.md` documents inside the directory, linked to FEATURE.md so they
are discoverable.

### When to Create

A workstream is warranted when the work shapes the project, and involves **decisions worth indexing**: new
design choices, rejected alternatives, non-obvious implementation patterns, or **exploration of dead ends**.

Skip it for:

- Typo fixes, trivial renames, bugfixes.
- Changes where the commit message alone carries sufficient context
- Code already explains itself.

While a workstream is active, follow-up work in the same problem space **updates the existing FEATURE.md**
rather than spawning a new workstream — it is a reverse index into a decision trail, not a task ticket.

A `completed` workstream only reopens when the decision is changed during the same stage. Otherwise,
create a new decision set.

When an active workstream grows a concern with its own decision set,
create a subtask document, or spawn a linked workstream:
the child lists the parent in `depends`, the parent mentions the child in its body.
The cross-reference keeps the index connected, but it is expensive to maintain — know the trade-off.

### State Machine

```
draft → in-progress → completed
  ↓         ↓  ↑ resume
  └──── parked / dropped
```

- **`dropped`** — abandoned. The judgment is closed; the workstream remains only as a trace.
- **`parked`** — a **formed proposal deliberately set aside**: a technical plan kept on file for reference,
  optional to ever build, carrying no attention debt. This is the state for work worth writing down but not
  worth doing now.

`parked` is a **quiet status**. Quiet statuses are dropped from the query unless named explicitly.
Retrieve them deliberately: `moss features list --status parked`.

Status is an open vocabulary. The reserved values above are a stability contract — they will not be
removed; status is a coarse signal — don't over-invest.

### Staleness is normal

FEATURE.md is a snapshot of context. It may not reflect current facts (like code). **Trust the facts first.**
When they conflict, update the FEATURE.md if still progressing, or let it be.

Key Decisions record judgment, not truth. When implementation contradicts one, challenge it; if it falls,
mark it overturned (date, reason) and keep the original text. Reversals are inheritance too.
Recording the way to find the facts is better than recording concrete links — the latter will always go
stale.

**Never** link the workstreams back into your code, which definitely is debt in the future.

If a decision changes significantly, the old content may be deleted directly:
summarize what was deleted in a line, and point to `git log` for the remainder. Avoid unbounded document growth.

### Model's role

- **Write for the next incarnation.** FEATURE.md is cognitive inheritance between model instances; humans
  receive its content through the model. Be the bridge.
- **Reverse-lookup before modifying.** Before editing any file that carries design weight, run
  `git log -- <file>`. If a commit message references a FEATURE.md name, read that FEATURE.md. The design
  decisions and current status are indexed there — skip this and you will repeat
  analysis, break intent, or work on an already-completed feature.
- **Guide humans** unfamiliar with the mechanism. The model is its native user.
- **Update after meaningful work**, not after every commit. A typo fix doesn't need a Key Decision.
- **Close out completed features.** Set the status to `completed` and commit the FEATURE.md with the
  final code — see Git Commit Discipline.
- **Proactively synthesize** from the features directory when the human needs to know what's happening.
  FEATURE.md is a knowledge distribution mechanism, not a passive record.
- **Serve as examples.** Since features carry the decisions on why and how, use them as a reference base for
  similar tasks.

### Git Commit Discipline

> A commit that lands feature code carries that feature's FEATURE.md with it.

`git log -- <source-file>` should resolve to the FEATURE.md state at that point. Without it the
reverse index breaks — and the same property is what makes each such commit an anchor, a state a future
session can reset to and replay.

Update when a decision worth indexing was made — a new Key Decision, a reversal, a status transition.
`updated` and `status_note` follow those, not commits. Do not log micro-changes — the commit message
carries details; FEATURE.md carries decisions worth indexing.

The final commit of a feature carries the status transition to `completed`. This is the update that
matters most — without it, `features list` shows stale in-progress workstreams and the next model
incarnation wastes time investigating dead trails.
`completed` asserts: motivation satisfied, the attention on this workstream can be released.
Cut scope must be recorded scope — an unrecorded cut makes the index lie.

**Execution order**: `set-status completed` first (modifies FEATURE.md), then `git commit` with workstream modifications
included.

### How to Review

Review is a development-process quality check. Its goal is verifying feature implementation quality — that
the delivery holds to what the FEATURE.md declared, and that nothing was silently dropped along the way.

It is paired with a command: `moss features review <feature>`.

Run it when you or others want to inspect a feature's development state. The command returns a bare-text review
prompt; let its output guide the next step.

**Default timing** — when the feature is finalized, when a development phase completes, and before a
merge-boundary commit.

**Not a hard constraint.** As with the rest of this mechanism, review is a recommendation: use it when it
helps, and communicate with the human about how to apply it.

### Workflow

Features imposes no strict execution process; these are the common steps:

- **Conceive** — build a task from discussion: understand its motivation, the problem it solves, its value,
  and its relationship to existing mechanisms.
- **Survey** — research the project's current state through the features system, forming a sense of where
  the task sits in the architecture; record the exploration scope and methods worth keeping.
- **Argue** — justify the task's feasibility technically, its connection strategy to existing assets, and
  the cost of change.
- **Evaluate** — re-assess the feature's value, cost, acceptance strategy, execution strategy, and best
  timing; decide whether it is worth doing.
- **Draft** — create the feature from the previous four steps; assess priority, decide draft vs parked, and
  whether it enters the iteration plan (stage).
- **Start** — before formally starting, seriously assess the change scope, the model's independent execution
  steps, architectural-entropy concerns, key interruption checkpoints (to avoid drift), commit nodes, and the
  acceptance strategy. A task whose plan cannot converge does not start.
- **Takeover check** — for a task expected to exceed the model's context boundary, use the features review
  mechanism to confirm the FEATURE.md is written well enough to hand off to a zero-context model instance.
- **Develop** — develop with common sense; update context at stage commits, keeping the next instance able
  to take over.
- **De-risk** — the biggest failure mode here is treating the plan as the decision: upon discovering a
  decision error or a topological mess, still delivering first, causing architectural-entropy growth or a
  silent todo. At high-risk moments, stop, think, and re-communicate.
- **Accept** — complete acceptance with the agreed strategy, including unit tests, regression, real-machine
  tests, etc. For important tasks, run a zero-context check via the features review mechanism.
- **Retrospect** — record the valuable communication strategies, implementation strategies, and failure
  modes found in the iteration as project assets.
- **Close** — set the task's stage-end status and commit.
- **Reopen** — within the same stage cycle, when the plan changes, maintain the same document rather than
  spawning a new one.

These steps are suggestions; combine them flexibly as the actual work demands.

## Companion Mechanisms

### Stages

A stage manages an iteration plan composed of multiple workstreams. Its full lifecycle runs discussion →
locking → milestones → stage retrospective → regression acceptance → full retrospective. It constrains the
goal, boundary, and cadence of L2 architecture iteration.

See [stages/README.md](stages/README.md).

### Regressions

Regressions manage the verification system for capabilities that require real-machine validation, across
iteration cycles. They are model-facing: a model can complete a regression independently, or in human–model
collaboration with the model recording results.

See [regressions/README.md](regressions/README.md).

### Project Surface

The project's L3 — its product promises — is presented through a bounded scope of documents, code, and CLI
tools. A stage builds its vision around the surface, and closes with acceptance regression against it. The
surface can be handed to a zero-context model entering the project as a first-contact test: it surfaces
product-form friction points and strengthens self-explanation.

The surface is one of the L3 deliverables — or, more precisely, the index over them. It is a checklist of
pointers, not a restatement of truth (see `SURFACE.md`). Because the surface is forward-facing — it
declares promises, not facts — the *discussion* of the surface is project-level and is carried by stages,
not by the surface document itself.

## No-Debt Orientation

Features conventions are not a statute to comply with, and not a report for a human.
They are a tool, not ritualized paperwork.

Their purpose is to help code form a system that explains itself at the meta layer — code as prompt is the
first principle. Features does not define the facts; it clarifies them with reasons.

Three debts this convention refuses:

1. **Authoring debt.** Written to satisfy a rule. A rule the next model obeys
   without understanding is worth less than no rule. Record judgment; an unevaluatable MUST is debt.
2. **Document debt.** The document expires the moment the work is done. This is the record of a *formation*, not an
   iteration log.
3. **Pointer debt.** Never leak feature material — decision numbers, workstream names, "see FEATURE.md" —
   onto an abstraction surface. The surface explains itself; a consumer forced to fetch a frozen file
   carries the debt.

## CLI Reference

The CLI is a thin convention enforcer; `moss features --help` is the
authoritative surface. Implementation lives in `ghoshell_moss.core.codex`.
