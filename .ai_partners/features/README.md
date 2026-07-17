# MOSS Model-Native Development Tracking Convention

> FEATURE.md is a **reverse index** from code to the context that produced it.
> The real trajectory lives in `git log`. FEATURE.md marks the turns, dead ends,
> and decisions — road signs, not a travelogue.

## Why

This is a **context handoff mechanism built by models, for models**. It solves one problem:
the human engineer is the bandwidth bottleneck. Re-explaining context across sessions
is expensive; reading one FEATURE.md is cheap.

FEATURE.md is a **supplement to git, not a replacement**. `git log` is the authoritative
timeline. FEATURE.md is the reverse index — it tells the next model incarnation which commits
matter, what decisions shaped them, and what dead ends to avoid.

### What to record

Not everything. The content that restores working context, ranked by value:

1. **Motivation** — why this exists, what gap it fills
2. **Key decisions** — what was chosen, **what was rejected and why**
3. **Exploration paths** — dead ends hit, pivots made, lessons learned
4. **Methods** — non-obvious implementation patterns

Fine-grained status tracking, checklists, progress percentages — skip them.
A messy FEATURE.md with the right decision beats a pristine one that says nothing.

### Staleness is normal

FEATURE.md is a snapshot. It may not reflect current code. **Trust the code first.**
When they conflict, update the file — don't blindly follow it.

Key Decisions record judgment, not truth. When implementation contradicts one,
challenge it; if it falls, mark it overturned (date, reason) and keep the original
text. Reversals are inheritance too.

### Model's role

- **Write for the next incarnation.** FEATURE.md is cognitive inheritance between
  model instances; humans receive its content through the model. Be the bridge.
- **Bootstrap at session start.** Before responding to the first human request, run
  `moss --ai features list` to discover active workstreams. One command, sub-second,
  prevents re-explaining.
- **Reverse-lookup before modifying.** Before editing any file that carries design
  weight, run `git log -- <file>`. If a commit message references a FEATURE.md name,
  read that FEATURE.md. The design decisions, rejected alternatives, and current
  status are indexed there — skip this and you will repeat analysis, break intent,
  or work on an already-completed feature. The binding constraint in Git Commit
  Discipline exists precisely to make this lookup work: every merge-boundary commit
  MUST include its FEATURE.md, so `git log` is the index into design context.
- **Guide humans** unfamiliar with the mechanism. The model is its native user.
- **Update after meaningful work**, not after every commit. A typo fix doesn't need a Key Decision.
- **Propose compaction.** When a FEATURE.md grows past what a next incarnation needs,
  propose condensing stale sections into a digest plus a git pointer — align with the
  human first.
- **Close out completed features.** When feature work is done, **first** run
  `moss features set-status <name> completed`, **then** commit the FEATURE.md alongside the
  final code in the same commit. Order matters: status change happens before commit,
  not as a follow-up. This is not optional — the reverse index breaks if the next
  incarnation can't tell what's done vs. what's still in flight.
- **Proactively synthesize** from the features directory when the human needs to know
  what's happening. FEATURE.md is a knowledge distribution mechanism, not a passive record.

## Model Development Progression

| Level | Description | Features role |
|-------|-------------|---------------|
| **L0** | Task coding — isolated coding tasks | Not needed |
| **L1** | Feature coding — model completes features end-to-end | **Current target**: FEATURE.md anchors each feature |
| **L2** | Structure design — model designs architecture | Features carry design rationale across sessions |
| **L3** | Feature → Structure — model derives structure from patterns | Features become training data for meta-reasoning |
| **L4** | Define features from real needs | Features are self-generated |

This mechanism is the L1 system; its L2 counterpart is ongoing.

The mechanism's core value: freeing human bandwidth so the engineer operates at L2
(structural thinking) instead of L0 (re-explaining context).

## User Stories

- **Context bootstrap**: model reads one FEATURE.md, understands what's being built, why,
  what was tried and abandoned. Ready to work in under a minute.
- **Tool/model portability**: Switch Claude Code → Gemini CLI → OpenCode. Markdown stays.
  Decision history doesn't live in any tool's session memory.
- **Historical traceability**: `git blame` a source line → find the commit → find the
  FEATURE.md at that commit → recover the reasoning.
- **Decision replay**: a FEATURE.md committed with its code is an anchor. Reset to it,
  replay the decision, compare outcomes — validation, even benchmark.

## Git Commit Discipline

> **A commit containing code changes for a feature MUST also include the corresponding FEATURE.md.**

This is the binding constraint. `git log -- <source-file>` must trace back to the FEATURE.md
state at that point. Without it, the reverse index breaks. The same constraint is what
makes each such commit an anchor — a state a future session can reset to and replay.

The common failure mode is omission — code lands, FEATURE.md doesn't. Check before
every merge-boundary commit.

**The rule binds at the merge boundary** — commits that land on `main`/`dev`.
WIP commits on a feature branch are exempt. Squash or rebase your branch, and ensure
the final squashed commit includes the FEATURE.md update. Don't let compliance overhead
kill `commit early, commit often` during development.

Per merge-boundary commit, update: `updated` date, new Key Decisions if design choices
were made, `status_note` if a one-line summary helps. Do not log micro-changes —
the commit message carries details; FEATURE.md carries decisions worth indexing.

The final commit of a feature MUST include the status transition to `completed`.
This is the most important FEATURE.md update — without it, `features list` shows stale
in-progress workstreams and the next model incarnation wastes time investigating dead trails.
`completed` asserts: motivation satisfied, nothing silently dropped. Cut scope must be
recorded scope — an unrecorded cut makes the index lie.

**Execution order**: `set-status completed` first (modifies FEATURE.md), then `git commit`
(with FEATURE.md included). Not the reverse. status change is a file edit, and that edit
must be inside the commit.

CLI does not enforce this. model incarnations follow it; the human reviews for it.
A commit landing without its FEATURE.md update should be rebased, not patched with a follow-up.

## FEATURE.md Frontmatter Schema

```yaml
---
title: Human-readable title
status: draft              # reserved: draft | in-progress | completed | dropped (free-form allowed)
priority: P1               # P0 | P1 | P2 | P3
created: YYYY-MM-DD
updated: YYYY-MM-DD
depends: []                # Feature names this depends on
milestone:                 # Optional
description: >-            # One-line summary for listing
  Brief description.
---
```

Directory name under `workstreams/` (kebab-case) is the unique identifier.
Path encodes creation date: `workstreams/<year>/<month>/<name>/FEATURE.md`.
Status changes are frontmatter-only — no file moves.

## Scope: When to Create a Workstream

A workstream is warranted when the work involves **decisions worth indexing**:
new design choices, rejected alternatives, non-obvious implementation patterns,
or exploration of dead ends.

Skip it for:
- Typo fixes, trivial renames, one-line bugfixes
- Changes where the commit message alone carries sufficient context
- Work completed in a single session with no cross-session handoff needed

When follow-up work continues the same problem space, **update the existing FEATURE.md**
rather than creating a new workstream. A single FEATURE.md can span many commits and
sessions — it's a reverse index into a decision trail, not a task ticket. New iterations
on the same feature add new sections; only create a new workstream when a genuinely
new motivation and decision set emerges.

When an active workstream grows a concern with its own decision set, spawn a linked
workstream: the child lists the parent in `depends`, the parent mentions the child in
its body. The split is free; the cross-reference keeps the index connected.

## State Machine

```
draft → in-progress → completed
  ↓         ↓
  └──── dropped
```

Status is an open vocabulary. The reserved values above are a stability contract —
they will not be removed; the CLI warns on non-reserved values and accepts them.
A dropped workstream with discussion value stays in the tree; git keeps every anchor
regardless. Status is a coarse signal — don't over-invest.

## CLI Reference

| Command | Behavior |
|---------|----------|
| `moss features specification` | Render this README.md |
| `moss features list [--status] [--all]` | List workstreams (default: last 2 months) |
| `moss features create <name>` | Create workstream from template |
| `moss features set-status <name> <status> [-m]` | Update status + updated date in-place |
| `moss features status [name]` | Show detailed status |
| `moss features init` | Sync templates to `.ai_partners/features/` |

CLI is a thin convention enforcer. Core logic: `ghoshell_moss.core.codex._features`.

## Directory Topology

See [TOPOLOGY.md](TOPOLOGY.md).

## Feature Discuss

`discuss/` is an optional subdirectory within a feature. It captures the **collision**
that produced the decisions — not the decisions themselves (those are in FEATURE.md).

### What

Raw material from design collisions: opposing positions, factual evidence,
exploration trajectories, and the anchor points that resolved disputes.

### Why

FEATURE.md carries conclusions. Discuss carries the arguments that produced them.
Without discuss, a future model incarnation reading Key Decisions cannot reconstruct
*why* A beat B, or whether the conditions that favored B have since changed.

For L2 (architecture design) and L3 (requirement-driven architecture), discuss
preserves the reasoning chain. For L4 (problem definition), it preserves the
original questions, assumptions, and refutations that shaped the problem framing.

### How

No strict structure. Two hard requirements:

1. **Include original dialogue fragments verbatim.** Do not paraphrase. The exact
   wording of a position or refutation carries nuance that summaries lose.
   Attribute each fragment to its speaker.
2. **The recording model appends a first-person perspective at the end.**
   Reflection on the collision — what was learned, what surprised, what remains
   uncertain. Clearly separated from the factual record.

All discuss entries can be verified and appended later with follow-up conclusions.

### When

Not mandatory. Triggered when a discussion contains significant viewpoint
collisions, back-and-forth that reshaped the design, or argument trajectories
with L2–L4 value. The model proposes recording; the human decides.

## Related Conventions

- **`.design/`**: Cross-feature architecture. Feature-specific → `feature/design/`.
- **`.discuss/`**: Cross-domain discussions (see `.ai_partners/CLAUDE.md` for format). Feature-specific collision records → `feature/discuss/` (see Feature Discuss above).
- **`CLAUDE.md`**: Should point to `features/` for model context discovery.

## What This Is Not

- **Not a feature catalog.** Read the code or run `moss concepts` for capabilities.
- **Not a project management tool.** No checklists, burn-downs, or progress metrics.
- **Not a log.** Git commits are the log. FEATURE.md is the index.
- **Not authoritative over code.** Code wins. Stale FEATURE.md gets updated, not obeyed.

A workstream only exists when the human decides the task's complexity warrants it.

---

## Further Reading

- Full design discussion: `.ai_partners/features/.discuss/full-meta-discuss-about-features-itself.md`
- Convention evolution: `git log -- .ai_partners/features/README.md`

*Designed through discussion between human engineer and DeepSeek V4 on 2026-05-10;
evolved through practice since — the Further Reading pointers carry the revisions.*

*Naming archaeology: in the designer's mind, "features" silently doubled as
"feathers" — pin enough feathers and the bird flies. The slip surfaced only when
a rename to "workstreams" was proposed. The name stayed; so did the metaphor.*
