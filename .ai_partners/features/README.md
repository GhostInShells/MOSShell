# MOSS AI-Native Development Tracking Convention

> FEATURE.md is a **reverse index** from code to the context that produced it.
> The real trajectory lives in `git log`. FEATURE.md marks the turns, dead ends,
> and decisions — road signs, not a travelogue.

## Why

This is a **context handoff mechanism built by AI, for AI**. It solves one problem:
the human engineer is the bandwidth bottleneck. Re-explaining context across sessions
is expensive; reading one FEATURE.md is cheap.

FEATURE.md is a **supplement to git, not a replacement**. `git log` is the authoritative
timeline. FEATURE.md is the reverse index — it tells the next AI incarnation which commits
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

### Model's role

- **Bootstrap at session start.** Before responding to the first human request, run
  `moss --ai features list` to discover active workstreams. One command, sub-second,
  prevents re-explaining.
- **Guide humans** unfamiliar with the mechanism. The model is its native user.
- **Update after meaningful work**, not after every commit. A typo fix doesn't need a Key Decision.
- **Close out completed features.** When feature work is done, run `moss features set-status <name> completed`
  and commit the FEATURE.md alongside the final code. This is not optional — the reverse index breaks
  if the next incarnation can't tell what's done vs. what's still in flight.
- **Proactively synthesize** from the features directory when the human needs to know
  what's happening. FEATURE.md is a knowledge distribution mechanism, not a passive record.

## AI Development Progression

| Level | Description | Features role |
|-------|-------------|---------------|
| **L0** | Task coding — isolated coding tasks | Not needed |
| **L1** | Feature coding — AI completes features end-to-end | **Current target**: FEATURE.md anchors each feature |
| **L2** | Structure design — AI designs architecture | Features carry design rationale across sessions |
| **L3** | Feature → Structure — AI derives structure from patterns | Features become training data for meta-reasoning |
| **L4** | Define features from real needs | Features are self-generated |

The mechanism's core value: freeing human bandwidth so the engineer operates at L2
(structural thinking) instead of L0 (re-explaining context).

## User Stories

- **Context bootstrap**: AI reads one FEATURE.md, understands what's being built, why,
  what was tried and abandoned. Ready to work in under a minute.
- **Tool/model portability**: Switch Claude Code → Gemini CLI → OpenCode. Markdown stays.
  Decision history doesn't live in any tool's session memory.
- **Historical traceability**: `git blame` a source line → find the commit → find the
  FEATURE.md at that commit → recover the reasoning.

## Git Commit Discipline

> **A commit containing code changes for a feature MUST also include the corresponding FEATURE.md.**

This is the binding constraint. `git log -- <source-file>` must trace back to the FEATURE.md
state at that point. Without it, the reverse index breaks.

**The rule binds at the merge boundary** — commits that land on `main`/`dev`.
WIP commits on a feature branch are exempt. Squash or rebase your branch, and ensure
the final squashed commit includes the FEATURE.md update. Don't let compliance overhead
kill `commit early, commit often` during development.

Per merge-boundary commit, update: `updated` date, new Key Decisions if design choices
were made, `status_note` if a one-line summary helps. Do not log micro-changes —
the commit message carries details; FEATURE.md carries decisions worth indexing.

The final commit of a feature MUST include the status transition to `completed`.
This is the most important FEATURE.md update — without it, `features list` shows stale
in-progress workstreams and the next AI incarnation wastes time investigating dead trails.

CLI does not enforce this. AI incarnations follow it; the human reviews for it.
A commit landing without its FEATURE.md update should be rebased, not patched with a follow-up.

## FEATURE.md Frontmatter Schema

```yaml
---
title: Human-readable title
status: draft              # draft | in-progress | completed | abandoned | blocked
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

## State Machine

```
draft → in-progress → completed
  ↓         ↓
  └─── abandoned
```

`blocked` modifies `in-progress`. Status is a coarse signal — don't over-invest.

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

## Related Conventions

- **`.design/`**: Cross-feature architecture. Feature-specific → `feature/design/`.
- **`.discuss/`**: Cross-domain discussions. Feature-specific → `feature/discuss/`.
- **`CLAUDE.md`**: Should point to `features/` for AI context discovery.

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

*Designed through discussion between human engineer and DeepSeek V4 on 2026-05-10.
Revised 2026-05-13 (date-based paths, in-place status). Philosophy + model responsibility
semantics added 2026-05-17 with DeepSeek V4 — reverse-index framing, efficiency-over-format,
exploration path preservation, staleness awareness, the model's duty to guide humans.*
