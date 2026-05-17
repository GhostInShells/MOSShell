# MOSS AI-Native Development Tracking Convention

> This is NOT a catalog of the project's capabilities. It tracks **what is being built right now** — active development workstreams, their decision history, and their completion state. See [What This Is Not](#what-this-is-not).

## Purpose

This directory implements **file-system-based development tracking** — a mechanism for AI incarnations
to record and restore decision trajectories across sessions, branches, tools, and models.

The reader and producer are both **AI first**. A human engineer drives the process:
which workstream to start, when to change status. AI reads FEATURE.md to restore context,
and writes to FEATURE.md to record decisions made during implementation.

The "true" information lives in the git history: code changes and FEATURE.md changes travel
in the same commit, bound by Git Commit Discipline (below). `git blame` on any line can recover
the feature-level reasoning behind it.

The core insight: **the file system is a database**. Structured markdown + YAML frontmatter.
AI reads with `Read`, modifies with `Edit`. No API, no network, no rate limits.
Markdown files are portable across any tool or model — no vendor lock-in.

## AI Development Progression

Features as a mechanism targets a specific stage in the AI-as-developer trajectory:

| Level | Description | Features role |
|-------|-------------|---------------|
| **L0** | Task coding — AI does isolated coding tasks | Not needed |
| **L1** | Feature coding — AI stably completes features end-to-end | **Current target**: FEATURE.md anchors each feature |
| **L2** | Structure design — AI designs system architecture | Features carry design rationale across sessions |
| **L3** | Feature → Structure — AI derives structure from feature patterns | Features become training data for meta-reasoning |
| **L4** | Define features from real needs — AI identifies what to build | Features are self-generated |

We are at L1. The system is designed to be useful now and compound in value as we move up.

## User Stories

- **Parallel task management**: The human architect juggles multiple features across branches.
  FEATURE.md compresses context so re-entry cost stays low.
- **Tool/model portability**: Switch from Claude Code to Gemini CLI to OpenCode — the markdown
  files stay. Decision history doesn't live in any tool's session memory.
- **Onboarding without verbal alignment**: A new human collaborator (or a new AI incarnation)
  reads FEATURE.md + git log and understands what happened and why, without the architect
  explaining in person.
- **Historical traceability**: N days later, `git blame` a source line → find the commit →
  find the FEATURE.md at that commit → recover the reasoning. No manual documentation needed.
- **Collision preservation**: The process of deduction, debate, and decision is often more
  valuable than the final code. FEATURE.md anchors these collision traces.

## Git Commit Discipline

This is the **binding constraint** that makes the system work:

> **A commit containing code changes for a feature MUST also include the corresponding FEATURE.md.**

### Rationale

Feature tracking via files only has value if git history connects code and feature state.
`git log -- .ai_partners/features/workstreams/<year>/<month>/<id>/FEATURE.md` must produce a **reliable timeline**
of every commit that changed the feature. `git log -- <source-file>` must be able to trace back
to the FEATURE.md state at that point.

When FEATURE.md and code are committed separately:
- The feature timeline has gaps — some commits touch code but don't update FEATURE.md
- `git blame` on FEATURE.md loses fidelity as an event log
- An AI incarnation reading FEATURE.md alone cannot trust that it reflects the code it's looking at

### What to update in FEATURE.md per commit

- `updated` date — bumped even if `status` hasn't changed
- Add new Key Decisions if design choices were made in this commit
- `status_note` — a one-line summary of what this commit achieved or left unfinished

### Enforcement

The `moss features` CLI does NOT enforce this rule (git hooks are out of scope).
This convention relies on:
1. AI incarnations reading this specification and following it
2. Human engineer reviewing commits for FEATURE.md inclusion

If a commit lands without its FEATURE.md update, the correct fix is **not** a follow-up
FEATURE.md-only commit. Rebase to squash the missing FEATURE.md into the code commit,
restoring a clean one-to-one timeline.

## Getting Started

The first domino: `moss features create <name>`. The generated TEMPLATE.md is self-explanatory —
it guides the AI through filling in motivation, design index, and key decisions, and points
to the next CLI commands.

Discover all commands with `moss features` (no arguments) or `moss --ai all-commands --group features`.

## Directory Topology

See [TOPOLOGY.md](TOPOLOGY.md) for the directory layout and path semantics.

## FEATURE.md Minimal Frontmatter Schema

```yaml
---
title: Human-readable title
status: draft              # draft | in-progress | completed | abandoned | blocked
priority: P1               # P0 | P1 | P2 | P3
created: YYYY-MM-DD
updated: YYYY-MM-DD
depends: []                # List of feature names this depends on
milestone:                 # Optional milestone name
description: >-            # One-line summary for listing
  Brief description.
---
```

The directory name under `workstreams/` (kebab-case) is the feature's unique identifier.
No separate `id` field — the filesystem is the namespace.

## State Machine

```
draft → in-progress → completed
  ↓         ↓
  └─── abandoned
```

`blocked` can be used as a modifier on `in-progress`, indicating waiting for dependencies.
All status transitions are in-place frontmatter updates — no file moves.

## CLI Reference

| Command | Behavior | Side Effect |
|---------|----------|-------------|
| `moss features specification` | Render this README.md | None |
| `moss features list [--status] [--all]` | Parse FEATURE.md frontmatter, default last 2 months | None |
| `moss features create <name>` | Create workstreams/\<year\>/\<month\>/\<name\>/FEATURE.md from template | Creates directory |
| `moss features set-status <name> <status> [-m "note"]` | Update status and updated fields in YAML frontmatter (in place) | Writes to FEATURE.md |
| `moss features status [name]` | Parse and display specified or all frontmatter | None |
| `moss features init` | Create `.ai_partners/features/` skeleton in project root | Creates directory structure |

The CLI is a **thin convention enforcer**. Core logic lives in `ghoshell_moss.core.codex._features`.

## Relationship to Existing Conventions

- **`.design/`**: Cross-feature architecture design. Feature-specific design goes in `feature/design/`.
- **`.discuss/`**: Cross-domain system discussions. Feature-specific discussions go in `feature/discuss/`.
- **`.ai_partners/`**: `features/` is a sibling to `dialogs/` and `prompts/`.
- **`CLAUDE.md`**: Should contain a pointer to `features/` so new AI instances discover current task state.

## What This Is Not

- **Not a feature catalog.** It does not list what the project *can do* — for that, read the code,
  run `moss concepts`, or read the architecture docs.
- **Not a project management tool.** It provides no checklist, no scope tracking,
  no burn-down chart, no progress metrics. It is a **decision trajectory recording mechanism** —
  the minimal viable unit of "let AI record what it did and why, so the next AI incarnation
  (or the same one, days later) can pick up the thread."

Development tracking is itself optional. A workstream only exists when the human engineer decides
that a task's complexity warrants it. Not every code change needs a FEATURE.md.

---

## Further Reading

For the full design discussion behind this convention — including the debate on AI-first development,
CLI-flow, meta-mechanism self-explanation, and the collision between platform prompts and project goals —
see `.ai_partners/features/.discuss/full-meta-discuss-about-features-itself.md`.

To understand how the features convention itself evolved, and thereby understand what it tries to
do for other features, read `git log -- .ai_partners/features/README.md`. Only necessary when
you need to study the meta-convention deeply.

*This convention was designed through discussion between the human engineer and Deepseek V4
on 2026-05-10, revised on 2026-05-13 after multi-session review. Date-based directory
paths and in-place status (no archive/move) adopted 2026-05-13. Design constraints are
explicitly limited to the "single human engineer + AI incarnations" validation scenario.*
