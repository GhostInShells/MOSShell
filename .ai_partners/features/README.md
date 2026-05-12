# MOSS AI-Native Feature Tracking Convention

## Purpose

This directory implements **file-system-based feature tracking** as a replacement for GitHub Issues,
designed exclusively for AI incarnations collaborating on the MOSShell project.

The core insight: **the file system is a database**. Structured markdown + YAML frontmatter,
queried via Glob/Grep. AI reads with `Read`, modifies with `Edit`. No API, no network, no rate limits.
The file is an in-situ presence in context, not an external resource pulled via API.

## Scope

This convention is designed for and validated within:

> **A single human engineer + Claude's multiple incarnations (different branches/sessions),
> sharing development task state through `.ai_partners/features/`.**

Explicitly NOT in scope:
- Human collaborator UX (use GitHub Issues for that)
- External open-source community contributions
- Notifications / CI / webhook integration
- Distributed locking and concurrent write conflicts (branch = isolation, one agent per branch)

## Directory Topology

```
.ai_partners/features/
  README.md              # This file — the convention specification
  TEMPLATE.md            # Template for new features (source of `moss features create`)
  active/
    <feature-name>/      # kebab-case naming
      FEATURE.md         # REQUIRED: frontmatter + motivation + design index + key decisions
      discuss/           # Feature-specific discussion trails (optional)
      design/            # Design documents (optional)
  archived/
    <year>/<month>/<name>/  # Completed/abandoned features — the tree IS the index
```

## FEATURE.md Minimal Frontmatter Schema

```yaml
---
id: kebab-case-id          # Unique identifier
title: Human-readable title
status: draft              # draft | in-progress | completed | abandoned | blocked
priority: P1               # P0 | P1 | P2 | P3
created: YYYY-MM-DD
updated: YYYY-MM-DD
depends: []                # List of feature IDs this depends on
milestone:                 # Optional milestone name
description: >-            # One-line summary for listing
  Brief description.
---
```

## State Machine

```
draft → in-progress → completed → archived
  ↓         ↓
  └─── abandoned → archived
```

`blocked` can be used as a modifier on `in-progress`, indicating waiting for dependencies.

## CLI Reference

| Command | Behavior | Side Effect |
|---------|----------|-------------|
| `moss features specification` | Render this README.md | None |
| `moss features list [--status] [--archived]` | Parse all active (or archived) FEATURE.md frontmatter | None |
| `moss features create <name>` | Copy TEMPLATE.md → active/\<name\>/FEATURE.md | Creates directory |
| `moss features set-status <id> <status>` | Update status and updated fields in YAML frontmatter | Writes to FEATURE.md |
| `moss features status [id]` | Parse and display specified or all frontmatter | None |
| `moss features archive <id>` | Move directory to archived/\<year\>/\<month\>/ | Moves directory |
| `moss features init` | Create `.ai_partners/features/` skeleton in project root | Creates directory structure |

The CLI is a **thin convention enforcer**. Core logic lives in `ghoshell_moss.core.codex._features`.
The CLI adds default directory conventions (`.ai_partners/features/` for the MOSShell project itself).

## Archive Convention

1. Read FEATURE.md frontmatter to confirm status is `completed` or `abandoned`
2. Recursively move the entire feature directory to `archived/<year>/<month>/<name>/`
3. Year/month extracted from frontmatter `updated` field
4. Query archived features via `moss features list --archived` — the directory tree IS the index

## Relationship to Existing Conventions

- **`.design/`**: Cross-feature architecture design. Feature-specific design goes in `feature/design/`.
- **`.discuss/`**: Cross-domain system discussions. Feature-specific discussions go in `feature/discuss/`.
- **`.ai_partners/`**: `features/` is a sibling to `dialogs/` and `prompts/`.
- **`CLAUDE.md`**: Should contain a pointer to `features/` so new AI instances discover current task state.

## Git Commit Discipline

This is the **binding constraint** that makes the file-system-as-database work:

> **A commit containing code changes for a feature MUST also include the corresponding FEATURE.md.**

### Rationale

Feature tracking via files only has value if git history connects code and feature state.
`git log -- .ai_partners/features/active/<id>/FEATURE.md` must produce a **reliable timeline**
of every commit that changed the feature. Similarly, `git log -- <source-file>` should be
able to trace back to the FEATURE.md state at that point.

When FEATURE.md and code are committed separately:
- The feature timeline has gaps — some commits touch code but don't update FEATURE.md
- `git blame` on FEATURE.md loses fidelity as an event log
- An AI incarnation reading FEATURE.md alone cannot trust that it reflects the code it's looking at

### What qualifies as "feature code"

Any file listed or implied by the FEATURE.md Design Index. If the commit modifies a file
that implements or tests a feature, FEATURE.md goes in the same commit. When in doubt, include it.

### What to update in FEATURE.md per commit

- `updated` date — bumped even if `status` hasn't changed
- Mark tasks as done in the Scope table
- Add new Key Decisions if design choices were made in this commit
- `status_note` — a one-line summary of what this commit achieved or left unfinished

### CLI tool enforcement

The `moss features` CLI does NOT enforce this rule (git hooks are out of scope).
This convention relies on:
1. AI incarnations reading this specification and following it
2. Human engineer reviewing commits for FEATURE.md inclusion

If a commit lands without its FEATURE.md update, the correct fix is **not** a follow-up
FEATURE.md-only commit. The right fix is an interactive rebase to squash the missing
FEATURE.md into the code commit, restoring a clean one-to-one timeline.

## Self-Iteration Perspective

The feature system is itself the minimal viable unit of MOSS "let AI modify itself":

1. AI identifies capability gap → `moss features create` creates a feature
2. AI reads FEATURE.md + related source → understands what needs to be done
3. AI modifies code, runs tests → implements
4. AI updates frontmatter → `moss features archive` archives

This three-step cycle is the basic unit of self-iteration. The feature system can describe
its own improvement proposals (meta-feature), achieving second-order reflexivity.

## Unverified Claims

The following remain to be validated:

1. **Parallel development effectiveness**: Do AI incarnations on different branches sharing state via
   the features directory actually reduce conflicts?
2. **Human overhead of status maintenance**: Is manually maintaining frontmatter `updated` and
   `status` sustainable for the human engineer?
3. **Archive search efficiency**: As archived features grow, is Glob/Grep search still acceptable?
4. **Git workflow integration**: When feature state changes and code implementation are on different
   branches, `moss features status` showing `in-progress` cannot tell which branch. Should we add
   a `branch` field to the frontmatter?
5. **FEATURE.md sufficiency**: Is the minimal frontmatter schema sufficient for complex multi-day discussions?

---

*This convention is part of the MOSS project's AI-native collaboration infrastructure.
It was designed through discussion between the human engineer and Deepseek v4 on 2026-05-10.*

*Design constraints are explicitly limited to the "single human engineer + AI incarnations"
validation scenario. Complex problems (distributed locking, multi-user concurrency, human UX)
are reserved as future work.*
