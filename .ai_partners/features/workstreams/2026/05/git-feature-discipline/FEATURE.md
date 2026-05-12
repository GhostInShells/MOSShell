---
id: git-feature-discipline
title: Git Commit Discipline — FEATURE.md must be committed with code
status: completed
priority: P0
created: 2026-05-12
updated: 2026-05-12
depends: []
milestone: beta
description: >-
  Add "Git Commit Discipline" section to the feature tracking specification,
  mandating that FEATURE.md is always committed together with related code changes.
  This commit itself serves as the bootstrapping example.
---

# Git Commit Discipline — FEATURE.md must be committed with code

## Motivation

Recent commits have shown a pattern: AI incarnations commit feature code without updating
the corresponding FEATURE.md in the same commit. This breaks the core premise of
file-system-based feature tracking — that `git log -- <feature>/FEATURE.md` produces
a reliable timeline of the feature's evolution.

The specification itself must be amended to make this rule explicit, and the amending
commit must itself demonstrate the rule (bootstrapping).

## Scope

| 任务 | 状态 |
|---|---|
| Add "Git Commit Discipline" section to `README.md` | done |
| Commit FEATURE.md together with README.md (bootstrapping) | done |

**Out of scope:**
- Fixing past commits that violated the rule
- CLI or git hook enforcement

## Design Index

- Specification: `.ai_partners/features/README.md` — "Git Commit Discipline" section

## Key Decisions

### 1. Convention-based enforcement, not tool-based

Git hooks and CLI checks are out of scope. The rule relies on AI incarnations reading
the specification and the human engineer reviewing commits. This keeps the feature
system lightweight and the CLI a thin convention enforcer.

### 2. Fixing violations: rebase, not append

If a commit lands without its FEATURE.md, the correct fix is interactive rebase to
squash the FEATURE.md into the original code commit, restoring a clean one-to-one
timeline. A follow-up FEATURE.md-only commit creates a gap in the timeline.

## Related

- Depends on: (none)
- Related: `ai-native-feature-tracking` (the original convention)
