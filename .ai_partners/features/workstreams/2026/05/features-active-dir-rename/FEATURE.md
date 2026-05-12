---
title: Features Active Dir Rename
status: completed
priority: P2
created: 2026-05-13
updated: 2026-05-13
depends: []
milestone:
description: >-
  Rename or remove the "active" intermediate directory in features/ topology to
  resolve semantic mismatch with in-place status convention.
---

# Features Active Dir Rename

> Use `moss features set-status features-active-dir-rename <status> -m "note"` to update state.

## Motivation

The current directory topology places all features under `active/`:

```
.ai_partners/features/
  active/
    <year>/
      <month>/
        <feature-name>/
          FEATURE.md
```

The spec adopted **in-place status** on 2026-05-13 — `completed` and `abandoned` features
stay where they were created, never moving to an archive. This means `active/` contains
features in ALL states: draft, in-progress, completed, abandoned, blocked.

The name "active" is semantically misleading. A completed feature from 2025 sitting under
`active/2025/03/` is not "active" by any common definition. This creates confusion for:

- New AI incarnations scanning the tree
- Human collaborators browsing the directory
- Any tooling that interprets "active" as "currently being worked on"

The mismatch was noted during the 2026-05-13 spec revision but deferred for a dedicated
discussion. That discussion is this feature.

## Design Index

- Key design documents: `design/`
- Key discussion records: `discuss/`
- Spec to update: `.ai_partners/features/README.md` — Directory Topology section
- Code to update: `ghoshell_moss.core.codex._features` — all path-building logic
- Code to update: CLI commands that reference path conventions

## Options

### Option A: Rename to `workstreams/`

The spec and CLI help text consistently call features "workstreams." The name "workstreams"
captures the intent: these are development tracks with a lifecycle, not necessarily "active."

```
.ai_partners/features/
  workstreams/
    <year>/
      <month>/
        <feature-name>/
```

**Pros**: Accurate to what the CLI says. One extra hop in path building. `git mv` preserves
history cleanly — it's a single rename, not a structural change.

**Cons**: Still has an intermediate directory whose sole purpose is grouping. The year/month
nesting already provides organization.

### Option B: Remove the intermediate directory

Flatten `active/` out entirely:

```
.ai_partners/features/
  <year>/
    <month>/
      <feature-name>/
        FEATURE.md
  README.md
  TEMPLATE.md
```

**Pros**: Cleanest topology. No misleading name. Less nesting = easier to mentally model.
Year/month already groups features by creation time.

**Cons**: `git mv` across multiple features is noisier — but since features are few (currently
~10), the churn is manageable. `.design/` and `.discuss/` siblings at the `features/` level
remain untouched and unambiguous.

### Option C: Keep `active/` as is, document the semantics

No code change. Just add a clarifying sentence to README.md: "'active' means 'the active set
of all features' — a living collection, not a filter on status."

**Pros**: Zero churn. Zero risk.

**Cons**: Every new AI incarnation and human reader will trip over this. "Document it" is the
universal last resort of naming problems. The name still suggests a status filter to anyone
who doesn't read the fine print.

## Key Decisions

### KD1: Rename `active/` to `workstreams/` (Option A)

Chose Option A over B (flatten) and C (keep as is). Rationale:

- **Option A over B**: Keeping an intermediate directory (`workstreams/`) reserves namespace
  for future siblings — `reports/`, `analysis/`, etc. Flattening would create ambiguity between
  top-level docs (README.md, TOPOLOGY.md, TEMPLATE.md) and year/month feature dirs.
- **Option A over C**: "active" is objectively misleading given the in-place status convention.
  "workstreams" matches the CLI terminology exactly and describes what these ARE (development
  tracks), not what state they're in.

### KD2: Extract Directory Topology to TOPOLOGY.md

Removed the inline topology tree from README.md into a standalone TOPOLOGY.md. Rationale:

- README.md is the "why" (convention specification); TOPOLOGY.md is the "where" (layout reference)
- TEMPLATE.md now references TOPOLOGY.md, making the `moss features create` flow self-contained
  — AI reading the generated FEATURE.md can follow the pointer to understand where it lives
- Topology changes (like this one) are scoped to a focused document

### KD3: Update all artifacts consistently

- `_features.py`: all `active_dir` → `workstreams_dir`, all `"active"` string → `"workstreams"`, docstrings updated
- DEFAULT_TEMPLATE (code fallback): synced with TEMPLATE.md changes
- `init_features()`: now copies TOPOLOGY.md alongside README.md and TEMPLATE.md
- Existing features: `git mv` from `active/` to `workstreams/`

## Implementation Notes

- `git mv` preserves git history (rename detection) — no `git log` breakage
- `_feature_path` in list results is `relpath` from `workstreams/` — displays as `year/month/name`
- The `_find_feature_dir` helper uses recursive glob, so no path logic change needed there
