# Features Directory Topology

The features system is the project's human–AI collaboration layer (pillar 2).
Three axes plus one surface index live under `features/`.

```
features/
  README.md              # Convention specification — "why" and "how"
  TOPOLOGY.md            # This file — "where"
  TEMPLATE.md            # Template for new workstreams (source of `moss features create`)
  SURFACE.md             # L3 deliverables checklist — the project's outward face (pointers only)
  review/                # Project-level review perspective docs (optional)
  workstreams/           # L1 intent axis — all workstreams in all states (never move)
    <year>/              # Created year (features stay in place for entire lifecycle)
      <month>/           # Created month
        <feature-name>/  # kebab-case, unique across the entire tree
          FEATURE.md     # REQUIRED: frontmatter + motivation + key decisions + design index
          discuss/       # Feature-specific discussion trails (optional)
          design/        # Design documents (optional)
          review/        # Feature review perspective docs (optional; override global)
  stages/                # L2 intent axis — development periods (forward-facing)
    ROADMAP.md           # Cross-stage index (active/planned/completed/cancelled)
    _template/           # Stage + milestone templates
    YYYY-MM-<id>/        # One directory per stage, anchored by STAGE.md
  regressions/           # Verification axis (orthogonal) — long-lived, keyed by scope
    README.md TEMPLATE.md
    <scope>/             # Semantic scope name, not date path
      REGRESSION.md      # + baselines/YYYY-MM-DD_vN.md
```

## Path Semantics

**workstreams/** — path encodes creation date at `create` time. Features stay in place for their
entire lifecycle — `completed`/`dropped`/`parked` are just a `status` field update, no file move
(clean git history). `workstreams/` is the single source of truth for all workstreams in all states;
there is no `archive/`. Each FEATURE.md owns its internal organization; the feature name is the
directory name (unique across the tree, not just within a month).

**stages/** — keyed by start period (`YYYY-MM-<id>`), forward-facing, terminating. `STAGE.md`
travels from intent to record (planning → active → completed). Associations are by name, never path.

**regressions/** — keyed by the *scope being verified* (not date, not stage, not necessarily a
feature name — scopes like `dependency-install` are cross-cutting). Regression sets are long-lived
and evolve across versions (`expired ≠ completed`).

**SURFACE.md** — the L3 deliverables checklist. Pointers only, no duplicated content. Changed only
at a big Stage's close, and only on checklist-level change. Single file; history is `git log`.
