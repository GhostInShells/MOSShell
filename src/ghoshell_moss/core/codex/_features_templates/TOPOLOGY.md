# Features Directory Topology

```
.ai_partners/features/
  README.md              # Convention specification — "why" and "how"
  TOPOLOGY.md            # This file — "where"
  TEMPLATE.md            # Template for new features (source of `moss features create`)
  workstreams/           # All workstreams in all states — never move, only update frontmatter
    <year>/              # Created year (features stay in place for entire lifecycle)
      <month>/           # Created month
        <feature-name>/  # kebab-case, unique across the entire tree
          FEATURE.md     # REQUIRED: frontmatter + motivation + key decisions + design index
          discuss/       # Feature-specific discussion trails (optional)
          design/        # Design documents (optional)
```

## Path Semantics

- **Path encodes creation date** at `create` time. Features stay in place for their entire
  lifecycle — `completed`/`abandoned` are just a `status` field update in frontmatter, no file move.
  This preserves clean git history without path-forking from rename detection.
- **`workstreams/` is the single source of truth** for all features in all states. There is no
  `archive/` or `completed/` sibling — terminal states live alongside active ones.
- **Each FEATURE.md owns its internal organization.** The `design/` and `discuss/` subdirectories
  are suggestions, not requirements. A feature may define its own document structure directly
  in its FEATURE.md.
- **The feature name is the directory name.** No separate `id` field — the filesystem is the namespace.
  Name must be unique across the entire tree (not just within a month).
