---
id: nuclei-manifests-discovery
title: Nuclei Manifests Discovery
status: completed
priority: P1
created: 2026-05-11
updated: 2026-05-11
depends: []
milestone:
description: >-
  Add NucleusFactory environment discovery via manifests, following the
  ProviderInfo/TopicInfo pattern. Declared for test/dev infrastructure.
---

# Nuclei Manifests Discovery

## Motivation

NucleusFactory is the declarative layer of Nucleus in Mindflow (similar to
ResourceStorageFactory for ResourceStorage). Without manifests discovery,
developing a Nucleus requires a full Ghost runtime. This feature enables
"list, filter, find" — quick verification during development.

## Scope

**In scope:**
- `NucleusMetaInfo` frozen dataclass in `core/blueprint/manifests.py`
- Discovery functions in `host/manifests/nuclei.py`
- `PackageManifests` lazy scan of `.nuclei` sub-package
- `MergedManifests` merge (right-side priority)
- CLI: `moss manifests nuclei [search] [--json]`
- REPL inspector support
- Stub example in workspace

**Out of scope:**
- Matrix bootstrap integration (wait for Ghost runtime)
- `PackageNuclei` ResourceStorage wrapper (not a Resource pattern)
- Shell/Mindflow runtime integration

## Design Index

- Plan: `.discuss/` (embedded in the task plan from Claude Code)
- Implementation: `src/ghoshell_moss/host/manifests/nuclei.py`
- Convention spec: `.ai_partners/features/README.md`

## Key Decisions

1. **Simple dataclass pattern** over ResourceMeta pattern — NucleusFactory is
   not a ResourceStorage, doesn't need VFS routing.
2. **`NucleusMetaInfo` defined in core/blueprint/manifests.py** — follows
   ProviderInfo/TopicInfo precedent, avoids core→host dependency inversion.
3. **Keyed by `name()`** in the dict — NucleusFactory.name() is the natural
   unique key, unlike resource storages keyed by `scheme:host`.

## Implementation Notes

- `find_nucleus_metas()` uses `scan_package(max_depth=2)` for lazy import —
  same pattern as `find_resource_storage_metas()`.
- `match_nucleus_infos()` searches name, description, and signal_names.
- `NUCLEI_SUB_PACKAGE = 'nuclei'` on PackageManifests for the sub-package
  name convention.
- Right-side priority in MergedManifests means mode-level nuclei override
  environment-level nuclei with the same name.

## Related

- Related: `resource_storages.py` (followed the discovery pattern)
- Related: ghost manifests (future, when Ghost runtime is ready)
