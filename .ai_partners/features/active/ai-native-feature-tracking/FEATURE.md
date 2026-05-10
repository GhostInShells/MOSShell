---
id: ai-native-feature-tracking
title: AI-Native Feature Tracking via File System Convention
status: in-progress
priority: P0
created: 2026-05-10
updated: 2026-05-10
depends: []
milestone: moss-self-bootstrap
description: >-
  Replace GitHub Issues with file-system-based feature tracking (FEATURE.md + YAML frontmatter)
  designed exclusively for AI incarnation collaboration. CLI as thin convention enforcer,
  core logic in ghoshell_moss.core.codex.
---

# AI-Native Feature Tracking via File System Convention

## Motivation

MOSShell has entered parallel development phase. The human engineer pushes 3~4 tasks concurrently,
AI incarnations assist on different branches. A shared state whiteboard is needed so different AI
instances can understand "what is being done, where is it at" through the file system.

GitHub Issues is an anti-pattern for AI: API rate limits, requires network, format is uncontrollable,
discussions scattered across PR/Issue/Comment dimensions. AI needs in-situ file presence, not API-fetched
external resources.

## Scope

- Design and implement FEATURE.md + YAML frontmatter convention
- Implement core functions in `ghoshell_moss.core.codex._features`
- Implement CLI as thin convention enforcer (`moss features init/create/list/status/archive`)
- Create specification and template files in `.ai_partners/features/`
- Dogfood: use this system to track its own development

Out of scope:
- Distributed locking and concurrent write conflicts (branch = isolation)
- Human collaborator UX (use GitHub Issues)
- CI/webhook integration
- External contributor scenarios

## Design Index

- Design document: `.design/2026-05-10-ai_native_feature_tracking_file_system_convention.md`
- Discussion summary: `.discuss/2026-05-10-ai_native_feature_tracking.summary.md`

## Key Decisions

1. **FEATURE.md (not README.md)**: README implies a human "reading" entry point. FEATURE.md is AI's "understand state + act" interface.
2. **File system as database**: Markdown + YAML frontmatter, query via Glob/Grep. No API, no network.
3. **AI-only design**: This is for AI collaboration exclusively. Humans use GitHub Issues when needed.
4. **Branch = context isolation**: Each branch has only one agent. No two AI instances modify the same feature file simultaneously.
5. **CLI as thin convention enforcer**: Core logic in codex, CLI just adds default directory conventions.
6. **Minimal frontmatter**: `id, status, title, created, updated, priority, depends, description`. Add fields only when proven necessary.

## Implementation Notes

Implementation order:
1. Specification files (README.md + TEMPLATE.md)
2. First feature (this file, dogfood)
3. Core functions in `ghoshell_moss.core.codex._features`
4. CLI integration in `ghoshell_moss.cli.features_cli`

The feature system can describe its own improvements (meta-feature), achieving second-order reflexivity.

## Related

- Related features: None yet (this is the first feature)
