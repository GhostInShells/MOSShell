---
created: 2026-05-13
depends: []
description: 'Optimize moss features list/status CLI output: shorter paths, smarter
  column layout, and better scannability for both human (rich) and AI (--ai plaintext)
  consumers.'
milestone: null
priority: P2
status: completed
status_note: CLI output optimization + conceptual reframing done
title: Features CLI Output Optimization — Compact, Scannable List & Status Display
updated: '2026-05-13'
---

# Features CLI Output Optimization

> Use `moss features set-status features-cli-output-optimization <status> -m "note"` to update state.

## Motivation

Current `moss features list` output has two pain points:

1. **Path column wastes tokens and screen space.** It renders the full absolute path
   (e.g. `/Users/.../GhostInShells/MOSShell/.ai_partners/features/active/2026/05/zenoh-fractal/FEATURE.md`),
   which dwarfs the actual feature metadata. In `--ai` mode this is pure token waste.

2. **Description column stretches the table horizontally.** Long one-line descriptions
   force very wide columns, making the table hard to scan in both terminal and
   markdown renderers.

3. **status --all output is panel-bloated.** Each feature gets its own panel with
   redundant field labels. 9 features = 9 panels. Hard to scan comparatively.

The features system targets L1 (AI stable feature coding) — where FEATURE.md anchors
context across sessions. The CLI should surface the right information density:
compact enough to scan, complete enough to decide which feature to open next.

## Design Index

- Plan: `.claude/plans/generic-stirring-giraffe.md`

## Key Decisions

1. **One feature for both surface (CLI) and root (concept) fixes.** The CLI output problems are downstream symptoms of the "features = catalog" misconception. Fixing both in one workstream avoids the surface fix being immediately obsolete.

2. **Keep CLI command names (`moss features ...`).** Backward compatibility constraint. The fix is in narrative framing (README.md, help text, output labels), not namespace.

3. **List table: drop Description, use relative Path.** Description is too wide for tables and belongs in `status <name>`. Path column now shows `active/2026/05/foo/` instead of full absolute path.

4. **Create: rich output eliminates follow-up Read.** After creation, display generated frontmatter fields + next steps. AI no longer needs a second round trip to read the file.

5. **Set-status: next-step hints per transition.** A small dict maps `(old_status, new_status)` to a one-line hint. Kept terse — one `print_info` call after the success message.

6. **Template: remove `## Related` section.** "Related features" only makes sense in a feature catalog, not a dev tracking system. Dependencies live in frontmatter `depends:`.

7. **Status --all: heading-based format instead of panel-per-feature.** `### name [status pri] — title` followed by indented fields. More scannable than 10 individual panels.

## Implementation Notes

- All output changes are in `features_cli.py` — the core `_features.py` data layer needed zero changes.
- `_feature_path` was already returned by `list_features()` as relative-to-active-dir — the CLI was just building absolute paths from it.
- `is_ai_mode()` imported from utils to branch between markdown heading format (AI) and rich-colored output (human) in the status --all view.
- The `_STATUS_HINTS` dict and `_ABANDONED_HINT` are module-level constants, easy to extend.