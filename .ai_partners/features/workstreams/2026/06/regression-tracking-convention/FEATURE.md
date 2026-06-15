---
title: Regression Tracking Convention
status: in-progress
priority: P0
created: 2026-06-15
updated: 2026-06-16
depends: []
milestone:
description: >-
  Document-led regression tracking in .ai_partners/ — methodology + test cases + baselines,
  a sibling mechanism to features for verification trajectory handoff.
---

# Regression Tracking Convention

> `moss features set-status regression-tracking-convention <status> -m "note"` to update state.

## Motivation

Automated tests (pytest, integration, e2e) tell you what passes and fails. They don't
hand off the verification context across sessions: methodology, prerequisites, execution
steps, root cause diagnoses. The G1 integration workstream revealed a working pattern —
SKILL.md + atomic scripts + execution sequences + baseline results — but it was embedded
in a single feature's workflow, not reusable.

This workstream formalizes that pattern into a `.ai_partners/regressions/` convention,
making it a first-class sibling to features. Features track what was built and why;
regressions track how it was verified and what broke.

## Design Index

- Convention spec: `.ai_partners/regressions/README.md`
- Template: `.ai_partners/regressions/TEMPLATE.md`
- Validation case (TBD): `ghost-runtime` regression set

## Key Decisions

### KD1: No CLI — copy-and-edit workflow

Regressions don't need status management tooling. Discovery is `find -name "REGRESSION.md"`.
Template → REGRESSION.md → baseline file is a file copy + edit chain. No tooling overhead.

### KD2: Semantic directory names, not date-path

Unlike features (which start and end), regression sets are long-lived entities that evolve
across versions. `ghost-runtime` not `2026/06/ghost-runtime`.

### KD3: "baseline" not "milestone"

"Milestone" implies achievement. A baseline is a reference point — the first complete run
of a version's test case set, against which future runs are compared.

### KD4: Version counter in frontmatter

`version: N` increments when case structure changes. It's a quick signal; git is the audit
trail. Each baseline file carries `_vN` suffix, tying it to the REGRESSION.md version it
was run against.

### KD5: Five live columns + three result columns

REGRESSION.md: Case ID | Priority | Description | Test Steps | Expected Result
Baseline file: above + First Test | Fix (root cause, not commit hash) | Final Result

### KD6: Three-way result states for baselines

PASS / FAIL / BLOCKED (can't run) for first test. PASS / FAIL / SKIP (doesn't apply) for
final result. BLOCKED → SKIP is a valid trajectory.

### KD7: In .ai_partners/, not start.md

Regressions are a development verification tool, not a cognitive entry point. Mentioned in
CLAUDE.md only.

### KD8: Cross-validated with features convention

Same philosophical foundation (filesystem as database, markdown portability, model-to-model
handoff), same sub-convention structure (discuss/, design/), same frontmatter style.
Intentional divergences documented in README.md §Cross-Validation.

## Validation Plan

The convention will be validated by creating the first regression set:

1. Another model instance (fresh session) reads `regressions/README.md` + `TEMPLATE.md`
2. Creates `ghost-runtime/REGRESSION.md` with real test cases
3. This instance reviews the result

If the other instance can produce a correct REGRESSION.md from the spec alone, the
convention is functional.

## Validation Session — 2026-06-16 Friction Patches

First validation attempt (other instance creating ghost-runtime) surfaced three issues:

### FP1: Empty directory pre-creation
The other instance ran `mkdir -p .../baselines` before any baseline existed. Root cause:
the Directory Topology tree implied baselines/ is always present. Fix: topology now shows
only REGRESSION.md as "created initially," all subdirs as "created when content exists."
TEMPLATE.md now warns against pre-creating empty directories.

### FP2: Model-as-solo-author framing
Both the spec and template implied the model designs cases alone. Reality: the human
defines scope ("what matters if it breaks"), the model proposes cases, the human decides.
Fix: added co-authorship note to Model's Role and TEMPLATE.md Methodology section.

### FP3: Missing exploration index
When a model enters a regression set, it needs a compressed path to build minimal
understanding of the subsystem. Fix: added Exploration Index section to TEMPLATE.md —
`moss` commands, not paragraphs.

## Implementation Notes

- First cases expected: run full unit tests, `moss-run-ghost echo`, say hello to echo
- "Fix" column records root cause diagnosis, not solution or commit hash — compound
  diagnostic value across sessions
- `expired` is not `completed` — a regression set becomes irrelevant when the target
  system drifts, not when it's done