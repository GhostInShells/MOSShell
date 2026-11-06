---
title: 
version: 1
status: draft
priority: P1
created: 
updated: 
scope: 
depends: []
description: >-
  Brief description.
---

# 

<!--
Do NOT pre-create empty subdirectories (baselines/, discuss/, design/).
Only REGRESSION.md is created now. Subdirectories appear when content is
committed to them — not before.
-->

## Exploration Index

```
<!-- DELETE after filling: minimum moss commands to understand this subsystem.
Replace each line with real commands. Generic example:
moss codex blueprint <relevant-blueprint>
moss codex get-interface <package.module:Class>
moss --ai all-commands --group <relevant-group>
git log --oneline -10 -- src/<relevant-path>
-->
```

## Methodology

<!-- DELETE after filling.
Co-authored by human and model.

Human: defines scope and priorities — "what matters if it breaks."
Model: proposes cases by mapping concerns onto system interfaces.
Human: accepts, rejects, or reshapes the proposed cases.

Also: is this fully automated or human-in-the-loop?
What hardware/environment prerequisites exist?
What execution order — sequential, parallel, grouped by priority?
-->

## Prerequisites

<!-- DELETE after filling.
Environment setup, hardware connections, configuration files, API keys —
everything needed before starting.
-->

## Test Cases

<!-- DELETE after filling.
These are template instructions — remove them when the table is populated.

Copy the case table to create a baseline file after the first complete run:
  baselines/YYYY-MM-DD_v1.md

The baseline file appends three result columns: First Test | Fix | Final Result

Version increments when case structure changes (cases added, removed, or
materially modified). A new version needs a new baseline.

DELETE the example row below and add your own cases.
-->

| Case ID | Priority | Description | Test Steps | Expected Result |
|---------|----------|-------------|------------|----------------|
| TC-001  | P0       | <!-- DELETE this example row and add your own cases --> Example: verify the system starts without errors | 1. Run `command`<br>2. Check output | System starts, no error in stderr |

## Execution Notes

<!-- DELETE after filling.
Tips, known pitfalls, diagnostic commands — anything that saves the next
person 15 minutes of debugging.
-->