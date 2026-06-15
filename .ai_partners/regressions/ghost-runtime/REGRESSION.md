---
title: Ghost Runtime Regression
version: 1
status: draft
priority: P0
created: 2026-06-15
updated: 2026-06-15
scope: subsystem
depends: []
description: >-
  Verify GhostRuntimeImpl three-loop protocol (main → articulate → action),
  5-step wiring lifecycle, interrupt/pause semantics, and error recovery.
---

# Ghost Runtime Regression

## Exploration Index

```
moss codex blueprint ghost
moss codex get-interface ghoshell_moss.host:GhostRuntime
moss codex get-source ghoshell_moss.host.ghost_runtime
git log --oneline -10 -- src/ghoshell_moss/host/ghost_runtime.py
moss ghosts list
moss ghosts show echo
moss-run-ghost --help
```

## Methodology

Human-in-the-loop. Human operates terminal, model observes and records results.
Execution in case order (TC-001 → TC-002 → TC-003), each depends on prior passing.

## Prerequisites

- Full-clone MOSS workspace with `uv sync --active --all-extras`
- `.moss_ws/.env` configured with API keys (LLM provider)
- TTY-capable terminal (for TUI interaction)

## Test Cases

| Case ID | Priority | Description | Test Steps | Expected Result |
|---------|----------|-------------|------------|----------------|
| TC-001  | P0       | Run all unit tests | 1. `cd MOSShell`<br>2. `.venv/bin/pytest tests/ -x` | All tests pass, no failures |
| TC-002  | P0       | Start echo ghost and send hello world | 1. `.venv/bin/moss-run-ghost echo`<br>2. Type `hello world` and press Enter<br>3. Observe the reply | Ghost replies coherently; no duplicate moment messages in output |
| TC-003  | P0       | Emergency stop (Ctrl+G) during ghost reply | 1. `.venv/bin/moss-run-ghost echo`<br>2. Type a message that triggers a streaming reply<br>3. During streaming, press `Ctrl+G`<br>4. Observe system behavior | System pauses immediately; TUI shows pause indicator; no crash |

## Execution Notes