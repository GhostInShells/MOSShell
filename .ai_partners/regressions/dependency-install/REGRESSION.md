---
title: dependency-install
version: 1
status: draft
priority: P1
created: 2026-07-27
updated: 2026-07-27
scope: >
  Verify each pip extra installs correctly and the corresponding CLI surface
  degrades gracefully when deps are missing.
depends: []
description: >-
  Post-restructuring smoke test: install each extra in isolation and verify
  the expected command surface.
---

# dependency-install

## Exploration Index

```
moss --ai all-commands
moss --ai help nodes networks manifests
moss codex get-source ghoshell_moss.depends
git log --oneline -5 -- pyproject.toml src/ghoshell_moss/depends.py
```

## Methodology

Human-in-the-loop. Requires a clean venv for each extra to avoid cross-contamination
from the dev environment.

Human: defines the expected surface for each install level.
Model: proposes cases and verifies output matches expectations.
Human: confirms pass/fail, signs off on regression.

Tests run sequentially by install level: cli → matrix → host → ghost.

## Prerequisites

- Python 3.10+ venv (clean, no existing ghoshell-moss installed)
- Working directory: MOSS repo root
- `uv` available (for `uv lock` already run after pyproject.toml changes)

## Test Cases

| Case ID | Priority | Description | Test Steps | Expected Result |
|---------|----------|-------------|------------|----------------|
| TC-001 | P0 | `pip install .[cli]` — moss --help works | 1. Create clean venv<br>2. `pip install .[cli]`<br>3. `moss --help` | moss --help shows groups: start, codex, project, ctml, howtos, features, docs, modes, ghosts, ground, memento. nodes/networks/manifests NOT shown. |
| TC-002 | P0 | `pip install .[cli]` — moss ground works | 1. Same venv as TC-001<br>2. `moss ground --help` | ground subcommands shown |
| TC-003 | P0 | `pip install .[matrix]` — nodes/networks/manifests appear | 1. `pip install .[matrix]` into same venv<br>2. `moss --help` | nodes, networks, manifests now appear in help |
| TC-004 | P1 | `pip install .[matrix]` — moss nodes list runs | 1. Same venv<br>2. `moss nodes list` (in a workspace) | Command runs (may show empty list, but no ImportError) |
| TC-005 | P0 | `pip install .[host]` — moss-shell starts | 1. `pip install .[host]` into same venv<br>2. `moss-shell --help` | Help shown, no ImportError |
| TC-006 | P1 | `pip install .[host]` — moss-ghost starts | 1. Same venv<br>2. `moss-ghost --help` | Help shown, no ImportError |
| TC-007 | P1 | `pip install .[host]` — moss-mcp starts | 1. Same venv<br>2. `moss-mcp --help` | Help shown, no ImportError |
| TC-008 | P0 | `pip install .[ghost]` — pydantic_ai importable | 1. `pip install .[ghost]`<br>2. `python -c "import pydantic_ai, anthropic"` | No ImportError |
| TC-009 | P1 | Clean install .[host] — all entry points work | 1. Fresh venv<br>2. `pip install .[host]`<br>3. `moss --help`<br>4. `moss-shell --help`<br>5. `moss-ghost --help`<br>6. `moss-mcp --help` | All four show help, no ImportError |

## Execution Notes

- TC-001 verifies the core Option B behavior: without matrix, nodes/networks/manifests are hidden from help.
- TC-003 verifies that installing matrix makes them appear.
- The regression does NOT require a running MOSS workspace — ImportError is the main thing we're testing against. Runtime behavior (actual node discovery, ghost launch) is tested elsewhere.
- `uv lock` was run 2026-07-27 after pyproject.toml restructuring. If new deps are added since, re-run before testing.
- If any TC fails: check that the corresponding `depend_*()` call site is correct in the source file that raises the error.
