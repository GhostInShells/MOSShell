---
created: 2026-05-19
depends: []
description: moss script CLI — dev-time one-shot scripts that connect to the running
  matrix to send signals, query state, and debug the ghost runtime.
milestone: null
priority: P1
status: completed
status_note: 'v1: moss script CLI, zenoh config merge, connect() helper. Liveness
  dynamic discovery deferred.'
title: Script Cell
updated: '2026-05-19'
---

# Script Cell

> `moss script run <name>` — launch a lightweight, one-shot script that joins the matrix network,
> does its job, and exits. No Circus, no persistent runtime.

## Motivation

When a ghost is running, the developer needs ways to interact with it beyond TUI/GUI. The simplest
interaction is sending a signal — a text percept that enters the mindflow perception loop. Script
cells fill this gap: CLI-launched, one-shot Python scripts that connect to the running Zenoh
network, inject signals (or query topics, or use channel proxies), and exit.

They complement persistent apps by serving the **dev-time** use case — debugging, probing,
manual triggering — rather than providing persistent runtime capabilities.

## Design Index

- Key discussion: `discuss/01-convergence.md`

## Key Decisions

### 1. First version: lightweight signal injection, no liveness discovery

The Zenoh signal path is already fully wired: `MOSS/{session_scope}/signals` → Session →
Mindflow → Nucleus → Impulse → Attention. A script just needs a Zenoh session and the right
key expressions to inject signals. No need for the main node to "discover" the script cell.

**Why not dynamic liveness discovery now**: the liveness infrastructure exists but only monitors
pre-registered cells. Expanding to wildcard discovery is technically straightforward but introduces
a passive discovery surface — potential future security concern. Defer until the threat model
is clearer.

**Note for future**: when dynamic liveness discovery is added, scripts will show up in
`Matrix.list_cells()` as `type=script`. For now they operate as transient Zenoh clients.

### 2. `moss script` CLI, not standalone executables

Scripts are launched via `moss script run <name>`, following the same pattern as `moss apps test`:
Host discovers environment → build MOSS env vars → subprocess. This keeps lifecycle management
unified under the moss CLI.

### 3. SCRIPT.md manifest, minimal

Scripts live in `[ws]/scripts/<name>/` with a `SCRIPT.md` manifest. Much simpler than APP.md:
no `executable`, `workers`, `respawn`, `max_age`. Just metadata. The script itself is a Python
file that reuses the moss runtime — no pyproject.toml isolation.

### 4. No pyproject isolation

Scripts strictly reuse the moss Python runtime. Unlike apps (which can have independent
dependencies via uv), scripts are dev-time tools that share the project environment.

### 5. CellType.script already exists

`CellType.script = "script"` was defined in the original Matrix design. This workstream
implements the long-planned concept.

## Implementation (2026-05-19)

### Files changed

| File | Change |
|------|--------|
| `host/matrix.py` | `_default_providers()`: elif chain → host vs non-host binary |
| `host/providers/zenoh_provider.py` | `HostEnvZenohProvider`: `== 'app'` → `== 'host'`, param rename |
| `configs/zenoh_config_app.json5` → `zenoh_config_cell.json5` | Renamed: all non-host cells share connector config |
| `cli/main.py` | Registered `script` command group |
| ~~`script.py`~~ | Removed — `Matrix.discover()` is already self-explanatory, no wrapper needed |
| `cli/scripts_cli.py` (new) | `moss script list/run/init` commands |
| `stubs/workspace/scripts/` (new) | Template: README + `_example/SCRIPT.md` + `main.py` |

### Zenoh config fix (bundled)

The old `_default_providers()` had `if main: ... elif app: ... else: RuntimeError`.
Every new cell type required a new branch. Fixed by inverting the logic:
host is the special case (listener), everything else uses the connector config.

`HostEnvZenohProvider` had the same bug — also fixed.

### Verification

- `moss script init/list/run` all functional
- `connect()` import works
- 310/311 tests pass (1 pre-existing CTML failure, Claude Code env specific)
- Zero references to old `zenoh_config_app.json5` remain