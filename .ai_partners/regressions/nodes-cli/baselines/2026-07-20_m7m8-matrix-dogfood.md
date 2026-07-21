---
version: 1
completed: 2026-07-20
operator: claude-opus-4-7 (dogfood) + human architect
---

# Baseline 2026-07-20 M7/M8 Matrix Channel Dogfood

First end-to-end dogfood of matrix channel cell governance + desktop channel
OS tooling via MCP (system_test mode). Verified full CTML chain: node discovery
→ spawn → proxy mount → cross-process CTML → signal chain → stop.

## Results

| Case ID | Priority | Description | First Test | Notes | Final Result |
|---------|----------|-------------|------------|-------|--------------|
| MC-001  | P0 | `matrix.nodes:list` discovers installed nodes | PASS | — | PASS |
| MC-002  | P0 | `matrix.nodes:run` spawns node cell | PASS | — | PASS |
| MC-003  | P0 | Spawned cell appears as `matrix.mesh.<fullname>` proxy | PASS | — | PASS |
| MC-004  | P0 | Cross-process CTML command via proxy | PASS | `signal_sender:send` works | PASS |
| MC-005  | P0 | `matrix.nodes:status` shows running + dead cells | PASS | — | PASS |
| MC-006  | P0 | `matrix.nodes:stop` clean shutdown (exit=0) | PASS | — | PASS |
| MC-007  | P0 | `matrix.nodes:read_output` reads stderr | PASS | — | PASS |
| MC-008  | P0 | `matrix.mesh:events` shows network events | PASS | — | PASS |
| MC-009  | P0 | Session signal bus: `add_signal` → `on_signal` → janus | PASS | NotifySignalMeta end-to-end | PASS |
| MC-010  | P0 | `desktop.file_editor:view/str_replace` works | PASS | — | PASS |
| MC-011  | P0 | `desktop.bash:exec` works via CDATA | PASS | — | PASS |
| MC-012  | P0 | Channel tree: matrix + desktop as peers | PASS | No OS tools under matrix | PASS |
| MC-013  | P1 | `nodes_mgr()` callable bug caught by dogfood | FAIL → FIX → PASS | ProjectNodeManager is a property, not callable | PASS |

## Summary

- **Total cases**: 13
- **First-test failures**: 2 (MC-013: callable bug; MC-009: Signal.body → Signal.messages attr typo)
- **Final PASS**: 13 / 13

## Channel Tree Verified

```
__main__
├── matrix              ← cell governance
│   ├── nodes           ← list/read/run/stop/status/read_output
│   └── mesh            ← accept/reject/set_auto_accept/events + proxies
└── desktop             ← OS tools
    ├── bash            ← exec/run/read_output/stop
    └── file_editor     ← view/create/str_replace/insert/undo_edit
```

## Signal Chain Verified

```
sender cell (CTML) → session.add_signal(NotifySignalMeta)
    → session.on_signal(callback) → janus.Queue.sync_q
    → async task: janus.Queue.async_q → print → received[] list
    → receiver.received() (CTML) → returns captured signals
```

## What This Baseline Establishes

- matrix + desktop channel split is functional and clean
- Cell spawn → proxy mount → cross-process CTML chain works end-to-end
- Session signal bus (add_signal/on_signal) works across cell processes
- janus queue pattern for nonblocking signal consumption verified
- Dogfood found and fixed: `nodes_mgr()` callable bug, Signal attribute naming
