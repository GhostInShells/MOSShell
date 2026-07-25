# interleaved_probe

Semantic instrument panel for the `interleaved-ctml-thinking` workstream.

## What it does

Provides channel `probe` (+ sub-channels `probe.a` / `probe.b`) where every
command is a controlled test case for one cursor-projection semantic:
live progress, empty outcomes (observed vs plain), runtime failure, critical
failure, and parallel FIFO tracks for per-channel cut-point testing.
No real-world side effects. The command↔semantic mapping lives in NODE.md.

## Setup

No install needed — depends only on the repo venv.

## Usage

```bash
moss nodes run .moss/system_test_nodes/interleaved_probe
```

Or spawn from a model session via `matrix.nodes:run`. Once the cell is
accepted, `probe` appears under `matrix.mesh` and the test tracks in NODE.md
can be laid via the interleaved MCP tools (`ctml_append` etc.).

## Development

Context and payload rules:
`.ai_partners/features/workstreams/2026/07/interleaved-ctml-thinking/FEATURE.md`.
When a new projection semantic gets decided, add one command here that
exercises exactly that semantic — keep the 1 command : 1 semantic discipline.
