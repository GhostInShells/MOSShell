# CLI Entry Point Governance

2026-07-26. Naming + help text cleanup for the four (→ three) interactive entry points.

## Rename Plan

| Before | After | Rationale |
|--------|-------|-----------|
| `moss-cli` | **deleted** | Historical; type-channel validation complete. |
| `moss-repl` | **`moss-shell`** | "repl" misstates — it's a full Textual TUI debugger for the Shell runtime. Humans test CTML/channels/matrix here before Ghost runs. |
| `moss-run-ghost` | **`moss-ghost`** | `moss-{role}` pattern. "run-" prefix is misleading — it launches a persistent TUI session, not a one-shot run. |
| `moss-as-mcp` | **`moss-mcp`** | `moss-{role}` pattern. "as-" is unnecessary. |

Pattern: `moss-{role}` — shell / ghost / mcp. Each is a human-only interactive tool, distinct from the `moss` CLI which serves both human and model.

## Help Text Changes

### moss-shell (was moss-repl)

Current: `启动 MOSS ToolSet TUI 调试终端。` (Chinese only, vague)

Proposed:
```
MOSS Shell runtime debugger — interactive TUI for testing CTML,
inspecting channels, and debugging the MOSS runtime before a Ghost runs.

Usage: moss-shell [--mode MODE] [--scope SCOPE] [--network NETWORK]
```

Also add `--network` option (currently missing from moss-repl).

### moss-ghost (was moss-run-ghost)

Current: `启动 Ghost TUI 交互终端 — 与 Ghost 实时对话。`

Proposed:
```
Launch a Ghost interactive terminal — stream logos, inspect output,
operate the SafeMode approval gate. Meta-control surface for Ghost
development; real Ghost interaction lives in the nodes system.

Usage: moss-ghost [GHOST] [--mode MODE] [--scope SCOPE] [--network NETWORK]
```

### moss-mcp (was moss-as-mcp)

Current: `MOSS MCP 服务启动程序`

Proposed:
```
Expose MOSS runtime as an MCP server for AI coding platforms (Claude Code,
Gemini CLI, etc.). Registers CTML execution and runtime introspection as
MCP tools.

Usage: moss-mcp [--mode MODE] [--transport sse|stdio|streamable_http] [--port 20773]
```

## Files to Touch

| File | Change |
|------|--------|
| `pyproject.toml` `[project.scripts]` | rename 3 entries, delete moss-cli |
| `src/ghoshell_moss/cli/cli_controller.py` | delete whole file |
| `src/ghoshell_moss/cli/moss_debug_repl.py` | rename function, update Click docstring |
| `src/ghoshell_moss/cli/ghost_run.py` | rename function, update Click docstring |
| `src/ghoshell_moss/cli/moss_as_mcp.py` | rename function, update Click docstring |
| `src/ghoshell_moss/cli/start.md` | update the 4-command table |
| `CLAUDE.md` | update embedded start.md content |
| `README.md` | update references (moss-as-mcp → moss-mcp, moss-run-ghost → moss-ghost) |

## Not in Scope

- Moving MCP tool logic (bootstrap/_drain_and_project/_spawn_interpreter) out of moss_as_mcp.py into host/ — deferred to v0.1.0
- moss-shell adding `--network` option — simple 1-line add, include here
