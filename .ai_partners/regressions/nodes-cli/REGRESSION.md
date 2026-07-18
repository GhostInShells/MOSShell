---
title: nodes CLI
version: 1
status: active
priority: P0
created: 2026-07-18
updated: 2026-07-18
scope: subsystem
depends:
  - cells-cli
  - matrix-cell-governance
  - cell-run-cycle
description: >-
  Dogfood verification of `moss nodes` CLI (9 commands) — path-based target
  resolve, spawn ownership, singleton lock, address matching, install gate,
  and the runtime/status/kill/prune cleanup surface.
---

# nodes CLI Regression

> First dogfood after `moss cells` → `moss nodes` rename + path-based redesign
> (§Rewrite 2026-07-17 in cells-cli FEATURE.md). Verification round exposed
> a fossil `spawn_cwd` concept in `nodes_cli.py` that bypassed
> `NodeLauncher.cwd` — surfaced as env leak + child could not find `main.py`.

## Exploration Index

```
moss codex get-interface ghoshell_moss.core.blueprint.cell:NodeLauncher
moss codex get-interface ghoshell_moss.core.blueprint.cell:NodeManifest
moss codex get-interface ghoshell_moss.core.blueprint.cell:CellRuntimeInfo
moss codex get-source ghoshell_moss.core.blueprint.cell
moss --ai all-commands --group nodes --depth 3
moss --ai help nodes create nodes run nodes status nodes kill
git log --oneline -- src/ghoshell_moss/cli/nodes_cli.py src/ghoshell_moss/core/blueprint/cell.py
```

## Methodology

Human-in-the-loop dogfood. Model runs commands, human catches drift the model
misses. Each surprise → discussion → decision → fix in same round. Focus on:

1. **Contract fidelity** — Every CLI action should delegate to blueprint
   abstractions (`Cell.home`, `NodeLauncher.cwd`, `enter_cell_lifecycle`,
   `env.dump_runtime_scope`). Fossil variables that bypass the contract are
   the primary defect class.
2. **Full lifecycle** — create → show → install → run → status → kill → prune
   across at least one node.
3. **Kernel drift caught during dogfood** — `project_id`, stub MOSS.md/HOST.md
   generation, M11-B default_mode fallback, `moss modes create` gap.

Sequential execution, single operator. No hardware prereqs. `.moss/system_test_nodes/`
is the dogfood-owned discovery path (added to a `system_test` mode's `node_paths`).

## Prerequisites

- Workspace initialized (`moss init -c -f -y`).
- `system_test` mode created via `moss modes create system_test`, with
  `HOST.md` `node_paths: [$MOSS_WORKSPACE/system_test_nodes]`.
- `.moss/system_test_nodes/` directory exists (dogfood-only, not tracked).

## Test Cases

| Case ID | Priority | Description | Test Steps | Expected Result |
|---------|----------|-------------|------------|----------------|
| TC-001  | P0 | `moss init -f` preserves project_id (no config wipe) | 1. Note `.moss/project_id` before<br>2. `moss init -c -f -y`<br>3. Diff `.moss/MOSS.md` and `cat .moss/project_id` | project_id file unchanged; MOSS.md is fresh MossMeta dump (no `project_id` field, no `cell_paths` residue) |
| TC-002  | P0 | Stub MOSS.md is dumped from MossMeta defaults | `cat src/ghoshell_moss/stubs/workspace/MOSS.md` | Frontmatter shows real field values (not comments), no `project_id` field, `default_mode: default` |
| TC-003  | P0 | M11-B fallback: empty env falls through to `default` mode | `moss --ai project where` after empty MOSS.md | Default Mode = `default` |
| TC-004  | P1 | `moss modes create <name>` copies default stub | `moss modes create system_test` → `moss modes show system_test` | `.moss/modes/system_test/` created with HOST.md + src/HOST scaffold; `show` displays it |
| TC-005  | P0 | `moss nodes list` mode-aware discovery | 1. `moss --mode system_test nodes list` (empty)<br>2. Create node in `.moss/system_test_nodes/`<br>3. Re-list | Empty on empty dir; lists node after create |
| TC-006  | P0 | `moss nodes create <path>` scaffolds stub with `{name}` substitution | `moss nodes create .moss/system_test_nodes/hello_world` | Directory contains NODE.md/main.py/README.md/INSTALL.md; NODE.md has `name: 'hello_world'` |
| TC-007  | P1 | `create` prints INSTALL.md hint when stub carries INSTALL.md | Read `create` command output | Output includes `Read <path>/INSTALL.md — declares install steps` + guidance to delete or install |
| TC-008  | P0 | `moss nodes show` prints verbatim NODE.md + install status warning | `moss nodes show <path>` before install | Shows NODE.md content verbatim + `[WARN] Not installed. INSTALL.md declares steps...` |
| TC-009  | P0 | `moss nodes install <path>` touches `.installed` marker | `moss nodes install <path>` → `ls <path>/.installed` | `.installed` file exists; `show` no longer prints install warning |
| TC-010  | P0 | `moss nodes run` uses `manifest.cwd` (NODE.md dir), not fossil `spawn_cwd` | `moss nodes run <path>` (background) then check child cwd via `lsof`/`ps` | Debug section prints `cwd: <node dir>` (not `.moss/runtime/cells/...`); child successfully imports `main.py` |
| TC-011  | P0 | `moss nodes run` env debug section leaks no secrets | Read `[run] Starting node cell` debug output | Env section shows only 9 MOSS runtime scope keys (from `env.dump_runtime_scope()`); no `MOSS_LLM_API_KEY` or other os.environ inheritance shown |
| TC-012  | P1 | `moss nodes run` refuses when node is not installed | Remove `.installed`, then `moss nodes run <path>` | Exit code 1 with `[ERROR] Node ... is not installed` + guide to `moss nodes install` |
| TC-013  | P1 | `moss nodes run` singleton lock refuses second instance | Run twice concurrently | Second attempt fails with `Singleton conflict for '...': cannot acquire lock` + status/kill hints |
| TC-014  | P0 | `moss nodes status` lists live entries; `status <query>` matches via uid prefix | 1. Run node<br>2. `moss nodes status`<br>3. `moss nodes status <uid[:8]>` | List shows the address; detail matches by uid prefix (not `endswith`) |
| TC-015  | P0 | `moss nodes status` detail shows `cell.home`, no fossil `spawn_cwd` row | Look at the detail table | Table has `home` row (correct value); no `spawn_cwd` row |
| TC-016  | P1 | Ambiguous query is rejected with clear error | With ≥2 running nodes with overlapping uid prefix, `moss nodes status <short>` | Error lists all matches + advises full address |
| TC-017  | P0 | `moss nodes kill <query> --force` sends SIGKILL to pgid; ledger removed | 1. Run node<br>2. `moss nodes kill <uid[:8]> --force`<br>3. `ls .moss/runtime/cells/` | Node terminated; ledger JSON removed; no orphan process (via `ps -ef`) |
| TC-018  | P1 | Crash-fast: run of unlaunchable command emits `[ERROR] Node exited abnormally (returncode=N)` | Modify NODE.md `exec.args` to nonexistent script → run | Debug prints ok, child stderr shows import error, CLI prints `[ERROR] Node exited abnormally (returncode=2)` |
| TC-019  | P1 | `moss nodes prune` cleans stale + kills alive orphans by default | 1. Manually corrupt ledger or leave stale entry<br>2. `moss nodes prune` | Reports pruned count; runtime dir cleaned |
| TC-020  | P2 | `moss nodes prune --keep-alive` preserves live orphans | Run node, `moss nodes prune --keep-alive` | Live entry preserved (skipped count > 0), dead ones removed |
| TC-021  | P2 | `moss nodes link <workspace> <script> --command python` shortcut | Create a `hello.py` outside workspace, link it into `.moss/system_test_nodes/link_test/` | NODE.md created with `exec.command: python`, `exec.args: <abs path>`; description mentions absolute-path fragility |
| TC-022  | P2 | `moss nodes link` fails without `--command` (no auto-detection) | Same but omit `--command` | Exit code 1 with `--command is required` guidance |

## Known Limitations Recorded This Round (Not Blockers for v1 baseline)

- **`Matrix.discover().run(pass)` blocks under CLI Popen env inheritance** —
  child launched via `moss nodes run` inherits full `os.environ` (per
  `env.dump_cell_env(with_os_env=True)`). Direct `python main.py` with a
  minimal MOSS-scoped env exits in ~1.6s; CLI-launched child blocks in
  matrix `arun` until SIGTERM. Not a nodes-CLI bug — env-sensitive path in
  matrix discover/mode assembly. Assigned to cell-run-cycle **M7** for the
  next round.
- **Crash-fast has no persistent ledger** — child that crashes before
  entering `enter_cell_lifecycle` leaves no `.json` in runtime dir. CLI
  emits `[ERROR] Node exited abnormally (returncode=N)` to stderr but no
  historical record. Design question: whether to introduce a CLI-owned
  crash ledger (new blueprint concept) or route to future Jobs layer.
  Deferred to a separate discussion (§R-4 in cells-cli FEATURE.md).
- **`CellNamePattern` forbids hyphens** — `moss modes create system-test`
  is rejected (`system_test` works). Same issue for node names.
  cell-run-cycle FEATURE.md M4a already tracks pattern-widening; not
  addressed in this commit.
- **`moss --ai project where` does not strip rich formatting** — `--ai`
  flag inconsistent across commands. Not in nodes CLI scope but recorded
  here as `project` CLI health.
- **Stub node's `if __name__ == "__main__": Matrix.discover().run(main)`
  behavior differs between direct-python and CLI-Popen contexts** — see
  first bullet. Related to the M7 investigation.

## Execution Notes

- Do not run any CLI command from inside a node's own directory unless
  intentionally testing `find_upward` behavior — target resolve prefers
  path arg over cwd walk.
- `--force` on kill is safer than default 3s grace during dogfood — the
  stub node's blocking behavior means SIGTERM alone may not be enough.
- `.moss/system_test_nodes/` is not tracked (dogfood-only) — recreate as
  needed by rerunning the setup section.
- The `_match_address` helper (uid-prefix + full-address) replaces earlier
  `endswith` semantics. Historic short-form kills relying on trailing chars
  will break — retest any external tooling against uid prefix.
