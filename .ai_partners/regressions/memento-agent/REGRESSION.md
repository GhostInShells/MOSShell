---
title: Memento Agent Regression
version: 1
status: draft
priority: P0
created: 2026-07-26
updated: 2026-07-26
scope: subsystem
depends: [memento-cli-and-agent, momento-mori]
description: >-
  Verify memento agent lifecycle: invoke → staging → commit → show,
  multi-turn moment accumulation, degenerate no-memento path, and CLI 4 verbs.
---

# Memento Agent Regression

## Exploration Index

```
moss --ai all-commands --group memento
moss --ai memento agent --help
moss --ai memento agent invoke --help
moss codex get-interface ghoshell_moss.agents.contract:MementoAgent
moss codex get-interface ghoshell_moss.agents.memento_pydantic_agent.impl:MementoPydanticAgentImpl
git log --oneline -10 -- src/ghoshell_moss/agents/
```

## Methodology

Human-in-the-loop. Each case is a CLI command sequence with observable
output. Model proposes the commands; human or model executes; both verify.

Agent file: `agents/memento_agents/calc.agent.py` (stdlib math only, zero deps).
Memento root: use a temporary directory for each test run to isolate state.

Execution grouped by phase:
- **Phase 1** (TC-001–004): single-turn recording, manual commit
- **Phase 2** (TC-005–006): multi-turn accumulation
- **Phase 3** (TC-007–009): degenerate paths and CLI coverage

## Prerequisites

- Full-clone MOSS workspace with `uv sync --active --all-extras`
- `ANTHROPIC_MODEL` and `ANTHROPIC_API_KEY` in environment
- `agents/memento_agents/calc.agent.py` exists and imports only `math`

## Test Cases

| Case ID | Priority | Description | Test Steps | Expected Result |
|---------|----------|-------------|------------|----------------|
| TC-001  | P0       | Single invoke records moment to staging | 1. `rm -rf /tmp/_reg_memento && moss memento init -r /tmp/_reg_memento`<br>2. `moss memento branch create calc/main -r /tmp/_reg_memento`<br>3. `moss memento agent invoke agents/memento_agents/calc.agent.py "compute sqrt(42)" -r /tmp/_reg_memento`<br>4. `moss memento branch staging calc/main -r /tmp/_reg_memento` | Staging shows 1 moment with type `pydantic_ai.messages/v2` and payload containing messages |
| TC-002  | P0       | Manual commit freezes staging | 1. `moss memento branch commit calc/main -m "test" --kind semantic -r /tmp/_reg_memento`<br>2. `moss memento branch staging calc/main -r /tmp/_reg_memento` | Commit succeeds (prints cmt_xxx). Staging is empty after commit. |
| TC-003  | P0       | commit show displays frozen moments | 1. `moss memento commit show calc/<cmt_id> -r /tmp/_reg_memento` | Shows commit with correct moment count and type |
| TC-004  | P1       | branch log shows commit chain | 1. `moss memento branch log calc/main -r /tmp/_reg_memento` | Shows the semantic commit with correct summary text |
| TC-005  | P0       | Multi-turn accumulates moments in staging | 1. Run 3 invokes: sqrt(42), sqrt(100), sqrt(2)<br>2. `moss memento branch staging calc/main -r /tmp/_reg_memento` | Staging shows 3 moments in order |
| TC-006  | P0       | Second commit after multi-turn | 1. `moss memento branch commit calc/main -m "round 2" --kind semantic -r /tmp/_reg_memento`<br>2. `moss memento branch log calc/main -r /tmp/_reg_memento`<br>3. `moss memento owner log calc -r /tmp/_reg_memento` | Branch log shows 2 commits. Owner log shows 2 entries in chronological order. |
| TC-007  | P1       | Invoke without memento root does not crash | 1. `moss memento agent invoke agents/memento_agents/calc.agent.py "say hello"` (no --root flag) | Agent responds normally, no error |
| TC-008  | P1       | parse displays instruction and sha | 1. `moss memento agent parse agents/memento_agents/calc.agent.py` | Output contains META narrative, verbatim source, sandbox_exec description, and sha line |
| TC-009  | P2       | export-context and describe return stubs | 1. `moss memento agent export-context agents/memento_agents/calc.agent.py -r /tmp/_reg_memento` (owner=calc, branch=main by default)<br>2. `moss memento agent describe agents/memento_agents/calc.agent.py -r /tmp/_reg_memento` | Both print info message about not yet implemented, exit 0 |

## Execution Notes

- Use `/tmp/_reg_memento` as memento root for isolation. Delete before starting.
- `ANTHROPIC_MODEL=claude-haiku-4-5` is sufficient — math needs no reasoning.
- calc.agent.py owner = `calc` (from file stem), branch default = `main`.
  This matches the `calc/main` line naming in test cases.
