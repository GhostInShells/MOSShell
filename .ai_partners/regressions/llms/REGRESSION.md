---
title: llms
version: 1
status: draft
priority: P1
created: 2026-08-08
updated: 2026-08-08
scope: >
  LLM config contracts, ModelRef, PydanticAIFuncs engine, CLI call/list,
  benchmark pipeline, and dependency coexistence (pydantic-ai-slim + mcp 2.0.0).
depends: []
description: >-
  Full-stack llms regression: contracts, ModelRef no-secret-leak, model-func
  engine (call + benchmark), CLI surface, and pyproject.toml dependency upgrade.
---

# llms

## Exploration Index

```
moss --ai all-commands --group llms
moss codex get-interface ghoshell_moss.contracts.llms:LLMFuncs
moss codex get-interface ghoshell_moss.llms.pydantic_ai_adapter.funcs:PydanticAIFuncs
git log --oneline -5 -- src/ghoshell_moss/contracts/llms.py src/ghoshell_moss/llms/funcs.py
```

## Methodology

Fully automated except TC-008 (requires API key in environment).

Model: runs pytest suite + CLI smoke commands.
Human: confirms dependency upgrade (uv sync) succeeds, signs off on regression coverage.

Tests run sequentially: contracts → engine → CLI. CLI structured-output tests
require a valid ANTHROPIC_API_KEY (or equivalent).

## Prerequisites

- Python 3.12+, uv available
- Working directory: MOSS repo root
- `uv sync` completed after pyproject.toml changes (2026-08-08: pydantic-ai → slim, mcp ≥ 2.0.0)
- ANTHROPIC_API_KEY set (for TC-006 through TC-008)

## Test Cases

| Case ID | Priority | Description | Test Steps | Expected Result |
|---------|----------|-------------|------------|----------------|
| TC-001 | P0 | Contract tests — LLMConfig, ModelConfig, ServiceConfig, Provider, ResolvedModel, converters | `pytest tests/ghoshell_moss/default/contracts/test_llms.py -v` | 43 passed (0.89s) |
| TC-002 | P0 | Engine tests — PydanticAIFuncs call + benchmark, ModelRef no-secret-leak | `pytest tests/ghoshell_moss/default/llms/test_funcs.py -v` | 15 passed — call structured/null/to_record, benchmark loop + output_file, ModelRef safety |
| TC-003 | P0 | `moss llms list` — ModelRef display, no secret leak | `moss --ai llms list` | Table with columns service/protocol/model/description/tags/content_types/max_out. No api_key or base_url in output. |
| TC-004 | P0 | `moss llms call` — plain-text call | `moss --ai llms call "reply with exactly: pong"` | Returns pong. No crash. |
| TC-005 | P1 | `moss llms call -r -j` — structured output JSON | `moss --ai llms call "the model you are running on" -r ghoshell_moss.contracts.llms:ServiceConfig -j` | Valid JSON with result/content/usage/cast/retries fields. result is dict matching ServiceConfig schema. |
| TC-006 | P1 | `moss llms call -r -v` — verbose structured output | `moss --ai llms call "reply pong" -r ghoshell_moss.contracts.llms:ModelConfig -v` | Structured result printed + table with usage/elapsed/retries columns |
| TC-007 | P1 | `moss llms call -n` — repetition | `moss --ai llms call "pong" -n 3` | 3 outputs. No crash. |
| TC-008 | P1 | Dependency coexistence — pydantic-ai-slim + mcp 2.0.0 | 1. Check pyproject.toml: `pydantic-ai-slim[anthropic,openai]>=1.90.0` in [ghost], `mcp>=2.0.0` in [dev]<br>2. `uv sync` succeeds<br>3. `python -c "import pydantic_ai, mcp; print(mcp.__version__)"` | uv sync no conflict. mcp version ≥ 2.0.0. pydantic_ai imports clean. |
| TC-009 | P1 | ModelRef safety — no key leak in serialization | `python -c "from ghoshell_moss.contracts.llms import ModelRef, ResolvedModel, ServiceConfig, ModelConfig; ..."` | model_dump_json() contains no api_key or base_url substring |

## Baseline — 2026-08-08

| Case ID | First Test | Fix | Final Result |
|---------|-----------|------|-------------|
| TC-001 | 44 passed ✓ | — | — |
| TC-002 | 15 passed ✓ | — | — |
| TC-003 | PASS ✓ | — | — |
| TC-004 | PASS ✓ | — | — |
| TC-005 | PASS ✓ (result:dict, cast:~1.8s, usage:406in/62out) | — | — |
| TC-006 | PASS ✓ (cast:~1.4s, usage:539in/72out) | — | — |
| TC-007 | PASS ✓ | — | — |
| TC-008 | PASS ✓ (no conflict, mcp 2.0.0 + pydantic-ai-slim coexist) | — | — |
| TC-009 | PASS ✓ (no 'sk-' or 'secret' in JSON) | — | — |

## Execution Notes

- TC-005/006 require a live API key — if env var is missing, these will fail with "env var ... is not set".
- The pydantic-ai-slim migration (2026-08-08) replaced `pydantic-ai>=1.90.0` (full bundle with mcp/evals/logfire/
  google/web extras) with `pydantic-ai-slim[anthropic,openai]>=1.90.0`. This removed the transitive
  fastmcp-slim → mcp<2.0 constraint, enabling independent mcp 2.0.0 upgrade.
- API surface in pydantic-ai-slim is identical: AnthropicModel, OpenAIChatModel, Agent, Agent.run — all in the
  slim base. No code changes needed.
- The full test suite (TC-001 + TC-002) runs in under 1s with no network dependency.
