---
title: Dolores Ghost Regression
version: 1
status: draft
priority: P1
created: 2026-09-01
updated: 2026-09-01
scope: subsystem
depends: [ghost-prototype-dolores]
description: >-
  Verify Dolores ghost moment mapping (context/inputs split), yield loop
  (wait_next_moment), thinking exit cancel classification, and interrupt.
---

# Dolores Ghost Regression

## Exploration Index

```
moss codex blueprint ghost
moss codex get-interface ghoshell_moss.ghosts.dolores._ego:DoloresEgo
moss codex get-interface ghoshell_moss.core.blueprint.moment:Moment
moss ghosts list
moss ghosts show moss
moss-ghost run --help
moss-ghost send --help
git log --oneline -10 -- src/ghoshell_moss/ghosts/dolores/
```

## Methodology

Human-in-the-loop. Human starts the ghost and operates the web UI for interrupt
tests; model drives `moss-ghost send` and observes the logos / moment stream.

Two execution surfaces:
- **Unit**: `pytest` covers the python-side mapping (moment → context/inputs) and
  yield-tool recognition. Run first — it is the fast, deterministic gate.
- **Live**: `moss-ghost run` + `send` exercises the cross-language path (dsh plugin
  inject/steer, yield lock, exit cancel). Run in case order; TC-003 depends on TC-002.

## Prerequisites

- Full-clone MOSS workspace with `uv sync --active --all-extras`
- `.moss` workspace initialized in the repo root
- LLM provider configured (`.env`) — the ghost launches dsh at `127.0.0.1:3083`
- A web UI reachable against the dsh session (for the interrupt case)

## Test Cases

| Case ID | Priority | Description | Test Steps | Expected Result |
|---------|----------|-------------|------------|----------------|
| TC-001  | P0       | Python 侧单测全通过 | 1. `.venv/bin/python -m pytest src/ghoshell_moss/ghosts/dolores/ tests/ghoshell_moss/default/core/mindflow/test_mindflow.py tests/ghoshell_moss/default/core/blueprint/test_moment.py` | 全通过，无失败 |
| TC-002  | P0       | moment 拆两条：context inject + inputs steer | 1. `.venv/bin/moss-ghost run moss --surface output`<br>2. `.venv/bin/moss-ghost send --ghost moss "请描述你现在看到的上下文"`<br>3. 观察模型回复 | 模型分层感知：背景（echoes/dynamic，inject 注入）+ 输入（percepts，steer 注入）；无 percept 内容不进入 context |
| TC-003  | P0       | yield 闭环 + resolve "ok" | 1. send "请调用 wait_next_moment 等待下一帧" → 模型 yield（无 logos）<br>2. 再 send "继续" → 触发解锁<br>3. 观察模型回复 | 模型调 wait_next_moment 后 yield；新帧到达解锁，tool 返回 "ok"（str），模型继续 |
| TC-004  | P0       | thinking exit 分类 | 1. yield 场景：模型调 wait_next_moment 后 MOSS break → exit 不 cancel（tool 保持 pending）<br>2. 非 yield 场景：模型仍在跑时 exit → cancel（interrupt） | yield 时 tool 保持 pending；非 yield 时 agent 被 cancel（不空跑失速） |
| TC-005  | P1       | session.cancel 中断 yield tool | 1. send 触发 yield（tool 阻塞）<br>2. 界面点"停止生成"（session.cancel）<br>3. 观察 tool 状态 + 会话是否可继续 | yield tool 被 abort（走 dsh 默认，与其它 tool 一致）；会话不崩，可继续输入 |

## Execution Notes

- **yield 判定在 MOSS 侧**：`_is_yield_tool_call`（`_runtime.py`）识别 `tool/call == wait_next_moment` 即 break 收线，`run.yielded` 置位 → `exit_thinking(yielded=True)`。dsh 侧不猜 `pendingYield`（曾是竞态源）。
- **moment 映射在 python 侧**：`DoloresEgo._context_message`（= `as_moment_message(with_percepts=False, with_hint=False)`，inject）与 `_inputs_message`（percepts 平铺 + optional `<hint>`，steer）。plugin 只收两条现成 content，不组装 moss message。
- **yield tool 返回值是 str**：正常解锁 `resolve("ok")`；被 cancel 走 dsh 默认 abort（reject → error），与其它 tool 一致，不做特殊处理。
- **调试 session 状态**：`curl -s -X POST http://127.0.0.1:3083/api/session.list -H 'Content-Type: application/json' -d '{"type":"client-request","rpcId":"rpc-list-1","method":"session.list","payload":{}}'` 看 `projections.values.sessionStats`（turns/steps）。
- **已知现象**：yield 时 send 命令退出但无 logos 输出（模型只发 tool call、无文本），属正常，不是失败。
