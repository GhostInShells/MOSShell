---
id: eval-debug
title: Moss Eval — Live Runtime Code Evaluation for AI Debugging
status: draft
priority: P1
created: 2026-05-12
updated: 2026-05-12
depends: []
milestone:
description: >-
  CLI tool that executes arbitrary Python code in the live MOSS runtime,
  returning captured stdout and return value as structured JSON.
  Uses Compiler+Executor infrastructure already in place.
---

# Moss Eval — Live Runtime Code Evaluation

## Motivation

AI 协作者在调试 MOSS 运行时时，当前只能通过 `python -c` 写单行表达式，
无法访问运行中的 Host、Matrix、Cell、Channel 等对象状态。
当调试对象变成 Ghost 运行时内部状态、多进程通讯时序、async 协程调度时，
需要一个能写真正多行代码、注入运行时对象、并捕获输出的工具。

基础设施 (`Compiler` + `Executor`) 已就绪，CLI 包装成本很低。

## Scope

### In Scope

- `moss eval <code>` — 执行一段 Python 代码，返回 JSON `{returns, std_output}`
- `moss eval --module <import_path> <code>` — 在指定模块上下文中执行（共享类型和 import）
- `moss eval --file <path>` — 从文件读取代码执行
- `--help` 输出作为 AI 的唯一使用说明（不另设 instruction 命令）
- 代码在隔离的临时 Module 中执行，不污染真实模块
- `__result__` 变量作为结构化返回值
- `print()` 输出自动捕获到 `std_output`

### Out of Scope

- 暴露 Executor 的 `--func-name`、`--with-local-args` 等参数（保持 CLI 最简）
- REPL 模式（已有 `moss-repl`）
- 持久化执行历史
- 代码补全或语法检查

## Key Decisions

1. **不强制 `def main()`**。代码形态由调试场景决定——有时一行 `print(type(host))` 就够。
   约定 `__result__` 作为返回值，`print()` 自动捕获。两个可同时用。

2. **协议用 `--help`，不搞隐式 instruction**。`--help` 输出控制在 10 行以内，
   AI 每次调用看一眼开销可忽略。不需要先 `moss eval instruction` 再 `moss eval <code>` 的两步协议。

3. **`--module` 解决"二次反射"问题**。AI 已经读过某模块源码后，
   直接在那模块上下文中执行代码，import 和类型现成，不需要重新反射。

4. **async 不需要特殊处理**。模型知道用 `async def` 封装 + `asyncio.run()` 即可。

## Implementation Notes

- CLI 入口：`src/ghoshell_moss/cli/eval_cli.py`
- 注册到 `main.py` 的 `eval` 命令组
- 核心依赖：`ghoshell_moss.core.codex.compiler:Compiler` + `executor:Executor`
- `--module` 参数的实现：通过 `importlib.import_module` 加载，作为 `Compiler(origin=module)` 传入
- 输出格式：`{"returns": <value>, "std_output": "<captured stdout>"}`，repr 不可序列化的对象

## Related

- Depends on: (none — Compiler + Executor already in `core/codex/`)
- Related: `moss-repl` (human-oriented interactive debug), `moss-as-mcp` (MCP debug channel)
