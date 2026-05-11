---
id: cli-all-commands-and-batch-help
title: CLI Self-Introspection — all-commands + batch help
status: completed
priority: P1
created: 2026-05-11
updated: 2026-05-11
depends: []
milestone: moss-self-bootstrap
description: >-
  Reduce AI's CLI discovery round trips from 40+ to 2: add `all-commands` for
  hierarchical tree view and extend `help` to accept multiple command paths.
  Additionally update CLAUDE.md to reflect the new tools.
---

# CLI Self-Introspection — all-commands + batch help

## Motivation

AI 当前使用 CLI 时需多轮交互才能发现完整命令树:

1. `moss --help` → 看到 9 个命令组
2. `moss codex --help` → 4 个子命令, `moss concepts --help` → 3 个子命令, ...
3. `moss codex get-interface --help` → 了解具体参数, ...

即使了解命令树结构就要 10 轮，理解所有子命令参数要 40+ 轮。这在 Claude Code 每次 tool call 都有开销的情况下极度低效。

核心问题: 标准 CLI help 系统是为人类设计的逐层探索模式，不适合 AI 的一次性上下文加载。

## Scope

- 实现 `moss all-commands` 命令，递归输出命令树，支持 `--depth` 和 `--group` 控制
- 扩展 `moss help` 命令，接受多个命令路径参数，一次性输出多条 help
- 两个命令均支持 `--ai` 输出纯文本
- 更新 `CLAUDE.md`，添加这两个命令的使用指南，优化 AI 的 CLI 发现流程

Out of scope:
- 改变现有命令的 help 输出格式 (保持标准 Typer/Click help)
- auto-completion 增强 (moss-cli 已有 Tab 补全)
- 交互式 help 浏览

## Design Index

- Discussion: `.discuss/` (TBD)

## Key Decisions

1. **`all-commands` default depth = 2**: 显示命令组 + 子命令名称，不展开参数详情。depth=3 才显示 option 列表。这是 AI 发现最常用的粒度。
2. **`all-commands --group` 支持子树聚焦**: `moss --ai all-commands --group codex` 只输出 codex 子树。
3. **`help` 变长参数按顺序解析**: `moss help codex get-interface concepts core` → codex 是命令组，get-interface 是它的子命令，concepts 是命令组，core 是它的子命令。找不到路径时优雅报错。
4. **`help` 无参数保持原行为**: 向后兼容，无参数时显示顶层 help。
5. **CLAUDE.md 更新作为 feature 的一部分**: 工具实现后 CLI 的 AI 使用流程变了，文档必须同步更新。

## Implementation Notes

**Commit**: 待提交

**Changed files**:
- `src/ghoshell_moss/cli/main.py` — 新增 `all-commands` 命令 + 扩展 `help` 支持多路径
  - `all-commands`: 递归遍历 Typer 的 `registered_groups` 和 `registered_commands`
  - `--depth 1/2/3` 控制深度 (深度 3 通过 `get_command()` + Click params 反射参数)
  - `--group <name>` 限定子树
  - `help`: `context_settings={"allow_extra_args": True}` 接受变长参数
  - 路径解析器按序匹配 group→command，匹配到终端命令后输出 help 并重置
  - 未知路径 `print_warning()` 不崩溃
  - `_is_hidden_*` 过滤 `DefaultPlaceholder` 的 hidden 组/命令
- `CLAUDE.md` — 在 "常用工具" 后新增 "CLI 命令发现" 章节
- `src/ghoshell_moss/cli/CLAUDE.md` — 补充 `all-commands` 和 `help` 自省命令文档

**验证结果** (全部通过):
1. `moss --ai all-commands --depth 1` → 9 个命令组 ✓
2. `moss --ai all-commands --depth 2` → 所有组 + 子命令 (约 30+ 行) ✓
3. `moss --ai all-commands --depth 3` → 包含参数信息 ✓
4. `moss --ai all-commands --group codex` → 仅 codex 子树 ✓
5. `moss --ai help codex get-interface` → 输出 get-interface help ✓
6. `moss --ai help codex get-interface concepts core` → 连续两条 help ✓
7. `moss --ai help nonexistent` → `[WARN]` 不崩溃 ✓
8. `moss --help` → 人类模式正常 (含颜色/TUI) ✓

## Related

- Related features: `codex-get-source-does-not-support-module-attr-syntax` (同为 CLI 改进)
