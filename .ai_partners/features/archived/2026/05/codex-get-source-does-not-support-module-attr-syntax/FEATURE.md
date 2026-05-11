---
created: 2026-05-11
depends: []
description: moss codex get-source 和 get-interface 对 module:attr 格式传参时，将冒号误当作模块路径的一部分，导致
  import 失败。
id: codex-get-source-does-not-support-module-attr-syntax
milestone: null
priority: P2
status: completed
status_note: 'Fixed in 05cf633: get-source now uses import_from_path() to parse module:attr
  syntax.'
title: codex get-source/get-interface 不支持 module:attr 语法
updated: '2026-05-11'
---

# Codex Get Source Does Not Support Module Attr Syntax

## Motivation

`moss codex get-source` 和 `moss codex get-interface` 的 CLI 参数约定是 `modulepath:attr`（参考 CLAUDE.md 中的示例：`moss codex get-interface ghoshell_container:IoCContainer`）。但当传入 `ghoshell_moss.core.concepts.channel:ChannelCtx` 时，实际报错：

```
No module named 'ghoshell_moss.core.concepts.channel:ChannelCtx'
```

说明 CLI 没有正确解析 `:` 分隔符，直接把 `module:attr` 整个字符串当成了模块路径去 import。需要检查 `codex_cli.py` 中的参数解析逻辑。

### 影响范围

同样影响不带 `--ai` 的人类模式。`get-source` 的 help 里写的是 `MODULEPATH`，没有提示 `:attr` 语法，但 `get-interface` 文档里明确提到了 `modulepath:attr` 用法。

### 正常工作的命令

- `moss --ai codex list ghoshell_moss.core.concepts.channel` — 可以列出模块成员（不含 `:` 分隔符，不受影响）

## Scope

- 修复 `get-source` 的 `module:attr` 参数解析
- 修复 `get-interface` 同样可能存在的问题
- 更新 help 文本，明确说明 `modulepath[:attr]` 用法

## Design Index

- Key design documents: `design/`
- Key discussion records: `discuss/`

## Key Decisions

*（待实现时填写）*

## Implementation Notes

涉及文件：
- `src/ghoshell_moss/cli/codex_cli.py` — get-source 和 get-interface 两个 typer command

## Related

- Depends on: (list feature IDs)
- Related features: (list feature IDs)