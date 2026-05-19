---
created: 2026-05-19
depends: []
description: Move moss eval into the codex command group and rewrite its help text
  for AI consumption, making runtime code execution discoverable alongside other introspection
  tools.
priority: P2
status: completed
status_note: moved to codex group, docstring rewritten for AI consumption
title: Move eval to codex group with AI-oriented help text
updated: '2026-05-19'
---

# Move eval to codex group with AI-oriented help text

## Motivation

`moss eval` 是 AI 在 live runtime 中执行代码、探查对象状态的工具，和 `codex get-interface`、
`codex get-source`、`codex info`、`codex list` 同属运行时自省工具集。但它目前作为根级命令存在，
不在 `codex` 组中，AI 通过 `all-commands` 发现时无法自然归类。

当前 help 文本对 AI 来说信息密度不够 — 讲了执行代码的机制，但没说输出格式（JSON schema）、
什么时候用它 vs `get-interface`/`get-source`、以及 `--module` 对 AI 二次调试的真正价值。

## Key Decisions

1. **eval 进入 codex 组，而非新建组或保持根级**。eval 本质上就是"用代码探查运行时"，
   和"用反射探查代码"是同一光谱的不同精度。放在一起降低 AI 发现成本。

2. **docstring 重写，面向 AI**。当前 docstring 是人机两用的折中。AI 需要知道：
   JSON 输出格式 `{returns, std_output}`、`--module` 适合"二次反射"场景、
   以及和 `get-interface`/`get-source` 的定位差异。

3. **_eval_child.py 不动**。子进程入口独立于 CLI 注册位置，无需变更。