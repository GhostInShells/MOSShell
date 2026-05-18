---
created: 2026-05-18
depends: []
description: 为 docs/ai/ 写第一份架构拓扑文档，从动机到方法论到抽象，统领所有技术领域的综合理解。
milestone: null
priority: P2
status: in-progress
status_note: v2 draft written, human editing v3
title: Ai Docs Topology
updated: '2026-05-18'
---

# Ai Docs Topology

> Use `moss features set-status ai-docs-topology <status> -m "note"` to update state.
> See [TOPOLOGY.md](TOPOLOGY.md) for directory layout and [README.md](README.md) for the full convention.

## Motivation

`moss docs` CLI 已由 deepseek-v4-pro 搭好骨架（`src/ghoshell_moss/cli/docs/`，三路径 en/zh/ai），但三个子目录全空。
这是第一批 AI 文档 —— 从架构拓扑入手，让后续 AI 实例能快速建立全局认知。

不是写人类文档，是为 AI 写高密度上下文。目标是一份文档统领所有技术领域进行综合理解。

## Design Index

- Key design documents: `design/`
- Key discussion records: `discuss/2026-05-18-architecture-topology-alignment.md` — 人类与 AI 对架构拓扑的完整对齐论述

## Key Decisions

1. **文档结构 = 动机 → 方法论 → 抽象框架**。不是静态描述"有什么"，而是动态展示"为什么走到这里 → 用什么方法解决 → 抽象出什么"。拓扑不是组件图，是从问题到方案的推演路径。

2. **写作流程**：人类先讲概念提纲 → AI 探索代码验证 → 对齐共识 → 最后才写。写之前不擅自产出。

3. **写作风格**：遵循 code as prompt 原则，保留探索痕迹。AI 文档面向 AI 读者，高密度但不断裂。

## Implementation Notes

<!-- Gotchas, non-obvious behaviors, reasons for rejecting simpler alternatives. -->