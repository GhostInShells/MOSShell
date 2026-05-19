---
created: 2026-05-18
depends: []
description: 为 docs/ai/ 写第一份架构拓扑文档，从动机到方法论到抽象，统领所有技术领域的综合理解。
milestone: null
priority: P2
status: completed
status_note: v5 finalized, 结构稳定，后续按需演进
title: Ai Docs Topology
updated: '2026-05-19'
---

# Ai Docs Topology

> Use `moss features set-status ai-docs-topology <status> -m "note"` to update state.
> See [TOPOLOGY.md](TOPOLOGY.md) for directory layout and [README.md](README.md) for the full convention.

## Motivation

`moss docs` CLI 已由 deepseek-v4-pro 搭好骨架（`src/ghoshell_moss/cli/docs/`，三路径 en/zh/ai），但三个子目录全空。
这是第一批 AI 文档 —— 从架构拓扑入手，让后续 AI 实例能快速建立全局认知。

不是写人类文档，是为 AI 写高密度上下文。目标是一份文档统领所有技术领域进行综合理解。

## Design Index

- Preserved draft: `architecture-topology-v2.md` — 二稿快照，供未来回溯拓扑演进轨迹
- Key discussion records:
  - `discuss/2026-05-18-architecture-topology-alignment.md` — 人类与 AI 对架构拓扑的完整对齐论述
  - `discuss/2026-05-18-draft-review.md` — 人类审阅二稿的八个改进方向 + CLI 重构影响

## Key Decisions

1. **文档结构 = 动机 → 方法论 → 抽象框架**。不是静态描述"有什么"，而是动态展示"为什么走到这里 → 用什么方法解决 → 抽象出什么"。拓扑不是组件图，是从问题到方案的推演路径。

2. **写作流程**：人类先讲概念提纲 → AI 探索代码验证 → 对齐共识 → 最后才写。写之前不擅自产出。

3. **写作风格**：遵循 code as prompt 原则，保留探索痕迹。AI 文档面向 AI 读者，高密度但不断裂。

4. **方法论不独立成章**（v3 引入）：六条方法论溶入各层描述。拓扑文档的核心是"结构即理论"——方法论会演进，拓扑位置是稳定的。

5. **每层统一模板**（v3 引入）：拓扑定位 → 核心描述 → 行业对标 → 知识入口/引用。八层同一格式，AI 扫描式阅读。

6. **引用面 = Python 包路径**（v2→v3 修正）：CLI 命令名随目录结构调整，Python import path 是稳定面。知识入口用文件路径 + `moss codex` 工具入口。

7. **CTML → Logos 重定位**（v2→v3 修正）：CTML 是 Logos 的一种 XML 流式解析实现。Logos 对标 Function Call，但语义是"边说边做"而非"描述后等结果"。

## Implementation Notes

- 文档位置：`src/ghoshell_moss/cli/docs/ai/architecture-topology.md`
- v2 快照备份在 feature 目录：`architecture-topology-v2.md`
- 堆积木图用 markdown 表格实现，后续可用实际图形替换

<!-- Gotchas, non-obvious behaviors, reasons for rejecting simpler alternatives. -->