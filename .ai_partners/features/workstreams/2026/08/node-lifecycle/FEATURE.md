---
title: Node Lifecycle — 身份、入口、验证与记忆
status: in-progress
priority: P1
created: 2026-08-04
updated: 2026-08-14
depends: []
milestone: 0.1.0
description: >-
  Node 生命周期治理，从 node-migration 独立。重心已从"四层治理方案"
  转向 node 最佳实践探索：四地址发现组合、一次性 node、event 分级。
---

# Node Lifecycle

> 人类架构师 + claude-opus-4-7 + deepseek 家族。node 生命周期治理 workstream。

## Motivation

node 的就绪状态没有进入管理：`.installed` marker 只回答"装没装过"，不回答"环境现在
能不能用"；启动失败只在 stderr 和 bounded FIFO 里，不进模型上下文。需要一个覆盖生命
周期的治理链。早期方案是"身份 → 入口 → 验证 → 记忆"四层，随后演进为按需生长的
最佳实践探索（见 Compaction Note）。

## Compaction Note (2026-08-14)

历史决策 1–10.x 与三轮调研增补（启动成本实测、事件分级、一次性 node）已折叠，详细
轨迹在 git log：

```
git log -- .ai_partners/features/workstreams/2026/08/node-lifecycle/FEATURE.md
```

| commit | 主题 |
|---|---|
| `9f4ecbd9` | 初始四层治理：identity / entry / probe / ghost memory |
| `9c636408` | 启动成本调研 + 决策 5–8（砍 zenoh 否、anthropic import、事件分级）|
| `98c2cf5a` | 事件分级 + MatrixOperator 方向（决策 9）|
| `19a42de8` | 一次性 node 角色 + event_level gating（决策 10.x）|

已落地（代码即真相）：event_level 五档 + persist 字段（`19a42de8`）、images.py 去
anthropic import（`9c636408`）、project IoC 拆分（`d107dc05`）。

## Current Consensus

### 四地址组合 — node 发现路径前缀

node 发现路径有四个语义锚，对应四个确认方：

| 前缀 | 解析到 | 确认方 |
|---|---|---|
| （无前缀）| `project_dir` | 使用者 |
| `$MOSS_WORKSPACE` | `workspace_path` | 管理者 |
| `$MODE` | `workspace/modes/<mode_name>` | mode 开发 |
| `$GHOST` | `workspace/ghosts/<ghost_name>` | ghost 自己 |

默认 `node_paths` 扩展为四组合。普通使用者无需看懂组合语义，能力就位、自现。

### Matrix.new 默认 persist=False

- `persist` 参数进 `Matrix.new` 表面，默认 `False`（脚本启动式 = 一次性 run-to-completion）。
- `event_level` 不暴露，由 `persist` 推导（persist=false → DEBUG 静默）。
- 主动 `publish_event` 的级别语义见 Open Questions。

### 决策 4 纠正 — 非独立 memory 契约

原 `NodesMemoryContract`（store/load/forget interface 级契约）是模型误读，非人类设计。
node 级记忆的正确落点是 **skills 声明式约定 + ground 认知**（g1 体系已有），不是独立契约。

### node id / probe 降级为候选

`.node_id` 身份锚、probe 启动前闸口，从"决策"降级为"待评估候选"。落地前需重新论证价值。

## Open Questions

- **publish_event 级别**：persist=false 脚本主动 `publish_event` 目前被 cell.event_level
  锁死（`zenoh_presence.py:177` 硬编码继承），"默认静默但能喊"做不到。是否加显式
  event_level 覆盖参数，待定。先不加。
