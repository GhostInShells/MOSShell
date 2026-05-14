---
title: First Ghost Prototype
status: design_done
priority: P0
created: 2026-05-14
updated: 2026-05-14T19:00
step: 7
depends: []
milestone:
description: >-
  从零开发第一个完整的 Ghost 原型——将 Ghost/Mindflow 抽象转化为可运行的智能体实现，打通 "感知→思考→执行" 三循环。
---

# First Ghost Prototype

> 入口文档。新的 AI 实例读这个即可恢复上下文，然后按索引进入子文档。

## 这是什么

Ghost 是 MOSShell 最核心的模块——持久化智能体的运行时。ABC 已定义，Mindflow 调度链路已验证，但不存在完整可运行的 Ghost 实现。本 workstream 从零构建第一个原型。

## 文件夹结构

```
first-ghost-prototype/
├── FEATURE.md              # 本文件：状态追踪 + 子文档索引
├── DESIGN.md               # 设计结论汇总（随讨论推进逐步填入）
├── TASKS.md                # 任务分解 + 当前进度 + commit 记录
└── discuss/                # 每步讨论的完整记录
    ├── 01-ghost-abc-positioning.md
    ├── 02-minimal-prototype-goals.md
    ├── 03-infrastructure-preparation.md
    ├── 04-runtime-integration.md
    ├── 05-matrix-tui-integration.md
    ├── 06-full-link-testing.md
    └── 07-documentation.md
```

**三个顶层文件的分工**：

| 文件 | 角色 |
|------|------|
| `FEATURE.md`（本文件） | 入口 + 状态 + 索引。不存详细内容，只存指针。 |
| `DESIGN.md` | 最终设计结论。每个讨论步骤结束后更新。精简、声明式。 |
| `TASKS.md` | 实施追踪。任务分解、依赖、进度、已完成 commit。跨 session 追踪。 |
| `discuss/` | 每步讨论的现场记录。保留决策理由和权衡过程。 |

## 推进方法论

**模式**：人类引导讨论 → AI 记录决策 → 生成子文档 → FEATURE.md 更新索引。

**纪律**：
1. 一步一文档。完成一个讨论步骤后，才进入下一步。
2. 不提前调研。避免上下文窗口膨胀和压缩丢失。
3. 每个 discuss 文档记录：背景、讨论要点、决策结论、保留的锚点对话。
4. DESIGN.md 在每个讨论步骤结束后更新，只记录"结论"不记录"过程"。
5. FEATURE.md 在每个讨论步骤结束后更新 `updated` 日期和状态。

## 推进步骤

| # | 讨论主题 | 产出 | 状态 |
|---|---------|------|------|
| 1 | Ghost ABC 重新审视 + GhostFactory 定位 | `discuss/01-ghost-abc-positioning.md` | done |
| 2 | 最小原型技术目标 | `discuss/02-minimal-prototype-goals.md` | done |
| 3 | 基建依赖准备 | `discuss/03-infrastructure-preparation.md` | done |
| 4 | Ghost runtime 集成方式 | `discuss/04-runtime-integration.md` | done |
| 5 | Matrix / TUI 集成 | `discuss/05-matrix-tui-integration.md` | done |
| 6 | 全链路测试方案 | `discuss/06-full-link-testing.md` | done |
| 7 | 文档准备 | `discuss/07-documentation.md` | done |

## 认知准备（已完成 2026-05-14）

首轮探索确定了以下认知基线：

1. **测试体系**：`tests/` 下 ~70 个测试文件。Ghost 零测试。Mindflow 有三个测试文件覆盖三循环调度。
2. **CTML**：流式命令标记语言。Code as Prompt、时间第一公民、树形 Channel、结构化并发。
3. **Mindflow ABC**：三循环（感知/思考/执行）全双工调度中枢。Signal → Nucleus → Impulse → Attention → Articulator/Action。
4. **Ghost ABC**：`system_prompt()` + `memories()` + `nuclei()` + `articulate()` + `channel()`。核心是 `articulate()`。
5. **Mindflow 测试**：`BaseMindflow` / `BaseAttention` / `BufferNucleus` 已通过测试验证三循环链路可运行。

### 架构关系速览

```
Signal (感知信号) → Nucleus (加工/降频) → Impulse (动机)
                                              ↓
Mindflow (调度中枢) → Attention (单次运行态)
                                              ↓
                        Articulator (推理) + Action (执行)
                                              ↑
                              Ghost.articulate() 提供 Logos
```

---

*本文件在每个讨论步骤完成后更新。新 AI 实例：读此文件 → 检查 TASKS.md 了解进度 → 进入当前讨论步骤。*
