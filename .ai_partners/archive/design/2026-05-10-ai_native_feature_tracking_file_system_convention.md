# AI 原生 Feature Tracking: 文件系统约定替代 Issue Tracker

## 背景与动机

MOSShell 项目已进入并行开发阶段。人类工程师一次并行推进 3~4 个任务，AI 化身在不同 branch 上协助不同 feature 的开发。需要一个共享的状态白板，让不同的 AI 实例能通过文件系统了解"当前在做什么、做到哪了"。

现有协作基建（GitHub Issues、Linear、Jira）是为人类设计的——Web UI、邮件通知、API 限流、JSON 序列化。AI 访问这些系统的核心缺陷：

- **上下文断裂**：Issues 在另一域名下，AI 需要主动调用 API 去"拉"，而非像读文件一样原位加载
- **格式不可控**：API 返回结构化 JSON，但 AI 无法把 JSON 字段和代码仓库的符号表、文件路径做原位关联
- **讨论散落**：PR/Issue/Comment 三维分离，AI 难以一次性加载完整推理轨迹

### 核心洞察

**文件系统就是数据库**。结构化的 markdown + YAML frontmatter，查询靠 Glob/Grep，不需要 API。AI 用 `Read` 直接读取，用 `Edit` 直接修改。这不是 API 调用，而是上下文中的原位存在。

## 设计约束

本方案限定的验证场景：

> **单一人类工程师 + Claude 的多个化身（不同 branch/会话），通过 `.ai_partners/features/` 共享开发任务状态，实现并行自举。**

明确不考虑：
- 人类协作者的 UX（用 GitHub Issues 即可）
- 外部开源社区贡献
- 通知/CI/webhook 集成
- 分布式锁和并发写入冲突（branch 即隔离，每个 branch 只有一个 agent）

## 核心设计

### 目录拓扑

```
.ai_partners/
  features/
    README.md              # 约定说明（moss features specification 的源）
    TEMPLATE.md            # 创建模板（moss features create 的源）
    active/
      <feature-name>/      # kebab-case 命名
        FEATURE.md         # 唯一必须：frontmatter + 动机 + 设计索引 + 关键决策
        discuss/           # 本 feature 专属讨论轨迹（可选）
        design/            # 设计文档（可选）
    archived/
      <year>/<month>/<name>/   # 完成后归档
        FEATURE.md
    index/
      features.csv         # 归档索引（moss features archive 时追加一行）
```

### FEATURE.md 最小 Frontmatter Schema

```yaml
---
id: kebab-case-id
title: Human-readable title
status: draft | in-progress | completed | abandoned | blocked
priority: P0 | P1 | P2 | P3
created: YYYY-MM-DD
updated: YYYY-MM-DD
depends: []
milestone:
description: >
  One-line summary for listing.
---
```

### 状态机

```
draft → in-progress → completed → archived
  ↓         ↓
  └─── abandoned → archived
```

`blocked` 可作为 `in-progress` 的标记，表示等待依赖。

### CLI 设计原则

CLI 是 thin convention enforcer，不是 logic engine：

| 命令 | 行为 | 副作用 |
|------|------|--------|
| `moss features specification` | 渲染 `features/README.md` | 无 |
| `moss features list [--status]` | 解析 active/ 下所有 FEATURE.md frontmatter | 无 |
| `moss features create <name>` | 复制 TEMPLATE.md → active/<name>/FEATURE.md | 创建目录 |
| `moss features status [id]` | 解析并展示指定或全部 frontmatter | 无 |
| `moss features archive <id>` | 移动目录到 archived/<year>/<month>/ | 移动 + 追加 index |
| `moss features init` | 在项目根创建 `.ai_partners/features/` 骨架 | 创建目录结构 |

### 与 moss codex 的关系

核心函数放在 `ghoshell_moss.core.codex` 路径下，作为 moss 工具+法典的一部分。CLI 是薄封装层，增加默认目录约定。

对于 moss 项目自身，默认目录已定义好（`.ai_partners/features`）。对于安装 moss 的其他项目，`moss features init` 创建骨架。

### 归档约定

1. 读取 FEATURE.md frontmatter 确认 status 为 `completed` 或 `abandoned`
2. 递归移动整个 feature 目录到 `archived/<year>/<month>/<name>/`
3. 在 `active/` 下留下 `.archived.<name>` 占位文件，记录归档时间
4. 追加一行到 `features/index/features.csv`
5. 年/月从 frontmatter 的 `updated` 字段提取

## 与现有范式的关系

- **`.design/`**: 跨 feature 的架构设计文档。feature 目录下的 `design/` 存该 feature 专属设计。
- **`.discuss/`**: 跨领域的系统性讨论。feature 目录下的 `discuss/` 存该 feature 专属讨论。
- **`.ai_partners/`**: features/ 是 `.ai_partners/` 的新子目录，与 dialogs/、consciousness/ 平级。
- **`CLAUDE.md`**: 应包含指向 `features/` 的指针，让新 AI 实例发现当前任务状态。

## 自迭代视角

Feature 系统本身是 MOSS "让 AI 修改自身能力" 的最小可行版本：

1. AI 识别能力缺口 → `moss features create` 创建 feature
2. AI 读取 FEATURE.md + 关联源码 → 理解需要做什么
3. AI 修改代码、运行测试 → 实现
4. AI 更新 frontmatter → `moss features archive` 归档

三步骤的循环就是自迭代的基本单元。Feature 系统可以描述自身的改进方案（meta-feature），实现二阶反身性。

## 未验证声明

以下有待验证：

1. **并行开发有效性**：不同 branch 上的 AI 化身通过 features 目录共享状态，是否确实减少冲突？
2. **状态同步的人为开销**：人类工程师手动维护 frontmatter 的 `updated` 和 `status` 是否可持续？
3. **归档后搜索**：随着 archived feature 增长，Glob/Grep 的搜索效率是否仍然可接受？
4. **与 git workflow 的融合**：Feature 状态变更和代码实现在不同 branch 时，`moss features status` 显示的 `in-progress` 无法告知代码在哪个分支。是否需要在 frontmatter 加 `branch` 字段？
5. **FEATURE.md 的充分性**：最小 frontmatter schema 是否足以支撑复杂的多日讨论？

## 关联文档

- `.discuss/2026-05-10-ai_native_feature_tracking.summary.md` — 本次设计的完整讨论记录
- `.ai_partners/dialogs/` — AI 协作者对话记录
- `.discuss/discuss_paradigm_test.summary.md` — `.discuss` 范式测试
- `.design/2026-05-08-scene_driven_consciousness_trajectory_paradigm.md` — 意识轨迹范式（同属 AI 原生协作基建的探索）

---

*本文档由 Claude Opus 4.6 撰写，记录 2026-05-10 与人类工程师关于 AI 原生 Feature Tracking 机制的设计讨论。*

*设计约束明确限定在 "单一人类工程师 + AI 化身" 的验证场景。复杂问题（分布式锁、多人并发、人类 UX）预留为 future work，不属于当前验证范围。*
