---
name: moss-skills
description: MOSS skills 治理——复合任务的行动导向技能。写前必答入口判定三问。
---

# MOSS Skills

**"该用时有"**——复合任务的可交付技能，不是文档。skill 是行动导向
（可执行脚本 / 编排知识），模型在任务开始时刻经 `moss skills recall`
主动获得，不是模型"知道自己该问"时才会去读的参考。

## 定位

MOSS 项目的自解释是分层的，skills 补的是**交付层**：

| 层 | 承担 | 例子 |
|---|---|---|
| L0 code | 项目第一自解释原则 | 源码本身 |
| L1 CLI-flow | 认知地图 | `moss codex` / `moss --ai all-commands` / `moss ctml` |
| L2 目录 README | 目录承担什么 | `<subdir>/README.md` |
| L3 docs | 对外的系统化陈述 | `moss docs read` |
| **skills** | **复合任务的行动入口** | **本目录** |

skill 存在的合法性只有一种：**上述四层都覆盖不了，但它是一个真实的复合任务入口，
且模型在任务开始时需要被主动交付**。skill 是单元，recall 是交付。

## 入口判定 · 写前必答三问

不能全部答"是"的，**不写**。

1. **是复合行动吗？** 需要跨多个组件/系统协作、有明确的可执行编排，不是单一命令或单一接口能覆盖的。
2. **模型会在任务开始时需要它吗？** skill 的价值在"该用时有"——如果只是文档查询需求，
   `moss docs` 已经覆盖，不需要做成 skill。
3. **入口路径半年内稳定吗？** 依赖的抽象是否已经过实战、不在活跃演进中？不稳定的领域写了就是自制过期源。

## 反模式（已被历史验证）

- ✗ **组件顺带写 skill**——组件的 interface 已经是 prompt，抄一遍就是自制 stale。
- ✗ **操作步骤级 skill**——CLI 帮助 + `moss --ai all-commands` 已经能自解释操作。
- ✗ **决策/架构讨论**——那属于 `.design/` / `.discuss/` / `features/`，不属于 skill。
- ✗ **接口用法说明**——`moss codex get-interface` 已经是权威且不过期的来源。
- ✗ **复制文档当 skill**——skill 是行动导向，正文指向可执行编排；纯阅读内容留在 docs。

## 写作纪律

一旦通过判定三问，动笔时：

- **YAML frontmatter 必需**——`name` + `description`（跨工具 SKILL.md 标准）。
  `description` 是 `moss skills recall` 的语义召回信号，写"模型在什么任务下需要这个技能"。
- **只引 interface，不复制**——`moss codex get-interface <modulepath>` 是权威来源，不抄进正文
- **不硬编码具体值**——枚举成员、CLI 命令字面量、CTML 命令字面量都会变；引导读者用 codex 反射拿最新的
- **描述"复合行为的形状"**——独特价值是编排知识（先做什么、再做什么、组件间怎么协作），不是接口目录
- **可执行脚本放目录内**——`<name>/SKILL.md` 同目录可承载脚本/引用文件，SKILL.md 是入口

## 交付方式

skill 的主动来自**标准放置 + recall 交付**，不与 harness 抢：

- 本目录 `*/SKILL.md` 由 `moss skills list` 发现（glob+frontmatter 索引）
- `moss skills recall <query>` 语义召回（LLMFuncs 多标签分类，需 LLM 配置）
- 模型在任务开始时刻主动 `moss skills hint` / `recall`——**开始任务时先 `moss skills recall <你的任务>`**
- 硬约束：MOSS 碰不到 coding agent 的上下文（归 harness 管），所以不自作主张注入，
  skills 放标准位置，harness 自己亮给模型

## 命令

```bash
moss skills list                # 发现技能
moss skills list -q keyword     # 关键词过滤
moss skills recall <query>      # 语义召回（需 LLM 配置）
moss skills list --root <path>  # 探索任意目录的 skills
```

## 现存技能

由 `moss skills list` 动态生成，本文不硬编码——硬编码目录列表本身就是过期源。
