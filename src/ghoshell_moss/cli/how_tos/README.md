---
title: MOSS How-Tos
description: 复合任务入口——师傅领进门。不是操作手册，不写组件用法。写前必答入口判定三问。
---

# MOSS How-Tos

**"师傅领进门"**——复合任务的入口索引，不是 manual。

## 定位

MOSS 项目的自解释是分层的：

| 层 | 承担 | 例子 |
|---|---|---|
| L0 code | 项目第一自解释原则 | 源码本身 |
| L1 CLI-flow | 认知地图 | `moss codex` / `moss --ai all-commands` / `moss ctml` |
| L2 目录 README | 目录承担什么 | `<subdir>/README.md` |
| L3 docs | 对外的系统化陈述 | `moss docs read` |
| **howtos** | **复合任务入口** | **本目录** |

howto 存在的合法性只有一种：**上述四层都覆盖不了，但它是 MOSS 项目内一个真实的复合任务入口**。

## 入口判定 · 写前必答三问

不能全部答"是"的，**不写**。

1. **复合任务吗？** 需要跨多个组件/系统协作，不是单一命令或单一接口能覆盖的。
2. **CLI/codex 覆盖不了吗？** `moss codex get-interface` + `moss --ai all-commands` 组合为什么不够？说不出理由就说明够了。
3. **入口路径半年内稳定吗？** 依赖的抽象是否已经过实战、不在活跃演进中？不稳定的领域写了就是自制过期源。

## 反模式（已被历史验证）

- ✗ **新组件顺带写 howto**——组件的 interface 已经是 prompt，抄一遍就是自制 stale。
- ✗ **操作步骤级 howto**——CLI 帮助 + `moss --ai all-commands` 已经能自解释操作。
- ✗ **决策/架构讨论**——那属于 `.design/` / `.discuss/` / `features/`，不属于 howto。
- ✗ **接口用法说明**——`moss codex get-interface` 已经是权威且不过期的来源。

## 写作纪律

一旦通过判定三问，动笔时：

- **只引 interface，不复制**——`moss codex get-interface <modulepath>` 是权威来源，不抄进正文
- **不硬编码具体值**——枚举成员、CLI 命令字面量、CTML 命令字面量都会变；引导读者用 codex 反射拿最新的
- **描述"复合行为的形状"**——独特价值是编排知识（先做什么、再做什么、组件间怎么协作），不是接口目录
- **YAML frontmatter 必需**——`title` + `description`。description 是 `moss howtos recall` 的语义召回信号，写"读者在什么场景下需要这篇"

## 命令

```bash
moss howtos list              # 列表
moss howtos list -q keyword   # 关键词过滤
moss howtos read <path>       # 读一篇
moss howtos recall <query>    # 语义召回（需 ANTHROPIC_SMALL_FAST_MODEL）
```

## 现存文档

由 `moss howtos list` 动态生成，本文不硬编码——硬编码目录列表本身就是过期源。
