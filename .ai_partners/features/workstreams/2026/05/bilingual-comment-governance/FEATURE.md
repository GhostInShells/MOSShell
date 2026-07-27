---
title: Bilingual Comment Governance — 核心抽象层英文释义 + 注释治理
status: design-locked
priority: P1
created: 2026-05-29
updated: 2026-07-28
depends: []
milestone: beta-release
description: >-
  contracts / concepts / blueprint / channels / architecture.py 的 docstring
  英文释义（中文概念体系为锚点），过期内容清理，docstring vs comments 边界治理。
---

# Bilingual Comment Governance

> 为 Beta 正式发布做准备：核心抽象层 docstring 英文释义，过期内容清理，
> docstring vs comments 边界约定。

## Motivation

contracts、concepts、blueprint 是 MOSS 架构的抽象定义层，channels 是预制能力目录，
architecture.py 是认知地图。它们共同构成 `moss codex` 和运行时 code-as-prompt 的
自解释接口。

当前问题：

1. **大量中文 docstring 未英文化**。中文版本理解稳定，但非中文读者/模型无法直接消费。
   翻译不是机械双语并列，而是以中文概念体系为锚点做英文释义。
2. **部分描述已过期**。项目经过多轮重构，某些注释中的术语或描述不再准确。
3. **docstring 和 comments 边界模糊**。实现笔记、历史残留、临时 TODO 混入 docstring，
   而这些不是给运行时模型看的。

## Core Strategy

### 中文概念锚点，英文释义

中文概念体系是 MOSS 的认知底座，理解稳定。英文是**释义（paraphrase）**，不是替换。

- docstring 主力语言 → 英文
- 翻译方式 → 释义，不是直译。目标是英文读者/模型能无歧义理解
- 不对中文原句做逐行翻译，不保留双语并列格式

### 关键术语：英文词 + 中文 + gloss

当 docstring 中首次引入或定义承载 MOSS 架构语义的关键术语时，提供 gloss：

```
The Channel (经络, "meridian") is the module-like capability container —
commands flow through it like signals through a nerve pathway.
```

需要 gloss 的术语（枚举，非穷尽）：

- Logos / 道 — 模型输出的流式控制讯息，包含 CTML
- Channel / 经络 — 能力组织单元，树形嵌套
- Matrix / 矩阵 — 通讯网络在进程内的投影
- Mindflow / 心流 — 三循环全双工调度中枢
- Nucleus / 核 — 感知信号的加工与仲裁单元
- Impulse / 冲动 — 感知信号加工后的调度原语
- 反身性 / reflexivity — 系统通过 Channel 控制自身运行时状态的能力
- 双工 / duplex — 感知输入与躯体输出并行
- 三循环 / three-loop — 感知 / 思考 / 执行 的并行循环

普通技术术语不需要 gloss（如 "command", "interpreter", "runtime", "session"）。

### docstring vs comments 边界

同一份代码同时面向两类读者，需要区分信息放在哪里：

**docstring**（给运行时模型看，`get-interface` 反射输出）：

- 这是什么、怎么用
- 设计动机和架构哲学（code-as-prompt，模型需要理解"为什么"）
- 与其他抽象的关系

**`#` 注释**（给读源码的人和模型看，不出现在 `get-interface` 输出）：

- 实现细节、为什么这样实现
- 临时的 workaround、hack
- 历史上下文（"这个函数在 v2 重构时保留因为..."）
- TODO / FIXME（仍在有效期内的）
- 对代码模式的元评论（"反范式实现抽象可执行"）

**判断标准**：删掉这段文字后，模型通过 `get-interface` 还能不能正确使用这个类/函数？
不能 → 进 docstring。能 → 进注释。

### 过期内容清理

翻译过程中发现以下情况，直接处理：

1. **已解决的 TODO**：删除。如 `Priority` 的 "todo: 检查 python 3.10 是否支持"
2. **历史残留标记**：删除。如 `resource.py` 的"验证版，验证通过后覆盖回"
3. **仍在有效期内的 TODO**：从 docstring 移到 `#` 注释。如 `shell.py` 的
   `stop_interpretation` 临时实现标记
4. **与代码行为不一致的 docstring**：修正 docstring 使其匹配实际行为
5. **不确定是否过期的**：保留，加 `# NOTE(verify): ...` 注释标记

## Scope

| Package | Path | Files | 中文密度 |
|---------|------|-------|---------|
| contracts | `src/ghoshell_moss/contracts/` | 7 | 重度 |
| concepts | `src/ghoshell_moss/core/concepts/` | 7 | 重度 |
| blueprint | `src/ghoshell_moss/core/blueprint/` | 12 | 重度 |
| channels | `src/ghoshell_moss/channels/` | 10+ | 轻度（主要是注释） |
| architecture | `src/ghoshell_moss/architecture.py` | 1 | 轻度 |

### 优先级

contracts > concepts > blueprint > channels > architecture

contracts 是被最多模块引用的基础依赖。blueprint 的 matrix 和 mindflow
承载最重的设计哲学内容，翻译质量要求最高。

## Key Decisions

### 不做的事

- **不重构代码**。不修改接口签名、不调整逻辑、不重新组织模块结构。
- **不纯英文化**。中文注释中的设计讨论移至 `#` 注释保留，不删除。
- **不机械翻译**。不对每行中文做逐句英文对应。读原文，用英文重述。
- **不双语并列**。docstring 里不出现"中文原文 + 英文翻译"的叠放格式。
- **不在 docstring 里罗列 commands**。命令签名已由 interface 自动反射，
  手写重列必然随代码漂移。

### 翻译质量标准

- **准确性优先**：不能因为翻译而导致语义缺失或歧义。blueprint 和 concepts
  的 docstring 承载架构语义，翻译错误比不翻译更糟。
- **术语一致**：同一个中文概念在所有文件中用同一个英文词。
- **简洁**：不追求文学性。用项目已有的英文 docstring（如 channel_builder）
  作为风格参照。保留中文 docstring 中已有的英文内容（如 Field description
  的中英混杂部分，统一为英文）。
- **可验证**：翻译完一个文件后，跑 `moss codex get-interface <模块路径>`
  确认自解释输出正常且语义无损。

### 执行方式

- 8 月执行，等 memento agent 打磨好后用自己的 agent 做
- 每个 package 单独 commit
- 每完成一个文件，`get-interface` 验证

## Implementation Notes

- 发现明显的 bug 可以顺手修，但不要在这个 workstream 里做架构改动。
- 英文翻译风格参照：`channel_builder.py` 模块 docstring、`matrix_channel.py`
  模块 docstring。
- `__init__.py` 如果只有 re-export 没有实质注释，跳过。
- channels 目录模块 docstring 已采用机器可解析格式（`描述 | 类型 | status`），
  翻译时保留这个格式，只翻译描述部分。
