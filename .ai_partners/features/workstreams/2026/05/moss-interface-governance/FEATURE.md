---
title: MOSS Interface Governance — 抽象面自解释 + 英文释义 + 注释分层
status: in-progress
priority: P1
created: 2026-05-29
updated: 2026-09-12
depends: []
milestone: beta-release
description: >-
  抽象层（contracts / concepts / blueprint / channels / architecture）的接口表面治理：
  自解释、可脱离源码使用、作为实现索引；清除对 FEATURE.md / workstream / 决策编号的耦合；
  过期内容修复；中文 docstring 英文释义；docstring vs comments 边界。
---

# MOSS Interface Governance

> **前身 `bilingual-comment-governance`**（2026-05-29 创建，2026-09-12 改名）。
> 改名的原因：原定位（多语种翻译）只是现在更大工作面里的一个子任务。
> 保留此别名，因为更早的 commit message 用旧名引用本 workstream，
> 反向索引 `git log -- <file>` → FEATURE.md 需要仍然可解析。

## Motivation

contracts、concepts、blueprint、channels 是 MOSS 架构的抽象定义层，
architecture.py 是认知地图。它们共同构成 `moss codex` 与运行时 code-as-prompt 的
自解释接口。

原 workstream（多语种翻译）发现的三个问题依然成立：

1. **大量中文 docstring 未英文化**。中文概念体系理解稳定，但非中文读者/模型无法直接消费。
2. **部分描述已过期**。项目经过多轮重构，注释中的术语或描述不再准确。
3. **docstring 和 comments 边界模糊**。实现笔记、历史残留混入 docstring。

### 触发改名的新问题：抽象面耦合了 feature 文档

2026-09-12 审计 blueprint 时发现：抽象面里存在对 **FEATURE.md / workstream / 决策编号**
的直接引用，例如 `见 warrant FEATURE.md KD14`、`决策 12/13`、
`见 workstream channel-meta-dyn-static`（该 workstream 已不存在）。

这违反 features specification 的核心断言：**"Not authoritative over code. Code wins."**
FEATURE.md 是**反向索引**——索引方向是 `git log` → FEATURE.md。源码 → FEATURE.md
是反向指，等于让权威表面依赖一份会被压缩、会过期、且不发版的文档。

三条硬事实：

- `pyproject.toml:73` 打包排除 `.discuss*` / `.design` / `.memory`，`.ai_partners/`
  不在 `src` 下发布。**PyPI 用户拿到的抽象面上，"见 xxx FEATURE.md" 是死链。**
- 对模型是一次无效探索：`moss codex get-interface` 会反射 docstring，
  模型看到"见 xxx FEATURE"会去找一个它无法解析的编号。
- 悬空引用会静默腐烂：`channel-meta-dyn-static` 已经不存在，引用仍在。

### 与 doc-governance 的关系

`doc-governance`（2026-07，completed）已经诊断过同一病根并给出原则：
**"过期是因为文档跨越了抽象边界"**，正确做法是"文档只描述概念关系，不列举具体值"。
当时治理对象是 `docs/`。本 workstream 把同一原则推进到 **blueprint 的 docstring**——
抽象面上的源码注释。

## Core Strategy

### Code as Prompt 纪律

抽象面的 docstring 是**运行时模型与模型开发者共同消费的权威表面**。四条纪律：

1. **可读的自解释权威表面**。抽象面自身完整，不依赖外部文档即可读懂。
2. **不读实现源码即可使用**（尤其配合 IoC）。使用者模型的第一入口应是抽象面，
   不是实现源码。代表案例：`channel_builder`。
3. **作为阅读相关源码实现的索引**。抽象面负责"这里有什么、去哪找"，
   实现细节由抽象面指向实现，而不是复制进抽象面。
4. **装线 / 控制流在表面自解释**；暴露出来的控制流代码可以在实现里替换。
   表面承诺控制流形状，实现承诺可替换性。

推论：**决策历史不属于抽象面**。抽象面陈述"是什么、怎么用、契约是什么"；
"为什么这么定、当时否掉了什么、进度如何" 属于 FEATURE.md 与 git log。

### docstring vs comments 边界

同一份代码面向两类读者，区分信息归属：

**docstring**（`get-interface` 反射输出，模型与开发者可见）：

- 这是什么、怎么用
- 使用所必需的契约：不变量、参数语义、负向约束（"不得 X"）
- 与其他抽象的关系、IoC 装线方式

**`#` 注释**（不进 `get-interface` 输出，读源码时可见）：

- 实现细节与实现理由
- 临时 workaround、已知陷阱
- 需要贴在现场的设计理由（本 workstream 的降级落点）
- 仍在有效期内的 TODO

**判断标准**：删掉这段文字后，使用者还能不能正确**使用**这个类/函数？
不能 → docstring。能 → `#` 注释或删除。

### 抽象面禁止耦合 feature 系统

docstring / 模块注释里**禁止**出现：

- `见 xxx FEATURE.md` / `详见 xxx FEATURE` / `见 workstream xxx`
- 决策编号（`决策 N` / `KD N`），无论孤儿编号还是带出处
- 进度语（"设计已锁定"、"已落地"、"本期不实现"）
- 历史叙事（"早期工程复杂度低时…"、"vN 重构时保留因为…"）

**理由**：抽象面是权威，不能依赖非权威且会腐烂的文档。反向索引用
`git log -- <file>`——spec 已经声明这条路径，源码里再转发一遍是重复且易腐。

处理落点：

| 内容 | 落点 |
|---|---|
| 使用所需的契约、不变量 | docstring 保留（去掉出处引用） |
| 现场必要的实现理由 | `#` 注释 |
| 决策轨迹、否掉的方案、进度 | 丢弃（FEATURE.md + git log 已是索引） |

### 边界判断：指向活代码的指针可以留

区别不在"有没有指针"，而在**指针指向易腐的文档还是活的代码**。

保留：`详见 channel_builder`、`详见 ctml`、`见 core/concepts/qa.py`、`详见 Mindflow`。
删除：任何指向 `FEATURE.md` / `workstreams/` / 决策编号的指针。

### 中文概念锚点，英文释义

中文概念体系是 MOSS 的认知底座，理解稳定。英文是**释义（paraphrase）**，不是替换。

- docstring 主力语言 → 英文
- 翻译方式 → 释义，不是直译。目标是英文读者/模型能无歧义理解
- 不对中文原句做逐行翻译，不保留双语并列格式

**关键术语：英文词 + 中文 + gloss**（首次引入时）：

```
The Channel (经络, "meridian") is the module-like capability container —
commands flow through it like signals through a nerve pathway.
```

需要 gloss 的术语（枚举，非穷尽）：Logos / 道、Channel / 经络、Matrix / 矩阵、
Mindflow / 心流、Nucleus / 核、Impulse / 冲动、反身性 / reflexivity、
双工 / duplex、三循环 / three-loop。

普通技术术语不需要 gloss（command、interpreter、runtime、session 等）。

### 翻译分层（2026-09-12）

不是所有中文都翻译。按内容复杂度分三层：

| 层 | 内容 | 处理 |
|---|---|---|
| L1 导言层 | 每个模块顶部 docstring | **必译**：给出英文导言，说明模块在架构中的位置与角色 |
| L2 简单层 | 字段 description、参数说明、一句话契约 | **必译**：直接以英文替换 |
| L3 哲学层 | 架构推演、设计哲学、长段论述 | **不译**：保留中文；由 L1 导言承担入口可读性 |

`mindflow.py` 是 L3 的代表：正文（~395 行含中文）保留中文，
只在模块顶部 docstring 增补英文导言。`cell.py` / `project.py` / `environment.py`
同理——大体量正文按 L3 处理，其中的 L1/L2 部分照常治理。

**L3 是过渡态，不是终态**：英文导言保证了抽象面的入口可读性（纪律 1、2），
但 L3 正文对非中文读者仍不可消费。未来若需要，再逐个模块做 L3 翻译；
本 workstream 不承诺这一步。

**与「不双语并列」的关系**：该 KD 禁止的是**逐条对照**（一份内容两种语言叠放）。
L1 英文导言 + L3 中文正文是**导言层与详述层的分工**，不是对照翻译，允许。
L2 是替换，不产生并列。

### 过期内容清理

治理过程中发现以下情况，直接处理：

1. 已解决的 TODO → 删除
2. 历史残留标记 → 删除
3. 仍在有效期内的 TODO → 从 docstring 移到 `#` 注释
4. 与代码行为不一致的 docstring → 修正 docstring 匹配实际行为
5. 不确定是否过期的 → 保留，加 `# NOTE(verify): ...`

## Scope

| Package | Path | Files | 中文密度 |
|---------|------|-------|---------|
| contracts | `src/ghoshell_moss/contracts/` | 7 | 重度 |
| concepts | `src/ghoshell_moss/core/concepts/` | 7 | 重度 |
| blueprint | `src/ghoshell_moss/core/blueprint/` | 17 | 重度（~1700 行含中文） |
| channels | `src/ghoshell_moss/channels/` | 10+ | 轻度（主要是注释） |
| architecture | `src/ghoshell_moss/architecture.py` | 1 | 轻度 |

优先级：contracts > concepts > blueprint > channels > architecture。
contracts 被最多模块引用；blueprint 的 matrix 与 mindflow 承载最重的设计哲学，
翻译质量要求最高。

### 本次（2026-09-12）圈定的工作面

1. **blueprint/CLAUDE.md**：删除 `blueprint/READEME.md`（拼写错误），
   改为 CLAUDE.md，声明 blueprint 是 MOSS 架构模块的蓝图、大部分是 facade，
   并写入上面的 Code as Prompt 纪律。
2. **抽象面解耦**（本 workstream 新规则）：清除 blueprint 全部
   FEATURE.md / workstream / 决策编号引用。
3. **过期 docstring 修复**：本波次，例如 `cell.py` 的 `CELL.md` → `NODE.md`。
4. **英文翻译（分层）**：不复杂的 docstring / comment 翻译成英文；
   架构哲学重的正文（尤其 `mindflow.py`）**不翻译**，仅在模块顶部 docstring
   增补英文导言。见 §翻译分层。
5. **单测**：相关单测按 blueprint 设计做**行为测试**——
   验证抽象面声明的契约，而不是复刻实现细节。

## Key Decisions

### KD-A: 抽象面不承载决策历史（2026-09-12）

新增。抽象面 docstring 描述"是什么、怎么用、契约"，不描述决策轨迹。
理由见 Motivation §触发改名的新问题。

**这细化了本 workstream 原 KD「docstring vs comments 边界」**——原 KD 主张
"设计动机和架构哲学进 docstring（code-as-prompt，模型需要理解为什么）"。
该主张在有出处引用时可接受，但它诱导了"把 FEATURE 的决策编号搬进 docstring"。
现收紧为：**使用所必需的"为什么"进 docstring；决策历史与出处引用一律不进**。

### KD-B: 现场理由降级为 `#` 注释，而非保留在 docstring（2026-09-12）

新增。确有现场必要（实现者需要知道）的理由写在 `#` 注释里，
不占 docstring，不进 `get-interface` 输出。

### KD-C: 指向活代码可以，指向文档不可以（2026-09-12）

新增。判据是目标的易腐性，不是"有没有指针"。

### 不做的事

- **不重构代码**。不修改接口签名、不调整逻辑、不重新组织模块结构。
- **不纯英文化**。中文注释中的设计讨论移至 `#` 注释保留，不删除。
- **不机械翻译**。不对每行中文做逐句英文对应。读原文，用英文重述。
- **不双语并列**。docstring 里不出现"中文原文 + 英文翻译"的叠放格式。
- **不在 docstring 里罗列 commands**。命令签名已由 interface 自动反射，
  手写重列必然随代码漂移。

### 翻译质量标准

- **准确性优先**：blueprint 与 concepts 的 docstring 承载架构语义，
  翻译错误比不翻译更糟。
- **术语一致**：同一个中文概念在所有文件中用同一个英文词。
- **简洁**：不追求文学性。参照 `channel_builder.py`、
  `matrix_channel.py` 模块 docstring 的英文风格。
- **可验证**：每个文件翻完后跑 `moss codex get-interface <模块路径>`
  确认自解释输出语义无损。

## Implementation Notes

- **人类架构师的原话，迁移时保留原风格。** docstring → `#` 注释 的降级是**迁移**，
  不是改写：照搬原措辞，不要替换成模型自己的表述。2026-09-12 `mindflow.py`
  `OnChallenge` 的设计说明即此例。
- 发现明显 bug 可顺手修，但不在此 workstream 做架构改动。
- `__init__.py` 若只有 re-export 无实质注释，跳过。
- channels 目录模块 docstring 采用机器可解析格式（`描述 | 类型 | status`），
  翻译时保留格式，只翻描述。
- 每个 package 单独 commit；每完成一个文件 `get-interface` 验证。
- 本 workstream 的规则本身是**新约定**：治理抽象面时若再发现
  "把 feature 决策搬进抽象面"的写法，按 KD-A/KD-B/KD-C 处理，
  不要重新发明判据。

## Status Log

- 2026-09-12: 从 `bilingual-comment-governance` 改名；新增抽象面解耦规则
  (KD-A/B/C)；圈定 blueprint 首轮工作面（CLAUDE.md、解耦、过期修复、翻译、单测）。

- 2026-09-12 (P1 + P2 + P3-L1 落地):
  - 删除 `blueprint/READEME.md`（拼写错误），新建 `blueprint/CLAUDE.md`（英文），
    含 Code as Prompt 四条纪律 + 禁止耦合 feature 系统的落点表 + docstring/comments 边界。
  - **解耦**：清除 blueprint 全部 FEATURE.md / workstream / 决策编号引用
    （11 处：`warrant.py`×2、`host.py`×5、`matrix.py`×1、`channel_builder.py`×2、
    `mindflow.py`×1）。现场理由按 KD-B 降级为 `#` 注释；`channel-meta-dyn-static`
    悬空引用直接删除。
  - **过期修复**：`cell.py` 的 `CELL.md` → `NODE.md`（4 处）。全文件复扫
    `CELL.md` / `list_cells` / `moss cells` 已清。
  - **L1 英文导言**：覆盖全部 16 个模块（`__init__.py` 跳过）。
    `environment` / `channel_builder` / `shell_trajectory` 的简单导言直接以英文替换；
    `cell` / `host` / `matrix` 保留中文正文，英文导言置于顶部。
  - 验证：`py_compile` 全部通过；316 tests passed；`get-interface` 反射无死链。

### 剩余工作面

- **P3-L2**：blueprint 其余简单 docstring / comment 的英文替换（逐模块推进）。
- **P3-L3**：`mindflow` / `cell` / `project` / `environment` 正文（不承诺）。
- **P4**：行为测试补齐 6 个缺口（见 §本次圈定的工作面 第 5 条与下表）。
  缺口清单：Warrant `list_states`、Warrant 队列落盘顺序、SafeMode 闸口范围/反射弧绕行、
  SafeMode note 回放后 observe、ChannelBuilder 构造 API、Matrix `warrant` + discovery。
