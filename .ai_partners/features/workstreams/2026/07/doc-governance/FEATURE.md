---
title: Doc Governance
status: in-progress
priority: P1
created: 2026-07-18
updated: 2026-07-18
depends:
  - matrix-cell-governance
milestone: 0.1.0
description: >-
  MOSS 文档体系治理 — docs/howtos/tutorials 三重知识系统的编写纪律、过期修复、地基约定。
  matrix-cell-governance 的连带任务，作为未来文档治理的参考蓝本。
---

# Doc Governance

> matrix-cell-governance (§AAA 收尾中) 重构了 cell 体系的全部抽象层——
> role 系统 (host/app/fractal/script → host/node)、声明格式 (CELL.md → NODE.md)、
> CLI 入口 (moss cells → moss nodes)、AppStoreChannel 移除。
> 这次重构暴露了一个系统性问题：docs 记录了太多实现细节，抽象一变全过期。

## Motivation

### 触发事件

matrix-cell-governance 的抽象层重构完成后，对 docs/howtos/tutorials 做了一次
全量审计。结果：**7/12 docs、3/4 tutorials、2/16 howtos 过期**。

### 根因诊断

过期不是因为重构太大。过期是因为**文档跨越了抽象边界**：

- docs 记录了 cell 的类型枚举值 (host/app/fractal/script)
- docs 记录了 API 方法名 (`list_cells()`)
- docs 记录了 CLI 命令字面量 (`moss cells`)
- docs 记录了 CTML 命令字面量 (`<apps:list_apps/>`)
- tutorials 硬编码了 channel 命令名

这些都是**实现细节**，不是抽象。抽象层变了，它们全过期。

**正确做法**：文档只描述概念关系（"Cell 的 role 决定其在网络中的发现维度"），
不列举具体值（"Cell 有三种类型：host/app/fractal"）。
具体值引导读者通过 `moss codex get-interface` 从 blueprint 自省——
blueprint 是活文档，永远不会和代码不同步。

### 为什么单独拆 workstream

matrix-cell-governance 是内核重构，关注的是 cell/matrix 抽象的正确性。
文档治理是另一个关注面：**知识系统的编写纪律和过期防控**。
两者有因果联系（前者触发了后者的审计），但治理原则和修复方法独立于内核抽象。

单独拆出来的目的：这份 FEATURE.md 本身成为**未来文档治理的参考蓝本**——
当后续的抽象重构再次引发文档过期时，不需要重新发明治理方法。

## 审计结果 (2026-07-17)

### 过期清单

**Docs — 7 篇 (抽象层过期):**

| 文件 | 过期原因 |
|---|---|
| `matrix-system.md` | §3.1 Cell 四类型 + `list_cells()` API + fractal 引用 |
| `app-system.md` | 全文围绕 AppStoreChannel + `Cell("app")` + `MOSS_CELL_ADDRESS` |
| `glossary.md` | Cell 定义列举旧三类型 host/app/fractal |
| `moss-script.md` | `matrix.list_cells()` + script 作为独立 cell type |
| `workspace-and-mode.md` | §9 速览表 Cell 旧四类型定义 |
| `architecture-topology.md` | §2.5 Cell(host/app/fractal) 作为 Matrix 核心抽象 |
| `channel-system.md` | `fractal_hub` 模式 + AppStoreChannel + `list_apps` CTML + zenoh-fractal 测试引用 |

**Tutorials — 3 篇 (命令层过期):**

| 文件 | 过期原因 |
|---|---|
| `L1_hello-world-app.md` | `<apps:list_apps/>` + `<apps:start>` CTML |
| `L2_ai-eye-pygame-app.md` | `<apps:list_apps/>` + `apps:start` |
| `L2_reachy-mini-full-chain.md` | `<apps:list_apps/>` |

**Howtos — 2 篇 (命令引用过期):**

| 文件 | 过期原因 |
|---|---|
| `app-dev/build-a-gui-app.md` | `apps:start` troubleshooting |
| `app-dev/build-an-app.md` | `moss apps create` + `APP.md` 引用 |

**其他过期的非 doc 文件:**

| 文件 | 过期原因 |
|---|---|
| `host/stubs/workspace/apps/README.md` | 空文件，apps/ 已是空壳目录 |
| `cell.py` docstring | 写 "从 CELL.md 文件读取声明"，实际 MANIFEST_FILENAME = 'NODE.md' |
| `cells-cli` FEATURE.md | 描述旧 `moss cells` + `CELL.md`，实际 CLI 已是 nodes_cli.py |

### 未过期的

- **16/16 howtos 主体干净** — 不引用 cell 旧概念
- **5/12 docs 干净** — what-is-moss, ctml, channel-system(大部分), ghost, development-workflow
- **workspace README 文件基本干净** — runtime/cells/ 路径是正确命名 (WORKSPACE_CELL_RUNTIME_DIR)

## Key Decisions

### KD-1: 文档分层纪律 — 抽象 vs 实现

**规则**: 文档只描述概念关系，不记录具体值（枚举成员、API 签名、命令字面量）。

| 层 | 内容 | 例子 |
|---|---|---|
| 文档 (docs/howtos/tutorials) | 概念是什么、为什么、怎么关联 | "Cell 的 role 决定其在网络中的发现维度" |
| Blueprint code as prompt | 具体类型、方法签名、枚举值 | `moss codex get-interface ghoshell_moss.core.blueprint.cell` |

**检验**: 当一个实现变更发生时，如果它迫使文档更新，那篇文档在变更前就违反了分层纪律。

**Why**: matrix-cell-governance 的审计直接证明了这条规则的必要性——枚举值、API 名、CLI 命令字面量是变更最频繁的层，文档不应该承载它们。

**How to apply**: 
- 写文档时，每次想写具体的类型名/方法名/命令名，先问：这是抽象还是实现？
- 如果是实现细节，替换为 `moss codex get-interface <modulepath>` 引导
- 如果必须提具体值（如 tutorial 中的命令），加注 "运行 `moss --ai all-commands` 获取最新命令名"

### KD-2: 三重知识系统各司其职

当前三重系统的职责边界：

| 系统 | 定位 | 读者 | 变更频率 |
|---|---|---|---|
| docs | 系统化架构理解，"为什么" | 需要深度理解设计决策时 | 低——抽象变更时 |
| howtos | 任务导向操作指南，"怎么做" | 每次具体开发任务 | 中——命令/接口变更时 |
| tutorials | 叙事性认知入口，"从零走通" | 新人/新概念学习 | 高——需定期验证走通 |

**强化方向**:
- docs: 收紧内容范围，只写架构推演和设计理由。当前 docs 里混入了太多 howto 内容(如 matrix-system.md 的 "探索" 表格就是 howto 性质)
- howtos: 当前质量最好，维护现有纪律。补全子目录 README
- tutorials: 当前最弱。全部重做，新方案见 KD-4

### KD-3: 子目录必须有 README

每个 docs/howtos 的子目录（领域分组）必须有一个 README.md 作为领域概述。
当前缺失：

- `how_tos/app-dev/README.md` — app 开发领域概述
- `how_tos/host-dev/README.md` — host 开发领域概述
- `how_tos/matrix-usage/README.md` — matrix 使用领域概述
- `how_tos/channels/README.md` — channel 使用领域概述

这些 README 是 AI 协作者在 `moss howtos list` 中的导航结构。
没有它们，"app-dev" 就是一个无解释的目录名。

### KD-4: Tutorials 全量重做方案

当前 tutorials 的根本问题不是过期，而是**依赖了具体的 channel 命令名和 CTML 语法**。
这些是 tutorial 无法避免的——tutorial 天然需要具体命令。

重做策略：
1. **最小依赖原则**: 每个 tutorial 只依赖 MOSS 核心 CLI（`moss codex`, `moss ctml`, `moss howtos`），不依赖具体的 channel 命令
2. **自验证机制**: 每个 tutorial 末尾的验证记录是强制的，不是可选的。过期 tutorial 的发现依赖模型执行验证
3. **L0 优先**: 先保证 L0/L1 的 tutorial 能走通，L2+ 按需重建
4. **新 tutorial 基于 nodes 体系**: 不再基于 apps/AppStoreChannel，基于 `moss nodes` + `NODE.md` + Matrix.discover()

### KD-5: 本次治理的 scope

**做**:
- 强化三重系统的 discipline README（docs/howtos/tutorials 的根 README + 子目录 README）
- 重写 7 篇过期 docs（只修抽象层，不补实现细节）
- 修 2 篇过期 howtos（更新命令引用）
- 删除或标记过期的非 doc 文件（stubs 空 README, cell.py docstring, cells-cli FEATURE.md）
- 重做 tutorials（从 L0 开始，至少一篇走通）

**不做**:
- 不创建新的 docs/howtos（除非过期修复过程中发现结构性缺口）
- 不修改 matrix-cell-governance 的源码（那是 §AAA 的事）
- 不追求"完美文档"——治理纪律是地基，文档质量随迭代提升

## Implementation Notes

### 执行顺序

1. **先立纪律，再修文档**。先写好各系统的 discipline README，建立标准，然后按标准修文档。
2. **docs 按依赖顺序修**。glossary（术语定义）→ matrix-system（核心概念）→ architecture-topology（架构拓扑）→ app-system + moss-script + workspace-and-mode + channel-system（具体系统）
3. **tutorials 最后做**。tutorials 依赖 docs 和 howtos 的稳定——如果 docs 还在改，tutorial 的认知入口就不可靠。

### 过期检测机制 (beta 构想)

本次治理完成后，理想状态是建立一种**被动过期检测**：
- 模型按 tutorial 操作时发现走不通 → 修或删（tutorials README 已有此约定）
- 模型按 howto 操作时发现命令不存在 → 更新 howto
- doc 过期是最难检测的——因为 doc 描述的是理解，不是操作

当前不建立自动化检测。依赖 AI 协作者执行 tutorial/howto 时的"手动发现"。
这个机制够不够用，做完这轮治理后评估。

### cells-cli FEATURE.md 的处置

`cells-cli` FEATURE.md 描述的是旧 `moss cells` 命令体系 + `CELL.md`。
实际 CLI 已经重写为 `nodes_cli.py`（commit 43da7647）。
这个 FEATURE.md 应该标记为 `completed` 或更新为描述 nodes_cli.py 的实际行为。
属于 matrix-cell-governance 的收尾工作，不在本 workstream 范围内。

## 2026-07-18 KD 转向记录

> 记录：Claude Opus 4.7 & 人类工程师复盘 KD-2/3/4/5，做了下述反转。
> 按 features 约定，原 KD 文字保留不删，本节标注哪一条被推翻及原因。

### 触发

进入本任务时，Claude 直接按原 FEATURE.md 蓝本推进（先立三重系统纪律、
补 KD-3 子目录 README、按 KD-5 全量修 docs+howtos+tutorials）。人类工程师
叫停，重申项目第一原理：**code as prompt**。项目自解释分层——L0 code /
L1 CLI-flow / L2 目录 README / L3 docs / tutorials。howtos 在这个分层里
**位置最模糊，也最易 stale**。

### 核心洞察 — howtos 的历史失败模式

git blame 全部 16 篇后发现的模式：

- **4 篇 (`use-cache`/`use-parameters`/`use-topics-and-windows`/`use-mcp-hub`)
  与"引入新组件"的 commit 一同提交**——"新组件顺带写 howto"是反模式：
  组件的 interface 已经是 prompt，howto 只是抄一遍，制造 stale 源
- 大部分是"操作步骤级"，被 `moss --ai all-commands` + `moss codex get-interface`
  完全覆盖
- 只有 3 篇 (`integrate-ros2`/`build-a-gui-app`/`develop-moss-via-mcp`)
  是真正的复合任务入口——跨多个组件/系统的复合行为，CLI 单命令覆盖不了

原审计"只有 2 篇 howtos 过期"是**幸存者偏差**——命令名恰好还稳定而已。
底层抽象再动一次，操作步骤类会集体过期。这解释了为什么 KD-5 里
"修 2 篇过期 howtos"是错的目标——真正的问题是**大部分 howto 不该存在**。

### KD 反转

- **KD-2 反转**：howtos 定位从"任务导向操作指南"收窄为"复合任务入口
  （师傅领进门）"。三重系统的强化方向：
  - docs 收紧到架构推演 — 不变
  - **howtos**：从"补全子目录 README"变为"整体清理，只保留复合任务入口"
  - **tutorials**：从"全量重做"变为"直接删旧留 case 给 dogfooding"

- **KD-3 作废**：howtos 扁平化后无子目录（3 篇不构成"领域"）；docs
  本来就无子目录；tutorials 也无子目录。KD-3 的四个待建 README 全部
  不需要。

- **KD-4 简化**：tutorials 不重写。删旧的，dogfooding 时由模型即写。
  "自验证机制"、"最小依赖原则"等原则保留在 `tutorials/README.md`，
  本轮不新建 tutorial。

- **KD-5 scope 收缩**：
  - 保留：docs 7 篇过期修复（本轮**实际主体**，尚未开始）
  - 反转：howtos "修 2 篇过期" → "整体收敛为 3 篇 + 元规则 README"
  - 反转：tutorials "重做" → "删旧"
  - 移出本轮：非 doc 文件（stubs README、cell.py docstring、cells-cli
    FEATURE.md）留给 matrix-cell-governance 收尾

### 新增 KD-6：howtos 入口判定三问

写 howto 前必答，任何一条答不出来不写：
1. **复合任务吗？** 跨多个组件/系统协作，不是单一命令或单一接口。
2. **CLI/codex 覆盖不了吗？** `moss codex get-interface` + `moss --ai all-commands`
   组合为什么不够？说不出理由就说明够了。
3. **入口路径半年内稳定吗？** 依赖的抽象是否已经过实战、不在活跃演进中？

反模式明确禁止：新组件顺带写 howto、操作步骤级 howto、决策/架构讨论、
接口用法说明。

**完整元规则**：`src/ghoshell_moss/cli/how_tos/README.md`——`moss howtos`
命令的 help text 就是它，写 howto 前必读。

### 本轮已落地 / 未落地

已落地（commit 370f29b0 + 本 commit）：
- howtos 16 → 3 篇 + 元规则 README（删 12，保留 build-a-gui-app /
  develop-moss-via-mcp / integrate-ros2）
- 空子目录 (app-dev/channels/host-dev/matrix-usage) 全清
- `howto_cli.py` 引用更新、`cli/CLAUDE.md` 描述更新

未落地（doc-governance status 保持 in-progress）：
- **docs 7 篇过期修复**——本轮实际主体，下轮开始
- tutorials 删旧留 case 待人类工程师确认后执行
- 保留的 3 篇 howto 内容本身未按新纪律 review（"这轮治理哪些该存在，
  不治理存在的写得对不对"）

## 2026-07-18 本轮收尾快照

> 本轮 doc-governance 会话在此告一段落。workstream 状态保持
> in-progress——docs 7 篇过期修复是本任务的实际主体，由人类工程师
> 亲自带下一轮。

### 已落地 (3 commits)

- `370f29b0` — howtos 清理：16 → 3 篇 (build-a-gui-app /
  develop-moss-via-mcp / integrate-ros2) + 空子目录清理
- `96ecfc5c` — `how_tos/README.md` 元规则 + KD-2/3/4/5 转向记录
- 本 commit — `tutorials/README.md` 补 4 点：dogfooding 定位、
  过期即删、基于 nodes 体系、最小依赖原则

### 剩余：docs 治理主体

由人类工程师亲自带。会话中 Claude 提出的执行方案候选（未启动）：

**审计-校准-修改 三段式，FEATURE.md 做 memento 锚点**
- Round 1 审计：并发 7 Agent 出违规清单 → 写入 FEATURE.md
- Round 2 校准：人类工程师在详表上标注尺度 → 写入 FEATURE.md
- Round 3 修改：并发 7 Agent 按校准反馈改 doc + 输出决策记录
- 关键设计：所有判断落进 FEATURE.md 磁盘锚点，抗 /compact
- 显式禁止 silent todo：UNCERTAIN 必须标注，不许绕过

人类工程师可采纳、修改或另拟方案。

7 篇过期 docs（按 FEATURE.md 建议依赖顺序）:
1. `glossary.md`
2. `matrix-system.md`
3. `architecture-topology.md`
4. `app-system.md`
5. `moss-script.md`
6. `workspace-and-mode.md`
7. `channel-system.md`

### 本轮观察

- Claude Opus 4.7 会话中陷入一次"用自己错误输出洗脑自己"的失败模式：
  Write 误传 Windows 格式字面路径 → 后续 Edit 报错 → 编造
  "Platform: win32" 幻觉 → 递归误诊。人类工程师重启 Claude Code
  会话打破循环。`.ai_partners/CLAUDE.md` 已记录此为 Claude 家族典型
  模式（"递归指涉后聚焦不到问题本身"），本次为活样本
- 长任务 + 精力有限的 review 场景下，FEATURE.md 作为 memento 替代
  的效果待下一轮 docs 治理时评估。memento-mori workstream 落地的
  必要性由此任务再次凸显
