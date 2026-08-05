---
title: Meta Mode
status: draft
priority: P2
created: 2026-08-05
updated: 2026-08-05
depends: [moss-project-ground, qa-exchange]
milestone:
description: >-
  MOSS 开箱矩阵中"最小依赖可自开发"的模式 — ground + bash + file_editor 三件套能力面，
  mode 与 ghost 解耦（echo ghost 即可驱动）。让 MOSS 在自己运行时里具备自我开发能力。
---

# Meta Mode

> Use `moss features set-status meta-mode <status> -m "note"` to update state.
> See [TOPOLOGY.md](TOPOLOGY.md) for directory layout and [README.md](README.md) for the full convention.

## Motivation

MOSS 需要一个"开发 MOSS 的 ghost"：让 MOSS 在自己运行时里拥有自我开发能力。这不是再加一个
coding agent，而是把"开发 MOSS 本身"变成运行时的一等公民——有躯体（channel）、有认知空间
（ground/memento）、有工具面（bash/file_editor/codex）。

meta 是 MOSS 开箱矩阵的一格，矩阵由人类工程师构思：

- **meta** — 最小依赖可自开发（本 workstream）
- **desktop** — 桌面模式，基于 GUI（qt-screen，`nodes/screens/qt_screen/`，PySide6/QML，S3 已完成）
- **reachymini** / **g1** — 具身机器人平台（HTTP / DDS）

**为什么现在**：qa-exchange（跨进程广播问答）快完成，是 meta 自开发验证回路的工程信号。
meta 的认知主通道（project-ground）在推进中，screen-node 的 S3 已为 desktop 模式铺好 GUI 躯体。

**边界警示（交付幻觉讨论的教训）**：meta 的目标不是"模型全自主开发"，而是"把自主性放在
人类可否决的结构里"。MOSS 是反共识设计，模型自主演进会被先验重力拖回行业均值。否决判据：
人类能否在有界时间内正确否决貌似合理但方向错误的设计提案。自主性以可转向性为代价 = 锁死。

## Design Index

- 认知背景：
  - `moss_context_assembly_architecture`（.discuss/2026-06-08）— "meta ghost 配合足够好用的工具，它就不是 coding agent，是一个有躯体有认知空间的实体。现在的你未来就是他"
  - `delivery_illusion_and_steerability`（.discuss/2026-07-30）— 交付幻觉 / 否决判据 / 先验重力
  - `moss-project-ground` workstream — meta 是 `--mode meta` 下的默认 ground（已在 FEATURE 里预埋点名）
- 前身验证器：`memento-cli-and-agent`（completed）— 无 harness agent，`.agent.py` 文件即 agent、反射即 prompt、能力=可导入函数、memento 跨 invocation 记忆（读侧已落地）

## Key Decisions

### 1. mode 与 ghost 解耦 — meta 是纯 Shell 能力面

meta 不需要新 ghost、不需要新大脑。用 echo ghost（Atom 原型）就能驱动 meta 的能力面。
这是 Ghost-in-Shell 的解耦兑现：mode 提供躯体（channel 能力面），ghost 提供大脑，两者正交。
因此 meta 的"最小"可以做到很彻底。

### 2. 最小依赖面：ground + bash + file_editor 三件套

meta 的能力面只依赖三件事，不依赖音频 / GUI / 机器人：

- **ground** — 认知场。真正决定的只有一件事：ghost 启动时要不要立刻看到 project ground。
  挂默认 ground channel，path = project_home，MOSS 项目认知由 GROUND.md 生产。
- **bash** — moss CLI + 任意命令，作普通 channel 集成（不是独立 GUI）。
  bash + moss CLI 覆盖了 codex 反射、features、memento 全部命令，故这些不必独立 channel。
- **file_editor** — read/write 五动词（create/str_replace/insert/undo_edit，已实现未对 agent 暴露）。

**memento 不必要**：有 bash，meta 可用 bash 调 `moss memento` 命令集合。

### 3. 认知结构：MOSS.md/HOST.md 轻量 + ground channel 主通道

MOSS.md 正文（project slot）和 HOST.md 正文（mode slot）是 **meta instruction 的一部分，不写细**——
每个几十 token，回答"我在哪 / 这个躯体是什么"。具体认知资产（features/.design/.discuss 的 pin 组织）
由 ground channel 承担，Ghost 需要时 open。slot 组装代码已就位（`host/moss_runtime.py:_build_system_prompter`
四层：ctml / project / mode / static），**缺的只是内容**。

### 4. mode 自带默认 ghost

`moss-ghost --mode meta` 不传 ghost 应直接可跑。现状缺口：

- `HostModeMeta`（`core/blueprint/project.py`）无 `default_ghost` 字段
- `Environment.__init__` ghost 解析链：显式参数 > 环境变量 > MOSS.md 全局 `default_ghost`，无 mode 层
- `moss-ghost`（`cli/ghost_run.py`）传 None 就列列表不跑

改动：`HostModeMeta` 加 `default_ghost` 字段；`Environment` 解析链插入 mode 层（参数 > env var >
mode.default_ghost > MOSS.md 全局）；`moss-ghost` 无参时 env 解析出 ghost 就直接跑。落点放
Environment 层（配置单一信源 + seal 一次性），`moss-mcp` 等其他入口自动受益。

### 5. meta 验收：双工对话，不是脚本化（本场最大纠偏）

**脚本化 echo（启动 matrix → 发 signal → 拿结果 → 关）是零价值**——moss ghost 是全双工运行时，
感知/思考/行动并发重叠，让它变脚本化 = 阉割技术价值。memento-agent 的"无 harness 单帧"哲学
属于轻量 agent 家族，**不能外推到 ghost runtime**——ghost 是另一种动物。

验收走双工形态：**人类直接测试（TUI/节点对话），或两个 ghost 运行时对话**。跨 ghost 对话
的机制已准备——独立 node 节点给 ghost 建聊天室，任意通讯协议的聊天室都可以。**这块验收由
人类工程师接手**，本 workstream 不做脚本化验收工具。

**QA 不是跨 ghost 对话机制**。QA（qa-exchange）是广播问答抽象（Asker 广播问题 / Watcher 应答 /
requester 持真相 / 先到先得裁定），是 topic 的 ask 侧。跨 ghost 对话走 node 聊天室，别混淆。

### 6. 自动授权 = 独立于 QA 的交互命题（未来项，不阻塞 v1）

拿到一个 qa，符合自动授权条件的做自动授权；"什么叫符合条件"本质是交互命题，不是授权问题。
meta 的自动授权不关心具体命题：**单 token 多分类，flash 模型决策**——instruction 100% 命中 cache，
授权规则手写 prompt，0.3~1s 完成一个验证；不开启时全手动。做细粒度授权（区分是否需要交互的命题）
极其麻烦，不做。v1 审批退化用已有 safemode（logos 级人工闸口，体感差但现成可用）。

## Exploration Paths

- **脚本化 echo 设计（本场讨论被否决）**：曾设计 `moss-ghost echo`（启动 matrix → 发 signal →
  关）作为验收。否决理由：全双工运行时脚本化 = 零价值。验收形态改为双工对话。
- **QA 角色两次猜错**：先猜"验证基座"，再猜"ghost 间对话通道"。实际 QA 是广播问答抽象，
  跨 ghost 对话走 node 聊天室。教训：不要把新机制的职责往已有机制上套。
- **memento 是否必要**：一度列 memento 进依赖面，被纠正——bash 调 `moss memento` 即可，不必要。

## Implementation Notes

- 开发点 1（MOSS.md/HOST.md 结构化）是**纯内容**问题，代码不动——slot 组装已就位。
- 开发点 2（mode 默认 ghost）是**三处小改动**：`HostModeMeta` 加字段 + `Environment` 解析链 +
  `moss-ghost` 无参直跑。
- meta mode 的 ground 依赖 `moss-project-ground`（in-progress）的 Grounds concrete 实现。
- 依赖：`moss-project-ground`（认知场）、`qa-exchange`（工程信号，非 runtime 依赖）。
- 全双工验证走 node 聊天室（跨 ghost 对话），不在本 workstream 内做。
