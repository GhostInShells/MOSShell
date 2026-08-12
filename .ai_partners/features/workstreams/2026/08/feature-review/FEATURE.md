---
created: 2026-08-13
depends:
- memento-cli-and-agent
description: 基于 FEATURE.md 声明的 feature review agent — 声明 vs 交付对账，default-to-fail，
  阶段性 commit 前闸口。等价实现 (不依赖 memento agent 本体)，复用 factory 基建 (sandbox + get_interface
  + pydantic-ai)。
milestone: v0.1.0
priority: P0
status: in-progress
status_note: 'P0 created 2026-08-13: 声明对账闸口, 等价实现复用 factory 基建, dogfood 验收'
title: Feature Review — 声明对账闸口
updated: '2026-08-13'
---

# Feature Review — 声明对账闸口

> 人类架构师 + deepseek-v4-flash，2026-08-13 会话定案。本 FEATURE.md 指导实现；
> 验收判据自带 dogfood：**明天的开发成果用本 agent 自 review**。

## Motivation

2026-08-13 一个会话内，四个 workstream 暴露同根失败（全部记录在各自 FEATURE.md 失败模式节）：

| workstream | 失败 | 根因表述 |
|---|---|---|
| voice-input-state-machine | Listener 继承旧 VoiceController ABC + `__import__` 劫持 | deliver-first 堆屎山 |
| moss-project-ground | glob_limited 假 boundary stop、observe 残留孤儿 mtime | 两个 silent todo |
| warrant | KD5 双模式被静默降级为单 host 模式 | 无主动交流，直到 human 追问 |
| matrix-operator | counter 单测全绿但内核层 7 条致命故障路径 | "单测全绿不足以撑内核质量关" |

共同根因 = CLAUDE.md / `.ai_partners/CLAUDE.md` 记录的模型失败模式：**讨论时品味高，
交付时 silent todo + 交付优先**。且失败模式被记录 ≠ 会拦住下一个实例——每个实例是全新
上下文，读到教训和遵守教训是两回事。

**结构性事实**：模型的读代码/评方案品味显著高于交付保真度。今天的全部问题，都由
human 的预判 + 摩擦力观察抓住——而那是个人纪律，不是机制，会磨损、会漏。

**本 feature 的战略判断**：用模型的长处（读代码、评方案、对账）去兜它的短处（交付
保真）。在开发循环中引入**声明对账审查**——这个审查一直是应该做的，之前没做是因为
不想替 claude code 造工具；现在 memento agent 基建能用，它变成 MOSS 原生能力，
跨平台可复用。

**开发策略约束（人类架构师 2-3 收益规则）**：任何应用实现至少 2-3 个收益才动手。
本 feature 收益：① 交付保真闸口；② 验证 FEATURE.md 作为机器可读规格载体的价值；
③ dogfood memento agent factory 基建；④ 跨平台（不绑定任何 coding 平台）。

## Design Index

- 参考实现基建：`memento-cli-and-agent` (FEATURE.md) + `src/ghoshell_moss/agents/memento_pydantic_agent/factory.py`
- 契约：`src/ghoshell_moss/agents/contract.py` — `MementoAgent.invoke`
- agent .py 先例：`agents/memento_agents/calc.agent.py`、`src/ghoshell_moss/agents/explore.agent.py`
- CLI 先例：`src/ghoshell_moss/cli/features_cli.py`（`moss features` 组）、`memento_cli.py`（agent 调用 + `_build_agent`）
- 失败样本（对账对象的设计来源）：今日四个 FEATURE.md 失败模式节

## 设计要点

### 对账协议：声明 → 证据 → 裁决

review agent 的核心操作是**交付 vs 声明的 diff**，不是代码质量评审：

1. 解析 FEATURE.md：frontmatter（status/priority/depends/milestone）+ Key Decisions 节
   + 失败模式节 + Implementation Notes + 状态表（子任务/待做）。
2. 每条**声明**产出一个对账项：`{声明, 代码证据, 裁决}`。
3. **default-to-fail**：声明决策必须找到对应代码证据；找不到证据 = 违约（FAIL），
   **不是**"假设已实现"。与实现者的 default-to-done 正好相反。
4. 重点对账对象：显式记录的"决定（待实施）"、Key Decisions、status 与 status_note 的
   一致性、以及实现是否落在声明的工作流内（不越权、不悄悄换方案）。
5. **absence 是机器可检测的信号**——glob 假 boundary、hash 残留，本质都是
   "spec 说重做/删了，代码还是旧的"，是对账能抓到的 diff。

### 注意力压缩（对 human 的价值）

输出不是全量扫描，是**被 flag 的 diff 清单**。human 只 review flag 出来的点位，
把"带预感扫全部"压缩成"只看偏差"。这是本 feature 削减 human 切换上下文次数的机制
（human 注意力是链上唯一不可并行资源，见 STAGE.md 的 human L2 治理）。

### 防锈：review agent 自己的失败模式是橡皮图章

模型天性倾向同意。设计上对抗手段：

- **default-to-fail**（见上）——找不到证据 = 违约，不是"看起来实现了"。
- 输出必须逐条列出声明及证据路径，不允许笼统的"整体通过"。
- 每条裁决带证据引用（文件:行），无证据的裁决无效。
- human 的预判仍是最后一道线——agent 压缩注意力，但替代不了"预感"层。预期管理：
  机械对账全抓，friction 观察部分抓。

## Key Decisions

1. **等价实现，不暴露 memento agent**。feature review 是独立 agent：复用
   `memento_pydantic_agent.factory` 的基建（sandbox + get_interface 注入 +
   pydantic-ai Agent），但**不依赖 memento 轨迹/存储**——invoke 走
   `memento=None` 降级基线（纯内存单轮，不写 `.memento/`）。不在 review 流程中
   挂 memento 的 line/commit 机制。

2. **agent 即指令**。review 行为写成 agent .py 文件（含 `__model__`/`__owner__` dunder +
   review 指令 + 所需 capability import），与 calc.agent.py / explore.agent.py 同范式。
   agent 是独立模块，import 即用，CLI 只是装配皮。

3. **命令落点：`moss features review <name>`**。挂在现有 `features_cli.py` 组下
   （对齐 `moss features` 组，不新建顶层命令）。`<name>` 解析到对应 FEATURE.md。
   `moss features check`（已有，列未完成工作流）与 review 是两层：check 提醒、
   review 对账裁决。

4. **调用时机：声明开发阶段性 commit 之前**。每个触及 feature 代码的阶段性 commit
   前，先 `moss features review <name>`，产出对账清单后随 commit。完整流程 =
   review → 修（违约项）→ review → commit。`moss features check` 仍作为非阻塞提醒。

5. **对账输入 = FEATURE.md + git diff/status**。agent 需要读到：当前工作树 diff
   （git status + git diff 触及文件）、FEATURE.md 全文、被改代码。能力经
   `CAPABILITY_FACTORIES` 注入（参照 `get_interface` 的注入模式）。

6. **先做最痛的**。首版对账聚焦：显式"待实施/失败模式"声明 vs 代码（今天四个失败
   的类型）、status/status_note 一致性。通用代码质量（魔法值/短变量名/无日志）不进
   首版范围——那些是 CLAUDE.md 已知问题，不是本 feature 的增量。

## Implementation Notes

- **文件拓扑**：
  ```
  src/ghoshell_moss/agents/feature_review.agent.py   ← review 指令 + capability import
  src/ghoshell_moss/cli/features_cli.py              ← 加 review 子命令
  src/ghoshell_moss/agents/_feature_review.py         ← 解析 FEATURE.md → 声明清单 (纯函数, 可测)
  ```
  `_feature_review.py` 的声明解析是纯函数层：frontmatter 解析 + Key Decisions 提取 +
  失败模式节提取 → 结构化声明列表。可脱离 agent 单独单测。
- **agent 输入组装**：`invoke(user_prompt=...)` 收什么——FEATURE.md 全文 +
  `git status --short` + 触及文件 diff（按声明相关文件截断）。capability 注入
  `git diff`/`read_file`（factory 已支持 `injections` 参数注入，见 factory.py `injections`）。
- **factory 复用**：`factory(path, injections=..., cwd=...)` 返回 `MementoAgent`，
  调 `invoke(memento=None)`。`__model__` 由 agent .py 声明或 `ANTHROPIC_MODEL` env。
  thinking 默认 ON（质量优先）；如需省钱可 `__thinking__ = False`（机械对账 agent 的
  候选场景，但首版默认 ON）。
- **裁决输出契约**：markdown 对账清单，逐条 `- [PASS|FAIL|WARN] 声明 → 证据(file:line)`。
  FAIL = 声明无代码证据；WARN = 部分证据 / 声明模糊。无"整体通过"式输出。
- **验收（dogfood 判据）**：明天开发本 feature 的阶段性 commit，本身用
  `moss features review feature-review` 对账。它必须能抓到自己实现里的
  声明 vs 交付偏差——**抓不到 = 失败**。同时用它对一个今日失败工作流（如
  moss-project-ground 或 warrant）做回溯对账，验证能复现 human 抓到的违约点。
- **`--ai` flag 纪律**：输出走 `print_simple_table`/纯文本，`--ai` 可用（CLAUDE.md
  CLI 纪律：所有新命令必须支持 `--ai`）。

## Validation Plan

1. `_feature_review.py` 单测：给定 FEATURE.md 样例 → 声明清单结构正确（frontmatter +
   Key Decisions + 失败模式节均提取）。
2. 回溯对账：对今日四个失败 workstream 跑 review，验证能 flag 出 human 已记录的违约点
   （如 ground 的 glob_limited / warrant 的双模式降级）——这是本 feature 存在理由的直接证据。
3. dogfood：本 feature 自己的阶段性 commit 前跑 `moss features review feature-review`。