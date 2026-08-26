---
title: Feature Review — 零上下文 review (遗忘测试)
status: completed
status_note: feature 已经成立，后续在实际迭代中优化。
priority: P0
created: 2026-08-13
updated: 2026-08-26
depends: []
milestone: v0.1.0
description: >-
  `moss features review <feature>[@<perspective>]` — 旁路零上下文 review (遗忘测试)，
  机制化"声明 vs 交付"对账，兜住模型 silent todo / 交付漂移。MOSS 只生成 prompt + 文件发现，
  不做执行层。
---

# Feature Review — 零上下文 review (遗忘测试)

> 历史：2026-08-16 one 实现彻底漂移 (deepseek 家族第 8 例"声明-交付漂移")，会话烂尾、产出未留。
> 本文件 2026-08-26 按人类架构师还原的最终方案整体重写，废弃此前全部中间态
> (brief 生成器 / 确定性层 / 五切面分层路由)。详细历史见 `git log -- <本文件>`。

## Motivation

四个 workstream 同日暴露同根失败：模型讨论时品味高，交付时 silent todo + 交付优先。
失败模式虽已写入 `CLAUDE.md`，但每个模型实例是全新上下文——读到教训 ≠ 遵守教训，教训兜不住下一个实例。

**结构性事实**：模型读代码 / 评方案的品味显著高于交付保真度。本 feature 用模型的长处
(读代码、对账、评方案) 去兜它的短处 (交付保真)：在开发循环里引入**声明对账审查**。
这是一直该做、之前没做的——现在成为 MOSS 原生能力，跨平台可复用。

### 核心原理：零上下文旁路 review (导演看不懂观众的观感)

模型深陷开发上下文，自满于"知道本来要做什么"，看不到用户 / 其他模型 / 其他人看到的东西——
如同电影导演常不懂观众的观感。**旁路 review** 强制切一个零上下文化身：只读声明 (FEATURE.md) +
交付 (代码)，从零重建认知。它不知道你的意图，只会拿 declaration 对比 delivery，于是
silent todo 与声明漂移自然涌现。

review 的 stance 是**重建** ("你是下一个化身，从零试着重建，卡在哪报什么")，不是**审计**
("按清单打勾")。重建让 default-to-fail 自然涌现——"我卡在这"本身就是 fail，且比审计更抗橡皮图章。

## Design

### 轻量定位：MOSS 不做执行层

完全不做 sub agent / 语义 review / 机械检查 / brief 生成，**保持 features 轻量**。
MOSS 唯一作用 = **生成两种 prompt + 文件发现**。执行层 (真正读代码、报卡壳) 住在
coding-agent 集成架构——严格说来，允许模型自己用 harness 的 sub-agent / 别的 agent，
给它 prompt 自行探索。**零上下文的实现 = "派 sub-agent"这个动作本身** (sub-agent 天然全新上下文)，
不是 MOSS 造零上下文 agent。

### 命令两套形态 (由 `@` 有无区分)

| 形态 | 返回 | 内容 |
|---|---|---|
| `moss features review <feature>` | **meta prompt** | 顶层"怎么 review"指令 + 文件发现 (匹配到的 FEATURE.md + 可用视角文档路径与 description) + 告知调用者"要用某视角，用 `@[perspective]` 派 sub-agent / 别的 agent" |
| `moss features review <feature>@<perspective>` | **perspective prompt** | 该 feature 基本讯息 (不含全文为好) + 视角文档路径 + 内容呈现 + 提示完成后回报 |

- 没有 `@` = meta prompt；有 `@` = perspective prompt。
- stdout 为裸文本 synthesized instruction (可 pipe 进别的 agent run 命令)，不是 table / JSON。

### 名词解析

`<feature>` (feature_name_or_file)：复用 `get_feature` (对齐 `moss features status` /
`set-status` / `read`)，解析到 `workstreams/<year>/<month>/<name>/FEATURE.md`。

### 视角文档发现

`features/review/` 是**项目级可增加的约定空间**；per-feature `review/` 同名完全覆盖。
视角 (when 词汇表只三个，模型才匹配得上当前时点)：

| 视角 (file) | when | 核心提问 |
|---|---|---|
| **takeover** | `finalize` | 你是下一个化身，读声明+目录，知道怎么开始吗？卡在哪、缺什么？ |
| **reconcile** | `pre-delivery` | 逐条对账声明 vs 代码；声明说 X 而代码做 Y 或漏 X，用 file:line 指出；absence 也是信号 |
| **design** | `anytime` | 读声明+已有交付，设计层面矛盾/漏洞/静默降级？只 surface 候选点，交人终审 |

三视角覆盖四个 motivating failures (全是 reconcile 能抓的 silent todo) + 遗忘测试 (takeover) + L3 兜底 (design)。

### 命令落点与触发

- **落点**：挂在现有 `features_cli.py` 组下 (`moss features` 组，不新建顶层命令)。
- **触发 (懒披露)**：`moss features specification` 新增"如何 review feature"小节，只
  *提示*有这个实机 (用 `moss features review '<name_or_file>'`)，**不预载** review 内容
  进模型注意力；调用时才呈现 meta prompt 与视角文档。

### 输出纪律 (防锈)

- 只输出被 flag 的 diff 清单 / 视角核心提问，不讲笼统"整体通过"。
- perspective prompt 只给基本讯息 + 视角文档路径与内容，不灌完整 FEATURE.md (零上下文纪律)。
- 完成后提示回报。

## Key Decisions

1. **不做执行层**。MOSS 只生成 prompt + 文件发现；语义 review 由模型派 sub-agent 完成。
2. **不做机械检查**。status/status_note 不一致、文件拓扑缺失等 100% 机械可查的违约，
   交 CLI / 未来 pre-commit，不归 review 管。
3. **不做 brief 生成**。命令只呈现原理 + 视角文档，材料由模型 / sub-agent 组装——MOSS 聚合材料
   会引入开发模型偏差。
4. **`@` 区分 meta / perspective**。`<feature>` 出 meta prompt；`<feature>@<perspective>` 出 perspective prompt。
5. **视角文档由项目约定 + per-feature 同名覆盖**，不硬编码进 CLI。

## Implementation Notes

- `moss features review` 纯只读：读 FEATURE.md (经 `get_feature`)、发现 `review/` 视角文档、
  生成 prompt 输出。无副作用。
- 支持 `--ai` (CLAUDE.md 纪律：所有新命令必须支持 `--ai`)，输出裸文本 (可 pipe)。
- **已定实现选择** (2026-08-26)：
  - 目录用 `review/` (单数)；per-feature `review/` 同名覆盖全局 `features/review/`。
  - meta 模式只给基本讯息 + 路径，不内联 FEATURE.md 全文 (零上下文纪律，子 agent 自己读)。
  - meta prompt 文本 v1 写死在 CLI (元模板不外提)。
- **已建** (2026-08-26)：`moss features specification` 的"如何 review feature"小节 (本地 README +
  内置模板同改)；视角文档 `takeover.md` (finalize)、`reconcile.md` (pre-delivery)。
- **待补**：视角文档 `design.md` (anytime)，逐个讨论后写。

## Validation Plan

验收轨迹 = dogfooding 循环 (本 feature 用自己验证自己)：

1. **实现 CLI 套件** — `moss features review` 两形态 + 视角文档发现。
2. **用 subagent 测试** — 派零上下文 subagent 跑 `moss features review <self>@<perspective>`。
3. **按结果倒过来优化当前 feature** — 把 subagent 报的摩擦点反馈回 FEATURE.md / 实现 / 文档。
4. **观察真实摩擦点** (subagent 不好用的地方)，倒过来优化，重复实验消除摩擦。

3 → 4 是 dogfooding 效果：review 抓到自己声明里的漂移，本身就是它有效的证据。

当前进度 (2026-08-26)：
- ① 完成 — CLI 已实现并提交 (b55c7cf8)。
- ② 完成 — takeover 视角 review feature-review 自己，报出 5 个摩擦点 (待共建滞后、当时三视角只 authored 一个、调研轨迹延后、命名自指、计划停在 idea 层)；reconcile 视角复 review，报出静默吞坏 review 文档 (except 兜底) 等摩擦点。
- ③ 完成 — 据此校准"待共建"、补验收轨迹、修静默吞错误。
- ④ 持续 — feature 已成立，后续在实际迭代中观察真实摩擦、重复消除。
