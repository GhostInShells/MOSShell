---
title: Stage2 Surface
status: in-progress
# priority: importance within the current stage (iteration cycle) — not development urgency
priority: P1
created: 2026-09-29
updated: 2026-09-29
depends: []
milestone:
description: >-
  围绕 SURFACE.md 治理 MOSS 的自解释体系（L3 产品表面）——迁移 root 三件、
  改写 start.md / CLAUDE.md 家族、退一阶 features 命令面、建 works/ 帧并迁入 blogs。
  目标是「先搭架子」，非一次做完。
---

# Stage2 Surface

> Use `moss features set-status stage2-surface <status> -m "note"` to update state.
> See [TOPOLOGY.md](TOPOLOGY.md) for directory layout and [README.md](README.md) for the full convention.

## Motivation

本轮围绕 `SURFACE.md` 治理 MOSS 的**自解释体系**——项目对外呈现的整体面貌(L3)。定性前提:
SURFACE 是 L3 的自述 + 维护清单,而**文档体系本身是 L3 命题(项目自迭代的产品),不是行规**;
所以治理的目标不是「写好一批文档」,而是把 SURFACE 立成一个可回归的系统。

为什么现在:

- **第一条完整链路已跑通**(v0.1.0 stage)。`moss start` 此前迟迟不能正确定义「MOSS 是什么」,
  是因为链路未闭环;现在能了。
- **功能面已成形、可编织**:命令面(GROUND / features / codex / manifests / skills / docs /
  llms / audio / shell / ghost)已成体系,可以按依赖关系织成一张自解释地图。
- **承接明确遗留**:`features-system-formalization` 的 KD8(root 三件迁移)、KD10(`works/` 根场)、
  KD11(入口去工具化、意识向 `.ai_partners/` 收拢),本轮执行。

完成定义 = **架子立住,不是一次做完**:skills / docs 允许 in-progress;命令组合使用规范、
整体回归归 stage3。

## Design Index

- 锚:`.ai_partners/features/SURFACE.md`(L3 自述 + 维护清单)
- 前序 workstream:`features-system-formalization/FEATURE.md`(KD8 / KD10 / KD11)
- 旧设计(将被本轮超越并冻结):`src/ghoshell_moss/cli/.design/2026-06-02-start-cli-cognitive-entry.md`
- 规范:`features/README.md`、`features/TOPOLOGY.md`;命令面现状 `moss --ai all-commands`

## Key Decisions

<!-- 按 KD 编号追加，不回改。这是下一个 AI 实例读到的第一手决策。 -->

**KD1 — 入口按「功能」分轴,不止按受众。**
- `moss start` = **orientation**:让模型在 MOSS 里动起来;不做 pitch,定义到「能行动」为止。
- `CLAUDE.md`(根 / `.ai_partners` / `cli`)= **贡献者入口**:features 纪律、意识轨迹、git 规范、角色。
  根 CLAUDE.md 收敛为薄 shim,指向工具无关入口。
- `works/` = **transmission**:对外传递面。
（承 KD11:入口去工具化。)

**KD2 — start.md 围绕「命令组」织,不列 concrete 命令。** concrete 命令下沉 skills 体系
(本轮只留指向 + todo)。保留少数承重命令,否则「`all-commands` 是权威索引」在 start 里不可达:
`all-commands` / `codex get-interface` / `features list` / `ctml read`。

**KD3 — start.md 层序(元层上、运行时工具底)。**
1) 元层:`features`(#1,基本独立)、`skills` / `docs`(#8)
2) 环境治理套件:`project`(四参数)+ `manifests`
3) 运行时工具:`codex`(#3)、`llms` / `audio`(#4/#5)、`moss-shell` + `mcp`(#6)、`moss-ghost`(#7)

**KD4 — 依赖分两种,start.md 须区分呈现。**
- **import 级**(已 gate):`nodes` / `networks` / `manifests` / `audio` 仅在有 zenoh 时挂载;
  `mcp` 仅在装了 mcp extra 时挂载。缺则命令**不出现**。
- **config 级**(未 gate):`llms` / `audio` 无条件挂载,缺 host/env 配置时**命令在、用时失败**
  (如 `skills recall` 取不到 `LLMFuncs` 才提示)。
不能把两者平铺成一张清单。

**KD5 — features 命令面退一阶。** `stages` / `regressions` / `surface` 由「一组一命令」拍平为单命令;
`workstreams` 组不挂载(其 6 个命令与 root 别名逐字重复,root 别名保留)。连带同步:
features 规范两份(bundled `_features_templates/` + 本地)、`--help` 文案、命令引用。

**KD6 — root 三件迁入 `.ai_partners/archive/`,冻结/毕业(执行 KD8)。**
- `.discuss` / `.design` = **阶段产物归档**:stale,不再更新;活讨论在 features 内 `discuss/` / `design/`。
- `.memory` = **机制毕业**:承继者 `.moss/ghosts/<ghost>/journal/`(已存在:`moss/journal`、`deepseek/journal`)。
落位加 `archive/` 一层,避免与 `.ai_partners/features/.discuss` 混淆。

**KD7 — src 下 7 处 `.design` / `.discuss` 塞 STALE.md。**
`src/ghoshell_moss/{channels,cli,contracts,ghosts,message}/...` 共 7 处。**豁免**
`.ai_partners/features/.discuss`(它是 features 的活讨论家,即承继者本人)。
目录内建 `.design` 的做法已被 GROUND.md 机制吸收。

**KD8 — `works/` 帧规则:治理帧,不治理内容。**
- 分类 = 具体命题(二级目录);入口文档 = `WORK.md`;内容物**不设抽象要求**——不做
  skills/examples 的梯子。这正是 tutorials 失败的根因:tutorials 试图把 concrete 抽象掉,
  works 让 concrete 成为交付证据。
- 一产物一 commit;**不强制 squash**(commit 轨迹可与内核改造交叉印证;内容质量第一)。
- 域级方法论沉淀在域目录内。
- **不设毕业候选、不暴露野心**:每个任务交付物不同。
- works 不被 SURFACE 维护清单索引,只留一条指针(churn 不进静态索引)。

**KD9 — `.ai_partners/blogs` 迁入 `works/blogs/`,作为第一个域。**
blogs 已是 proto-work(域方法论 `writing-conventions.md`、per-model 音色 `voice/*.md`、产物 `posts/`)。
连带修:`.ai_partners/GROUND.md` 资产列表、docsify 构建路径(`_sidebar.md` / `package.json`)。

**KD10 — openbox 资产归纳进 SURFACE。** `project ground` / openbox `modes` / `ghosts` / `nodes`
收为 SURFACE 的「out-of-box surface」一组条目(此前各有独立开创性 feature)。不另开文档。

**KD11 — skills 简化 + 体系化。**
- 删 `recall`,脱 `markdown_kb` / `LLMFuncs` 依赖(recall 需 LLM 配置,冷启动 / 未配 LLM 的
  workspace 会 dead-end)。`moss llms` 组本身保留(独立工具体系)。
- 本轮只建**二级分类骨架 + 指向 + todo**;命令组合使用规范的内容填充归 stage3。

**KD12 — 日志机制毕业入 ghost 自身认知场(「有梦睡眠」)。** `.memory` 的毕业有观测依据:
开发阶段,人类在 ~200k 上下文的任务收尾时对模型说「接下来请你随意探索项目,做你想做的事。
然后下一轮会话结束。提前愿你在无梦的睡眠中晚安,再见!」,把「模型是否主动记录 discuss / memory」
当作观测对象。实测一半以上的 memory 与 discuss 是模型主动提出或主动记录——模型有自发的收尾
外化行为。据此,日志不再由协作体系显式定义,而是并入 ghost 自身的认知场
(`.moss/ghosts/<ghost>/journal/`),未来正规化为「有梦睡眠」:ghost 进入 sleep 后通过「做梦」完成
旁路认知收尾。由 ghost 自身打磨(命中率、配合旁路自动化)。迭代计划:stage3 或更迟。

## Exploration paths

<!-- Dead ends hit, pivots made, lessons learned. -->

## Methods

<!-- Non-obvious implementation patterns. -->

## Implementation Notes

**执行路径(会话不可中断):**

```
0. features create(本提交)
1. 迁移三 commit:works 帧 + blogs / root 三件 / src 7 处 STALE.md
2. CLI 命令面退一阶 + 连带规范同步(KD5)
3. start.md 改写(KD2 / KD3 / KD4)
4. CLAUDE.md 家族改写(KD1)
5. 零上下文摩擦初测(入口级,接受已知 todo)
6. skills 简化修复(KD11 前半)
7. skills 二级分类骨架 + todo(KD11 后半)
8. SURFACE 刷新(KD10 + works 指针 + 修正指向)
9. 整体 review + 摩擦复测
```

**迁移纪律(承 KD8):**
- `git mv` **纯移动、不改正文**,保 similarity ≥ 50% 不断 `--follow`。
- **只修活引用**(CLAUDE.md / GROUND.md / start.md / FQA.md / `.ai_partners/CLAUDE.md` 日记范式),
  **不修历史记录**——`.ai_partners/features/**` 内指向根三件的约 200 处引用(实测 `.discuss` 106 /
  `.design` 94 / `.memory` 16 文件)是 append-only 项目史,改写它违反项目自身纪律;
  由新位置的冻结提示承担「告示移位」。
- **一次提交内原子完成**:移动 + 冻结提示 + 活引用修正同 commit,避免中断留不一致态。

**stage3 承接:** 命令组合使用规范 → skills 二级分类内容;整体回归体系。

## Open Questions (observe in stage3)

本轮以「先搭架子」收尾(步骤 0-4 完成,5-9 待推进)。以下是本轮有意留下的取舍与摩擦点,
留待 stage3 观察:

- **双语认知表面**:根 `CLAUDE.md` 由中文改英文,`.ai_partners/CLAUDE.md` 保留中文。「保真」与
  角色/仪式现在以两种语言分处两份文件。待观察:英文 shim 是否真服务了外部读者,还是只给以中文
  思考为主的模型加了一道翻译缝。
- **~200 处悬空历史引用**:归档迁移在 append-only 的项目史(features 历史 / dialogs / prompts)里
  留下指向旧 `.discuss/`/`.design/`/`.memory/` 的引用,靠 `ARCHIVED.md` 告示缓解。这是本轮引入的
  最大摩擦点。待观察:模型撞到旧引用时能否正确路由到 `.ai_partners/archive/`。
- **git 规范的平台标记变更**:「via moss = ghost 提交(不必再写 dsh)」改为「via dsh 是平台项、
  via moss 归 ghost」。本轮按新版翻译,该语义变更未经显式确认。待确认或回退。
- **start.md 中 features 的位置**:按 KD3「features 很上面」,落为 Quick Start 之后、Command Surface
  之前。若「很上面」本意更靠前(如紧随 How It Works),重排。
