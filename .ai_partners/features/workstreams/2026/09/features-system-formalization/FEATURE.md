---
created: 2026-09-28
depends: []
description: 将 features 目录从 workstream 追踪器升格为完整体系（三元开源第二元：人机协作体系）—— 四轴 workstreams/stages/regressions/surface，generic
  规范 + 一键初始化 + CLI + works/ 对外产出面。
milestone: null
priority: P0
status: in-progress
status_note: decision ledger recorded from stage2 discussion
title: Features System Formalization
updated: '2026-09-28'
---

# Features System Formalization

> Use `moss features set-status features-system-formalization <status> -m "note"` to update state.
> See [TOPOLOGY.md](TOPOLOGY.md) for directory layout and [README.md](README.md) for the full convention.

## Motivation

`features/` 现在只是一个 workstream 追踪器。这次把它升格为**三元开源的第二元：人机协作体系**
（三元的分野见 `.ai_partners/CLAUDE.md`）——与项目无关的 generic 体系，MOSS 只是它的一个实例。

MOSS 与其它项目开发的本质区别：**它是自己的 ghost 与人类协作开发自身，过程可观测，产物可分享。**
这句话给体系钉了两个端点：`surface` = 过程可观测的锚，`works/` = 产物可分享的出口。
普通项目的 features 体系不需要这两个；一个自举的 AI 项目需要。这是 generic 规范能立住的 Why。

升格后 `features/` 是一个**一键初始化**的体系目录，内含四条轴 + 一份静态索引（surface）。

## Design Index

- 母讨论：`.ai_partners/features/.discuss/2026-09-17-features-system-preparation.md`
  （Part 4 .ai_partners 四轴调研、Part 5 中心化收敛、Part 6 单帧真值/切片、Part 7 stage 产物清单）
- 现有轴规范：`.ai_partners/stages/README.md`、`.ai_partners/regressions/README.md`
- 先行 workstream：`stage-tracking-convention`、`regression-tracking-convention`
- bundled 模板（随包发布）：`src/ghoshell_moss/core/codex/_features_templates/{README,TEMPLATE,TOPOLOGY}.md`
- 本 workstream 的 `discuss/`、`design/`（待建）

## Key Decisions

<!-- 这是下一个 AI 实例读到的第一手决策。按 KD 编号追加，不回改。 -->

**KD1 — 定位：generic 第二元。** features 体系与项目无关，可被任意项目 `moss features init`。
`README.md` 标题的 "MOSS" 前缀去掉；bundled 规范必须去 MOSS 特化，本地副本承载 MOSS 实例。
否则下一次 `init` 会把 MOSS 实例覆盖回 generic。

**KD2 — 四轴 + 一索引。** `features/` 下：`workstreams/`（L1 意图，连续）、`stages/`（L2 意图，大周期）、
`regressions/`（验证，正交）、`surface`（系统轴切片，阶段间静态）。核心机制：**stage = 时钟/触发点，
不是容器**——它在边界处圈定 workstreams 范围、提名 regression 关键集、刷新 surface。
（对应 Part 6「milestone=时间轴切片，架构面=系统轴切片」）

**KD3 — `facade` 改名 `surface`。** facade 在软件里是 GoF 简化接口，日常义"假面/表里不一"，
与本项目"保真于已发生"的底色冲突。项目已有术语"架构表面/架构面"（Part 7、Part 4/6）。
命令 `moss features surface`，文件 `SURFACE.md`。

**KD4 — workstream 准入 = 触发问题，不是规则。** 入 `workstreams/` 的必须是"对项目塑型的架构迭代"；
判据是一个问句：**"这是哪条领域命题的 few-shot？它给产品形态自述贡献哪一句？"**
产品形态自述 = surface 的第一段。可拓展的功能/应用有自己的 `features/`（外部项目默认 `$CWD/features`）。
（遵循 09-17「B 层要触发点，不要规则」）

**KD5 — surface 不是大文档。** 代码是第一真相。surface = 针对 features 建立的**维护清单**，
第一段是产品形态自述。守"索引树的顶"：指向 codex architecture / CLAUDE / skills / docs，**不重写**；
**不索引 churn 的 regressions**（静态索引不索引会腐烂的东西）。每个 stage 开始产生新预期，结束收尾。

**KD6 — regressions 是异种兄弟轴，不嵌套进 stage。** workstreams/stages 是意图轴（L1/L2）；
regressions 是验证轴，与两者正交。regression set 不终止（`expired≠completed`），
键 = **验证范围**（非 stage；scope↔feature 非 1:1，`dependency-install`/`ghost-runtime` 是横切范围）。
stage 只**按名字引用**关键集（沿用 stage-tracking-convention KD7: names, not paths）。

**KD7 — benchmarks 独立，debates 缓。** benchmarks 暂无验证定位，不属验证轴，独立留
`.ai_partners/benchmarks/`。debates 未来可能进 `works/`，现在不动。

**KD8 — root 三件冻结 + 迁入 `.ai_partners/`，最后一步。** `.design`/`.discuss`/`.memory`
冻结提示写在各自目录内。**两种冻结性质不同**：
- `.design`/`.discuss` = **阶段产物归档**：stale，不再更新，活讨论搬去 features 内 `discuss/`/`design/`。
- `.memory` = **机制毕业**：它验证了 ghost daily（开发完成后模型会主动写）。承继者是
  `.moss/ghosts/<ghost>/journal/`（同构 `Y/M/D/daily.md`）。提示须记录**验证结论** + 指向承继者，
  否则经验依据被归档到失联。

注：`.design`/`.discuss` 在 `src/` 下另有 7 处（contracts/ghosts/message/cli/channels），不在本次范围。

**KD9 — CLI。** 路径解析改为：`--dir` > `$CWD/features` > `$CWD/.ai_partners/features`（后者保留为
moss 历史遗产兼容）；**向上走**（git 式）。新增四组 `workstreams`/`stages`/`regressions`/`surface`；
平级旧动词 `list`/`create`/`set-status`/`status`/`check` **保留为 workstream 别名**（不破坏兼容）。
surface 的 CLI 不同类（读/刷新，无 create）。`init` 扩成四轴一键 scaffold。
`features_app` 的 help 文案要改（现写着 "not a project capability catalog"，与新定位正相反）。

**KD10 — `works/` 根场（新）。** 对外产出面：可丢弃、可分享。承载实用功能/文章/视频/观点；
`.ai_partners/blogs/` 迁入（它带 docsify 站点构建，是发布面不是笔记堆）。**不被 surface 维护清单索引**
（churn 不进静态索引），surface 最多留一条指针。需自己的 `GROUND.md` + `README.md`。

**KD11 — project ground 不再强读 `claude.md`。** 认知入口从 claude.md（工具私有）挪向
features/surface（工具无关）。这是"从 claude code 进入 → 在 ghost 里协作"这条弧线的第一步。

**KD12 — 规范有两份，须同步。** bundled `_features_templates/README.md`（15942B，与本地逐字节相同）
与本地 `.ai_partners/features/README.md`。重写规范 = 改两处；bundled 去 MOSS 化，本地承载实例。

**KD13 — surface 文档规格（`SURFACE.md`）。** 位置 `features/SURFACE.md`（MOSS 即
`.ai_partners/features/SURFACE.md`），由 `moss features surface` 呈现与维护。它是**清单，只指向，
不重复记录内容**：不陈述真相，只圈定要维护的范围。frontmatter：`description` / `version` / `updated`。
它描述 **L3 产物**（项目对外呈现的整体面貌），本身也是 L3 收官的一项。刷新时机：**大 Stage 收官时，
且仅当清单级改动才改**；单文件，无快照，历史靠 `git log`（此机制写进模板）。第一段是**更 meta 的介绍**
（愿景 + 迭代路径的压缩；定位细节指向 README）。末尾含「对 Surface 的回归」节：L1→L2 迭代收官时
逐条回归指向是否仍解析、能力是否仍存在。
`version` 语义（**待确认**）：暂按「本文档修订计数」（结构变化 +1）而非发布版本号——发布面貌由
`updated` + git 承载，避免 surface 成为版本的第二来源。

## Implementation Notes

**两步、两个提交：**
1. **改造**：四轴建立 + surface + spec 重写 + CLI + ground + `start.md` + `CLAUDE.md`
   + `stages/`、`regressions/` 用 `git mv` 迁入。
2. **迁移**：root 三件（`.design`/`.discuss`/`.memory`）冻结提示 + `git mv` 迁入。
   （`works/` 建立归 (1) 还是 (2)：待确认）

**`git mv` 保留历史**：git 不跟踪目录；纯移动 similarity 100%，`git log --follow <文件>` 可追溯，
目录级无历史。迁移提交里**只移动、不改正文**；stale 小头可加，但别在大改同提交（可能跌破 50% 阈值断链）。

**协作模式（本次）**：不用 planmode / tasks；走一步算一步，人类投入核心精力，每步对齐。
**会话不可中断**——半途而废 = 体系对自身的描述自相矛盾，比不改更坏。

**连带文档（必须同提交）：** 根 `CLAUDE.md`、`.ai_partners/CLAUDE.md`（含 `.memory/daily/` 日记范式一节）、
根 `GROUND.md`（`@claude.md`）、`.ai_partners/GROUND.md`（现平列 features/regressions/stages/benchmarks）、
`features/GROUND.md`、`stages/GROUND.md`、`src/ghoshell_moss/cli/start.md`；
`pyproject.toml:75` 的 exclude 项（根目录三件本在 `src/` 外，属 vestigial，可选清理）。

**待定：** `works/` 是否有"毕业进框架"路径（在 works 原型 → 证明有用 → 取得某领域命题的 few-shot 身份 +
workstream）。有则 works 与 features 有耦合，否则完全隔离——定性不同，需人类定。