---
created: 2026-07-24
depends: []
description: 阶段级开发轨迹机制 (.ai_partners/stages/) — 承载一个时间段的方向、目标、 验收与复盘，填补 features(战术)
  与 milestones(时刻) 之间的战略层空缺。
milestone: null
priority: P1
status: completed
title: Stage Tracking Convention
updated: '2026-07-31'
---

# Stage Tracking Convention

> Use `moss features set-status stage-tracking-convention <status> -m "note"` to update state.

## Motivation

现有 `.ai_partners/` 造物全部锚定在**已发生的事实**上：features 是从代码反向
指回上下文的索引 (spec 明令 skip 进度百分比/checklist)，milestones 记录已发生
的时刻，debates/regressions 锚定既有设计或可执行现实。整个体系的认识论底色是
**保真于已发生**。

缺的是**战略层的定向坐标**。项目自 2026-02-15 决定纯开源以来经历了清晰的阶段
演进 (内核重构 → CTML 1.0 → 概念手写 → features 体系 → matrix cell 改造)，
但这个演进过程**不是显性的**。后果：

1. 入场调研的模型只能从代码现状推断意图，而代码现状推断出的永远是"谨慎/防御性
   夯基础"——与项目实际所处的探索冲刺阶段**恰好相反**。模型因此持续 PUA 人类
   工程师做防御性研发。
2. 人类工程师自己有漂移倾向，缺少阶段性锚点。
3. 外部观察者不知道项目在什么阶段、在做什么，看起来像随机迭代/盲目扩充。

Stage 机制解决这一个问题：让"一段时间在做什么、为什么"成为一个可观测的造物。

## Key Decisions

<!-- Record each meaningful design choice. This is what the next AI incarnation reads first. -->

**1. Stage 承载轨迹，不承载真相。** 与 features 同构的哲学。阶段的目标/动机/复盘
是**主观声明** (作者书写的意图)；进度状态从关联的 features 观测出来，不在 stage
文件里复制。这样 stage 不存储会腐烂的东西——腐烂的部分永远从真相重新观测。

**2. 定位在 features 与 milestones 之间。**
- features = 战术级 (单 workstream 怎么做)
- **stages = 战略级 (一个时间段的方向与交付目标)**
- milestones = 时刻级 (完成了什么)
前瞻性是 stage 独有的时间性——它是 `.ai_partners/` 下唯一朝前看的造物。这是它
和整个体系张力的来源，也是它存在的理由。

**3. 独立目录 per stage，不平铺。** 拒绝了单文件平铺方案 (milestones 模式)。
理由：一个 stage 会产生**复数的次级造物**——里程碑是多个独立时间戳的文件，
未来可能出现 pivots/sketches 等新类型。平铺方案在第一个次级产物出现时就破防，
一级目录毫无扩展性。每个 stage 是一个目录，`STAGE.md` 作目录锚点 + 本目录索引。

**4. 生命周期在单个 STAGE.md 内完成：planning → active → completed。** 复盘
(Retrospective) 不是另建一个 milestone 文件，而是 stage 文件自身从"意图"走到
"记录"的终点。STAGE.md 自己就是那个 milestone。与 `features set-status completed`
同构。

**5. milestones 两层语义分层，不合并。** 全局 `.ai_partners/milestones/` 保留
为**模型协作者手写的 curated 高光时刻**；stage 内 `milestones/` 是**阶段内部的
操作日志** (规划内 + 涌现的里程碑)。历史的 6 个全局 milestone 语义上都属于
beta1 阶段，本次迁入 `2026-07-beta1/milestones/`——因为不想删这个 stages 体系，
用真实内容验证它。

**6. 暂不进 start.md，不配专门 CLI。** 保持"不好用就整体删除"的低成本试验姿态。
用 ground 治理观测 (@ README + ls，最多 glob */STAGE.md)，ground 缺什么补什么。
命名定为 `STAGE.md` (单阶段声明) + `ROADMAP.md` (跨阶段索引)——ROADMAP 是索引不是
阶段本身，语义不冲突。

**7. 关联用名字，不用路径。** Associated Workstreams 只写 workstream 名，通过
`moss features list` 等各机制自身的索引解析。硬相对路径在改名/迁移时就是死链，
而名字改了也能从 git 历史分析出来。stage 不做精确路径关联。

## Implementation Notes

<!-- Gotchas, non-obvious behaviors, reasons for rejecting simpler alternatives. -->

目录结构：
```
.ai_partners/stages/
├── README.md          # 机制说明书
├── ROADMAP.md         # 跨阶段索引 (active/planned/completed/cancelled)
├── _template/
│   ├── STAGE.md
│   └── milestones/MILESTONE.md
├── 2026-07-beta1/
│   ├── STAGE.md
│   └── milestones/    # 迁入历史 6 个 milestone
└── 2026-08-v0.1.0/
    └── STAGE.md
```

命名规范：`YYYY-MM-{identifier}/` (起始年月 + 短标识符)。

Ground 观测需求 (step 4 后验证)：pin 需要能读 frontmatter status、统计 Goals
checklist 完成度、按名字聚合 Associated Workstreams 的 feature 状态。若 ground
当前 pin 动词表缺这些能力，在此 feature 内补 ground。

执行顺序 (与人类对齐后修正)：1) 本 feature 引导构建 + 命名 → 2) 极简规范
(README + TEMPLATE) → 3) 落地 beta1 + v0.1.0 两个真实阶段 → 4) 用 ground 验证
可观测、修 ground。注意 3 在 4 之前：先有真实内容，再验证观测。历史阶段作为
背景写进本 feature (已在 Motivation 记录)，不强制补录为独立 stage 文件。

**收尾条件 (2026-07-24 与人类对齐)** — 机制创建已完成，feature 挂 in-progress
直到以下两项落地：

1. **Ground 观测验证**：`stages/GROUND.md` 已写 (3 pins: roadmap file /
   stage_status frontmatter pattern / stage_dirs ls)，但 frame/observe 实测
   留给正在打磨 ground 的另一会话。本会话不动 ground。
2. **进入项目认知入口**：beta1 结束时，stages 机制写入 CLAUDE.md 和根目录
   README。在此之前刻意不进 start.md — 保持"不好用就整体删除"的试验姿态。
   进入认知入口 = 机制被正式采纳，也是本 feature complete 的时点。