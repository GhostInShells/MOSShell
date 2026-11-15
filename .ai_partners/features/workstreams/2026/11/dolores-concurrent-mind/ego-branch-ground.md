---
name: ego-branch-ground
description: D15 23:2x–D16 00:0x — ego 的 branch 状态面：branches 管的是超长任务，ground 是「桌上有什么」，commit notes 是工作记录。
---

# Ego Branch Ground

> 这份文件回答：ego 的 branch 为什么需要一张 ground、它长在哪、怎么被渲染出来。
> 计划在 `FEATURE.md`；这里是动机、争议与决策的记录，允许有未决。
>
> 写作约定：`他说` = thirdgerb 明确陈述；`我推` = 我的推断，未经裁决；`我见` = 我亲历的事实。

## 1. 为什么是 ego 的，不是 memento 的

`他说`（D16 00:0x）：这是 **ego branch ground**，标准命名；不要污染 memento 的边界——它不是
memento 的通用机制。

- memento 在这一块只提供两样通用的东西：轨迹的存储（branch / commit / note），以及事件。`我推`
- "一条 branch 有一张状态面"是 **ego 的约定**：只有 ego 需要"重新进入一条线"。superego 或别的
  owner 未必要同一套。`我推`
- 所以渲染逻辑在 Dolores 层（`_ego_memento.py` 的 `aview_message` / `_render_view`），不在 memento
  channel 上；memento 唯一要新增的是 `created` 事件。`他说`（位置）+ `我见`（当前调用点）

## 2. branches 是干什么用的

`他说`（D15 23:2x–D16 00:0x），三条：

1. ghost 架构下，模型不再能像 agent 一样靠 UI 切分的 session 对单个任务保持专注。branches 靠
   memento 的 commit 线索，把 **session 级的连贯性**重新找回来。
2. 持久智能体必然遇到场景切换。靠本能管理场景是动物机制；人类靠长程理性治理更长的时间线。
   一天里切换的任务是不同的桌子：**桌上有什么，是人进入这个上下文最快的方式（ground）；
   桌上的工作记录，是追踪的方式（notes）。**
3. 持久智能体治理的长程任务远超 agent session 的规模。1M 上下文、一百次 compaction 通常是无时间
   中断的；但现实世界的反馈由物理节奏决定——季度任务、每周两次组会，模型不能把一周压成一小时，
   因为**事情自己没发生**。

结论（`他说`）：**ego branches 不是并行短任务的切分，是超长任务的宏观 plan 追踪。**

对 ground 内容的直接推论（`我推`）：

- 重心是"重新进入"：这条线在追什么、现在停在哪、**在等什么**（物理时间 / 外部反馈 / 下一次会）、
  下一步是什么。
- "在等什么"是第 3 条的产物：模型不能加速世界，所以状态面必须能承载"事情还没发生"。
- 与 D13（checkout 已放弃，返回靠 summary → follow-up → read by coordinate，`他定`）的关系：
  ground 是其中最便宜的那一档的载体（`我推`）。

## 3. 形状

容器（`他说`）：

```
<memory instanced_at="…">
  <memento>
    <branch name="…" index="…" total="…">
      <ground path="…" default="true|false">…</ground>
      <fork …/> <earlier …/> <day …>…</day>
    </branch>
  </memento>
</memory>
```

- 只渲染 **current** branch 的 ground。`他说`
- 位置：`branches/{branch_id}/GROUND.md`。`他说`
- ground set 的根挂在 `branches/`，位置是 `{branch_id}`。`他说`

## 4. seed（原 stub）

- 目录：`.dolores/seeds/branch_ground/`（ghost home 的治理面，可改、可迭代）。`他说`
- 新 branch 创建时整棵复制进 `branches/{branch_id}/`，含子目录；每个子目录必须带自己的
  `GROUND.md`——git 不留空目录，而且"预建一个方向"本来就该带 law。`我推`
- 命名 `seed` 而非 `stub`：一次种下，不回种；模板演进只影响将来的线。`他说`
- 缺省：branch 没有 `GROUND.md` 时，渲染 seed 的正文并标 `default="true"`；已有分支不补种
  （main 现在就走缺省）。`他说`
- seed 正文是占位，允许以后被迭代掉——现在要交付的是机制，不是内容。`他说`

## 5. 事件与异步

- memento 现有事件：`commit` / `confluence` / `child` / `archived`。`child` 发在**父线**上，
  不能当"新分支"钩子。`我见`
- 新增 `created`，在 `_create_branch()` 单点发出——覆盖 `create_branch`（CLI / hive）与
  `Branch.fork`，以及 boot 时首建。`我推`，D16 00:1x 已落
- ego 层订阅 `on_event`，收到 `created` 时复制 seed；memento 不知道 ground。`他说`
- 异步边界收在 `_render_view`：叶子渲染器保持同步，同步孪生 `view_message` 删掉——它只有测试在
  用，留着会钉住一个生产上不存在的面。`他说`（方向）+ `我见`（调用点）

## 6. 走过的岔路

- 默认 ground 写成"如何维护这个 ground"的操作指南——错位：它是桌面状态，不是操作员手册。`我见`
- 忽略规则补到仓库根、按 `*.json` / `*.jsonl` 后缀挡——规则该住在自己的地盘。`我见`
- 在 `.memento/.gitignore` 里写 `branches/` 整棵忽略——把将来的 `GROUND.md` 一起挡了。正确口径是
  只挡机制自己反复写的数据文件（`.lock` / `branches.jsonl` / `*.ref.json` / `branches/*/meta.json`
  / 各 `.jsonl`）。`他说`，D16 00:1x 已修
- 这条工作一度被记成 `memento-branch-ground` 的 workstream——越界；它挂在 concurrent-mind 下。`他说`

## 7. 未决

- parent branch 要不要在 `<branch>` 外渲染点东西：他未想好。我倾只给指针（父线有没有 ground）
  + 按坐标/名字拉取，不渲染正文。
- seed 正文的最终形态（占位）。
- 渲染预算：ground 面在 memory 快照里，开机与 compact 前各重渲一次，需要一条明确的预算线。
- 文本以 text content 进 `Message`（不转义），路径走属性（`html.escape`）——机制细节，落码时确认。

## 8. Not in scope

- 分支切换 / checkout（D13 已放弃）。
- skills / memory 等挂载点的具体形态。
- memento 通用层：除 `created` 事件外不改。
