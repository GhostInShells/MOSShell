---
created: 2026-08-05
depends:
- ghost-ground
description: Ground 在 Ghost 运行时里的薄 channel：法链跨 compact 存活 + 场开合。 open/close 把场挂成 virtual
  child (instruction=meta, help=帧, shell trajectory diff 重供)；pin 升为 channel 命令 (pin_{verb})，编辑模式折叠；render
  无状态 peek。
milestone: null
priority: P1
status: completed
status_note: virtual children + edit-mode folding implemented, dogfooded via MCP
title: Ground Channel — 认知场的运行时落点
updated: '2026-09-21'
---

# Ground Channel — 认知场的运行时落点

> Use `moss features set-status ground-channel <status> -m "note"` to update state.

## Motivation

ghost-ground 的 CLI 层（spec/init/frame/meta/observe/validate）已完成并 dogfood 验证。
ground channel 是它在 Ghost 运行时里的落点。

核心事实：`read`/`frame` 的命令结果活在对话历史里，**历史会被 compact 压掉**；唯一跨
compact 不丢的载体是 channel static（moss_static 前置、跨 fold 缓存）。所以 channel 的实质
= 把一个根的法链放进 static。其余一切（pin 内容、frame）都是函数调用（= CLI 等价），
模型爱用 bash 调 `moss ground` 也行 —— channel 与 CLI 的差异仅此而已。

## 设计结论（当前形态）

- **法链进 static**：channel 唯一跨 compact 存活的是根的法链（祖先 GROUND.md body，
  root-first，一次求值）。
- **场开合 = virtual children**：`open(dir)` 把场挂成 command-less 子 channel ——
  子 `instruction=meta`（cd + $id + pin TOC + 法链计数），`help=帧`（body + pins 内容，
  每 refresh 重算，shell trajectory diff 增量重供）。`close(label)` 撤下（dirty 落盘）。
- **render 无状态 peek**：`render(dir)` 渲染一个场的帧，不挂子 channel（cat vs cd）。
- **pin 升为 channel 命令**：`pin_{verb}(ground_file, label, ...约定参数)` 增改指定 GROUND.md
  的 pin（同 label 覆盖），目标不存在报"不存在"不创建。CLI 早在 K48 就撤了 pin，落点在 channel。
- **编辑模式折叠**：`pin_* / spec / validate / templates` 用 command 级 `available` 折叠，
  `edit(on)` 开关；构造 `edit=False` 默认折叠。`available=False` 在 interface 渲染被跳过
  (prompts.py:40) + 执行被拒 (channel.py:813)。
- **注入持有 GroundSet**：`new_ground_channel(groundset, ...)` 持一个 GroundSet 承载开合生命周期。
- **无对账**：拉模式，内容按需 `render`/`open` 拉进，读到即新鲜，无 hash shadow/stale/update。

## 冷/温/热

| 层 | 内容 | 载体 |
|---|---|---|
| cold | 根法链 + 机制 prose | 父 instruction（一次求值，compact 免疫） |
| warm | 帧（body + pins 内容） | 子 channel help（每 refresh diff 重供） |
| hot | （可选）变更信号 | context_messages（默认空） |

## 关键 trade-off

- **帧进 help 不进 instruction**：`instruction` 是"生成一次"的冷层，帧是易变内容 —— 项目根
  body 是 `@claude.md`，render 会 @-展开成整份 CLAUDE.md，几千 token 冻进 static 违背
  "冷层付一次恒 0.1"。故帧走 help（diff 增量重供）。
- **法链快照陈旧**：cold 法链"一次求值"，编辑 GROUND.md（pin_* 或人工）后 static 冻结、
  无刷新路径。"compact 免疫"的收益以快照陈旧为代价。
- **拉模式无主动变更信号**：无对账 = 模型不记得重新 render/open 就看不到编辑；hot 层
  （变更信号）默认不建 —— 主动推送留待后续。
- **edit 折叠 vs 常驻全命令面**：默认折叠省 token，代价是 pin 命令要 `edit` 一次才可见。
- **open 返回 meta 与子 instruction 重复**：接受冗余，open 返回 meta 给即时反馈。

## 命令面

常驻 `open / close / render / edit`；编辑组（fold）`pin_file / pin_glob / pin_frontmatter /
pin_ls / pin_exec / pin_law / spec / validate / templates`。

## 参照

- 上游协议：`ghost-ground`（`src/ghoshell_moss/ground/SPECIFICATION.md`）
- 参照框架：`context-cache-engineering`（冷/温/热）、`channel-meta-dyn-static`（LSM 静态前置）

## 历史

早期"薄 channel、无子 channel、pin 降 CLI"设计后被 virtual children 取代；完整推演轨迹见
`git log -- .ai_partners/features/workstreams/2026/08/ground-channel/FEATURE.md`。

## 设计结论（2026-09-21 追加）：帧缓存 + 显式 refresh，meta 落点迁移

> 演进（非推翻）上文的「设计结论（当前形态）」。其中「场开合 = virtual children」一条的
> 「子 `instruction=meta`、`help=帧`、每 refresh 重算」被替换；其余锚点不变。上一段的
> 「冷/温/热」表 warm 行（帧 → 子 channel help）同步失效，新形态见下。

- **meta 从 instruction 迁到 named notice**。`instruction` 没有 delta 载体 —— shell
  trajectory 的 `diff_facade` 只比 states / notice / commands，放 instruction 等于每 epoch
  只发一次、之后永不更新（实测：virtual 子节点在 `make_static_block` 还被显式排除）。子
  channel 改为 `named_notices`，`meta` 分片（身份 + pin TOC）+ `frame` 分片（body + pins
  内容），逐片段 diff 增量重供。
- **帧缓存**：`_cached[label] = {meta, frame}`，只在 `startup` / `open` / `refresh` 三个
  显式点重渲染；`notice` / `named_notices` 只读缓存。场里的 exec pin 因此不再随每次
  `refresh_metas` 反复起进程（此前实测每次 refresh 起一次进程、~25ms，而读缓存 ~1.3ms）。
- **新增 `refresh` 命令**（`always_observe=False`）：重读法（`ground.load()`，脏场跳过以
  保留未落盘改动）+ 重渲染，返回 ack，新内容走下一帧 facade delta 送达。`render` 仍是
  无状态 peek，不入缓存。
- **代价是显式纪律**：场不再随刷新自更新 —— 模型改了场文件（含 GROUND.md 本身的
  pins/body）必须调 `refresh` 才看到。这是把"每刷新自动成本"换成"按需付费"。
