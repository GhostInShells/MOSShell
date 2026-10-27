---
title: Ghost Ground Governance
status: in-progress
priority: P1
created: 2026-10-27
updated: 2026-10-27
depends:
- ghost-ground
milestone:
description: >-
  A dolores ghost governs three grounds at once — ghost home (mind), mode home
  (body), project (environment). One channel, one control plane, content carried
  by notice; plurality lives in a manager, and GroundSet stays single-rooted.
---

# Ghost Ground Governance

> `moss features set-status ghost-ground-governance <status> -m "note"`

## Motivation

Ground 是模型可以自迭代的那一层（法在自己手上，harness 只是注入）。但它现在只能挂
**一个**根场：`DefaultGroundSet(workspace_root=ghost_home)`，而且只有根场的 body 会
被顶进 ghost 的 instruction（`_runtime.ground_instruction()` → `_ground_text` →
memories 里的 `ground` 消息）。

一个 ghost 活在三个 home 里，三者的法回答不同的问题：

| home | 回答 | 生命 |
|---|---|---|
| ghost | 我是谁、我怎么想、我记得什么 | 与 session 同寿 |
| mode | 我是什么身体、怎么动怎么听 | 与 session 同寿；**多个 ghost 共享** |
| project | 我在干什么活、这摊活什么规矩 | 同寿（单次启动内不可变，改要重启） |

三个 home 可能完全不在一个目录层次里，所以它们的地址是**启动事实**（matrix 手上的绝对
路径），不能靠 `../../..` 这类相对路径表达——那既是目录巧合，也会顺带继承上级的法。

## Context — what exists today (verified in code, not assumed)

- `DefaultGroundSet.__init__` 建 root，并按 root 的 `groundset:` **物化一层**子场。
- `_construct` **不**物化子场 → 非 root 场的 `groundset:` 现在是被静默忽略的。
- `_materialize_children` 内部写死 `self._root`，所以它只对 root 成立。
- `ground_channel.build(groundset, ...)`：root 的 body 进 channel instruction，其余
  active 场挂成**只带 notice（frame + meta）、不带命令**的子 channel。
- `_runtime.ground_instruction()` = `root_ground.render()`，含 body **与 pins 渲染结果**。
- `collect_chain` 默认向上爬到 `$HOME` —— 没有 GROUND.md 的目录会**盗取**上级的法。
- 已落地：根场 ignore `.*/` + `!.ai_partners/`，并用 `src/ghoshell_moss/ghosts/GROUND.md`
  的 `stop_on_match` 挡住 `stubs/`（commit `5fafccd5`）。

## Key Decisions

### D1 — Path 3's tree; plurality lives in a manager, not in GroundSet

树：一个根 channel → 每个 home 一个内容子 channel → 子场再一层。复数由
`GroundsManager` 持有 **N 个各自单根** 的 `GroundSet` 提供；channel 是 manager 的视图。

rejected：把多根塞进 `GroundSet`。它的每一处简化（`_materialize_children` 用
`self._root`、平铺 label 空间、构造期一次物化）都以"一个根"为前提；多根会把每一处都
变成特例。复数是父层的职责。

rejected：一个 channel 挂多个 `GroundSet`（命令面重复；CTML 发现面没有 interface 去重）。

### D2 — instruction 只讲 channel，不装内容；ghost 的 instruction 槽不再装场内容

`instruction` 是"这个 channel 怎么用、这三层是什么关系"，冷、稳。**渲染出来的场内容
一律不进 instruction。** ghost 的 instruction 槽因此空出来，内容全部走 notice。

理由：instruction 是**整段重发**（见 `channel_builder.instruction` 契约），任一段变动
都重发全部；named notice 是**逐片段**重发。同样的三份 body，放错地方成本是三倍。

### D3 — 内容面 = 子 channel + notice，路径即来源

key：`ghost-home` / `mode-ground` / `project-ground`。子场按 `前缀+label` 命名。保留
channel 路径（而不是把所有东西摊平成字符串 key），因为**来源要可寻址**：要 refresh 单个
子场、要指出"这个子场坏了"，都需要地址。

### D4 — 四值语义保持，并定义映射

| 现实 | 值 |
|---|---|
| 根场不存在（无 GROUND.md） | 缺席 → `removed` |
| 存在但解析/加载失败 | 业务零值 `broken: <原因>` |
| 存在且为空 | 业务零值 `empty` |
| 未变化 | `""`（零 token） |
| 有内容 | 文本 |

### D5 — 不自动热更新；过时性不进 notice，改走冷读

`refresh` 是命令，不是时钟。**过时性不能写进 notice 正文**：那会让每轮都判定"变了"，
每轮全量重发，等于把 delta 机制废掉。改由冷读 `stats()` 回答：每个根上次渲染的时刻、
源文件此后的变动。

### D6 — 存在性在装配期判定，缺席可见

装配时逐个 `(home / "GROUND.md").is_file()`；缺失的不挂，并按 D4 显示为 `removed`。
不静默、不继承。

### D7 — 子场常驻有上限，超出的折成 TOC

一个根的子场可能有几十个。超过阈值后只挂前 N 个 + 一条 TOC（对齐 `render --meta`）。
阈值是配置，不是常量。

### D8 — 法链 boundary 钉在挂载点

每个挂载根的场，`collect_chain` 的边界 = 它自己的根。否则一个没有 GROUND.md 的子目录
会爬出 home，把别的层的法当成自己的。

### D9 — 所有 mode 默认不挂 ground channel

mode 是能力组装，不是知识域；共享给多个 ghost，写成约束。

### D10 — AGENTS.md / CLAUDE.md 归 dsh，ground 不做兼容

两条发现链互不引用。ground 只管 GROUND.md。

## Acceptance criteria — 写完再看，重启后逐条对

1. facade 里**只有一个**带命令的 ground channel；三层的命令面不重复。
2. ghost-home / mode-ground / project-ground 各以一个内容子 channel 出现；缺 GROUND.md
   的那个显示为 removed（status line，而不是消失得无声无息）。
3. 每个根的 `groundset:` **都生效**（非 root 的 home 不再被静默忽略）。
4. ghost 的 instruction 里**没有**场内容。
5. 项目根的索引里没有 `.moss`（已达成，回归项）。
6. `refresh` 能指名刷新单个根；`stats()` 能说出每个根的上次渲染时刻。
7. 冷启动路径可回退：`MOSS_GROUND_MULTI_HOME=0` 走旧路径并能起来。

## Execution plan

1. 本记录 + commit（**先落盘再改代码**：破坏性重启丢了上下文，还能靠它恢复）。
2. 在本地分支 `ground-multi-home` 上实现，一次做完。
3. **重启前先做进程内装配验证**：构造 manager + channel，渲染三根的 notice，不依赖
   ghost 启停。新机制挂掉的症状是"我不会再回来"，没有第二个人能手修。
4. 重启，逐条对 Acceptance criteria，然后自己打磨。
5. 通过后合并回 dev。

## Open questions

- mode home 现在没有 GROUND.md（它有 `HOST.md`）。是让它长期缺位，还是承认它该有自己的场？
- 子场 TOC 的阈值定多少（先按 8 试）。
- `stats()` 属于 channel 命令还是外壳 CLI（暂定 channel，因为它是我要读的东西）。
