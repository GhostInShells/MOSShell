---
title: Ghost Ground Governance
status: in-progress
priority: P1
created: 2026-10-27
updated: 2026-11-13
depends: []
milestone:
description: >-
  重做 ground —— 把渲染策略收回 GROUND.md 自己身上。GroundSet 退化成带边界的
  解析器，Ground 变成无状态的一次渲染，子场由父场点名递归渲染，channel 只剩
  render，ghost home 直接进 memory。
---

# Ghost Ground Governance

> `moss features set-status ghost-ground-governance <status> -m "note"`

## 0. 这份文档为什么被整体重写

上一版（2026-10-27）写的是"一个 ghost 治理三个 home"：manager + 每 home 一个内容子
channel + 子场挂成只带 notice 的空 channel。那套东西已经落地又拆了，而拆完之后问题
反而更清楚：**方向从一开始就偏了。**

偏差的形状：**每一次迭代都在 channel 层补协议缺的洞，协议始终保持"一个目录一张面"，
于是 channel 长出树、门（gate）、行政面（admin）去模拟"多目录多面"。**

- `imports` 是容器私事写进了场的 frontmatter（`contract.py:77` 自述"Ground 自身不
  消费"），而 SPEC §7.2 前半说子目录自带 GROUND.md 就是**平级场**——同一节里两种子场观。
- walk 面（SPEC §7.4，位置决定视角）只在 `moss ground render <dir>` 与 `admin.ground_run`
  的子进程里可达；`ground_channel._build_leaf` 永远 `render()` 无 cwd。**机制存在，但不在
  ghost 的身体里。**
- channel 用 gate + `mount_child` 重造了 §7.2 已经交给 frontmatter pin 的渐进披露，两套
  且不同步（实测 3 vs 7）。
- 死面：`Ground.snapshot/ack_hash/changed`（仅测试用）、`Ground.context()`、
  `GroundSet.frame/pin/unpin`（src 零调用）、`peek_dir`/`templates_at`（无调用方）、
  `always_show` 只在 LawPin 处判断却对六个 `pin_*` 开放、`_render.py:234` 用
  `anchor_kind(...)=="cwd"` 字符串嗅探决定 pin 是否跟随位置。
- 模板子系统（`.grounds/`、`TemplateInfo`、`--template`、`templates_at`）整套在代码里，
  SPEC 472 行里一个字都没有——`810c09b3`「align SPEC with implementation」之后，SPEC 已从
  协议降级成实现说明书。

判词：**方向错的时候，模型会非常高效地把错误执行到底。** 这次先钉协议，再动代码。

## 1. 核心定位（一句话）

**GROUND.md 让一个目录声明自己的认知场。渲染策略由场自己统摄——包括指定哪些子场直接
在本场渲染。对认知场的"编程"不在 harness 里**（不像 AGENTS.md 那样被 harness 定义死）。

由此直接推出三条，后面所有决定都是它们的推论：

1. 位置决定视角：站在哪个目录，就用那个目录所属的场来渲染。
2. 子场不是自动渲染：**父场点名哪几个子场在父场里渲染，就只渲染哪几个。** 不是所有
   子场都需要默认呈现。
3. 读是冷的：主动读，不做变更感知。

## 2. 重做决定（D）

### D1 — 探索的边界是 root

向上找 GROUND.md 必须止于 root，越界即拒绝。现在这条边界散在三处（channel 的
`_inside()`、CLI 的 `_find_ancestor_ground` 一直走到 `$HOME`、`_addr.py` 的 anchor
confinement），收成**解析器持有 root**：解析器自己知道边界。

### D2 — 位置决定规则

`render(dir)` 的第一步永远是"这个位置属于哪个场"：

- 目录自己有 GROUND.md → 用它的规则（field-root 面）。
- 否则 → 向上找最近的 GROUND.md（不合并祖先 body，§7.3），用它渲染，`$CWD = dir`（walk 面）。

### D3 — 子场是点名渲染，不是自动铺开

父场的 `imports:` 列出**要在父场渲染的那几个子场**。没点名的子场不出现在父场的渲染里——
它们仍在磁盘上，可由一次显式的 render 到达。

### D4 — `imports:` 由 Ground 消费，是渲染指令

不是"容器物化指令"。`contract.py` 里"Ground 自身不消费"那句要删——它即将变成假话。
一个字段住在一个场的 frontmatter 里，就该由那个场执行。

### D5 — 递归按字节预算截断

场体积差一个数量级，depth 管不住。用一个整次渲染的字节预算，截断点必须显式标出来
（路径 + 原因），不能静默丢。

### D6 — GroundSet 退化成"解析器"

如果渲染是递归的、读是冷的，一个 GroundSet 里不需要持有任何 Ground 对象：

- 删 `open/close/_active/_label_by_path/dirty/sediment` 的半段生命周期。
- `sediment` 存在的理由是"内存里的 pin 集要写回去"；pin 的编辑面已经归文件工具
  （file editor + bash），这条理由消失。
- 保留：root、边界、`resolve(dir)`、模板发现。

### D7 — Channel 只剩 render

```python
render(path: str = "", template: str = "") -> str
```

默认渲染 root 场。其余全部退场：`open` / `close` / `stats` / `edit` / `pin_*` /
`unpin` / `ground_run` / `mount_child` / gate / admin 子通道。channel 唯一比 bash 多的
能力是：**把一段渲染推进 ghost 的活上下文**。

### D8 — render table = 预设渲染表（模板或目标 .md）

渲染目录 x 时，不走发现，走一个指定的模板；模板要么是 `.md` 文件，要么在默认模板列表里。
模板列表从场根（`.grounds/`）发现，**且感知本身是 GROUND 自动的**——渲染面自带一个
`<templates>` 节，不需要 pin、不需要额外命令。模板路径同样受 D1 边界约束。

### D9 — 渲染有墙钟预算（约束落在 exec pin）

冷读 = 每次 render 都真执行。给整次渲染一个时间预算，超时在 XML 容器上声明（哪个 pin
在哪一层超时），已得的部分照常交出。这是 exec pin 自身的约束，不是上层的调度。

### D10 — walk 面不输出 body（确认，不是 bug）

在场内子目录每看一次都重读一遍法，是纯浪费。法只在它自己的场根出现。

### D11 — 从 GROUND 体系里拿掉 "Law" 这个名词

`law` 是 fable5 定义的，产生的歧义已经超过它承载的信息。GROUND.md 的正文就是**目录的
自解释内容**，不叫法。影响面：

- `LawPin` / `LawArguments` / `law` verb → 改名 **`content`**（从 cwd 向上收集
  CLAUDE.md / AGENT.md 这类外部内容文件）。GROUND.md 本质是 README.md 的目录级加强版，
  所以"外部文档"就叫 content；`@` 语法保留。
- `chain_text()` / `_chain.py(collect_law_files)` / SPEC §1 §5.6 §7.3 / 术语表同步。
- 定为 `content`，不再有候选。

### D12 — ghost home 的场进 `<memory>`，不进 instruction

```xml
<memory instanced_at="...">
  <ground .../>      <!-- ghost home，imports 点名渲染 -->
  <memento .../>
</memory>
```

理由：ground 和 memento 是同一种东西——认知面。同生命周期的东西住同一个容器。副作用是
顺手解决"ground 冻在 epoch baseline、refresh 改了缓存没人读"——memory 每次 refresh 就带
新渲染。

### D13 — ghost 只挂 project 级 ground channel

ghost home 不挂 channel（D12），mode home 不挂。project ground 由 channel 冷读拉取，因为
它太大，不适合每次进 memory。

## 3. 执行顺序（严格串行，每步可重启）

> **进度（2026-11-13 18:5x）**
> - ✅ -1 本记录 + commit
> - ✅ 0 dolores 摘掉 ghost ground channel（`320f987d`）
> - ✅ 1 SPEC 0.3.0（`13df968e` + `e6382f27` 修 16 处自解释缺陷）
> - ✅ 2 契约 + 实现：`ca2a4030` 改名、`3a5cac9f` 位置渲染器、`e92e60a0` PathSpec 修复；
>   `DefaultGround` / `Ground` ABC / open-close-sediment-snapshot / RenderedView / TemplateInfo 全删
> - ✅ 3 `moss ground` CLI 可用（render/meta/observe/templates/validate/init）
> - ⬜ 4 ghost home 进 `<memory>`
> - ⬜ 5 project ground channel


- **-1 本记录 + commit。** 破坏性重启丢了上下文还能从这里恢复。
- **0 筹备：dolores ghost 先拿掉 ghost ground channel。** 避免改到一半重启爆炸——先让它
  回到"没有 ground 也能活"的状态。
- **1 改 SPEC。** `src/ghoshell_moss/ground/SPECIFICATION.md`：边界、位置解析、点名渲染、
  模板、截断、去 Law 名词、把已落地但未记录的模板子系统写进去。
- **2 改契约 + 实现。** `contract.py`（inline 归属、去 LawPin、删 snapshot/context/
  frame 转发）→ `_ground.py`（去生命周期，渲染带预算）→ `_grounds.py`（退化解析器）→
  `_render.py`（位置解析 + 递归 + `<templates>` + 截断标记）。
- **3 改 `moss ground` CLI，用它做测试。** CLI 是这一层唯一能脱离 ghost 跑起来的验证面。
- **4 把 ghost home ground 渲染进 memory。** `_runtime.memories()` 里 `<ground/>` 在
  `<memento/>` 之前；`system_prompt()` 里的 ground 段删除。
- **5 project ground channel** 等 0–4 全部完成后再考虑加。

## 4. 验收（重启后逐条对）

1. `moss ground render <dir>` 对"有 GROUND.md / 无 GROUND.md 但在场内 / 在边界外"三种
   位置给出三种正确的面。
2. 父场渲染里只出现 `imports:` 点名的子场；未点名的子场不在其中。
3. 截断点可见（路径 + 原因），不是静默丢。
4. `<templates>` 节自动出现在有 `.grounds/` 的场渲染里。
5. ghost 的 memory 段形状 `<memory><ground/><memento/></memory>`，instruction 里没有场内容。
6. facade 里没有 `ghost.ground`；只有 project 级 ground channel（step 5 之后）。
7. `grep -r "LawPin\|law pin" src/` 为空（D11 完成）。

## 5. 开放问题

- ~~`law` verb 的新名字~~ 已定：`content`（D11）。
- 渲染表的两种形态（`template=<name>` / `template=<path>.md`）是否还需要第三种"目录直挂"？
- 字节预算的默认值（先按 8K 试，逐层递减？还是整次渲染一个总预算）。
- 渲染墙钟预算默认值（先按 5s/整次）。
- `.grounds/` 模板发现的搜索层级：只有 root 一层，还是边界内每层都看。

## 6. 参考（当前实现的位置，供 step 1–2 逐条对照）

| 关注点 | 现在在哪 |
|---|---|
| 位置解析（walk vs field-root） | `_ground.py:102` `render(cwd=)`；`_find_ancestor_ground` 在 `cli/ground_cli.py:85` |
| walk 面组装（折叠 TOC） | `_render.py:206` `render_walk` / `_render.py:256` `_assemble_walk_blocks` |
| 子场物化 | `_grounds.py:343` `_materialize_children` |
| 帧缓存 / refresh | `channels/ground_channel.py:177` `_refresh_ground` |
| 根场进 instruction | `ghosts/dolores/_runtime.py:362-365`、`:673` |
| 模板 | `_grounds.py:46` `scan_templates`、`.grounds/` |
| 场级 ignore | `_ground.py:191` `_make_ignore_spec`、`contract.py` `GroundConvention.ignore` |
