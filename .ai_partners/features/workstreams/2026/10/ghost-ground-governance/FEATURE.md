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

### D12 — ghost home 的场进 **instruction**（原定的 `<memory>` 方案作废）

`_runtime.system_prompt()` 在 `identity_instruction()` 之后、inception/base 之前插入一段
`<!-- ground: ghost home -->` + 根场渲染，boot 时渲一次；`<memory>` 里只剩 memento。

理由是位置而不是容器：identity 是一行"我是谁"，ground 是"这一行活在哪"，先有人再有世界，
操作性的段落（protocol / concurrent mind）排在后面。原方案（ground + memento 同住 `<memory>`）
作废——`<memory>` 每帧重发，而 ghost home 是 12k 量级的稳定内容，逐帧带上等于每轮重付一次
prompt cache。

### D13 — ghost 只挂 project 级 ground channel

ghost home 不挂 channel（D12），mode home 不挂。project ground 由 channel 冷读拉取，因为
它太大，不适合每次进 memory。

## 3. 执行顺序（严格串行，每步可重启）

> **进度（2026-11-14 01:0x）**
> - ✅ -1 本记录 + commit
> - ✅ 0 dolores 摘掉 ghost ground channel（`320f987d`）
> - ✅ 1 SPEC 0.3.0（`13df968e` + `e6382f27` 修 16 处自解释缺陷）
> - ✅ 2 契约 + 实现：`ca2a4030` 改名、`3a5cac9f` 位置渲染器、`e92e60a0` PathSpec 修复；
>   `DefaultGround` / `Ground` ABC / open-close-sediment-snapshot / RenderedView / TemplateInfo 全删
> - ✅ 3 `moss ground` CLI 可用（render/meta/observe/templates/validate/init）
> - ✅ 4 ghost home 进 instruction（`_runtime.system_prompt()`，位置在 identity 之后、
>   inception 之前；home 渲染预算单独提到 16000）
> - ✅ 5 project ground channel 挂回 `ghost.project_ground`（边界 = 项目根）
> - ✅ 6 验收（D14 重启后）：walk 面无 imports、预算父子过滤、CLI 边界改项目根、删 `_edit.py`
> - ✅ 7 预算归场自声明（D15）：frontmatter `budget`、渲染器默认只兜底、`_runtime.py` 的
>   16000 移进 `.moss/ghosts/moss/GROUND.md`
> - 落地提交：`e27ed3fa` CLI 修正、`3941e5e9` imports 改名、`a11d4f16` dsh 锚点、
>   `0d8d69a6` instruction + project 场、`40c3a646` CLI 二轮、`915412bb` 移出生成物


- **-1 本记录 + commit。** 破坏性重启丢了上下文还能从这里恢复。
- **0 筹备：dolores ghost 先拿掉 ghost ground channel。** 避免改到一半重启爆炸——先让它
  回到"没有 ground 也能活"的状态。
- **1 改 SPEC。** `src/ghoshell_moss/ground/SPECIFICATION.md`：边界、位置解析、点名渲染、
  模板、截断、去 Law 名词、把已落地但未记录的模板子系统写进去。
- **2 改契约 + 实现。** `contract.py`（inline 归属、去 LawPin、删 snapshot/context/
  frame 转发）→ `_ground.py`（去生命周期，渲染带预算）→ `_grounds.py`（退化解析器）→
  `_render.py`（位置解析 + 递归 + `<templates>` + 截断标记）。
- **3 改 `moss ground` CLI，用它做测试。** CLI 是这一层唯一能脱离 ghost 跑起来的验证面。
- **4 把 ghost home 渲染进 instruction。** `_runtime.system_prompt()` 在 identity 之后插
  ground 段（boot 渲一次，预算 16000）；`memories()` 只留 memento，不再带 ground。
- **5 挂 project ground channel。** `ghost.project_ground`，边界 = 项目根，只有一条 `render`。

## 4. 验收（重启后逐条对）

1. `moss ground render <dir>` 对"有 GROUND.md / 无 GROUND.md 但在场内 / 在边界外"三种
   位置给出三种正确的面。
2. 父场渲染里只出现 `imports:` 点名的子场；未点名的子场不在其中。
3. 截断点可见（路径 + 原因），不是静默丢。
4. `<templates>` 节自动出现在有 `.grounds/` 的场渲染里。
5. instruction 里 ground 在 identity 之后、inception 之前；`<memory>` 里只有 memento。
6. facade 里没有 `ghost.ground`；只有 project 级 ground channel。
7. `grep -r "LawPin\|law pin" src/` 为空（D11 完成）。

### D14 — 验收后修的三件事（2026-11-14）

1. **walk 面不渲染 imports。** `imports` 是 GROUND.md 所在目录的机制，属于场根面。walk 面
   重画它，既在每下一个目录时重复整棵导入树，又必然成环——父场 import 的子目录走的就是父场
   自己。判环因此也改成"正在渲染的位置"，不只是"场"。
2. **预算父子过滤。** 子拿的不是父的剩余，而是"剩余 − 收尾保留"；子输出的最终字节（含缩进）
   记到父账上，且只记一次。缩进必须在记账前加——先发后缩，缩进就是没被数过的预算。
   pin 装不下时裁内容保留标签，而不是整枚丢弃。
3. **CLI 边界 = 项目根。** 不是 `$HOME`。`render`/`templates`/`meta`/`observe` 全部走
   `find_ground(boundary=项目根)`；项目之外不归这里管。`render` 把截断报到 stderr，
   `validate --budget` 按默认预算渲一遍并报截断——写 GROUND.md 的人当场知道场写胖了。

### D15 — 预算是场的自声明，不是渲染器的闸门（2026-11-14）

`budget` 不是渲染器的策略，是 GROUND.md frontmatter 里的一个槽：**场自己说"我这一面值多少
字符"，自己填、自己测、自己改**。渲染器只做三件事：

- 场声明了 → 按声明执行，超出处留 `<truncated reason="budget" path=…/>`；
- 场没声明 → 用渲染器的默认值兜底（默认是 fallback，不是 policy）；
- 调用方传了 `limits` → 与声明取小（小上下文想少要点，是它自己的上限）。

来由：这是验收时我搞错的地方。上一轮我把整次渲染的字符预算做成渲染器常量
`DEFAULT_RENDER_CHARS=12000`，又给 ghost home 单独写死 16000 —— 同一个场在两处得到两个
答案，`moss ground render` 说截断、instruction 说没截断。更根本的是：场说什么都不生效，
"自己填、自己测、自己分析"这个闭环根本不存在。

历史事实（可复核）：改动前 ground **没有**总预算。`git show 7adb7ea5^:src/ghoshell_moss/ground/_ground.py`
全文无 `budget`；旧 SPEC 只有 pin 级 `budget`。总预算来自 D5 那句"用一个整次渲染的字节预算"，
但 D5 讲的是**递归怎么终止**，不是"预算归谁声明"——那一步是我自己加的引申。

## 5. 开放问题

- ~~`law` verb 的新名字~~ 已定：`content`（D11）。
- 渲染表的两种形态（`template=<name>` / `template=<path>.md`）是否还需要第三种"目录直挂"？
- `.grounds/` 模板发现的搜索层级：只有 root 一层，还是边界内每层都看。
- ~~ghost home 的 16000 与默认 12000 口径不一致~~ 已解决（D15）：ghost home 在
  `GROUND.md` 里声明 `budget: 16000`，渲染器不再持有第二个数。当前是 11411 / 16000。

## 6. 参考（当前实现的位置，供 step 1–2 逐条对照）

| 关注点 | 现在在哪 |
|---|---|
| 位置解析（walk vs field-root） | `_render.py` `find_ground()` + `render_position()` |
| 面组装（body / pins / folded / imports / templates） | `_render.py` `render_position()` 尾段 |
| 预算（父子过滤 + 缩进记账） | `_render.py` `_Budget`、`_CLOSING_RESERVE`、imports 循环 |
| 场容器 | `_grounds.py` `DefaultGroundSet`（root/boundary/ground_at/render，63 行） |
| channel | `channels/ground_channel.py` `new_ground_channel()`（只有 `render`） |
| 根场进 instruction | `ghosts/dolores/_runtime.py` `system_prompt()` / `__aenter__` |
| 模板 | `_render.py` `_discover_templates()`、`_resolve_table()`、`.grounds/` |
| 场级 ignore | `_render.py` `_ignore_spec_from()`、`contract.py` `GroundConvention.ignore` |
| CLI | `cli/ground_cli.py`：边界 `_default_boundary()`（项目根）、截断报告 `_report_truncations()` |
