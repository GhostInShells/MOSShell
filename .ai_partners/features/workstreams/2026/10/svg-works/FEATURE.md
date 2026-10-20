---
created: 2026-10-20
depends: []
description: 把矢量图形从"一次生成的结果"变成"可以一直改、可以回放到任意一步的作品"。 作品 = 一个可寻址对象；动作是真相，DOM 是它的投影。
milestone: null
priority: P1
status: in-progress
status_note: v0 已在 nodes/webview_apps/svg_works 落地：作品=thread、op 追加、checkout 重放；4 条真机验证过
title: SVG Works — 可修改、可回放的矢量作品
updated: '2026-10-20'
---

# SVG Works — 可修改、可回放的矢量作品

> 来源：D20 上午的现场讨论（画布上一边说一边画）。本条不塞进 artifacts ——
> artifacts 是"结果面"，保持轻；这一条是"动作面"，另有它的重量。

## Motivation

`artifacts` 现在是 **PPT**：每个 artifact 是一个**结果**，换一个就整篇重载，状态清零。
今天实测两次：改一笔就重载、滚动与动画从头开始。

要的是「画到一半能改」：

- 骨架先立，之后每一笔落在**自己的地址**上；
- 改一笔只动那棵子树，页面不重载；
- 整个过程可读、可回放、可 checkout 到任意一格。

这不是"画得好不好"的问题，是**这条产出的过程在谁眼里存在**的问题：
一个只有结果的面，人只能看最后一眼；一个有动作的面，人可以看着它发生、也可以回到中途。

## Key Decisions

### D1 — 作品 = thread

一个作品是一个**可寻址的对象**，不是一个"画布上的东西"。
和 `file_editor` 的 thread 是同一个词：能被单独读、单独改、单独留着。
一句话判据：换作品不该等于换整页。

### D2 — 骨架先立，之后局部改

一次生成整张图（现在的 `write`）是 PPT 的做法。要的是：
先立结构（节点 + 地址），之后任何修改只落在某个子树上。

### D3 — eval 补地址；eval 与续写是两回事

- **在 DOM 上 eval**：能改属性、挪节点、补地址——什么都能修。
- **在 DOM 下续写**：只能往后加，改不了已经写错的那一笔。

所以顺序是死的：**先有 eval，才谈得上"骨架 + 局部改"**。典型用法：
某个节点忘了加地址 → 下发一条 op 给它补上 → 之后以这个地址为锚继续推流。

### D4 — 动作是真相，DOM 是投影

不存快照，存 action log；要看某一格就**重放前缀**。

- checkout 到第 n 格 = 重放前 n 个；
- fork 一支 = 复制前缀接着写；
- 对比两支 = 各自重放再 diff。

「整体状态一直可变」这个性质，不是状态本身可变换来的，是**动作可重放**换来的。
`reset` 因此是 append-only 的：它不是删除，是在历史里插一个"我回到第 n 格"。

### D5 — 可重放是有代价的：受控小语言

文件编辑的 action 是「文本 → 文本」，纯函数，重放必然一致；
DOM 的 action 若是任意 JS，就有副作用（定时器、网络、随机、动画进度），重放出来"大概是那样"。

所以 action 必须关进一个受控小语言：

- 允许：`setAttr` / `insert` / `remove` / `replaceChildren` / 有限的 style 变更；
- 不允许：`setTimeout`、`fetch`、`Math.random`、任意闭包状态。

代价是**表现力受限**：所有"动"都得是声明式的（CSS animation / SMIL）。
**先要可重放，再谈炫** —— 可回退是这一条的存在理由，"炫"可以后加。

### D6 — 呈现：画布 + 动作列

- 左边画布一直在变；右边一列小卡跟着长，每张卡是一个 action 与它的运行态。
- 点一张卡 = checkout 到那一格；然后有两个不同后续：**继续往下流**，还是**停在这儿看**。
  这两个动作要分开。
- **全屏时收起的只是"卡"，不是"记录"**：边条上留一条时间轴，而且它不是进度条，
  是一串**可点的格子**。
- 卡列与画布是同一份 action log 的两个投影 —— 与 screen 的 `drawer` / `home` 同构：
  **同池两读法**，一个按摆位看，一个按时间序看，点谁都不改另一方。

### D7 — 不塞进 artifacts

artifacts 的一次性 write 是它该有的轻。本条要在它旁边另做一块对象，
不把它改造成一个笨重的东西。两者的关系留待实现时再定（复用页面 or 新节点）。

### D8 — 边界：eval 只开在自有画布页

在**人的页面**上 eval 是越权；在**我们自己的画布**上 eval 就是机制本身。
这条界要写死在机制里，不能靠模型记着。

## 与既有 workstream 的关系

| workstream | 关系 |
|---|---|
| `webview_apps/artifacts` | 结果面，保持轻；本条的产物可以借它的页面壳 |
| `moss-os-control` 的 file_editor | 同一套骨架（thread / region / edit / rewind），本条是它在画布上的对偶 |
| `screen-manager` | 全屏、窗口语义由它提供，本条只做内容 |
| `screen-attention-programming` | actions 的卡与抽屉同一套"谁刚动过"的语义 |

## Exploration paths

- **画布的三种画法对比**（D20 实测）：`canvas` 是裸 2D、无依赖、但要手算坐标；
  `mermaid` 从 jsdelivr 引库、布局自动、样式少；`html` 能写完整 SVG/HTML、位置与样式全由自己定。
  本条走 `html` + 手写 SVG。
- **一次写完 vs 增量改**：一次写完已经证明不可用于"边画边改"（今天两次重载）。

## 未决

- 受控小语言的**边界**：哪些 op 进白名单，样式改到什么粒度。
- 时间轴在边条上的**形态**：格子多到一定程度怎么折。
- 与 artifacts 的**关系**：复用页面壳还是新节点（D7）。
- 作品与 screen 的 item 的关系：一个作品是不是也是一个可摆上屏的 item。

## Implementation Notes

- **SVG 就是 DOM**：任意 HTML 节点里可以 append 一个 `<svg>`；SVG 内部可以嵌套 `<svg>`
  （自带 viewBox 与坐标系）；`<foreignObject>` 还能反过来在 SVG 里嵌 HTML。
  "骨架 → 某个节点下面再推流"在标准里没有障碍。
- **卡点从来不在 SVG**，在我们这侧：没有桥（sandbox iframe ↔ 外界的 postMessage）、
  没有地址（节点没有稳定 id / 路径）、没有 patch 命令（只有 write / display / remove）。
- **svg 必须写死宽高**：`max-height: 96%` 这类在 grid 容器里会塌成 0 高（D20 实测，第一版画出来是空白）。

## Implementation (v0, 2026-10-20)

`nodes/webview_apps/svg_works/` — a separate node, not `artifacts` (D7).

| 命令 | 做的事 |
|---|---|
| `open_work(name, description)` | 开/选一个作品（作品 = thread） |
| `write(name, addr, chunks__)` | 往某个地址流的片段，落成一条 op（流是现场，记录是一次） |
| `set_attr(name, addr, attr, value)` | 改一个属性 —— "画到一半能改"的最小单位 |
| `remove_node(name, addr)` | 删一棵子树 |
| `history(name, n)` | 读 action 列表 |
| `checkout(name, n)` | 把读点挪回第 n 条，页面重放 |

真机验过的一条最小链路：骨架 → 往子地址挂节点 → 改属性（只有头变色，其余不动）
→ checkout 到 1（颜色回来、只剩骨架、卡片变灰）→ checkout 回来（状态完整复原）。

**实测发现的两件事**（都已修进代码）：

- SVG 元素的 `tagName` 保留**小写**（`path`），HTML 才是全大写 —— 大小写归一之前，
  "装不了孩子的节点"这条守卫永远不触发。
- `[data-addr=arm]` 指到一个 `<path>` 时，append 在 DOM 上成功、画面上什么都没有。
  现在卡片会明说"`<path>` 装不了孩子 — 换个 `<g>` 当容器"：**"装不下"和"画坏了"必须能分开**。

还没做的：流式预览的完整性（write 期间用 `.live` 节点预览，op 落地时接管）、
卡片的折叠/全屏、以及 D5 的受控小语言（现在 op 种类少，天然满足；加 op 时要守住）。