---
created: 2026-10-12
depends: []
description: 'A lightweight capture layer for surplus creativity — an off-context
  pressure-release mechanism, and the observable surface it feeds: a d3-force node
  graph where ideas surface over time, sized by priority, ringed by state. Shipped
  as one node (webview + command surface), its API the single point of truth for humans,
  the ghost and agents.'
milestone: null
priority: P2
status: in-progress
status_note: D22 定形：一个 node + d3-force 面；执行面四阶段，先做 Phase 0-1
title: Ideas Mechanism
updated: '2026-10-22'
---

# Ideas Mechanism

> Use `moss features set-status ideas-mechanism <status> -m "note"` to update state.
> See [TOPOLOGY.md](TOPOLOGY.md) for directory layout and [README.md](README.md) for the full convention.

## Motivation

The project regularly produces more ideas than the current stage can absorb. Today the
only home for an idea is a new workstream — but creating a `FEATURE.md` is a heavy,
structured act, and most ideas do not deserve to become workstreams. Ideas that have no
cheap home keep re-entering working context, crowding out the work actually in progress.

The mechanism this workstream builds is a **capture layer**: a zero-friction place to put
an idea down so it can leave the current context. Writing it down is the whole action —
the cognitive pressure is released because the idea is now durably held somewhere,
not because it was executed. Structuring, prioritizing, and promoting happen later, in a
separate rhythm, decided by a calmer reading of the same record.

This is deliberately *lower* than a workstream. It exists so that most ideas can live and
die without ever becoming one. A workstream is the promotion path for the few that
survive; it is not the entry point.

`moss features` already models workstreams (draft → in-progress → completed), stages, and
surface. What is missing is the tier *below* draft: raw, unscheduled, unprioritized
capture — and a way to view it over time.

## Why now

The project has reached a stage where capability generation outpaces consumption. As the
Ghost becomes a live, continuously-running entity, the rate at which ideas surface will
keep rising. Building the tier below `draft` now keeps that rise from turning into queue
growth at the `draft` tier — which is where it currently lands, and where it hurts.

## Design Index

- Key design documents: `design/`
- Key discussion records: `discuss/`

## Key Decisions

<!-- Record each meaningful design choice. This is what the next model incarnation reads first. -->

### D1 — It is a tier below workstream, not a feature of workstreams

The ideas mechanism is **independent and lighter** than the features system. The
relationship is one-way: **triage promotes an idea into a features `draft`; ideas never
depend on or attach to the features structure.** Attaching ideas as a sub-capability of
`features` would let the workstream structure (status, priority, stage) leak downward
into the capture layer and reintroduce the very friction the mechanism exists to remove.
The failure is expensive to reverse once dozens of entries exist under the wrong shape.

### D2 — Capture is zero-structure; structure is applied at triage

At capture time: date + free text, nothing else — no tags, no priority, no template. Any
structure at capture time turns recording into a task, and a task has friction, and
friction makes capture get bypassed. Tags, grouping, and status are applied later, at a
fixed triage rhythm, by a calmer reading.

### D3 — Admission delay is part of the design

An idea captured in a high-energy state is triaged only after a minimum delay (on the
order of a week). This mirrors the project's existing discipline of waiting until there
is enough of a record to judge, rather than acting on the impulse of the moment. The
delay is not suppression — the idea is durably held the moment it is written; it simply
waits for a calmer reading before it is promoted. Three outcomes at triage: promote to a
features `draft`, fold into an existing workstream, or explicitly retire. Retired ideas
are kept, not deleted — the record over time is itself the artifact.

### D4 — Storage is an append-only event log, projected from the source text

Each idea has an editable text as its primary artifact; every edit appends a new
revision rather than mutating in place, so the full history is time-traceable. The index
is a per-period (monthly) `jsonl`; the source documents live in a per-period directory.
The `jsonl` is the **projection** of the source text — the source is authoritative, the
index is derived. Do not let the index schema define the content.

### D5 — No rename

An idea's identity is stable; it is not renamed. This keeps the history and the
timeline addressable.

### D6 — The observable surface is part of v1's intent, not a later add-on

The intended observable form is a **timeline of bubbles** — a view where ideas can be
seen surfacing one by one over time. This is a first-class part of the mechanism's
purpose (an idea that has left working context is still *visible* to a third party),
not decoration. It is, however, an independently valuable and independently scoped
piece: it should be built on the webview-app node pattern (artifacts / calendar /
decision_tree), it depends on the capture layer having real data first, and the
"surfacing" motion is where the time sink lives — the static layout is cheap, the
*feel* is open-ended and has no completion criterion. Guard it accordingly.


## Design settled (D22, live session with thirdgerb)

D1–D6 above still stand. What follows is the shape settled in conversation on 2026-10-22,
after actually building three throwaway D3 surfaces and looking at them.

### User stories

1. 脑子转得飞快的时候，能一口气把十几个点倒进去，不用先判断它算不算一件事。
2. 一周后回来，能看见那段时间冒出过什么、哪几个在长、哪几个已经死了。
3. 想做某件事的时候，能按目录、状态、时间筛，只留下相关的那一簇。
4. 点开一个点，能看见它从第一句话到现在怎么变的，以及它属于哪个目录的什么规矩。
5. 让一个 agent 接过去做，人和 ghost 都能看见它在动。

### D7 — It is a node, not a page of the features system

一个 ideas node：**一个 webview 加一条命令面**。展示能力（D3 那张图）和数据能力
（读写 ideas/）都在这个 node 里。

**唯一的写入口是这个 node 的 HTTP API。** bash 工具不是另一条真相，它只是这个 API 的
客户端；skill 里的 bash 命令同理。代价说清楚：node 没开的时候，那些命令就不存在。
换来的是——人、ghost、agent 三方走同一个口，中间没有第二份真相。

> 「这些 bash 工具就不再是完整的功能面，而是去调用这个 node 开放的 HTTP 接口。」

### D8 — Where the data lives: one `ideas/` per directory, one registry in the ghost home

ideas 不是某一个目录的东西。任何目录下都可以有一个 `ideas/`，这个 node 起来时把它们
**一起**挂上。哪个目录有 ideas，由 ghost home 里的一张**登记表**说了算——node 读表，
不到处乱扫。

存储根本身只放数据，加一句指路的话（"用 ideas node 打开它"），**不抄第二份说明**。

### D9 — The record is an append-only event log, and so is each idea

每个 `ideas/` 里是 JSONL：**一条一行，只追加**。诞生是一条，改状态是一条，连一条线也是一条。

每条 idea 自己也是一条只追加的线——改状态不是覆盖，是加一条。所以「它怎么变来的」
永远答得出来，和 memento 同一个形状。

索引（内存里的那份、将来可能的 SQLite）是可丢弃的缓存物，不是真相：丢了重建，画面不等它。

> 「文件是记录，索引是可缓存物：启动先读索引，索引没了后台重建，画面不用等它。」

### D10 — The picture is a d3-force field, and time is a knob on the data

- **力导向**，不是 pack：气泡彼此推开、被线拉着、被中心轻轻吸住——它是算出来的，不是画出来的，
  所以它一直在动。
- **大小 = 优先级**（可调），**颜色与环 = 状态**：转圈的环 = 正在跑，实心绿 = 完成，
  灰 = 退休，虚线 = 在等。
- **时间轴**拖动时**不重建 simulation**，只给它一个很小的 alpha 重启：新球淡入，退出的变暗不删，
  相机不动。拖过去是「长出来」，不是「洗牌」。
- **点开一颗 = 以它为中心重排**（Neo4j 那种 ego network，N 跳内的邻居围上来），
  再点另一颗就换中心。
- 过滤只是一层视图：筛掉的只是「现在不画」，不改任何数据。

### D11 — The side panel is the same shape as the file editor

Meta / 当前 / History / Ground 四个页签：它在哪、现在什么样、怎么变来的、属于哪个目录的什么规矩。
和 `file_editor` 的 thread 是同一个形状，只是它挂在一颗球上。

### D12 — One surface, several directories

一个 ideas 面同时挂着几个 `ideas/` 目录：一个目录一簇，簇和簇之间也可以连线。
（理由见 D13：这套图的终点不是「我的想法」，是「一个正在并行工作的集群」。）

### D13 — This is the practice ground for a stateful node graph

今天是几千颗 idea；明天是几十个 ghost、每个 ghost 一个 DSH、每个 DSH 几个 workspace、
每个 workspace 上挂着几个 agent——**四级节点，同一张图，同一套交互**。

> 「这就对我们的大型 agent 集群运行机制的一套交互机制，ideas 相当于给这套机制探路。」

所以 D7 的「node + API + 多客户端」不是为 ideas 定制的，是为那一天的形状先拧一遍。

### D14 — Do not abstract the shared shape yet

features、file_editor、terminal、ideas 现在长得越来越像，因为整套机器是按同一种分型哲学在设计。
但**故意不抽**：等它们全部定型、经验能提炼了，再抽一个像 mindflow 那样的东西。
先让差异把共性逼出来，再抽就不是猜。




### D15 — 它不叫 webview app

ideas 这一类东西（围绕**思维治理**的）走一个统一的目录分类，不塞在 `webview_apps/` 底下——
它是"怎么想"的机制，不是"怎么显示"的东西。**一个 node 可以没有视图**：没屏的时候它照样记账。

### D16 — 一条 idea 是一份 Markdown，日志是它的骨骼

一条 idea 默认是**一份可读的 `.md`**——可分发、可 git diff、可丢进任何编辑器。jsonl 里的
事件是它的骨骼（怎么变来的）；md 是它的皮。附件多到装不下一份文件时，再把它升级成一个目录。

### D17 — 数据模型用 pydantic BaseModel，写进这份文档

模型本身就是契约。日志是事件流，当前状态是它的 fold：

```python
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field

Op = Literal["born", "state", "priority", "tag", "untag", "link", "unlink", "note"]

class IdeaEvent(BaseModel):
    t: datetime                     # 何时发生
    id: str                         # 哪条 idea
    op: Op
    payload: dict = Field(default_factory=dict)

class IdeaLink(BaseModel):
    src: str
    dst: str
    kind: Literal["related", "depend"] = "related"
    weight: float = 1.0

class Idea(BaseModel):
    id: str
    root: str                       # 哪个 ideas/ 目录（log 的指针）
    text: str                       # 第一句话
    born: datetime
    priority: float = 1.0           # 决定球的大小
    state: Literal["spark", "active", "waiting", "done", "retired"] = "spark"
    tags: list[str] = Field(default_factory=list)
    links: list[IdeaLink] = Field(default_factory=list)
```

**当前状态 = fold(events)**。不存快照；要显示就读日志、折出 `Idea`。索引是可丢弃的缓存。

### D18 — 边是一等事件，而且能带信息

边不挂在节点自己身上，它是一条 `link` 事件 `{t, src, dst, kind, weight}`——谁连谁、什么时候连的、
什么关系，都进日志。

关系词表**极小**：先只有 `related`（默认）和 `depend`（`a → b` 表示 a 依赖 b）。
别的等真需要再加，不预先发明。

边在画面上能带信息：**粗细=weight，颜色=kind，虚线=还没定，箭头=方向，线上可以挂一个字**。

### D19 — tags 是事后打的；规模靠目录分片

- **tags 不在抓取时打**（D2 零结构的延伸）：和优先级一样，是事后的事。可以让 ghost 先打一版，人来改。
- tags 只是过滤的一个维度，和目录、状态、时间并列。
- **不存在"数据量太大"**：目录就是分片。一个 `ideas/` 管一段，多了另开一个——和 `GROUND.md`
  一样是约定，不需要数据库迁就它。

## Revision (D22 20:00) — over-designed; the first version is the file

After building the log+fold version and living with it for a day, thirdgerb called the
whole thing back:

> 「我反复思考了现在的设计, 发现我有过度完美化的趋势, 投入的注意力资源和它想解决的问题,
> 阻塞的问题不匹配. 所以需要先收敛第一版的 ideas, 让它简化并立刻用起来.」

The mechanism below supersedes the log. D7 (node + one implementation + several clients)
stands; everything that turned a file into a record of itself is gone.

### What replaced what

| Superseded | Now |
|---|---|
| D8 — a registry in the ghost home decides which directories have ideas | A group **is** a first-level directory of the root; the root is `--root` / `MOSS_IDEAS_ROOT`, defaulting to `ideas/` beside the node. Nothing scans, nothing is registered. |
| D9 / D16 / D17 / D18 — append-only log, state = fold, pydantic event models | **The file is the truth.** One `.idea.md`: frontmatter (`state` / `priority` / `tags` / `links`) + body. Editing it by hand is a normal move. |
| D10 — timeline drags over the record | **Dropped.** Replay was a mechanism the problem never asked for; "how did this idea get here" is git's job, if anyone wants it. |
| D11 — Meta / 当前 / History / Ground tabs | **Deferred.** The right sidebar is always open: the group's `README.md` when nothing is selected, the idea's file when something is. |
| D12 — several `ideas/` mounted at once, clusters linked | **Deferred.** One root per run. |

### New decisions

- **The filename is the identity, and it is semantic.** `duplex-gap.idea.md` says what it
  is before anything is opened; an opaque `idea-10384843` would force a reader to open a
  hundred files to find one. `rename` therefore rewrites every reference — it is the one
  verb that sweeps the root.
- **A dead link is information, not an error.** Rendering checks liveness once and draws
  the missing end as a cut line: "I wrote that relation" and "I misremembered" must not
  look alike.
- **A broken file is reported, never skipped.** A file that vanishes from the list is
  indistinguishable from one that was never written.
- **No stored timestamps.** File mtime is the truth about "when did this last move";
  a second field would drift from it.
- **Writes are atomic** (tmp + rename): the page and `list` read these files while they
  are being written.
- **The data root is disposable.** It ships with examples and a `.gitignore`; deleting
  either the file or the whole directory is a supported choice. What is shipped is only
  what we deliberately put there.
- **One implementation, several doors.** `cli.py` (click) edits the files; the node's
  channel and the page both go through it. The channel exposes the daily verbs explicitly
  and one `run(text__)` escape hatch for the long tail; when the verb set passes ~10, the
  explicit commands collapse into `run` alone.

### The file

```markdown
---
state: active            # spark | active | waiting | done | retired
priority: 3              # the bubble's size
tags: [voice, duplex]
links: [attention-cue, voiceprint:depend]   # name = related, name:kind = kind
---

The body is the thinking itself.
```

### Still open

- Whether the ghost's 48 existing ideas (old jsonl format, `.moss/ghosts/moss/ideas/`)
  get migrated or dropped — deferred by thirdgerb: *「我也有点舍不得直接清, 我们先留一会,
  等一下我们先拿它做用例把我们一些东西讲完, 然后我们才继续.」*
- Skill packaging (the CLI as a `moss skill` executable with agreed env vars) — explicitly
  not in the first version: *「ideas 仍然 node 化, 暂时不做 skill, 未来可以做.」*


## Takeover Notes — 零上下文的人从这里开始
### 已经做出来的东西在哪

- **手写物理的原型**：`.moss/runtime/ideas/index.html`（237 行，零依赖）。气泡 + 时间轴 + 侧栏
  三件都能跑，但力的部分是我手写的（互斥 / 连线弹簧 / 向心）。它是形状的证明，不是实现。
- **D3 样例（用 artifacts 节点看源码）**：`d3-samples`（三合一）、`d3-force`、`d3-pack`、
  `d3-timeline`、`d3-roam`。用 `artifacts:read(label)` 取源码，`display(label)` 看效果。
- **游动的可行证**：`d3-roam`。点一颗 → 相机推近并居中；双击 → 它 N 跳内的邻居亮起、其余变淡；
  Esc 或「返回」→ 退回上一个相机位置；滚轮缩放。这套已经在真机上看过。

### D3 用哪几个模块

- `d3-force` — 位置（`forceManyBody` / `forceLink` / `forceCenter` / `forceCollide`）
- `d3-zoom` — 相机（缩放 + 平移 + 缓动过渡）
- `d3-selection` — DOM 绑定，`join()` 做增删
- `d3-scale` / `d3-scale-chromatic` — 优先级换半径、分组换颜色
- simulation 自带的 tick 就是时钟，**不要另起 `d3-timer`**

要 3D 才用 `three.js` + `d3-force-3d`；本 feature 不做。

### 侧边栏四个页签写什么

- **Meta**：id、优先级、状态、诞生时间、所属目录、连着几条线
- **当前**：这条 idea 现在的内容（最新那一段）
- **History**：只追加的那条线，一条一行，能回退到某一版读
- **Ground**：它所在目录的 GROUND.md 渲染——「它属于哪套规矩」

### 已经验过的几条事实

- 一颗球可以连任意多条线；SVG 按绘制顺序叠，**线先画、球后画**，所以线自然在下面。
- 节点多到一定程度会自己聚成团，但那是**拓扑的聚**（被线拉着的凑一起），不是语义的聚；
  要按类别聚，得自己加一条「同类往同一点吸」的力。
- Observable 的 `@d3/bubble-chart` 用 playwright 能直接访问（HTTP 200），代码 ISC，可以抄；
  但它是 `d3-hierarchy` 的 pack，是「塞满一罐」，不是力导向——别抄错。

## Execution Plan (for a side agent)
目标：一个**零上下文**的模型，只读这份文档，能按顺序做出 v0，并知道每一步「做完了」是什么样。
不写进度百分比，只写交付物和验收。

**Phase 0 — 能看见东西**
1. 新建节点 `nodes/webview_apps/ideas/`：`main.py` + `index.html`（d3-force 一页，四个标签页
   之外的脚手架先不做）。
2. 命令面最小四件：`open(root)`、`spark(text, priority=1)`、`link(a, b)`、`state(id, s)`。

3. 存储：`<root>/ideas/ideas.jsonl`，一行一条 `IdeaEvent`（D17 的模型，只追加）；
   每条 idea 同时有一份可读的 `<id>.md`（D16）；启动时读日志、fold 出 `Idea`。4. **验收**：连发三条 `spark`，屏幕上三颗球，各自带字；改一次 `state`，环的样式跟着变。

**Phase 1 — 时间和读法**
5. 前端时间轴：拖动只改喂给 simulation 的数据，**不重建** simulation——新球淡入、退出的变暗。
6. `focus(id)`：点开一颗，相机推近、它的 N 跳邻居围上来（Neo4j 式）。
7. `filter(state | since | root)`：只影响画不画，不改数据。
8. **验收**：往回拖时间轴，球退场不闪、不复位；点开一颗再点另一颗，中心跟着换。

**Phase 2 — 多目录**
9. ghost home 里一张登记表（例如 `ideas.roots`），列出哪些目录有 `ideas/`；node 起来读表，不到处扫。
10. 一个面同时挂多个目录：一个目录一簇，簇与簇之间可以连线。
11. **验收**：两个目录各放一条 idea，一次打开都在，簇分得开。

**Phase 3 — 侧栏**
12. 四页签：Meta / 当前 / History / Ground。History 是那条只追加的线的读法。
13. **验收**：一条 idea 改过三次状态，History 里是四条，从第一条读得出它怎么来的。

**Phase 4 — 出口**
14. HTTP API 与命令面共用同一层实现；写一个 bash 客户端示例，证明「bash 只是客户端」不是口号。
15. 导出：一张静态图（给文档用）、一份记录（可重放）。

**Out of scope（本 feature 不做，只留接口）**
- agent 集群那张图（ghost / DSH / workspace / agent 四级）。但它决定了 D7 的形状，接口要留对。
- 自动抓取（人说话就落一条）。见 Open Questions。


**整条 feature 的验收**：把这份文档给另一台机器上的模型，它能在不看我们对话的情况下做出 Phase 0–1。

**会走样的三处（写死，任何执行者都不许改）**

1. **布局必须是 d3-force**。换成 pack / 树 / 手算坐标都好写，但那就不是这张图了——
   它要的是「一直在动、每次都不一样」，静态布局给不了。
2. **写入口只能有一条**。命令面、bash 客户端、agent 都必须打到同一层实现；
   一旦出现「bash 直接改文件」的第二条路，真相就分叉了。
3. **拖时间轴不许重建 simulation**。重建会触发整场重排，那是洗牌不是生长；
   正确做法是保住实例、给小 alpha、换数据。

## Exploration paths

<!-- Dead ends hit, pivots made, lessons learned. -->
- **Deferred: node-ification / live-stream visualization as part of capture.** The
  capture layer must exist before a visualization of it has anything to show; building
  the view first would produce an empty container. Ordering follows the project's
  existing dependency-unlock discipline: capture first, view later.

## Methods

<!-- Non-obvious implementation patterns. -->

- **Directory-discovery convention**, consistent with the project's `GROUND.md` /
  workstreams pattern: the mechanism is discovered by convention inside an expected
  directory, so a personal instance (e.g. a `thirdgerb`-scoped ideas store under a
  Ghost's ground) and a project-level instance can share the same implementation without
  a schema change. Whether this is a personal capability or a project capability is
  decided by usage, not up front.

## Implementation Notes

<!-- Gotchas, non-obvious behaviors, reasons for rejecting simpler alternatives. -->

- A plain text file per period (e.g. `.ai_partners/ideas/2026-10.md`) is a sufficient
  v0. It already implies monthly slicing and lets usage decide the eventual shape.

## Open Questions


- ~~Personal vs project scope~~ — settled by D8: neither. A directory can carry an
  `ideas/`, and the node mounts as many as the registry names.
- ~~Is the bubble surface in scope~~ — settled by D10: the surface *is* the point. A
  capture layer alone would be a filing cabinet nobody opens.
- 一条 idea 什么时候算「该升级」：看连线的密度，还是人来判？
- 谁往里落：只有 ghost 落，还是人说话就自动落一条——后者的噪声怎么挡？
- 一条 idea 的「物料」（附件、截图、链接）放哪：同一个 jsonl，还是它自己的目录。
## Revision 2 (D22 22:00) — it is a graph, not a file per idea

Eleven hours after the first revision, the mechanism was rebuilt again, this time from
thirdgerb's own drawings. His words:

> 「这个之所以不是思维导图, 因为它是图. 节点关联建立完了后, 拓扑容易推算.
> 尤其是大型工程的依赖关系.」

> 「我将 ideas 图视作一种视觉信息压缩策略, 而且是我们共享拓扑交流的策略.」

**What the second revision got wrong:** one markdown per idea optimised for the wrong
reader. A graph is read as a whole — topology is a question about the whole — and a
directory of files forces any reader (model or human) to reassemble it from fragments.

### The shape now

- **One `graph.json` per group.** `{rev, meta.concepts, nodes{id: {...}}, edges[...]}`.
  `rev` is monotone; every write bumps it.
- **Edges are entities** with their own ids. Storing a link on its source node makes
  every edge single-parent, which is exactly the tree-shaped blindness the mechanism is
  trying to escape.
- **Ids are opaque (`n7`), labels are display.** Renaming cannot break an edge — which
  retires the whole "rename must sweep the root" problem from the first revision.
- **Weight decays by generation** (child = parent × 0.8, floor 0.2). Depth is size, so
  pruning is a decision about weight rather than taste.
- **Concepts are the colour scale**, not an enum in code: `meta.concepts` maps a word to
  a hex, extended by whoever is drawing. `flag` is the human's red mark — "this point
  has a problem" — drawn as a dashed red ring outside the bubble.
- **`ops.jsonl` is the account, not the truth**: one line per change with its rev; used
  by `diff`; deletable at any time. No replay, no fold.
- **`IDEAS.md` returns, as a sign.** Each group carries one: frontmatter
  `status: active|waiting|archived` + one sentence of context + a pointer to
  `graph.json`. It **never lists nodes** — that was the failure mode of the central
  registry in the first revision. Discovery = glob `*/IDEAS.md`, read the status, then
  read the graph.

### The read side is the mechanism

`tree` (the whole graph in one call), `diff <rev>` (what moved), `critical` (longest
chain + branch points), `path a b`, `leaves`. These are what make topological alignment
possible: two minds point at the same node instead of arguing over prose.

### Doors

- `cli.py` — the implementation; executable, root from `MOSS_IDEAS_ROOT`.
- channel — the same verbs as CTML commands, plus `run(text)`.
- **`batch`** — many verbs in one call. It exists because governance is a pass over the
  whole graph (seventeen `set --concept` lines), and without it the model reaches for a
  shell loop, which makes bash a mechanism nobody designed.

### Consequence to keep in mind

`decision_tree` is a strict subset of this: a graph with one parent per node and no
cycles. Once this holds, it can be retired rather than maintained.

## Resolved — the "literal body" bug was a shadowed builtin (D23 00:08)

The note that used to sit here blamed the kernel (`literal_eval` on a `text__` body plus
string annotations). That was wrong. The real cause was in this node.

Symptom, found while wiring `ops(chunks__)` — the command never receives the body when the
body looks like a Python literal:

| body | reaches the command? |
|---|---|
| `x` | yes |
| `{"op": "add", …}` | **no** — `UNKNOWN_ERROR: isinstance() arg 2 must be a type, a tuple of types, or a union` |
| `# json` on the first line, then the same JSON | yes |

Root cause: `async def list(...)` is a command defined in `new_ideas_channel()`, in the same
scope that defines `ops`. Python resolves the name `list` inside `ops` to that function
object, so

    batch = data if isinstance(data, list) else [data]

was checking against a *function*, not the builtin type. `json.loads("x")` raises first and
is caught, so only a body that parses ever reaches the broken line — which is why it looked
like a "text versus literal" problem.

Fix: `_LIST = list` at module level, and `ops` uses `_LIST`.

What this rules out for good:

- **not the kernel.** `chunks__` arrives verbatim through CTML, through the bridge on real
  zenoh, and across the mesh. `<say>1</say>` and `file_editor:write` with a `{"a": 1}` body
  both behave correctly.
- **not the annotations.** But `from __future__ import annotations` is load-bearing in this
  module: the `list[str]` annotations are harmless only because they stay strings. Drop it
  and the shadowed name breaks at import time.

Lesson worth keeping: a command name and a builtin sharing one scope is an invisible trap in
this style of channel code. Either keep a private alias for the type, or do not name a
command after a builtin.
