---
title: Decision Tree Node
status: in-progress
priority: P1
created: 2026-09-22
updated: 2026-09-22
depends: []
milestone:
description: >-
  面向对象的图交互 node — 用决策树做创作提纲。图是可视化分层状态机：人机围绕它讨论、
  形成决议、推进节点状态。结构走 append-only jsonl（单写者=通道），内容走节点目录（可懒建）。
  分形递归，也是模型社会化工程的核心实验。
status_note: >-
  v0 落地 + MCP dogfood 通过 (建树/建节点/改状态/fold/落盘/confirm 回传/目录 open)。概念成立。
  但「人机沟通」没做到: confirm 是非阻塞旁证, action 卡片是瞬态广播, 晚连/刷新即丢 —— 决议
  实际上是模型单方面定的, 人类确认是仪式。核心遗留 = 确认语义 (见 Open Problems 头一条)。
---

# Decision Tree Node

> Use `moss features set-status decision-tree-node <status> -m "note"` to update state.
> See [TOPOLOGY.md](TOPOLOGY.md) for directory layout and [README.md](README.md) for the full convention.

## Motivation

**图不是内容呈现工具, 是内容创作提纲。** 之前所有对"图"的实现 (mermaid_draw / artifacts) 都是
「已有内容 → 渲染成图」; 这个 workstream 是反的: **图先于内容**, 提纲是创作的输入端,
目录和内容是图被执行出来的产物。**decision 不是记录, decision 就是创作动作本身。**

要解决的问题: 复杂创作 / 技术调研 / 评估类讨论中, 人机双方大部分时间在**对齐一张图**,
但图只存在于对话里, 一会儿就腐烂。落成文档的形态 (FEATURE.md / debates/) 又只有终点、没有过程。

机制: 人机围绕图讨论 → 模型签发对象 (创建/变更节点) → 人点确认 → 形成节点决议 → 状态变更。
图本身是「可视化分层状态机提纲」, 一体覆盖 **讨论 → 协作 → 状态观察**。

与 tasks (模型私有的扁平执行台账) 的差别就是这个 workstream 的价值: 分层 / 共享 /
每节点一条自己的状态机 / 变更需人确认 / 每节点有独立目录承载物料。

**这是未来「模型社会化工程」的核心实验。** 因为结构集中、内容分形 (见 K4/K14),
层级、授权、见证自然落在"验收向上走、内容向下沉"这条缝上。

## Design Index

历史设计讨论全部在 2026-09-22 与人类架构师的一次会话中完成, 未另开 `discuss/`。
本文件的 Key Decisions 就是那次会话的判辞面。

## Key Decisions

### K1. 图 = 提纲, 不是内容呈现工具

图先于内容。渲染是副产品, 主要动作是「在图上做决策 → 状态变更 → 目录被生成」。
任何把它做成"画布 / 阅读器"的实现都偏离命题。

### K2. 本期只做决策树, 不做通用图

不做通用图类型抽象。但**边从第一天起就是列表, 不是标量 parent** —— 零成本,
免掉将来加 `link_node` 跨边时的全量迁移。树的形态 = 每个节点的出边列表里都是同一类边。

### K3. 对象模型: 节点 A → 边 → 节点 B

所有图的语义都是三元组。两个原语:

- `create_node(node, from_node, edge)` — 从既有节点长出新节点(树边)
- `link_node(node, linked_node, edge)` — 连两个既有节点(跨边; 本期保留接口, 树用不到)

`create_node` 长树(提纲), `link_node` 长图(将来做依赖/推翻这类跨边)。
本期单向边、无环。

### K4. 结构走 jsonl, 内容走节点目录

**访问模式决定存储布局, 不是背离 features。** features 是 144 个人工策展、低频变更的节点,
每个还带一篇大文档 → 一节点一文件是对的。决策树是**活讨论中高频增删改**的结构 →
一节点一文件在这里的代价是 N 次 open + read-modify-write + 部分写损坏。

三层, 各有生命周期:

| 层 | 载体 | 谁写 | 频率 | 性质 |
|---|---|---|---|---|
| 图的 meta | `DECISION_TREE_ROOT.md` | 人/模型手写 | 极低 | 自解释: statuses 表 + edges 词汇表 + 散文计划 |
| 节点结构 + 状态 | `tree.jsonl`(一树一文件) | **通道独占** | 高 | append-only, fold 出当前树 |
| 节点内容 | `graph-nodes/{name}/` | 任意工具 | 低 | **可选**, flag 控制是否创建 |

副产品: **jsonl 同时是状态和历史** — fold 出当前, raw 就是轨迹。per-node `history.jsonl` 因此不存在。
与 file_editor 的「content 是派生而非存储值」、memento 的 append-only 同一条哲学。

### K5. 边在日志里是一条事件, 不是一个字段

`create_node` = append 一行 `{ev: create, from, to, edge, title, ...}`;
`link_node` = 一行 `{ev: link, from, to, edge}`。**完全不需要回头改 source 节点** ——
没有 read-modify-write, 也就没有"两处 diff"。fold 时建邻接表。
**日志里没有可变结构, 只有事件。**

事件 schema 定成**能 compact 的形状**: 每条事件自包含 (`{ev, name, ...}`), fold 就是顺序 apply。
本期不做 compaction, 但格式不挡将来加。

节点名的自解释性靠**目录名** (与 features 一致): `graph-nodes/渲染选型/` 一眼知道是什么。

### K6. 图的 meta 自解释

`DECISION_TREE_ROOT.md` 的 frontmatter 声明 **statuses 表** 和 **edges 词汇表**, body 是散文计划。
jsonl 不需要自解释 —— 读它的人先读 meta 才能解释日志 (与 ground 的关系同形: GROUND.md 声明 pins,
目录装数据)。glob `DECISION_TREE_ROOT.md` 即发现图存在 (递归时发现所有子图)。

### K7. 渲染器只认 level, 不认 status

树 meta 里 status 声明成 `{status_name: level}`。levels 固定为
`info | success | warn | error | muted`。渲染器读到的永远是 level, 于是:

- 决策树声明 `{open: info, discussing: warn, researching: warn, decided: success}`
- 将来别的图声明自己那套, 映射到同一批 level
- **渲染器一行不改**

这是唯一能让「不同的图有不同的状态」和「渲染器不提前通用」同时成立的做法。

### K8. `pruned` 正交于 status

**剪枝是从状态机退出, 不是状态机的一个状态。** 因为它可以从任何状态发生 (讨论到 B 被剪、
调研到 C 被剪), 且剪完之后**原来的状态仍然有意义** ("它讨论到 B 就被剪了")。
做成状态枚举会丢掉这条, 状态机图上也会多一个从每个节点指向它的收口。

- **不叫 `delete`** —— `delete` 会招人真去 `rm -rf`, 而要点恰恰是**剪痕必须在场**, 文件留在原地。
- **不叫 `done`** —— 见 K9。

### K9. 不做状态传染; 选 `pruned` 不选 `done` 的理由

`pruned` 和 `done` 在树上的传播方向**相反**, 成本差一个数量级:

- **`pruned` = 向下传染。** 父被剪, 子树失效。读取时从根往下走, 遇到 `pruned` 就整棵子树置灰 ——
  一次遍历, 而**渲染本来就要遍历整棵树**, 边际成本接近零。写入只在被剪的那一个节点上写。
- **`done` = 向上汇聚。** 父是否完成取决于所有子 → **每次子节点变动都要回写祖先链** ——
  写放大、并发冲突。这就是"得不偿失"。

更根本: **`done` 是约定, `pruned` 是事实。** "父是否等于子的与"是规则, 规则会变;
一旦落盘, 改约定就要重算全部数据。**持久属性只该存事实, 不该存约定。**
而且 `done` 根本不需要存 —— 渲染遍历时顺手聚合即可。

**孤儿不是一个需要存储的状态, 是渲染期的一次推导。** 不给子节点打任何标记。
所以持久数据里**永远不存在"孤儿"这个东西**。

### K10. 决议是元, 讨论过程是内容

"节点决议最终的产物是清晰可执行的" → 决议进 jsonl (一两句、可执行), **不是散文**;
写不下说明它还没到"清晰可执行"。讨论过程 / 调研物料才落节点目录。
这正是 debates 的形状: `DEBATE.md` 的判辞面是元, `r1..rN.md` 是内容。

### K11. 人任何时候都不手写结构文件

**这个机制就是人机协作的。** 人只做三件事: **点节点、了解状态、确认 action**。
jsonl 由通道独占写 (单写者, 无竞争), 人点网页 → 通道 → 日志。
手编辑只保留在内容层 (节点目录里的物料)。

这条推翻了早期"模板复制 + 手动编辑"的姿态 —— 那个姿态继承自 `moss features create`,
但 features 是人工策展场景, 这里是协作运行场景。

### K12. 验收不能在子节点上自己做

子节点只能在自己目录上下文里按约定留物料, 不能给自己验收。**验收天然向上走, 内容天然向下沉。**
这是性质不是缺陷 —— 层级、授权、见证都落在这一层, 也是本 workstream 作为社会化工程实验的接口。

### K13. 分形递归

**结构集中(每层一个自有的 log), 内容分形(每节点一个目录)** → 递归不打架。
一个节点目录可以自己是一张子图的根:

```
{root}/
  DECISION_TREE_ROOT.md       # 图 meta
  tree.jsonl                  # 结构 + 状态(本级单写者)
  graph-nodes/{name}/         # 内容
    ...物料
    DECISION_TREE_ROOT.md     # ← 可选: 这个节点自己也是一张子图
    tree.jsonl
    graph-nodes/...
```

机制在**同一目录形态**上递归。glob `**/DECISION_TREE_ROOT.md` 自然发现所有层级。
这比目录式更利于分形递归 —— 目录式下结构与内容混在一处, 递归时两棵树的节点会互相污染。

### K14. 起名三元: name / title / description

- **`name` = 目录名 = id。** 不设独立 id 字段 (features TOPOLOGY.md: "The feature name is the directory
  name. No separate `id` field — the filesystem is the namespace")。唯一性范围 = 这棵树根下。
  本机是 Darwin, **文件系统大小写不敏感**, 所以只允许小写。
- **`title` 可以不唯一** —— 它是给人看的字。创建时从 title slug 出 name, 撞了加后缀。
- **`description` 是被机械扫的那一行。** ground 的 frontmatter pin 就是 `keys: [name, description]`,
  模型列表时读到的只有它 → 写成一眼可判的短句, 不是背景介绍。
- 另有 `status_note`: `description` 说"这节点是什么"(静态), `status_note` 说"现在到哪了"(动态)。
  图以状态为核心, 颜色只给 level, 那"为什么在这个状态"就需要 `status_note` 这一行。

### K15. 命令面

`text__` 载 JSON + pydantic 校验, 先例是 `host/listener/controller.py` 的 `set_etiquette_spec`
(「`text__` is a JSON string of EtiquetteSpec (schema in the instruction)」)。

K16 判据 (`file-editor-contract/FEATURE.md`): 少字段 + 无歧义类型 → example 够;
**多字段 / 有可选分支 / 有字段间约束 → 必须拼 schema**。本命令面属于后者, 所以 schema 拼进 instruction。

命令面围绕**图本身**和**节点本身**, 不掺文件编辑:

- 图: `create_tree` / `open_tree` / `trees`
- 节点: `create_node` / `link_node` / `update_node`
- 观察: `read` / `focus` / `history`

**创建图 → 编辑图是两个流程。** meta 建完 = 图锁定 = 可渲染; 未建完只能走新建流程约定。
锁定 = **边词汇表冻结** —— 编辑期只能在 meta 声明的边里选, 不能新造。**这就是"不扩大"的机制。**

### K16. 文件编辑不塞进这个概念

物料编辑用 file_editor 那套 (action 卡片 + 单闸口), 关联到 thread = **节点路径**。
图窗口只负责图本身和节点本身。两个网页窗口, 不合流。

file_editor 复用的是它的**路径边界制度** (`project_home ∪ tempdir`) 与 store —— 不是它的 surface。

### K17. 渲染: 左图右 detail, ECharts 起步

页面顶部 n 个图 (artifacts 的 label tabs 形态)。左边图, 右边 detail。**右边是内容面, 不是图面**:
点节点 = 切换右侧, 不是导航图。子节点 → 顶部是该节点的**目录路径** (不存在则 `(not exists)`,
存在则 ls 列表, 条目是 link 点击 `open`); 点根节点 → 树 root 路径 + plan 文档 (markdown)。

点击只切 panel —— 去掉了 ECharts 的 `expandAndCollapse`, 否则点节点会折叠子树、和"选中看内容"打架。

Tree rendering 选型 (CDN 已实测, 直接 `<script>` 引进, UMD 无构建):

| 库 | 路径 | 版本 | 实际传输 |
|---|---|---|---|
| **ECharts** | `jsdelivr/npm/echarts@5/dist/echarts.min.js` | 5.6.0 | **335 KB** |
| AntV G6 | `jsdelivr/npm/@antv/g6@5/dist/g6.min.js` | 5.1.1 | 393 KB |
| Cytoscape | `jsdelivr/npm/cytoscape@3/dist/cytoscape.min.js` | 3.34.3 | 137 KB |

压缩后差距远小于磁盘差距 ("G6 太重"不成立)。选 ECharts: 三十行配置拿到
树布局 + click + `roam` + `expandAndCollapse`, 符合"样式不要搞那么复杂"。
**渲染器输入规范化 → 选型可逆**: 渲染器只吃 `{nodes:[{name,title,level,pruned,children}], edges:[]}`,
通道和文件格式不许知道渲染库存在。换库 = 换一个渲染器。

**mermaid 出局**: click 需在图源码里预声明且要 `securityLevel: 'loose'`, 无 pan/zoom, 无折叠展开。
它是渲染器不是浏览器。artifacts 的 mermaid kind 因此不能复用于这一层。

**剪枝的视觉硬指标**: per-node 样式必须支持 (剪枝=置灰但不消失)。只能全局统一着色的库直接出局。

## Implementation Notes

- 节点落点 `nodes/webview_apps/decision_tree/`, 与 artifacts / zhihu 同族。
- 单端口 surface 复用 artifacts 模式: `websockets` 的 `process_request` 对 `/` 返回 index.html,
  对 `/ws` 返回 None 继续升级。零新依赖。
- 路径边界复用 file_editor 的 `resolve_target` 形态 (`_inside(r, root) or _inside(r, tempdir)`)。
- 树根 = `matrix.home / "trees"` —— **树是 node 自己的数据, 默认落在 node 下, 不落在 project 根**。
  `tree_root` 只收相对路径, 拒绝 `.` / `..` / 绝对路径 / 越界。`open_path` 仍在 `trees ∪ tempdir`。
- 上行为人类动作: 点节点 / 确认 action / 打开路径 → 回传模型 (artifacts 的 `user.action` 上行帧形态)。
- 验收闸口形态待 dogfood 定: file_editor 的 `export` 是「返回回执 → 人在 surface 决定 → 模型从 signal 得知结果」,
  非阻塞。本期沿用这个形态而非阻塞等待。

## Open Problems

- **确认语义 (dogfood 实锤, 核心遗留)。** v0 的 confirm 是非阻塞旁证: 动作先落盘, action
  卡片是 mutation 那一刻的瞬态广播, 晚连/刷新即丢。人类「confirm」在持久层是隐形的 ——
  「决议」实际是模型单方面定的, 人类确认是仪式。方向 (人类倾向, 未定):
  **confirm 进 log, 决议 = proposal + confirm 两个事件**。`create`/`update` 事件已是 proposal;
  加一个 `confirm` 事件; 「待确认」= log 里 proposal 之后无 confirm 的节点, 连接时 fold 出来
  重放, 天然不丢。不引入第二个队列, 与树状态同一套 append-only 哲学。代价: 节点状态机多一条 confirm 轴。
- **compact** —— jsonl 只增不减。本期不做, 格式已留口。
- **link_node 的跨边语义** —— 本期保留接口不实现; 将来做依赖/推翻类跨边时, "存活"从
  "祖先链无 pruned" 变成 "从任一根可达且链上无 pruned", 仍是一次遍历, 仍不用存。
