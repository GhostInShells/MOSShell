---
created: 2026-10-17
depends:
- screen-manager
- webview-artifacts-node
- dsh-fusion
description: 让 ghost 的搜索动作成为人和 ghost 共享的视觉面：搜索任务作为节点，来源是节点内的证据， 类型化边构成可复现的知识拓扑。当前只有验证
  node（mock 数据），真实事件流与 dsh search agent preset 交给 ghost 实现。
milestone: null
priority: P1
status: in-progress
status_note: 原型 node 已验证：nodes/research/web_search（mock 数据 + d3 拓扑/详情/阅读器/回放）。真实事件流、dsh
  search agent preset 与 screen 接线交给 ghost 实现。
title: Search Surface
updated: '2026-10-17'
---

# Search Surface

> Use `moss features set-status search-surface <status> -m "note"` to update state.
> See [TOPOLOGY.md](TOPOLOGY.md) for directory layout and [README.md](README.md) for the full convention.

## Motivation

ghost 的搜索目前发生在 dsh 内部：人只看到一次工具调用和最终结果，看不到 fanout 的
分支、逐条来源的到达、以及一次研究为什么转向。与此同时，搜索成本也因为在内部而不可见
（原生搜索一次 query 就是一个完整模型 turn，`searchMaxQueries × maxUses` 还会相乘）。

目标不是"给结果画一张图"，而是让**搜索行为本身**成为共享面：

1. 给搜索工具一个通用界面与 detail（来源、摘录、provenance）。
2. 给搜索动作一个可视化的实时视图（查询队列、fanout、来源到达）。
3. 给大型搜索任务一个可复现的视觉拓扑（任务图 + 关联路径）。

三件事是同一个机制的三个粒度，共享一个 SearchTask 身份。

## Design Index

- 验证节点：`nodes/research/web_search/`（`NODE.md` + `main.py` + `index.html`）
- 视觉机制：[screen-manager](../../2026/07/screen-manager/FEATURE.md)（item / group / focus）
- 产物语义：[webview-artifacts-node](../../2026/09/webview-artifacts-node/FEATURE.md)（artifact 即文档、append、version 是位置）
- dsh 融合：[dsh-fusion](../../2026/08/dsh-fusion/FEATURE.md)（session 事件、plugin 桥、preset）

## 实体层级

数据实体逐级上升；**只有 L2 上拓扑图**，L0/L1 收在节点详情里。

| 层 | 实体 | 内容 | 上不上图 |
|---|---|---|---|
| L0 | **Source（证据）** | url / title / snippet / 抓取正文 / 发布时间 / provider | 否 |
| L0' | **RawCall（审计）** | 一次 API 调用的入参与原始返回 | 否，附着在 Query 上 |
| L1 | **Query / Fetch（动作）** | 查询串、provider、参数、结果集、耗时/成本；详情抓取与 query 同属任务的 `actions` | 默认否 |
| L2 | **SearchTask（研究动作）** | 动机、研究问题、一组 actions、状态、预算、**产物出口** | 是，节点 |
| L3 | **TaskEdge（关联路径）** | `expands / supports / derives / compares / supersedes / contradicts` + `basis` + provenance | 是，边 |
| L4 | **TaskGraph（研究路径）** | L2 + L3 构成的可复现 DAG；可整体作为一个节点嵌入更高层 | 是 |

**单一节点 = 搜索任务 + 产物。** 不同任务类型（检索、对比、验证、深挖）共用节点壳，
差异在 `artifacts[]` 与 `exits[]`。

**分形要拆成两个方向**，否则图会被 source 淹没：

- **纵向（嵌套）**：L2 可包含子任务，展开成 cluster；只有一次详情查询获得独立动机时，
  才从 L1 提升为子任务。
- **横向（关联）**：L2 之间由 L3 连成知识拓扑。

节点最小契约：

```json
{
  "id": "task_xxx",
  "parent_id": null,
  "motivation": "为什么做这次搜索",
  "question": "要回答什么",
  "status": "open|running|closed|failed",
  "actions": [{"kind": "query|fetch", "input": "...", "result_ref": "...", "cost": {}}],
  "artifacts": [{"id": "...", "kind": "note|answer|dataset|claim", "ref": "...", "summary": "..."}],
  "exits": [{"artifact_id": "...", "role": "conclusion|evidence|lead"}]
}
```

边：`{from, to, type, basis, provenance}`。`basis` 必须由 agent 显式给出——语义关联无法
从 tool-call 时序推导，它同时是可复现路径和人对图的最有价值输入。

## 三档可视化

| 层 | 呈现 | 机制 |
|---|---|---|
| L0 Source | 卡片 + 详情页；外链；正文按不可信数据隔离 | webview-artifacts / renderer registry |
| L1 Query/Fetch | 工作台里的一行：状态、耗时、结果数，实时增长 | 活的 artifact（append 语义） |
| L2 Task | 同一身份两种渲染器：详情面板与 d3 图节点 | screen-manager item |
| L3 Edge | 有向边 + 类型；hover 显示 `basis`；点击高亮路径 | d3 force |
| L4 Graph | 力导向图；子任务 cluster 展开/折叠；时间轴回放 | d3-force + canvas（大图） |

人机共享面用 screen-manager 的 group 装三个 item：研究工作台（L1）、拓扑图（L4）、
阅读器（L0）。人在图上点节点 → 阅读器 focus；人的点击作为 uplink notice 回流。

## Key Decisions

- **dsh 只做 observer 的边界不够，任务边界与关联边必须显式产生。** dsh 的 `tool/call`
  / `tool/result` 只能给 L0/L1；原生搜索的 `maxUses` 在单个模型 turn 内部完成，没有事件。
  所以用专门的功能性 search agent preset（类似 ego 的包装 agent），并在 preset 里暴露
  轻量 framing 工具：`open_task(motivation, question)` / `link_tasks(from, to, type, basis)`。
  L2/L3 不能靠事件时序猜。
- **执行者是谁不重要，观察者拥有数据模型才重要。** Python 复刻 web search 协议并不天然
  "只能给 ghost 用"——fanout 是 agent 并行的 tool call，与实现语言无关。因此不先重写
  provider；先做 dsh search agent preset + 事件观测，执行可暂时留在 dsh 或换成 MOSS 工具。
- **append-only 事件日志是真相，图是投影。** 事件至少包括
  `search/task.open|close`、`search/action.start|result`、`search/source.add`、
  `search/link.add`、`search/artifact.add`。可复现 = 日志 + 输入参数，而不是存最终 JSON 图。
- **d3 是渲染器，不是数据模型。** 数据契约（节点/边/稳定 id/lineage）先定，d3 只消费
  `{nodes, links}`。不要让布局或交互反向污染实体。
- **迁移时禁用 dsh 内置 tool-web。** 在 Dolores 的 `dsh_preset/dolores-ego/agent.cordis.yml`
  里关掉内置 `tool-web`，由 plugin 用 `agentCtx.tools.register` 注册 MOSS 工具（plugin
  已有 `moss_*` 工具 + HTTP 回 MOSS 的通路）。工具返回值只给结构 + 样本 + `data_ref`，
  视觉负载走旁路事件，不进 dsh 上下文。

## Exploration paths

- **Python 复刻 web search 作为主路径**：协议本身简单，但只解决了 executor；仍需要 agent
  做 fanout 与任务规划。结论：不是第一优先级，除非要把执行与成本预算完全收进 MOSS。
- **只镜像 dsh 的 tool 事件**：够 L1 和浅层 L2，但拿不到子任务和关联边；目标 3 不成立。
- **把搜索完整留在 dsh 内部**：原生搜索的中间态不可见，拓扑最多是"每次 tool call 一个点"。
- **手工写一份最终图 JSON**：不可复现，无法回放；已被"日志是真相"替代。
- **d3 自定义布局 vs 现成图**：先用力导向验证概念；大图再考虑 canvas 与聚类。

## Methods

- 验证 node 沿用开箱 webview node 的通用风格：一个进程、一个 channel（模型面）+ 一个
  WebSocket/HTML surface（人面）；端口默认临时绑定，URL 走 warm `url` notice。
- 页面可脱离 node 直接打开（mock 数据内嵌）；由 node 提供服务时连接 `/ws`，模型侧可下发
  `focus` / `step` / `view`，人的点击上行 `user.action`，进入 channel `notice`。
- 三档共享同一份 mock 任务数据；`builtAt` / `closedAt` / `at` 驱动时间轴回放。

## Implementation Notes

- 文件：`nodes/research/web_search/{NODE.md, main.py, index.html}`。
- `main.py`：`SearchSurface`（HTTP + WS）+ `web_search` channel（`focus` / `replay` /
  `view` + `notice`）；`Matrix.discover().run(main)`。
- `index.html`：自包含演示数据；d3 v7 走 CDN；力导向图、子任务折叠、edge `basis`
  tooltip、任务详情、来源阅读器、产物页、时间轴、GHOST/HUMAN activity rail。
- 目前**没有真实搜索执行、没有事件日志、没有 screen-manager 接线**。页面清楚标注
  `PROTOTYPE · 演示数据`；来源正文全部是演示文本，不是真实抓取。
- 交给 ghost 的第一版实现建议：保留页面与交互，把内嵌 mock 换成事件日志的投影；先验证
  dsh search agent preset 能显式发出 task/link 帧，再接 screen-manager 的多 item 布局。
- 未决项：真实事件 schema 与持久化位置；provider 选型（DeepSeek 原生 / Exa / Perplexity）；
  search agent preset 的 framing 工具面；成本预算（queries、sources、tokens）如何按任务
  记账；不可信内容的渲染隔离；大图性能；多屏下人的焦点与模型布局的主权边界。

## Open Problems

- 大型研究是否由 MOSS 持有 research loop（dsh 只下目标），还是由 search agent preset
  自行规划、MOSS 只观测？当前倾向前者用于大型任务，后者用于简单搜索。
- "复现一次搜索路径"的语义：重放 API 调用、重放 agent 决策，还是只重放已存证据的图？