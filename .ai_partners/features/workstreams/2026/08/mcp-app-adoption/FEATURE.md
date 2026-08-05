---
title: MCP App Adoption — 评估 mcp-app 生态的引入价值与集成路径
status: draft
priority: P2
created: 2026-08-05
updated: 2026-08-05
depends: []
milestone:
description: >-
  独立于 mcp-fusion-point 的引入价值评估 workstream。核心问题：mcp-app（MCP Apps /
  mcp-ui）体系对 MOSS 怎么用、哪些项目值得引入。融合点裁决只影响集成成本，不影响引入
  价值的判断——高价值时即使不走 MCP 融合也用别的机制集成。
---

# MCP App Adoption — 评估 mcp-app 生态的引入价值与集成路径

> Use `moss features set-status mcp-app-adoption <status> -m "note"` to update state.
> See [TOPOLOGY.md](TOPOLOGY.md) for directory layout and [README.md](README.md) for the full convention.

## Motivation

MCP Apps（开源标准 ext-apps / mcp-ui 实现）是把交互 UI 挂到 MCP tool 上的标准协议：
tool 声明 `_meta.ui.resourceUri` → host 经 `resources/read` 取 HTML → 沙箱 iframe 渲染 →
JSON-RPC over postMessage 双向通信。已获 ChatGPT/Goose/LibreChat/Claude 等 host 支持。

MOSS 是否需要、以及如何引用这个生态，是一个**独立的引入价值命题**——与 mcp-fusion-point
（cell 间 RPC 底座 / MCP 在架构中的位置）原则不关联：

- **引入价值判断独立**：mcp-app 体系对 MOSS 的核心是"如何用它、哪些项目值得引入"。
  融合点（用什么协议/身份集成）只影响集成的**成本**，不影响值不值得。
- **高价值时另寻机制**：如果某些项目的引入价值很高，即使 mcp-fusion-point 裁决
  "MCP 留在外部边界"，也应当用别的办法（node / channel / 直接协议）集成它。

一句话定位：**这个 workstream 判断"要不要"和"要什么"，mcp-fusion-point 判断"怎么连"。**

## Design Index

- Key design documents: `design/`（暂无）
- Key discussion records:
  - 本 workstream 尚无讨论记录。调研输入来自 2026-08-05 的 mcp-ui 全量调研
    （repo clone 在 `/tmp/mcp-ui-research`，来源在 Implementation Notes 的调研轨迹）。
- 相邻轨迹（参考，非依赖）：
  - `mcp-fusion-point`（draft）— cell RPC 底座命题，与本 workstream 解耦
  - `mcp-hub-channel`（completed）— MCP 降级为纯 transport，CTML 接管调度
  - `screen-node`（in-progress）— "窗 = URL" 契约，mcp-app widget 是潜在窗源

## Key Decisions

### K1. 独立于 mcp-fusion-point，两边原则上不关联

mcp-app 的引入价值问题自成一体。融合点裁决（MCP 作为 cell 间 RPC 底座与否）只影响
集成成本，不改变"哪些项目值得引入"的事实。**若引入价值高，不走 MCP 融合也要用别的
机制集成。** 反向亦然：融合点即便走通，也不自动说明 mcp-app 生态值得引入。

### K2. 核心问题二段式

1. **如何用** — mcp-app 对 MOSS 的可用形态（见 Implementation Notes 的集成路径清单）。
2. **哪些项目有引入价值** — 用 K3 的评估标准过生态里的实际项目。

### K3. 引入价值评估标准

一个 mcp-app 对 MOSS 的引入价值由四条决定：

| 维度 | 高价值 | 低价值 |
|------|--------|--------|
| 资产 | **重资产**：真实 backend 数据/状态/产品（POS、项目管理、金融） | 薄壳：纯前端 widget，数据是 mock/可本地造 |
| 协议 | **开放 MCP Apps**（host 无关，MOSS 可接入） | 绑定 ChatGPT Apps SDK（`window.openai`） |
| 独占交互 | widget 有**超越其 tools 的独占交互状态**（内部状态机、协作、实时） | widget 只是 tool 结果的展示（MOSS 拿到 tools 数据可原生复现） |
| 引入成本 | 轻：无 OAuth、可本地跑、协议薄 | 重：商业闭源、OAuth 依赖、包体大 |

### K4. 调研结论（2026-08-05）：当前生态没有值得做成开箱示范 node 的 mcp-app

全量调研后裁决：

- **开放 MCP Apps 生态的 widget 全是商品货**：dashboard / chart / form / map / ecommerce /
  天气，本质都是"tool 结果展示"。连"实时"类也是绕过 spec 的 `connectedDomains`
  WebSocket hack——这些 MOSS 原生做更快，还带流式。
- **真正有资产价值的 app 不在开放协议里**：monday.com / Square 起家于 ChatGPT Apps SDK
  （商业闭源、OAuth），价值在 backend 产品不在 mcp-app 机制。
- **重资产的能力面已被 mcp-hub 拿走**：tools → 数据 → MOSS 原生呈现。mcp-app 层只额外
  加"渲染他们的 widget"，正是商品货部分。MOSS 并未错过重资产价值。

→ **"做一个 MOSS host node 消费 mcp-app" 的命题暂时封冻**，重开条件见 K5。

### K5. 重开条件

以下任一落地即重开：

1. **开放 MCP Apps 上出现重资产产品**，其交互 widget 提供超越其 tools 的独占体验
   （不是 monday 式"widget = 数据展示"，而是 widget 内部有独占交互状态）。
2. **ext-apps 规范官方加流式/订阅**（SEP-1724 的 `extensions` 字段成熟，或 `tool-input-partial`
   扩展为订阅/推送）。
3. **MOSS 出现明确的导出需求**：自己的 UI 要以 `ui://` 资源形式供给第三方 host。

## Implementation Notes

### 调研轨迹（2026-08-05，deepseek-v4-flash via claude code）

- 仓库：`MCP-UI-Org/mcp-ui`（5k stars，Apache-2.0，2026-08-04 活跃）。clone 于
  `/tmp/mcp-ui-research`。官方 spec：`modelcontextprotocol/ext-apps`（spec/draft/apps.mdx）。
- **mcp-ui 是什么**：TS-first 的 MCP Apps 实现。`@mcp-ui/server`（createUIResource,
  registerAppTool/Resource）、`@mcp-ui/client`（React 的 AppRenderer/AppFrame）、
  Ruby/Python server SDK。Content：`rawHtml` / `externalUrl`，MIME
  `text/html;profile=mcp-app`。
- **流式真相**：spec 唯一流式是 `ui/notifications/tool-input-partial`（agent 生成 tool 参数时
  流式推部分参数）。无订阅、无服务端推状态、无连续更新。widget 生命周期 turn-based
  （tool 调用 → 渲染 → tool-result → `ui/resource-teardown`）。
- **沙箱安全**：双 iframe + sandbox proxy + CSP，untrusted widget 隔离。这是 mcp-ui 的真工程。
- **生态抽样**：开放侧 = live dashboard builder、WeatherScope、mcp-chart-builder（ECharts）、
  KyuRish dashboards、HoloViz（Panel/Bokeh）——全是数据可视化 widget。重资产侧 =
  monday.com（battery widget）、Square（支付/POS）、OpenAI apps-sdk-examples（Pizzaz 等）——
  全绑 ChatGPT Apps SDK。
- **关键判据来源**：monday.com 工程博客（"widget 的职责是帮忙然后让开"）确认 widget 定位是
  tool 结果呈现；dev.to WebSocket streaming 文章暴露 `connectedDomains` CSP hack。

### 集成路径清单（"如何用"的候选形态，非决议）

- **引入方向 — MOSS 当 host**：screen-node 开一个 mcp-app 窗（"窗 = URL" 契约天然吻合），
  widget 的 `tools/call` 经 JSON-RPC 桥回落 hub。成本：要当 host 需实现 AppBridge
  （React 或 QML 自写 JSON-RPC over postMessage，协议薄但沙箱/生命周期要自己扛）。
- **导出方向 — MOSS 当 server**：channel 声明 `ui://` 资源，第三方 host 渲染 MOSS UI
  （"多皮囊"导出）。成本轻，但只在有分发需求时才有意义。
- **中间形态 — 不碰 mcp-app，直接 tools**：mcp-hub 已是此形态。重资产价值已在此处。
