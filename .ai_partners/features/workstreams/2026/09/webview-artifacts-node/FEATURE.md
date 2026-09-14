---
created: 2026-09-14
depends: []
description: 流式 artifact 表面：一个 write 命令 + 8 种 kind + duration 时间预算，head/delta/full
  三阶段流式，全屏单一看 + label 回溯面 + 人类动作上行 notice 反身性。
milestone: null
priority: P1
status: completed
status_note: 8 kinds + duration + notice uplink + non-singleton port, dogfooded via
  MCP
title: Webview Artifacts Node
updated: '2026-09-15'
---

# Webview Artifacts Node

> Use `moss features set-status webview-artifacts-node <status> -m "note"` to update state.
> See [TOPOLOGY.md](TOPOLOGY.md) for directory layout and [README.md](README.md) for the full convention.

## Motivation

MOSS 有 `channels/mermaid_draw.py`（一次性 data-url 浏览器渲染）和 `text_blocks`
（Reflex 共享文本表面），但没有一个**通用机制让模型把 artifact 流式写进一个活的视觉表面**。

要解决的问题：模型在生成时就该被看见。现在的做法是"生成完 → 整体渲染"，中间过程不可见。
这个 node 是"通用策略的一个 openbox 例子"——最小、可被复制：
- 流式源码在一个位置实时累加（可见的思考过程），
- 尾包按 kind 收尾，在源码**旁边**渲染出结果，
- 源码与渲染两个 pane 各自独立可切。

`kind` 只决定尾包怎么收尾，三阶段本身与 kind 无关——所以这是通用机制，不是 mermaid 专属。

## Design Index

- Key design documents: `design/`
- Key discussion records: `discuss/`

## Key Decisions

- **是 artifacts，不是 duplex 协议。** 一度尝试把 wire 帧做成 `core/duplex` 的
  `ChannelEvent` 形状以求"协议保真"——过度设计，被人类架构师叫停。CTML 的核心动机是
  让**流式**成立，不是 command 化。裸 WebSocket + 简单帧即可。
- **代码驱动，拒绝 DSL。** canvas kind 的内容 = 模型写的 JS，页面 eval 后对着裸
  canvas 2D 上下文画。不做绘图 DSL，不做场景图（Fabric 一类的对象模型是"wire 传状态"
  思路的遗留，与"wire 传流式源码"的机制相斥，已作废）。
- **三阶段流式是本体。** 命令吃 `chunks__`（CTML 特殊参数，任意命令可声明，接收
  `AsyncIterator[str]` 逐 token 流；内容须 CDATA 包裹、不可嵌套 CTML）。首包公告
  → 间包累加 → 尾包收尾。命令只是载体，流是核心。
- **一个 `artifacts` 通道，带 kind 的单个 write 命令。** 不拆多通道——回溯面
  (`read`/`history`/`display`) 是 kind 无关的，拆开割裂。label 是 artifact 的坐标。
- **回溯控制面**：`read(label)` 取回源码，`history(n)` 回 `label: description`（LRU tail），
  `display(label)` 重新点亮。这是模型的"pull 反身性"，不用 context_messages。
- **styles 也是 artifact。** `style` kind 注入 CSS——页面自己的长相也归模型管。
- **单端口零新依赖**：`websockets.serve(handler, host, port, process_request=...)`，
  `process_request` 对普通 GET 返回 `connection.respond(...)` 发 `index.html`，对 `/ws`
  返回 `None` 继续升级。`websockets` 已在 `[host]`，无需 aiohttp/fastapi。
- **节点零安装**：`exec.command: python` → `sys.executable`（MOSS venv，含 host 依赖），
  无自身 venv、无 INSTALL.md → 节点默认"已安装"。
- **时间预算是命令内建的，不是 sleep 原语。** `write` 带 `duration: float = 0`，从命令
  启动计时，尾包收尾后内部 `sleep` 到时间用尽。`duration > 0` 让命令自己占据通道持屏——
  一段多图序列无需 `sleep` 原语或 `say` 卡位即可编排成动画节奏。
- **模板化 kinds，token 最少化。** 一个 write 命令 + kind 决定 source 含义与渲染器：
  canvas / mermaid / markdown / image(MJPEG) / hls / term(黑板) / html(逃生舱) / style(元)。
  纯 html 永远可选，但常用类型用最小 payload，docstring 自解释每种 source 契约。
- **style 是元机制。** 全局注入 CSS，不入 tab、不入 history、不可 display，只在 `_items`
  留档供 read 召回。
- **上行链路 + notice 反身性。** 页面把人类动作（source 切换、tab 切换、连接）作为 WS
  上行帧传回，Python 侧 deque tail 存最近 N 条，挂进 channel 的 `notice`（warm data，带
  `[HH:MM:SS]` 时间戳）。模型由此能"看见"人类在页面上做了什么。
- **非 singleton + 端口入参。** `singleton: false` 允许多实例；`--port` argv 或
  `MOSS_ARTIFACTS_PORT` 环境变量定端口，同端口冲突自然 bind 失败。

## Implementation Notes

- 人类编辑 → 模型这条路 v0 未接。若要接：从 `main()` 把 `matrix.session`
  (`matrix.py:399`) 注入 channel，调 `session.add_input_signal(...)`。**注意**：
  duplex provider 的容器不携带 MOSS `Session` 契约，`CommandUtil.send_signal` 在此解析不到。
- canvas kind 在页面内 eval 模型写的 JS——这是**设计意图**（本地、模型自著），不是疏漏。
- 验收 = 人类读流。流式表面必须有人接住那串流：人类开 MCP，模型驱动绘制，人类旁观并切 pane。

## Implementation

- 节点：`nodes/webview_apps/artifacts/`（`NODE.md` + `main.py` + `index.html`），3 文件，零新依赖。
- 命令面：`write`(8 kinds + duration) / `read` / `history` / `display` / `source` / `remove` / `clear`。
- wire 帧（裸 WS，非 ChannelEvent）：`state` / `artifact.start` / `artifact.chunk` /
  `artifact.end` / `artifact.focus` / `artifact.remove` / `source` / `clear`；上行 `user.action`。
- 前端：单 view 全屏、label tabs、可折叠 source pane、绿点事件闪烁；mermaid/marked/hls.js 走 CDN。