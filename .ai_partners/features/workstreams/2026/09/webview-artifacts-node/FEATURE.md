---
created: 2026-09-14
depends: []
description: 流式 artifact 表面：一个 write 命令 + 8 种 kind + duration 时间预算，head/delta/full
  三阶段流式，全屏单一看 + label 回溯面 + 人类动作上行 notice 反身性。
milestone: null
priority: P1
status: in-progress
status_note: 8 kinds + duration + notice uplink + non-singleton port, dogfooded via
  MCP。2026-10-14 讨论出下一版方向（artifact 当成一份很小的文档：原地改动 + 版本位置 + 可导出关键帧），未实现。
title: Webview Artifacts Node
updated: '2026-10-14'
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

## 下一版方向（2026-10-14 讨论，未实现）

人类提的点：**给每个 artifact 的源码加 append / str_replace**，像编辑一个文件那样原地累加，
于是"单张幻灯片里也能一帧一帧地播"。我的判断与要绑定的约束：

- **身份说准**：artifact 就是**一份很小的文档**，指向一个会渲染的目标——照抄 file_editor 那套
  （线 / 一步 / 效果），别发明第二套语义：`write` = 整篇替换，`append` / `str_replace` = 改动，
  版本是**位置**不是快照，`read(label, n)` 能取回任意版本。这样"改一段、它重画一次"和改文件是
  同一种动作。
- **买到的是"位置不变"**：今天是写一张、切一张，人看到的是切页；原地改则黑板一直在，只是上面在
  长东西——讲机制时差别很大。
- **kind 分家（不定下来就会抖）**：`canvas` / `term` 天生容忍半截源码，可以流式重画；
  `mermaid` / `html` 不行——源码没写完就是错的，它们应当**停在上一张好图**，而不是闪一下报错。
- **不许抢人的视线**：谁在看什么是人的手。原地刷新的 artifact 要在页面上标明自己是**活的**；
  人一旦切走，不能用下一次 append 把它抢回来。（同日定下的"不给自己开绕开人的口子"是同一族。）
- **两种时间尺度别互相顶替**：写 JS 每帧重画 = **一张画里的运动**；改源码 = **一个故事里的步骤**。
  两个都要，别用其中一种去替另一种。
- **向上延展（人类给的框架）**：thread + action + effect 这套分形地用在每个方面——artifact 也可以
  **导出**，导出时把关键帧串起来（effect 连着 effect）本身就是一段**可重放的动画**。
  「你说的 20 年那套架构思想，分形地用在项目每一面。」

> 今天不改代码（下午有展示）。此节是讨论产物，等排期。

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