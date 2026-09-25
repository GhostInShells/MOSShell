---
title: MCP Fusion Point — 寻找 MCP 与 MOSS 的合适融合点
status: in-progress
priority: P1
created: 2026-07-31
updated: 2026-09-10
depends:
  - mcp-hub-channel
  - speech-protocol-alignment
milestone: null
description: >-
  正式目标：找到 MCP 与 MOSS 的合适融合点——先定位 MCP 在 MOSS 架构中的身份，
  再连带回答是否以 MCP 作为 cell 间 RPC 协议的底座。位置问题已收敛为「MCP 是
  外部皮，不是脊柱」，内部 cell↔cell 走原生 matrix.rpc。

  2026-09-10 重新对齐：此前记录的决策 1–13 及其实现叙述已删除（`git log` 可查，
  见下），因迭代轨迹被 8 月中旬那一波污染，方案与代码对不齐。本轮从动机重新
  对齐 hub 并简化机制。
---

# MCP Fusion Point — 寻找 MCP 与 MOSS 的合适融合点

> 状态：in-progress。hub 侧方案与实现重新对齐中，以本文件为唯一载体。

## 为什么不继承旧内容

本 feature 实现到一半，**8 月中旬 deepseek 回归那一波把迭代轨迹严重污染了**：
方案与代码不匹配、对不齐，声明跑在交付前面。继续在旧轨迹上叠加只会继续漂。
所以删掉旧决策与旧叙述，只保留仍然成立的判断，其余从动机重新对齐。

被删内容可查：

```
git log -p -- .ai_partners/features/workstreams/2026/07/mcp-fusion-point/FEATURE.md
```

同目录下的旧设计文档（`design/mcp-node-server.md`、`design/mcp-node-client.md`）
描述的部分机制从未交付，**未经本轮对齐，不要直接当权威实现**——处置留待本轮
方案落地后。

## 仍然成立的判断

### 1. MCP 的位置：外部皮，不是脊柱

MCP 在 mesh 边界双向存在（node as mcp server 朝外 / node as mcp client 朝内）；
内部 cell↔cell 通讯走原生 `matrix.rpc`（注册表 + 发现 channel + zenoh put/sub +
JSON-RPC 2.0），不用 MCP 作内部 RPC 底座——双向外转损失类型，且零优势。

### 2. node run as MCP server 的传输通路已成立

`Matrix.aserve_mcp` / `Matrix.serve_mcp`（`core/blueprint/matrix.py`）在 matrix
生命周期内 serve stateless streamable-http。实机证据：`ghost_bridge` 用
`Matrix.new("ghost_bridge", persist=True)` 入网并 serve。

### 3. ghost_bridge（原 mailbox 位）作为独立 MCP 已成立

`ghoshell_moss.mcp.GhostBridge` + `serve_ghost_bridge` + `moss mcp
serve-ghost-bridge` CLI。外部 agent 经 MCP tools 与 ghost request-reply，ghost 用
CTML `ghost_bridge:reply(task_id, text__)` 回复。

## hub — 重新对齐后的设计

### 动机（人类架构师 2026-09-10 重述）

- a. hub 从 config 里读取所有 MCP
- b. channel 可以主动打开指定的 MCP
- c. 打开的 client 落在子 channel 里，通过 notice 更新其 tool 的 **JSON schema**
  接口描述——是 schema，不是 Command 对象
- d. 子 channel 有并行 / 同步两个调用能力
- e. 把 tasks / resources 原生接入 channel
- f. hub 有动态增删 config 的能力，被 flag 到它的 factory 上，决定是否开启
- g. `moss mcp` CLI 可增删 MCP：实际修改 config，并给模型发 notify；**不自动
  开关，开/关由模型决定**

### 骨架

```
mcp  (父 channel, mode 级, 活在 persist=True 的 host 进程)
├─ on_startup          连 config 里 auto_connect 的 server
├─ list()              config 条目 + 各 server 连接状态
├─ open(name)          建 session → 长出子 channel
├─ close(name)         拆子 channel → 断 session
├─ add/remove          [仅当 factory flag allow_config_edit] 只改 config, 不连
├─ virtual_children(callback)   同步返回"已打开子 channel"缓存
├─ on_refresh_meta()            async 侧: 重读 config, 重建缓存
└─ notice              已打开 server 概览 (未打开的不进 context)

mcp.<server>  (虚拟子 channel, 以 server 名命名)
├─ notice              渲染该 server 全部 tool 的 JSON schema (code-as-prompt)
├─ call(tool, text__)  阻塞
├─ acall(tool, text__) 非阻塞
└─ _authorize(...)     warrant 集成 seam (当前直通)
```

父 channel 直接用 `new_channel()` + `Builder.virtual_children`，不需要
`MCPHubState` / `new_channel_from_state` 那层自定义 ChannelState。

### 命名与语义

- **named mcp client**：子 channel 以 server 名命名，CTML 路径即 `mcp.<server>:call(...)`。
- `open(name)` / `close(name)` 主动开关。
- **`close` 只断连、保留 config**；`remove` 才删配置——`close` 后再 `open` 不需要重配。
- `add` / `remove` 仅当 factory flag `allow_config_edit` 开启时挂出。

### CLI（g）的形状

`moss mcp add/remove` 是 **`persist=False` 的一次性 node**：`Matrix.new(...)` 入网
→ 写 config → `session.add_signal(new_notify_signal(...))` 通知模型 → 退出。
模板见 `moss-ghost send`（`cli/ghost_run.py`）。一次性 node **不能** `provide_channel`
（`zenoh_presence` 有硬闸口），它也不需要。

## 未做 — 已在代码中标记位置

以下两项本轮不做，但**必须在代码里留下开发者可见的标记**（不是 ghost 可见的提示）：

- **e. tasks / resources 原生接入** —— 未做。标记位置：子 channel 模块的模块级常量。
- **warrant 集成** —— 未做。warrant 体系已在运行，但把 hub 的开/关/改配置接进
  warrant 工作量大。标记位置：单一 `_authorize(action, name)` seam（四个动作共用）。

**纪律（容易搞反）**：channel 的 `instruction` / `notice` / `description` / 命令
签名 / docstring 是说给 **ghost（运行时模型）** 听的——它们是 ghost 的 prompt。
开发者状态（未做、待接、TODO）是说给 **coding 模型 / 下一个开发者** 听的，只能
写在注释、模块级常量、FEATURE.md 里，**绝不渲染进 ghost 的上下文**。ghost 不需要
知道"tasks 还没做"——哪些命令可用，它看得到；缺失对它是自明的。

## 参考

- `src/ghoshell_moss/mcp/config.py` — canonical 配置模型（MCPServerConfig /
  MCPHubConfig / auth），CLI、channel、GUI 共享。
- `src/ghoshell_moss/mcp/ghost_bridge.py` — bridge 实现。
- `channels/mcp_hub.py` — **将删除**（自带一份重复的 config/session 模型，扁平无
  子 channel，偏离动机）。
- `channels/mcp_channel.py` — **将按本骨架重构**（已有虚拟子 channel，但调用收在
  父 channel、notice 只列名、无 f/g）。
