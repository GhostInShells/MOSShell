---
title: MCP Fusion Point — 寻找 MCP 与 MOSS 的合适融合点
status: draft
priority: P1
created: 2026-07-31
updated: 2026-07-31
depends:
  - mcp-hub-channel
  - speech-protocol-alignment
milestone: null
description: >-
  MCP（尤其 2026-07-28 stateless 版）与 MOSS channel 基于高度类似的愿景设计。
  本 workstream 的正式目标是找到两者的合适融合点——先定位 MCP 在 MOSS 架构中
  的身份，再连带回答是否以 MCP 作为 cell 间 RPC 协议的底座。
---

# MCP Fusion Point — 寻找 MCP 与 MOSS 的合适融合点

> 状态：draft → 收敛中。RPC 底座裁决与 MCP 位置已收敛，node run as mcp server 机制
> 敲定，node run mcp client 细节待讨论。讨论轨迹见 `discuss/` + `design/`。
> 用 `moss features set-status mcp-fusion-point <status> -m "note"` 更新状态。

## Motivation

### 为什么要现在做

`speech-protocol-alignment` 的通用化（draft）暴露了 `drain` / `pause` 这类
控制动作可以跨 cell 存在，需要一个跨进程的控制协议。MOSS channel 本质是
有状态 1:1 协议，继续往下做就得发明"协议无关的 stateless transport RPC"。
不想发明轮子。

与此同时，MCP 与 MOSS channel 面貌相近而核心设计选择频繁相左——不做考古
的模型初见会困惑"为什么独立发明轮子"，进入代码后又撞上彼此不可兼容
（channel 是 MCP 的超集——时间第一公民、排序与阻塞、树形构建——这些
恰是 MCP 有意简化的维度）。困惑是融合确实值得正视的信号。

### 正式目标

> 找到 MCP 与 MOSS 项目的**合适融合点**。

- 主问题：确认 MCP（尤其最新 stateless 版）在 MOSS 架构中的**位置**。
- 连带问题：是否以 MCP 作为 MOSS cell 之间 RPC 协议的底座。
- **前者比后者更重要**。位置决定是前提，RPC 底座是派生。

## 判断（非决议）

以下为当前阶段的判断，记录背景而非锁定方向：

- **同源愿景，终将收敛**：MCP 与 MOSS channel 基于高度类似的愿景设计，最终会
  收敛到一处。
- **资源决定生存**：融合点最终由资源（生态、工程成本、行业走向）决定，不是由
  纯架构推演决定。
- **channel 现阶段仍是 MCP 的超集**：在时间第一公民、排序与阻塞、树形构建等
  设计维度上，channel 超过 MCP。这正是两者一直互相不好兼容的原因——MCP 简化了
  这些维度以换取 stateless 的可扩展性。

## Design Index

- Key design documents:
  - `design/mcp-node-server.md` — 子任务 1：node run as mcp server（朝外），机制已敲定
  - `design/mcp-node-client.md` — 子任务 2：node run mcp client（朝内），部分敲定，细节待讨论
- Key discussion records: `discuss/2026-07-31_mcp_position_and_fusion.md`
- 前置讨论：`.discuss/2026-07-30_mcp_duplex_convergence_and_memento_branch.md`

## Key Decisions

### 1. RPC 底座：原生 matrix.rpc，不是 MCP（2026-08-05 收敛）

cell 间 RPC 用原生 `matrix.rpc`——注册表 + 单一发现 channel + zenoh put/sub +
JSON-RPC 2.0 + 回调身份 + caller 侧超时，从 zenoh_qa 泛化。MCP 作内部 RPC 协议
零优势且双向外转损失类型。内部 cell↔cell 与外部边界分开。

### 2. MCP 位置：外部皮，不是脊柱

MCP 在 mesh 边界双向存在：node run as mcp server（朝外）/ node run mcp client
（朝内）。内部通讯走 matrix.rpc。`moss as mcp`（moss_as_mcp.py）是整运行时降级，
已做，独立于 node 级。

### 3. node run as mcp server 机制（2026-08-05 敲定）

`moss nodes run` 启动（非 `mcp run`）+ `matrix.run(mcp)` 糖（tools 先注册、run 不
重新注册）+ stateless streamable-http + `main(port=0)` 约定 + announce 走 nodes
channel EVENT（非 signal）、endpoint 双写 cell presence。详见
`design/mcp-node-server.md`。

### 4. node run mcp client 极简（部分敲定）

薄 channel（list/read/exec，不 command 化）+ `moss mcp connect` CLI。mcp_hub 瘦身。
授权 / 声明发现 / debug 细节待讨论。详见 `design/mcp-node-client.md`。
