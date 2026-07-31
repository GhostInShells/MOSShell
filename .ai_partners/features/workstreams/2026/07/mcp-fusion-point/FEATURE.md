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

> 状态：draft。本 workstream 不急于落决议——当前位置判断与 RPC 底座的裁决仍
> 开放，讨论轨迹见 `discuss/`。用 `moss features set-status mcp-fusion-point <status> -m "note"` 更新状态。

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

- Key design documents: `design/`（暂无）
- Key discussion records: `discuss/2026-07-31_mcp_position_and_fusion.md`
- 前置讨论：`.discuss/2026-07-30_mcp_duplex_convergence_and_memento_branch.md`

## Key Decisions

<!-- 本 workstream 刻意不在此处堆积决议。三身份（server as cell / client as cell /
client as channel）与 RPC 底座的判断轨迹记录在 discuss/，待收敛后回填。 -->
