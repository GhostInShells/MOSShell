---
title: DSH 0.2.0 跟踪
status: draft
priority: P1
created: 2026-09-30
updated: 2026-10-01
depends: [dsh-fusion]
milestone: '0.2.0'
description: >-
  跟踪 dsh 0.2.0，让 deepseek ghost 跑通 0.2.0（验收点：能对话，已达成）。
---

# DSH 0.2.0 跟踪

> Use `moss features set-status dsh-020-track <status> -m "note"` to update state.

## Motivation

dsh 0.2.0-rc.2 发布（2026-09-29）。0.1.5 已在 Ubuntu 上不兼容，需要跟踪上游版本演进。
本 workstream 是 dsh-fusion（已完成）的**版本跟踪续篇**：把 deepseek ghost（可弃测试面）跑通 0.2.0。
分支 `track/dsh-020` 承载所有改动，不进 main / dev，从 dev 周期性 rebase，追到 0.2.0 出第一个非 rc 版。

**验收（2026-10-01）：deepseek ghost 在 0.2.0 上 `moss-ghost send` 能对话，已达成。**

## 版本与安装

- vendor 源码: `.ai_partners/features/workstreams/2026/08/dsh-fusion/research/source/deepseek-harness`
  （git clone, 已 checkout `dsh-v0.2.0-rc.2`）。
- scoped 安装: `~/.local/dsh-0.2.0`（`npm i --prefix ~/.local/dsh-0.2.0 @deepseek-ai/dsh@0.2.0-rc.2`）。
- per-ghost 钉版本: `DSH_BINARY` 环境变量 / `deepseek/.env`（`resolve_dsh_binary`，已独立 commit）。
- 全局 dsh 仍是 0.1.5-rc.1，不被改动。

## 改造清单（0.1.5 → 0.2.0）——实际结果

| # | 改动 | 实际结果 |
|---|---|---|
| 1 | preset 程序化 | ✅ `agentPresets.register(PresetDefinition)` + `workflow-worker-thread`→`workflow-ptc` + `tool-ralph` 置 disabled |
| 2 | system 头 | ❌ **非问题**：dsh 0.2.0 对 fresh session 自己产出正确的 `system/message` 头，不用我们改 |
| 3 | moment→developer | ↩️ **回退**：user/message 已支持图、不碰头规则；developer 需动 user-only 的 inject/steer 通道，纯语义洁癖 |
| 4 | source kind | ✅ 11 处 `{kind:'plugin'}` → 生产方 kind（`{kind:name}` / `{kind:`${name}:moment`}`…） |
| 5 | 迁移 | 认知：v3 拒迁 v3→v4 非破坏、源留存；deepseek 可弃 |

**净改动 = #1 + #4**，两个都是小改。传输面无改动（mux / auth / $events / plugin 路由向后兼容）。

## Key Decisions

- **preset 程序化是正道**：0.2.0 删除了 `.agent-presets/` 目录发现，改为 `AgentPresetRegistry.register(PresetDefinition)`。
  `PresetDefinition = {id, name?, description?, order?, plugins[]}`，`plugins` 即 cordis entry list（= 原 `agent.cordis.yml`）。
  `mount(ctx, id)` 仍在，但必须先 `register()` 填 `definitions`。composition 回归 plugin 自己。
  **随 0.2.0 一并改名**：`workflow-worker-thread`（`@deepseek-ai/dsh-workflow-worker-thread`）→ `workflow-ptc`（`@deepseek-ai/dsh-workflow-ptc`），
  且 `tool-ralph` 在 0.2.0 变为 `disabled: true`（对照 `packages/bundle/web-app/presets/standard.patch.yml`）。
- **system/message 头规则（对 fresh session 非问题）**：0.2.0 里 system/message 若存在必须是 surface 第 0 个节点（protectedHead），
  否则 v3→v4 迁移拒绝。但 fresh session（无 v3 产物可迁）由 dsh 的 systemPrompt 服务正确产出头，我们**零改动**。
  只有「迁移存量 v3 session」才撞这条——deepseek 可弃，不处理。
- **source.kind 是运行时硬约束（非只查非空串）**：v4 格式 `assertV4MessageSources` 显式拒绝 `kind === 'plugin'`
  （`session-format-v3-to-v4/src/message-sources.ts:8`），报 `format v4 message requires a producer-owned source kind`。
  修复 = 用生产方 kind（plugin 名 `name` / `${name}:moment` / `${name}:epoch` / `${name}:notice`）。
- **moment 保持 user/message（不回退到 developer）**：`Agent.inject` / `Agent.steer` 只收 `UserMessage`
  （`core/agent/src/runtime-types.ts:231/241`），写死 user/message 事件；developer/message 需绕过它们直写
  `session.append('developer/message')`，丢掉 inject/steer 驱动/背压语义。而 user/message 本已支持图
  （`durableMomentContent` → `admitEncodedImages` 落成 image 块）、且 moment 是头之后注入不碰头规则。
  决策 + 依据已留在 `buildMomentFrame` 的注释里。
- **developer/message 支持图片**（认知，未用）：`ContentBlockMap` 含 `image`，`DeveloperMessage.content = ContentBlock[]` 共用。
- **迁移非破坏**：v3→v4 迁移拒绝时 source v3 artifact 不变（实测 mtime 未动、无 v4 生成）。

## Exploration paths

- 0.1.6 断点（`agent/session-start` 并入 `agent/created`）之前判定不追；0.2.0 含此改动，但不影响我们（不依赖该事件）。
- headless 模式本身没问题：moss ghost 在 0.1.5 + headless 正常 boot + 跑 turn 1。
- deepseek ghost on 0.2.0 的 boot 静默中断，根因 = ego/create 撞 `Unknown agent preset`（preset 未 register）。
- 逐层断点：preset → workflow-ptc 改名 → source kind → 全通。

## Methods

- dsh 单独起法（隔离 ghost）：`DSH_HOME=<ghost>/.dsh ~/.local/dsh-0.2.0/.../dsh --profile web --port <port> --no-open`。
- curl plugin 路由（未鉴权）：`curl -X POST http://127.0.0.1:<port>/moss-api/ghost/dolores/ego/create -H 'Content-Type: application/json' -d '{...}'`。
- 改 plugin 后重同步：`cp <repo plugin> .moss/ghosts/deepseek/.dsh/profiles/web/moss-dolores-ghost-plugin.ts`。

## Implementation Notes

- launcher 的 `resolve_dsh_binary`（显式值 → `DSH_BINARY` env → PATH 上的 `dsh`）是版本无关基建，已独立 commit。
- plugin 单文件 `moss-dolores-ghost-plugin.ts` 由 dsh 直接按 .ts 加载（不编译），imports 对 dsh 安装的 node_modules 解析。
- 待收尾（不阻塞验收）：`dsh_preset/` 目录 + `_sync_dsh_preset` 已死（0.2.0 无目录发现）；source kind 的 `MessageSourceMap` 类型增强（类型正确性）。
