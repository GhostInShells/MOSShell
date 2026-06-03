---
title: Feishu Channel Integration
status: draft
priority: P2
created: 2026-06-04
updated: 2026-06-04
depends: []
milestone:
description: >-
  飞书 SDK (lark-oapi) 集成：App 作为感知源独立进程运行，推轻量 Signal 到 Mindflow Nucleus，
  Ghost 通过 Channel proxy pull 消息详情。推拉分离，buffer 归 App 自身管理。
---

# Feishu Channel Integration

## Motivation

飞书 2026 年开放了面向 Agent 的 SDK 体系。Python SDK `lark-oapi` v1.6.0 的 `FeishuChannel` 模块
提供 WebSocket 长连接、消息归一化、流式输出、去重等能力。目标是让 MOSS Ghost 能在飞书中感知消息并
主动回复——作为一个独立 App 运行，不侵入 MOSS 核心。

## Design Index

- App blueprint: `ghoshell_moss.core.blueprint.app`
- Matrix blueprint: `ghoshell_moss.core.blueprint.matrix`
- AppStore channel: `ghoshell_moss.channels.app_store_channel`
- Mindflow/Nucleus: `ghoshell_moss.core.blueprint.mindflow`
- lark-oapi SDK: https://pypi.org/project/lark-oapi/1.6.0/
- Feishu Channel SDK 文档: https://open.feishu.cn/document/mcp_open_tools/integrating-agents-with-feishu/integrate-feishu-channel

## Key Decisions

### 1. Push-pull 分离

Signal 是敲门声，不是快递。App 发 Signal 只携带轻量元信息（badge、summary、pull_channel 路径），
不带消息正文。Ghost 被 Impulse 唤醒后，自行决定何时调用 App 的 channel pull 数据。

- **Push**: Signal → Session bus → Nucleus → Impulse → Ghost 被唤醒
- **Pull**: Ghost → CTML 调用 `im_feishu:pull_messages` → Channel proxy → App 返回 buffer 中的消息

### 2. App 而非 Nucleus

Feishu 集成是一个 MOSS App (Cell type=app, group=`im`)，不是 Nucleus。
Nucleus 是 Mindflow 内部的信号处理单元（如 InputSignalNucleus），App 只负责产生 Signal 和暴露 Channel。
每个 App 管理自己的 buffer（消息队列、长连接状态等），不泄漏到 Nucleus 层。

### 3. 分组命名: `im`

不使用 `perception`——这是架构角色，不是业务域。`im` 描述能力域，飞书/Slack/微信等同类感知源共享此分组。

### 4. 分层：IM Channel + Ops Channel

- **Layer 1**: `im/feishu` — 对话通道，封装 `lark-oapi` FeishuChannel。接收消息（感知），发送/流式回复（动作）。Python 原生，无外部进程依赖。
- **Layer 2**: `im/feishu_ops` — 操作通道（文档、日历、任务等）。飞书 CLI 是 Node.js 的，通过 MCP server 桥接，或复用社区 Python MCP server。按需，非首期范围。

### 5. 不新增抽象，不修改 Blueprint

所有能力已在现有抽象中：
- App 作为 Matrix Cell → `matrix.provide_channel()` + `session.add_signal()`
- Ghost 通过 AppStoreChannel → `start/stop` 管理生命周期 → virtual children 自动建 channel_proxy
- Signal 通过 Session bus 进入 Mindflow，由现有 InputSignalNucleus 处理

所需新增的仅有 App 实现本身和对应的 howto 文档。

### 6. App 自带 Channel，支持反身性管理

App 暴露的 Channel 使 Ghost 可以管理感知源自身状态：
`pull_messages`、`mark_read`、`get_status` 等。buffer 策略、过滤规则由 App 内部实现，
Ghost 通过 Channel 命令管理，不烦扰 Nucleus 或 Mindflow。

## Implementation Notes

- `lark-oapi` 的 `FeishuChannel` 是 async 的，App 进程内需要一个 event loop
- App 启动后调用 `matrix.provide_channel()` 注册管理接口，Ghost 侧通过 `AppStoreChannel.get_virtual_children()` 自动发现
- App 的 Signal 路由到哪个 Nucleus 由 `signal.name` 决定，默认 `"input"` 走 InputSignalNucleus
- `AppWatcher` 支持 `respawn`，长连接断开时可由 Matrix 自动重启 App
