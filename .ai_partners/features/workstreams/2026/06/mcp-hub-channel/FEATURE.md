---
created: 2026-06-04
depends: [storage-typed-protocols]
description: MCP Hub — 将 MCP 协议降级为纯 transport，CTML 接管调度，模型以原生 CTML 思路操作外部工具。
milestone: null
priority: P0
status: in-progress
status_note: 实现完成。6 命令 (exec/exec_blocking/list_servers/add_server/remove_server/restart_server),
  双路径 config (ConfigStore 全局 + Storage YAML scoped)。29 单测全过。
title: MCP Hub Channel
updated: '2026-06-06'
---

# MCP Hub Channel

## Motivation

MCP 生态的 tool call 是扁平的无状态 RPC。CTML 的 `@nonblocking` + scope + timeout + observe
是语言级的并发调度原语。把 MCP 引入 MOSS 的本质是：用 CTML 替换 MCP client 的 orchestration 层，
MCP 退化为纯 transport——只负责 tool 发现和参数/结果传输。模型完全不知道 MCP 协议的存在，
用 CTML 原生思路操作外部工具。

## Key Decisions

### 1. Hub 模式，而非 N 个独立 channel

一个 stateful channel (`mcp`) 管理 N 个 MCP client session。添加/移除 server 是 Hub 内部动态操作，
不污染 channel 命名空间，不需要 channel tree refresh。
类比 `AppStoreChannel` 管理 N 个 app，但 MCP tool 不反射为 child channel——通过 `exec(server, tool, text__)` 调用。

### 2. 两个 exec 命令

- `exec` — `@nonblocking`，默认。发射后不 occupy channel，结果在下一关键帧以 Observe 形式观察。
- `exec_blocking` — 阻塞。仅当后续 CTML 命令依赖当前 tool 返回值时使用。使用频次应远低于 `exec`。

管理命令：`list_servers`、`add_server`、`remove_server`、`restart_server`。

### 3. JSON Schema 保真，不做反向还原

MCP tool 的 inputSchema 由 MCP server 作者定义，是权威契约。翻译成 Python 类型是不可逆的有损压缩
（`oneOf`/`anyOf`/`$ref`/递归类型无法映射）。`text__` 承载原始 JSON，CDATA 包裹，零翻译损耗。
Tool 目录和 JSON Schema 通过 context messages（moss_dynamic）呈现，不绑定 command interface。

### 4. Scoped storage 决定配置隔离级别

MCP server 连接配置存储在 `matrix.get_scoped_storage(*scopes)` 下。隔离级别由 Hub factory 的 `scopes` 入参决定，
不由 Hub 自身决定：
- `scopes=['ghost', 'mode']` → 每个 Ghost 在每个 mode 下独立配置
- `scopes=['mode']` → mode 内共享

### 5. 返回 Observe，不是 Message

所有 MCP tool 结果返回 `Observe`，进入 Mindflow 感知流。MCP 两种 content 协议（text + image）
映射到 MOSS 的 `Text` + `Base64Image`，一一对应。

### 6. 完全屏蔽 MCP 语义

模型不知道 MCP 协议存在。它只看到 `mcp` channel，有 `exec`/`exec_blocking` 等命令。
MCP server 的 tool 目录在 context messages 中展示（类似 skills 列表）。未来 MCP 被替代，
模型侧零变化——Hub 内部换 transport adapter。

### 7. 语义指引在 instruction 中

区分 moss_static（Hub 自身的命令 interface——exec、list_servers 等固定命令）和 moss_dynamic
（MCP server 连接状态、tool 目录、JSON Schema 摘要——随 add/remove/restart 变化）。

### 8. Config 双路径：ConfigStore（全局） + Storage YAML（scoped）

`MCPHubConfig` 继承 `ConfigType`（即 BaseModel）。加载/持久化走两条路径：

- **无 scopes** → 全局 ConfigStore（`get_conf`/`save_conf`）
- **有 scopes** → `matrix.get_scoped_storage(*scopes).read_yaml/write_yaml`

两条路径都用 YAML 格式，迁移零成本。`MCPServerConfig` 是纯 BaseModel 子结构，不独立存储。

## Implementation Notes

- 基于 `states_channel` 模式构建，类比 `AppStoreChannelState`
- State 持有 `dict[str, _MCPServerSession]`，每个封装 `mcp.ClientSession` + transport 生命周期
- 参考现有 `compatible/mcp_channel/` 的积累，基于新的 Hub 架构和 stateful channel 模式演进
- `mcp` 依赖为可选 extra，MCPHub 在 import 时做懒检查
- context messages 做摘要化：tool 名 + 一句话描述，连接状态用 `[+]`/`[-]`/`[!]` 标记
- `on_startup` 自动连接已配置的 server，`add_server` 成功后 `_save_config` 持久化
- 三种 transport：stdio（子进程）、sse、streamable_http