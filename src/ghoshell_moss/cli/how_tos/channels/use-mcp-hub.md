---
title: Use MCP Hub
description: 通过 MCP Hub Channel 将外部 MCP server 的工具接入 MOSS CTML 调度体系。模型以 exec 命令调用外部工具，非阻塞发射 + Observe 观察，完全屏蔽 MCP 协议语义。面向 channel 开发者和 Ghost 开发者。
---

# Use MCP Hub

## 背景

MCP 生态提供了大量现成的工具 server（filesystem、database、web search 等），但 MCP 的 tool call
是扁平的无状态 RPC——没有时间语义，没有并发拓扑，没有 observe 流程。

MCP Hub Channel 把 MCP 降级为纯 transport：只负责发现工具有哪些、把参数传过去、把结果拿回来。
调度、并发、超时、取消、观察——全部由 CTML 接管。

```bash
moss codex channeltypes mcp_hub
```

## 心智模型

模型看到的只有 `mcp` channel，有 6 个命令（开启 `allow_model_config` 后 8 个）。模型不知道 MCP 协议存在。

**模型视角**（`allow_model_config=False`，默认）：

```
channel mcp:
  exec(server, tool, timeout=30.0, text__) -> Observe    @nonblocking
  exec_blocking(server, tool, timeout=30.0, text__) -> Observe
  list() -> str                                            always_observe
  connect(name) -> str                                     always_observe
  disconnect(name) -> str                                  always_observe
  reconnect(name) -> str                                   always_observe
```

**模型视角**（`allow_model_config=True`，额外暴露）：

```
  register(text__) -> str                                  always_observe
  unregister(name) -> str                                  always_observe
```

context messages 里提供当前连接的 server 和 tool 目录，模型从中选择。

## 配置

### 定义 MCP server 列表

所有 server 配置存储在 workspace 根目录的 `configs/mcp_hub.yml` 中。`MCPHubConfig` 是一个 `ConfigType`，
包含 `servers: dict[str, MCPServerConfig]`。

```yaml
# .moss_ws/configs/mcp_hub.yml
servers:
  filesystem:
    name: filesystem
    transport: stdio
    command: npx
    args: ["-y", "@anthropic/mcp-server-filesystem", "/tmp"]
    description: "Local filesystem access"
  websearch:
    name: websearch
    transport: sse
    url: "http://localhost:8080/sse"
    description: "Web search API"
```

env 值使用 `$VAR_NAME` 占位符，ConfigStore 在读取时自动从 os.environ 解析真值。
写文件时原样保存 `$VAR_NAME`，不泄露密钥。

**Scoped 配置**：传入 `scopes=['ghost', 'mode']` 时，config store 组装在 scoped storage 上。
首次创建时自动合并 workspace 预设，之后 scoped 配置可独立覆盖。

### 注册 channel

```python
from ghoshell_moss.channels.mcp_hub import MCPHubChannel

# 全局配置（无 scopes，读 workspace configs/）
main.import_channels(MCPHubChannel(name='mcp'))

# Scoped 配置（Ghost + Mode 隔离）
main.import_channels(MCPHubChannel(name='mcp', scopes=['ghost', 'mode']))

# 允许模型动态注册 server
main.import_channels(MCPHubChannel(name='mcp', allow_model_config=True))
```

## CTML 使用

### 非阻塞调用（推荐）

```ctml
<mcp:exec server="filesystem" tool="read_file">{"path": "/tmp/notes.txt"}</mcp:exec>
<mcp:exec server="websearch" tool="search" timeout="10.0">{"query": "CTML protocol"}</mcp:exec>
```

两个调用同时发射，不互相 occupy。结果在下一关键帧以 Observe 形式观察。

### 阻塞调用（少数场景）

```ctml
<mcp:exec_blocking server="filesystem" tool="read_file">{"path": "/tmp/config.json"}</mcp:exec_blocking>
```

### 管理命令

```ctml
<mcp:list />
<mcp:connect name="database" />
<mcp:reconnect name="filesystem" />
<mcp:disconnect name="websearch" />
```

### 动态注册 server（需 allow_model_config=True）

```ctml
<mcp:register>{"name": "myserver", "transport": "stdio", "command": "python", "args": ["-m", "mymod"], "env": {"KEY": "$SECRET"}}</mcp:register>
<mcp:unregister name="myserver" />
```

`register` 的 text__ 传完整 `MCPServerConfig` JSON，与 `exec` 传 tool arguments 的模式一致。

## 结果格式

MCP tool 的返回进入 Observe 流。text → `Text` content，image → `Base64Image` content。
模型在下一关键帧看到的是这两类原生 MOSS 消息。

## 架构要点

- **StatefulChannel**：`MCPHubState` 持有 `dict[str, MCPServerSession]`
- **MCPServerSession**：封装 `mcp.ClientSession` + transport 生命周期（连接/断开/重连）
- **Transport**：stdio（子进程）、sse、streamable_http 三种
- **ConfigStore 注入**：`MCPHubState` 依赖 `ConfigStore` 接口，不依赖 `Matrix`。factory 层决定 store 来源（scoped 或 workspace）
- **$VAR 解析**：`ConfigType.resolve()` 在读时递归解析 `$VAR` 占位符，写时原样保存
- **Context messages**：moss_dynamic 动态生成 server 连接状态和 tool 目录
