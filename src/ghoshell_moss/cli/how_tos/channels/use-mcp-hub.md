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

模型看到的只有 `mcp` channel，有 6 个命令。模型不知道 MCP 协议存在。

**模型视角**：

```
channel mcp:
  exec(server, tool, timeout=30.0, text__) -> Observe    @nonblocking
  exec_blocking(server, tool, timeout=30.0, text__) -> Observe
  list_servers() -> str                                    always_observe
  add_server(name) -> str
  remove_server(name) -> str
  restart_server(name) -> str
```

context messages 里提供当前连接的 server 和 tool 目录，模型从中选择。

## 配置

### 定义 MCP server 列表

`MCPHubConfig` 是一个 `ConfigType`，包含 `servers: dict[str, MCPServerConfig]`。

**全局配置**（无 scopes，走 ConfigStore）：

```yaml
# workspace/configs/mcp_hub.yml
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

**Scoped 配置**（有 scopes，走 scoped storage YAML）：

同一个文件放在 scoped storage 下，路径由 `matrix.get_scoped_storage(*scopes)` 决定。
格式相同，隔离级别不同。

### 注册 channel

在 manifests channels 中注册一个无副作用的 factory 闭包：

```python
# 全局配置（无 scopes）
from ghoshell_moss.channels.mcp_hub import MCPHubChannel
main.import_channels(MCPHubChannel(name='mcp'))

# Scoped 配置（Ghost + Mode 隔离）
main.import_channels(MCPHubChannel(name='mcp', scopes=['ghost', 'mode']))
```

## CTML 使用

### 非阻塞调用（推荐）

```ctml
<mcp:exec server="filesystem" tool="read_file">{"path": "/tmp/notes.txt"}</mcp:exec>
<mcp:exec server="websearch" tool="search" timeout="10.0">{"query": "CTML protocol"}</mcp:exec>
```

两个调用同时发射，不互相 occupy。结果在下一关键帧以 Observe 形式观察。

### 阻塞调用（少数场景）

当后续命令依赖当前 tool 的返回值时：

```ctml
<mcp:exec_blocking server="filesystem" tool="read_file">{"path": "/tmp/config.json"}</mcp:exec_blocking>
<_>
Hello, the config says...
</_>
```

`exec_blocking` 会 occupy channel，等返回后再执行后续命令。

### 管理命令

```ctml
<mcp:list_servers />
<mcp:add_server name="database" />
<mcp:restart_server name="filesystem" />
<mcp:remove_server name="websearch" />
```

## 结果格式

MCP tool 的返回进入 Observe 流。text → `Text` content，image → `Base64Image` content。
模型在下一关键帧看到的是这两类原生 MOSS 消息。

## 架构要点

- **StatefulChannel**：类比 `AppStoreChannel`，`MCPHubState` 持有 `dict[str, _MCPServerSession]`
- **_MCPServerSession**：封装 `mcp.ClientSession` + transport 生命周期（连接/断开/重连）
- **Transport**：stdio（子进程）、sse、streamable_http 三种
- **Context messages**：moss_dynamic 动态生成 server 连接状态和 tool 目录
- **Config 双路径**：无 scopes → ConfigStore；有 scopes → Storage YAML
