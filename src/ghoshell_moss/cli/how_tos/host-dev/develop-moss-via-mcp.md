---
title: Develop MOSS via MCP
description: 在 MOSS 生态中开发 App、Channel、Ghost 时，通过 MCP 接入 coding agent 做实时调试。覆盖三种 transport、agent 配置、与 moss-run-ghost 的分工边界。
---

# How to Develop MOSS via MCP

## 背景

MCP (Model Context Protocol) 是开发 MOSS App 和 Channel 时的辅助调试手段。通过 `moss-as-mcp` 启动 MCP server，coding agent 可以直接调用 MOSS tools（CTML 执行、动态信息获取、指令查询）——输出 CTML → 观察执行结果 → 修正 → 再试。

**MCP 不是运行时方案，也不用于 Ghost 开发调试。** Ghost 在运行时通过 Mindflow → Shell → Matrix 的完整链路自主运行。MCP 只服务于 App/Channel 开发阶段的快速验证。

什么时候用 MCP：
- 开发新 App，需要实时验证 Channel 命令是否按预期工作
- 探索 MOSS 的 Channel 树和能力拓扑

什么时候不用：
- Ghost 开发与运行 — 用 `moss-run-ghost`
- 人类直接交互 — 用 `moss-repl` 或 `moss-cli`

## 第一步：启动 MCP server

```bash
# 默认 mode，SSE transport（端口 20773）
.venv/bin/moss-as-mcp

# 指定 mode
.venv/bin/moss-as-mcp --mode reachymini

# stdio transport（适用于 Claude Code 等直接 spawn 的 agent）
.venv/bin/moss-as-mcp --transport std

# 自定义端口
.venv/bin/moss-as-mcp --port 20774
```

`--help` 看完整参数：

```bash
.venv/bin/moss-as-mcp --help
```

三种 transport：

| Transport | 适用场景 |
|-----------|---------|
| `sse` (默认) | agent 通过网络连接，适合远程调试 |
| `std` | agent 直接 spawn 子进程，无需端口 |
| `streamable_http` | 需要 HTTP 流式响应的场景 |

## 第二步：在 coding agent 中注册

以 Claude Code 为例，在项目 `.mcp.json` 或全局配置中添加：

```json
{
  "mcpServers": {
    "moss": {
      "command": ".venv/bin/moss-as-mcp",
      "args": ["--mode", "reachymini", "--transport", "std"]
    }
  }
}
```

注册后，agent 获得四个 MOSS tools：

- `moss_instruction` — 获取当前 mode 的完整指令（Channel 树 + Ghost prompt + CTML 版本）
- `get_moss_dynamic_info` — 获取实时动态信息（Channel 状态、context messages）
- `execute_ctml` — 执行 CTML 指令（阻塞调用，等待返回）
- `interrupt_execution` — 中断正在执行的指令

## 第三步：调试循环

典型的开发调试流程：

1. agent 调用 `moss_instruction` 加载能力视图
2. agent 输出 CTML 指令，调用 `execute_ctml`
3. 观察返回结果或报错
4. 修正 App 代码或 CTML，重复

```bash
# 开发时常用的并行操作
# 终端 1: MCP server
.venv/bin/moss-as-mcp --mode default

# 终端 2 (可选): 前台跑 app 看日志
.venv/bin/moss apps test my-app
```

## 与 moss-run-ghost 的分工

```
App/Channel 开发阶段 (你现在在这里)
  │
  └─ moss-as-mcp ──→ coding agent 通过 MCP 调试 App/Channel
                      改代码 → 重启 App → CTML 验证 → 再试

Ghost 运行阶段
  │
  └─ moss-run-ghost ──→ Ghost 自主运行，不经过 MCP
```

## 深入路径

- MOSS tools 的实现细节：`moss codex get-source ghoshell_moss.cli.moss_as_mcp`
- MCP Hub（将外部 MCP server 的工具接入 MOSS CTML 调度）：`moss howtos read channels/use-mcp-hub`
- Ghost 运行时：`moss codex blueprint ghost`
