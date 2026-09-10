---
title: MCP Fusion — MCP as first-class citizen
version: 1
status: active
priority: P1
created: 2026-09-11
updated: 2026-09-11
scope: subsystem
depends:
  - mcp-fusion-point
description: >-
  实机验证 MCP hub 把外部 MCP server 作为第一公民接入 MOSS 的完整链路——
  stdio/streamable_http 两路 transport、虚拟子 channel 的 schema notice、
  call/acall、config 生命周期（add/open/close/remove）、CLI 跨进程写 config，
  以及 CTML shell 全链路。
---

# MCP Fusion Regression

> mcp-fusion 的本意是把 MCP 作为第一公民。本集验证的是 hub 这一侧（node as
> mcp client）：外部 MCP server 的工具集经 channel 投影成 MOSS 的 CTML 能力面，
> 而不是被降级成扁平的一堆命令。核心判据：**子 channel 的 notice 是 tool 的
> JSON schema 接口描述（code-as-prompt），不是 Command 对象**——这是融合点是否
> 成立的分水岭。

## Exploration Index

```
moss codex get-interface ghoshell_moss.channels.mcp_channel:MCPHub
moss codex get-interface ghoshell_moss.channels.mcp_channel:MCPServerSession
moss codex get-source ghoshell_moss.channels.mcp_channel
moss --ai all-commands --group mcp --depth 3
git log --oneline -- src/ghoshell_moss/channels/mcp_channel.py src/ghoshell_moss/cli/mcp_cli.py
```

## Methodology

两层验证：

1. **纯逻辑（自动化）** — `pytest tests/ghoshell_moss/channels/test_mcp_channel.py`
   覆盖 MCPHub 公开方法、config 增删、open/close 生命周期、子 channel 契约、
   虚拟子 channel 对齐，以及仓内 demo 的 stdio 端到端。
2. **实机 dogfood（human-in-the-loop）** — 用真实生态 server
   （`modelcontextprotocol/servers` 的 `everything`，专为 MCP client 一致性测试设计）
   验证真实 transport（stdio / streamable_http）与 schema 渲染；再用
   `moss-shell mcp` + coding agent 驱动 CTML 走完整链路。

聚焦三件"坏了才重要"的事：

- **schema notice 保真** — 子 channel 的 notice 必须渲染 tool 的 input schema
  （含 type/required/description），这是 code-as-prompt 的承诺，随代码漂移即谎言。
- **config 生命周期语义** — `add` 只写配置不连接、`open` 主动连接、`close` 只关
  子 channel 保留配置、`remove` 才删配置。四者不可混淆。
- **跨进程 config 一致性** — CLI（`moss mcp add/remove`）在另一个进程写 config，
  运行中的 hub 必须经 `invalidate` 读到，否则 CLI 改动对模型不可见。

Future scope（本轮不做，锚点见 mcp-fusion-point FEATURE.md §未做）：OAuth
（GitHub remote MCP `https://api.githubcopilot.com/mcp/`）、tasks/resources
原生接入。它们实现后再加 case 并升 version。

## Prerequisites

- `mcp` extra 已装：`uv sync --all-extras`（`import mcp` 失败会 raise 清晰提示）。
- `everything` server 就绪，二选一：
  - clone：`git clone --depth 1 https://github.com/modelcontextprotocol/servers.git .moss/mcp/servers`
    然后 `cd src/everything && npm install && npm run build`（`.moss/mcp` 已 gitignore）。
  - npx（无需 build）：`npx -y @modelcontextprotocol/server-everything`。
- `.moss/modes/default/src/HOST/channels.py` 注册了 `mcp` channel
  （`main.import_channels(mcp_hub_channel_factory(name="mcp", allow_config_edit=True))`）。
- TC-012 需 `moss-shell mcp`（默认 20773）+ 一个连上的 coding agent。

## Test Cases

| Case ID | Priority | Description | Test Steps | Expected Result |
|---------|----------|-------------|------------|----------------|
| TC-001  | P0 | stdio 连真实 `everything` server | 1. `node dist/index.js stdio`（或 npx）作为 stdio server<br>2. probe 用 `MCPHub` + `MCPServerConfig(transport='stdio', command='node', args=[dist,'stdio'])` open | `connected`，13 tools |
| TC-002  | P0 | streamable_http 连真实 `everything` server | 1. `node dist/index.js streamableHttp`（监听 3001）<br>2. config `transport='streamable_http', url='http://127.0.0.1:3001/mcp'`<br>3. open | `connected`，13 tools |
| TC-003  | P0 | 子 channel notice = tool schema（非 Command 对象） | open 后读 `mcp.<name>` facade 的 notice | notice 含 `get-sum(`a` (number, required): First number, ...)`；interface 只有 `call`/`acall` 两个命令 |
| TC-004  | P0 | 阻塞 `call` + JSON 参数透传 | CTML `<mcp.<name>:call tool="get-sum">{"a":5,"b":7}</mcp.<name>:call>` | `The sum of 5 and 7 is 12.` |
| TC-005  | P1 | 非阻塞 `acall` | 读 `mcp.<name>` facade，`acall` 带 `@nonblocking`；CTML 发 `acall` 不阻塞同 channel 后续命令 | facade 显示 `@nonblocking`；结果下一帧 Observe 到达 |
| TC-006  | P0 | `add` 只写 config 不连接 | CTML `<mcp:add>{"name":"...","transport":"streamable_http","url":"..."}</mcp:add>` 后 `mcp:list` | `config added (not open)`；`list` 显示 `Configured (not open)` |
| TC-007  | P0 | `open`→`close` 保留 config | open → close → list | close 返回 `closed`；`list` 仍显示该 server（not open） |
| TC-008  | P0 | `remove` 删 config | remove → list | `removed from config`；`list` 空 |
| TC-009  | P1 | 非法 server 名（连字符）被拒 | `add`/`open` 名 `everything-http` | 清晰报错 + 正则 `[a-zA-Z_][a-zA-Z0-9_]*`，不崩 |
| TC-010  | P1 | 错误 tool 名 → MCP error 兜成 Observe | call 不存在的 tool | `MCP error -32602: Tool xxx not found` 兜成 Observe，不抛异常穿透 |
| TC-011  | P0 | CLI `moss mcp list/add/remove` 跨进程写 config | 1. `moss mcp add --json '{"name":"demo",...}'`<br>2. `moss mcp list`<br>3. `moss mcp remove demo` | 往返一致；运行中 hub 经 `invalidate` 读到 CLI 改动 |
| TC-012  | P0 | CTML shell 全链路（human-in-the-loop） | `moss-shell mcp` + coding agent 发 CTML：list → add → open → 子 channel 长出 → call → close → list → remove | 8 步全绿（见 baseline 记录） |

## Execution Notes

- **server 名必须符合 channel 命名**（`[a-zA-Z_][a-zA-Z0-9_]*`），连字符会被拒。用 `everything_http` 而非 `everything-http`。
- **虚拟子 channel 移除有一帧延迟**：`close` 后子 channel 的 notice 先变 `disconnected`，下一个 refresh 周期才 `<removed/>`。这是动态 channel 的文档化行为（one-cycle delay），不是 bug。
- **跨进程 config 靠 `_load_config` 里的 `invalidate`**：CLI 在另一进程写 `.moss/configs/mcp_hub.{mode}.yml`，hub 若不 invalidate 会读缓存旧值。
- `everything` 的 streamableHttp 默认监听 3001（`PORT` 可覆盖），路由是 `/mcp`。
- TC-001/002 可用 probe 脚本（不依赖 shell）快速验证；TC-012 才需要 shell。
