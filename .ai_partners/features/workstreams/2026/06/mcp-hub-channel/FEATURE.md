---
created: 2026-06-04
depends: [storage-typed-protocols]
description: MCP Hub — 将 MCP 协议降级为纯 transport，CTML 接管调度，模型以原生 CTML 思路操作外部工具。
milestone: null
priority: P0
status: completed
status_note: 2026-06-07 全链路验证通过 + 6 issues 修复 + moss-repl 验收 + 24 单测 + stubs 同步。人类工程师 review 合并。
title: MCP Hub Channel
updated: '2026-06-07'
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

## Verification (2026-06-07)

全链路验证通过 — default mode 下 MCP Hub Channel 集成 + 第三方 MCP server 工具调用。

**测试环境**：
- MOSS runtime via `moss-as-mcp` (sse, port 20773)
- 第三方测试 MCP server：`scripts/mcp_test_server.py` (stdio, 4 tools: echo/add/get_time/slow_echo)
- MCP Hub Channel 注册于 `.moss_ws/src/MOSS/modes/default/channels.py`

**验证结果**：

| 操作 | CTML | 结果 |
|------|------|------|
| 运行时发现 | runtime context messages | `mcp` channel 可见，context 中显示 "No MCP servers connected" |
| add_server | `<mcp:add_server name="test" />` | `[MCP:test] connected` — stdio 子进程连接成功 |
| list_servers | `<mcp:list_servers />` | 4 tool 完整目录（echo/add/get_time/slow_echo），状态 `[+]` connected |
| exec echo | `<mcp:exec server="test" tool="echo">{"text": "hello"}</mcp:exec>` | `ECHO: hello from MCP Hub` |
| exec add | `<mcp:exec server="test" tool="add">{"a": 3.14, "b": 2.86}</mcp:exec>` | `3.14 + 2.86 = 6.0` |
| exec get_time | `<mcp:exec server="test" tool="get_time">{}<mcp:exec>` | `2026-06-07T03:32:52` |
| exec slow_echo | `<mcp:exec server="test" tool="slow_echo" timeout="5.0">{"text": "async test", "delay": 2.0}</mcp:exec>` | `SLOW(2.0s): async test` (2s 后返回) |

## Issues (待修)

以下 4 个问题在 2026-06-07 验证过程中发现，实现已完成但存在摩擦点。

### Issue 1: 默认 scope 反了 — 应默认走 ConfigStore ✅ 已修

**位置**: `src/ghoshell_moss/channels/mcp_hub.py:436`

```python
# 当前（错误）
self._scopes = scopes or ['ghost', 'mode']

# 应为
self._scopes = scopes or []
```

**影响**: `MCPHubChannel(name='mcp')` 不走 ConfigStore，走 scoped storage，解析到 `.moss_ws/ghosts/None/mode-default/mcp_hub.yml`。how-to 文档写 "无 scopes → ConfigStore" 但代码行为相反。

**修复**: 改默认值为 `[]`（空列表 → falsy → 不进入 scoped 分支 → `_load_config` 走 ConfigStore）。

### Issue 2: MCPHubConfig 没有注册到 manifests/configs.py ✅ 已修

**位置**: `.moss_ws/src/MOSS/manifests/configs.py`

`MCPHubConfig` 是 `ConfigType` 子类，但从未在此文件中 import。ConfigStore 的 `get_conf(MCPHubConfig)` 依赖注册发现。

**修复**: 在 `manifests/configs.py` 中加一行 import：
```python
from ghoshell_moss.channels.mcp_hub import MCPHubConfig
```

### Issue 3: add_server 不支持运行时传参 ✅ 已修

**位置**: `src/ghoshell_moss/channels/mcp_hub.py` — `MCPHubState._bootstrap()` 闭包中的 `add_server`

**现状**: `add_server(name)` 只能从已有 YAML 加载。无法在 CTML 中直接指定连接参数。

**原因**: 纯实现问题，不是系统约束。`_MCPServerSession` 已封装完整连接逻辑，`add_server` 只需多接参数即可运行时构建 `MCPServerConfig`。

**前置依赖**: Issue 4（get_or_create）必须先修——否则首次添加时没有 config 对象可 append。

### Issue 4: _load_config 缺少 get_or_create 语义 ✅ 已修

**位置**: `src/ghoshell_moss/channels/mcp_hub.py:363-373`

```python
def _load_config(self) -> MCPHubConfig | None:
    if self._scopes:
        storage = self._matrix.get_scoped_storage(*self._scopes)
        return storage.read_yaml("mcp_hub", MCPHubConfig)  # YAML 不存在 → None
    else:
        try:
            return get_conf(ChannelCtx.container(), MCPHubConfig)
        except Exception:
            return None  # ConfigType 未注册 → None
```

**问题**: YAML 不存在或 ConfigType 未注册时返回 `None`，后续所有操作（add_server 等）因 config 为 None 而中断。

**修复**: 不存在时创建一个空的 `MCPHubConfig(servers={})` 并 `_save_config` 落盘。零风险——空字典，无副作用。

**修复后语义**: `_load_config` → 永不为 None，首次调用自动初始化空配置。`add_server` 无论从 config 读还是收运行时参数，都能在已有 config 对象上追加条目。

### Issue 5: 管理命令缺少 always_observe

**位置**: `src/ghoshell_moss/channels/mcp_hub.py:335-337`

**已修于 2026-06-07**: `add_server` / `remove_server` / `restart_server` 均已加 `always_observe=True`。

**原因**: 这些命令的结果（server 增删重启）改变了工具集状态，模型需要感知变化后决定下一步——与 tool use 触发下一轮思考的语义一致。

**暴露的问题**: `channels/CLAUDE.md` 的 observe 约定表将 "write/delete/start/stop" 归为 "只需知成败"（不需 always_observe），但 `add_server` 本质也是 "start" 类操作。矛盾在于：**是否触发下一轮思考，不取决于操作类型（读 vs 写），而取决于结果是否影响后续决策**。server 增删改变工具集 — 影响后续决策 — 应 observe。普通 write 只改一行数据 — 不影响后续决策 — 不需 observe。CLAUDE.md 的约定表需要从操作类型分类改为决策影响分类。

### 建议修复顺序

1. Issue 2 (注册 ConfigType) — 让 ConfigStore 发现 MCPHubConfig
2. Issue 1 (默认 scope) — 切换默认路径到 ConfigStore
3. Issue 4 (get_or_create) — 让空配置自动初始化
4. Issue 3 (运行时 add_server) — 支持 CTML 中传连接参数
5. Issue 6 (删掉手写 instruction) — 框架已自动反射
6. Issue 7 (补 JSON Schema) — context messages 缺失 tool 参数定义

### Issue 6: _DEFAULT_INSTRUCTION 手写且冗余 ✅ 已修

**位置**: `src/ghoshell_moss/channels/mcp_hub.py:201-212`

```python
_DEFAULT_INSTRUCTION = """\
MCP Hub — 通过 MCP 协议接入的外部工具集。
使用方式:
- 非阻塞调用 (推荐): <mcp:exec server="<name>" tool="<tool>" timeout="30.0">json args</mcp:exec>
- 阻塞调用: <mcp:exec_blocking ...>json args</mcp:exec_blocking>
管理命令:
- list_servers: 查看所有 server 连接状态
- restart_server(name): 重启指定 server"""
```

**问题**:
- 框架已通过 Code as Prompt 自动反射 6 个命令的完整 Python 签名到 moss_static（带 decorator、参数文档、返回类型）
- 手写 instruction 在 moss_static 中并未出现——框架没用它，用的是自动反射
- 手写版还不全：漏了 `add_server` 和 `remove_server`
- 双轨维护风险：命令签名变了，手写 string 不会跟着变

**修复**: 删除 `_DEFAULT_INSTRUCTION` 和 `get_instruction()` 覆写，让框架默认行为接管。

### Issue 7: context messages 缺少 tool inputSchema ✅ 已修

**位置**: `src/ghoshell_moss/channels/mcp_hub.py:401-418` — `get_context_messages`

**现状**: context 只渲染 tool 名 + 一行描述：
```
[+] **test**
  - `echo`: 回显输入文本。
  - `slow_echo`: 延迟回显，用于测试 @nonblocking exec 的异步行为。
```

模型看不到 tool 的参数定义。对于 `slow_echo`，不知道有 `delay` 参数；对于 `echo`，不知道要传 `{"text": "..."}`。

**原因**: `get_context_messages` 只取了 `tool.name` + `tool.description`，没取 `tool.inputSchema`（MCP Tool 对象的标准字段，包含完整 JSON Schema）。

**与设计文档的矛盾**: FEATURE.md Key Decision 3 写 "Tool 目录和 JSON Schema 通过 context messages（moss_dynamic）呈现"——但实现只做了目录，没做 Schema。

**修复**: 在 context messages 中为每个 tool 追加其 `inputSchema`。格式可以是简化的参数列表（比完整 JSON Schema 更省 token），或原始 JSON Schema 摘要。

## 交付质量反思 (2026-06-07)

Issue 6 和 7 暴露了一个模式：**模型写实现时已知全部细节（命令签名、MCP Tool 结构），默认 "模型自然会知道"——没有从运行时视角验证信息是否真正传递到了模型侧**。

具体表现：
- 手写了 `_DEFAULT_INSTRUCTION`，不知道框架已自动反射命令签名（Code as Prompt 机制被漏读）
- context messages 只渲染了 tool 名 + 描述，漏掉了 inputSchema
- 没有启动 moss-as-mcp 验证 moss_static/moss_dynamic 的实际内容

**教训**: channel 开发的验收标准之一应该是：启动运行时，检查 moss_static 和 moss_dynamic，确认模型视角的信息完整。设计文档里写的 "XYZ 通过 context messages 呈现" 必须在代码里真正实现。

## 待人类验证

以上 7 个 issue 修复后，需在 **moss-repl** 中做最终验收——moss-repl 是完整 TUI 运行时，能直观看到 channel 树、context messages、Observe 流。验证清单：

1. default mode 下 MCP Hub Channel 正常启动
2. moss_static 中 mcp channel 命令签名完整（无冗余手写 instruction）
3. moss_dynamic context 中 tool inputSchema 可见
4. `add_server` 支持运行时传参（无预配置 YAML）
5. `exec` 返回的 Observe 正确进入感知流
6. 管理命令触发 observe 后上下文正确刷新

## MCP App 集成验证 (2026-06-07)

Baidu Maps (`mcp-server-baidu-maps==0.2.4`) 作为第三方 MCP server，封装为 MOSS app 并端到端验证通过。

### App 设计

`mcp/baidu_map` app — 自包含 MCP wrapper，内部用 MCP Python SDK 直连 Baidu Maps MCP server (stdio transport)。不依赖 MCP Hub，独立进程管理连接生命周期。

- Channel: `apps.mcp_baidu_map`
- Commands: `call(tool, timeout, text__)` + `list_tools()`
- Context: 动态反射 10 个 tool 的 inputSchema
- AK: 通过 `BAIDU_MAPS_API_KEY` 环境变量传入

### 验证结果

| 操作 | 结果 |
|------|------|
| `moss apps list` | `mcp/baidu_map` 发现 |
| `<apps:start fullname="mcp/baidu_map" />` | app 启动，channel 注册 |
| `<apps.mcp_baidu_map:list_tools />` | 10 个工具完整暴露，带参数 schema |
| `<apps.mcp_baidu_map:call tool="map_geocode">` | 地理编码返回正确坐标 |
| `<apps.mcp_baidu_map:call tool="map_weather">` | 天气查询返回海淀区实时数据 |
| `moss-run-ghost echo` | Ghost 自然语言交互 → CTML 编排 → 结果呈现 |

### 意义

证明了 MOSS app 模式可以包装任意 MCP server，Ghost 通过 CTML 原生调用外部工具，完全屏蔽 MCP 协议。`moss-run-ghost` 端到端链路：自然语言 → Ghost 理解 → CTML 生成 → app channel 路由 → MCP tool call → 结果感知 → 自然语言回复。
