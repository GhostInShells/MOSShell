# 子任务 2 — Node Run MCP Client（朝内）

> 索引于 FEATURE.md。机制与细节于 2026-08-12 讨论收敛，已敲定。

## 定位

对朝内的 MCP 接入做治理——ghost 连接外部 MCP server、使用其工具。极简：client
不用独立 node，in-process channel 即可，CLI 化。`moss mcp connect` 只是配置糖。

## 已敲定

### 1. 薄 channel 表面 + interface 化

- 三个命令：`list`（分组）、`read <group>`（该分组 tools 签名 + 入参）、
  `exec <group> <tool> <json>`。
- **不 command 化**——工具保持 data，不物化成 MOSS Command。无论挂多少工具，
  表面恒 3 命令。
- 理由：MCP 工具是无状态扁平调用，command 化不买超集维度，只把面切碎成 N×M。

**Interface 化方向**（2026-08-12 追加）：mcp_hub channel 和 protocol translation
layer 共享同一套 `MCPToolSurface` Protocol——`list_tools()` + `call_tool()`。
ghost bridge 的 send/pull/wait_reply 也用同一套 interface 表达，只是多了
task 生命周期。

### 2. 渐进式披露（非全量展示）

外部 MCP client（Claude Code 等）是全量展示——所有 tools 一次性进 system prompt。
MOSS 做渐进式披露：

- **已连接的 server + tool 列表**进 context（当前 mcp_hub 已在做）。
- **未连接的 server 不进 context**——ghost 看不到就不会想用。
- ghost 按需 `mcp:list` 发现可用的 server，`mcp:connect <name>` 接入后 tool
  自动出现在下一帧 context。
- "有哪些 server 可用"本身是 `list` 命令的输出，不常驻 context。

### 3. CLI：`moss mcp connect` = topic 广播

`moss mcp connect` 不直接操作 session。通过 mesh topic 事件驱动：

```
moss mcp connect <url> --name github
→ CLI publish topic 事件: {action: "connect", url: "...", name: "github"}
→ mcp_hub channel 收到事件
→ 创建 MCPServerSession → connect → 写入 ConfigStore（持久化）
→ ghost 下一帧 context 自动包含新 tool
```

好处：
- CLI 不直接操作 session——只发事件，不管谁处理
- hub 可以在不同进程——事件是 mesh 级别的
- 断开同理：`moss mcp disconnect <name>` → topic 事件 → hub disconnect + 移除
- 可观测——连接/断开全在 mesh 事件流里

子命令方向：
- `moss mcp connect <url>` — 广播连接事件，hub 自动接入
- `moss mcp disconnect <name>` — 广播断开事件
- `moss mcp list` — 列出已配置 + 已连接的 server 状态
- `moss mcp refresh <name>` — 手动刷新 tools 列表

### 4. Ghost 主动接入 + 审批闸口

ghost 可以主动 `mcp:connect <name>`。不依赖 `allow_model_config` 布尔开关——
那是假自主。真正的审批流：

```
ghost 判断需要外部 MCP tool
→ mcp:connect <name>
→ SafeMode 闸口拦截: "Ghost wants to connect to MCP server 'github'. Allow?"
→ 人类 approve
→ hub 建立连接，tool 进下一帧 context
```

这是 SafeMode 的一个新审批类型，不是独立的权限系统。

### 5. CTML 编排 + MCP 叶子执行（不混用 pydantic_ai 直调）

MCP 有两个去向：
- **CTML 调用**（主路径）：MCP tool → channel 命令 → CTML 执行计划中的叶子节点。
  时序/并行/条件/超时全在 CTML 层。
- **pydantic_ai 直调**（不做）：失去时序逻辑后 MCP tool 退化为函数调用，
  pydantic_ai 原生已支持，不需要绕 MCP。

CTML + tool 混用经验风险低——tool 在 CTML 执行计划里是叶子节点，不参与编排。

**核心分层**：MCP 负责"能调什么"（能力发现 + 标准化调用协议），CTML 负责"怎么调、
何时调、和什么并行"（编排）。MCP 不需要也不应该有 sequence/parallel 语义——
这是两个协议清晰的分界线。

### 6. MCP 没有 sequence / parallel 协议（融合边界）

这是整个融合设计的核心张力，也是 MCP 不能替代 CTML 的根本原因：

- MCP：`call_tool(name, args) → result`，一次一个，调用之间无关系
- CTML：带时序的并行命令树，同 channel 顺序、跨 channel 并行、时间约束

融合边界：
```
CTML (编排层)
  ├── mcp:exec(server, tool, json)   ← MCP tool 是叶子节点
  ├── mcp:exec(server, tool2, json)  ← 可以顺序/并行调用
  ├── shell:exec(...)                ← 可以和本地能力混合
  └── ghost_bridge:reply(...)        ← 可以和 ghost 内部能力混合
```

如果未来 MCP 往时间感知方向演进（SEP-2322 MRTR 已有苗头），边界会移动。当下：
CTML 编排 + MCP 叶子执行，边界清晰。

### 7. mcp_hub 瘦身

- **砍**：connect/disconnect/reconnect（生命周期归 topic 事件）、register/unregister +
  ConfigStore + scope 解析 + workspace preset 合并（配置归 CLI/声明）、exec_blocking
  （阻塞语义归 channel 级）。
- **留**：MCPServerSession transport 选择 + initialize + list_tools + call_tool、
  result→Observe 转换。
- **配置复用**：`MCPServerConfig` + `MCPHubConfig` + `ConfigStore` + scoped storage +
  `$VAR` 环境变量解析——这套配置体系不变，只是写入方式从 hub 的 register 命令变成
  CLI 的 `moss mcp connect`。

### 8. tool 变更感知

MCP stateless 版无 session，无 server 端推送。三条路：

| 方式 | 机制 | 代价 |
|------|------|------|
| 惰性校验 | exec 时 tool 不存在 → 报错 → 提示 refresh | 第一次调用失败 |
| 定时轮询 | hub 每隔 N 分钟 list_tools 对比 | 浪费请求 |
| 事件驱动 | server 侧 publish 变更 → hub re-list | server 需配合 |

最务实路线：**惰性校验 + 手动 refresh**。调用失败返回 "tool not found, try
mcp:refresh"。`moss mcp refresh <name>` 手动触发。定时轮询作为可选
（`--watch-interval`）。

### 9. Resources / Prompts 决策

- **Resources**：MCP resource（`uri → data`）与 MOSS resources（matrix 网络内
  可寻址资源，原"上下文变量"，RESTful 风格）**是不同的概念，只是撞了名字**。
  MCP resources 暂不用——MOSS 端 manifests resources 向外部暴露用 server 侧
  resource bridge，client 侧不主动拉外部 resources（除非有明确用例）。

- **Prompts**：不用。MOSS 是 code-as-prompt + CTML，不是 template。ghost prompt
  来自 mode + ghost 定义 + channel context messages，不从 prompt registry 选。

### 10. Tasks 体系默认兼容

要做，但不是现在。MCP Tasks（SEP-2663）是异步长任务协议，MOSS 有两个对齐点：
- 外→内：外部 agent 调用 MOSS tool（可能慢）→ task_id → 轮询 result
- 内→外：MOSS 调用外部 tool（可能慢）→ task 句柄 → ghost 继续做别的事

ghost bridge 的 `send → task_id → wait_reply` 已是雏形。**等 MCP tasks 协议
稳定后做统一的 task 抽象层**，ghost bridge 和 mcp_hub 都退化为它的后端。
当下 ghost bridge 的 task 语义够用。

## 实施优先级

1. **mcp_hub 瘦身** — 砍 lifecycle/register 命令，保留 MCPServerSession
2. **`moss mcp connect/disconnect/list` CLI** — topic 广播机制
3. **interface 化** — `MCPToolSurface` Protocol 抽象
4. **SafeMode 审批闸口** — ghost 主动 connect 的审批类型
5. **渐进式披露** — 未连接 server 不进 context
6. **tool 变更感知** — 惰性校验 + `moss mcp refresh`
7. **Tasks 兼容层** — 等 MCP tasks 协议稳定
