# 子任务 2 — Node Run MCP Client（朝内）

> 索引于 FEATURE.md。机制部分敲定，关键细节待讨论。

## 定位

对朝内的 MCP 接入做治理——ghost 连接外部 MCP server、使用其工具。极简：client
不用独立 node，in-process channel 即可，CLI 化。`moss mcp connect` 只是配置糖。

## 已敲定

### 薄 channel 表面

- 三个命令：`list`（分组）、`read <group>`（该分组 tools 签名 + 入参）、
  `exec <group> <tool> <json>`。
- **不 command 化**——工具保持 data，不物化成 MOSS Command。无论挂多少工具，
  表面恒 3 命令。
- 理由：MCP 工具是无状态扁平调用，command 化不买超集维度，只把面切碎成 N×M。

### CLI

- `moss mcp connect --transport sse <url>` / `--transport stdio <cmd>`——配置糖，
  创建 client 接入。极简到不必建独立 node。

### mcp_hub 瘦身

- **砍**：connect/disconnect/reconnect（生命周期归治理）、register/unregister +
  ConfigStore + scope 解析 + workspace preset 合并（配置归 CLI/声明）、exec_blocking
  （阻塞语义归 channel 级）。
- **留**：MCPServerSession transport 选择 + initialize + list_tools + call_tool、
  result→Observe 转换。

## 待讨论（关键细节）

- **debug 类细节**：连接态（connected/error + 原因）、stdio 进程 stderr 捕获、probe
  验证"握手 + list_tools 成功"才 ready。node-lifecycle probe + MCPServerSession
  状态机的结合点。
- **授权/信任模型**：ghost 对每个 server/tool 建立信任判断（readOnlyHint /
  destructiveHint）、审批闸口（safemode/qa）、credential 进 scoped storage。
  mcp_hub 的 `allow_model_config` 布尔是假自主。
- **声明与发现策略**：RPC vs 规格上升（MCP 工具声明格式做统一标准）。
- **A/B/C 启动方式**：B（server+client→channel 入网，最重）/ A（只管 client）/
  C（subprocess/jobs 走 bash 跑 mcp 自动接线）。
- **"足够标准"治理**：ghost 自主治理 MCP 的办法——边界接口用 MCP 规格原样
  （tools/list、subscriptions/listen、tasks/MRTR），MOSS 特有的只有路由和决策。
