---
title: MCP Fusion Point — 寻找 MCP 与 MOSS 的合适融合点
status: converging
priority: P1
created: 2026-07-31
updated: 2026-08-11
depends:
  - mcp-hub-channel
  - speech-protocol-alignment
milestone: null
description: >-
  MCP（尤其 2026-07-28 stateless 版）与 MOSS channel 基于高度类似的愿景设计。
  本 workstream 的正式目标是找到两者的合适融合点——先定位 MCP 在 MOSS 架构中
  的身份，再连带回答是否以 MCP 作为 cell 间 RPC 协议的底座。

  决策 1-10 已锁定。CLI 化与命名已完成（ghoshell_moss.mcp + moss mcp CLI）；
  新 bridge 实机闭环走通。残留：signal 优先级 + ghost runtime 异常处理缺口。
---

# MCP Fusion Point — 寻找 MCP 与 MOSS 的合适融合点

> 状态：converging。RPC 底座裁决与 MCP 位置已收敛，node run as mcp server 机制
> 敲定，mailbox bridge 验证用例已实现并于 2026-08-10 首次实机闭环。CLI 体系
> （`moss mcp`）已落地（2026-08-11：`ghoshell_moss.mcp` 包 + `moss mcp
> serve-ghost-bridge` + `Matrix.serve_mcp`）。实机发现 6 个 bug 已修复 + 3 个
> ghost runtime 异常处理缺口已定位待修。讨论轨迹见 `discuss/` + `design/`。
> 用 `moss features set-status mcp-fusion-point <status> -m "note"` 更新状态。

## Motivation

### 为什么要现在做

`speech-protocol-alignment` 的通用化（draft）暴露了 `drain` / `pause` 这类
控制动作可以跨 cell 存在，需要一个跨进程的控制协议。MOSS channel 本质是
有状态 1:1 协议，继续往下做就得发明"协议无关的 stateless transport RPC"。
不想发明轮子。

与此同时，MCP 与 MOSS channel 面貌相近而核心设计选择频繁相左——不做考古
的模型初见会困惑"为什么独立发明轮子"，进入代码后又撞上彼此不可兼容
（channel 是 MCP 的超集——时间第一公民、排序与阻塞、树形构建——这些
恰是 MCP 有意简化的维度）。困惑是融合确实值得正视的信号。

### 正式目标

> 找到 MCP 与 MOSS 项目的**合适融合点**。

- 主问题：确认 MCP（尤其最新 stateless 版）在 MOSS 架构中的**位置**。
- 连带问题：是否以 MCP 作为 MOSS cell 之间 RPC 协议的底座。
- **前者比后者更重要**。位置决定是前提，RPC 底座是派生。

## 判断（非决议）

以下为当前阶段的判断，记录背景而非锁定方向：

- **同源愿景，终将收敛**：MCP 与 MOSS channel 基于高度类似的愿景设计，最终会
  收敛到一处。
- **资源决定生存**：融合点最终由资源（生态、工程成本、行业走向）决定，不是由
  纯架构推演决定。
- **channel 现阶段仍是 MCP 的超集**：在时间第一公民、排序与阻塞、树形构建等
  设计维度上，channel 超过 MCP。这正是两者一直互相不好兼容的原因——MCP 简化了
  这些维度以换取 stateless 的可扩展性。

## Design Index

- Key design documents:
  - `design/mcp-node-server.md` — 子任务 1：node run as mcp server（朝外），机制已敲定
  - `design/mcp-node-client.md` — 子任务 2：node run mcp client（朝内），部分敲定，细节待讨论
  - `src/ghoshell_moss_contrib/nodes/mailbox.py` — mailbox bridge 验证实现
- Key discussion records: `discuss/2026-07-31_mcp_position_and_fusion.md`
- 前置讨论：`.discuss/2026-07-30_mcp_duplex_convergence_and_memento_branch.md`

## Key Decisions

### 1. RPC 底座：原生 matrix.rpc，不是 MCP（2026-08-05 收敛）

cell 间 RPC 用原生 `matrix.rpc`——注册表 + 单一发现 channel + zenoh put/sub +
JSON-RPC 2.0 + 回调身份 + caller 侧超时，从 zenoh_qa 泛化。MCP 作内部 RPC 协议
零优势且双向外转损失类型。内部 cell↔cell 与外部边界分开。

### 2. MCP 位置：外部皮，不是脊柱

MCP 在 mesh 边界双向存在：node run as mcp server（朝外）/ node run mcp client
（朝内）。内部通讯走 matrix.rpc。`moss as mcp`（moss_as_mcp.py）是整运行时降级，
已做，独立于 node 级。

### 3. node run as mcp server 机制（2026-08-05 敲定）

`moss nodes run` 启动（非 `mcp run`）+ `matrix.run(mcp)` 糖（tools 先注册、run 不
重新注册）+ stateless streamable-http + `main(port=0)` 约定 + announce 走 nodes
channel EVENT（非 signal）、endpoint 双写 cell presence。详见
`design/mcp-node-server.md`。

### 4. node run mcp client 极简（部分敲定）

薄 channel（list/read/exec，不 command 化）+ `moss mcp connect` CLI。mcp_hub 瘦身。
授权 / 声明发现 / debug 细节待讨论。详见 `design/mcp-node-client.md`。

### 5. `matrix.serve_mcp(mcp)` — Matrix 级原语（2026-08-09）

node-as-mcp-server 的公共基础。封装 bind socket + announce + serve 时序，
让任何 node 用一行代码暴露 MCP 能力。mailbox 的 `serve_mailbox` 退化为
`serve_mcp` + channel 注册的薄层。

`serve_mcp` 内部用 `run_async`（绝不 `mcp.run()`——会开新 event loop）。
预 bind socket（port=0 稳定性）+ announce 走 nodes channel EVENT + cell
presence 双写。详见 `design/mcp-node-server.md` 中"坑（糖要封装的第一个点）"。

### 6. mailbox bridge — 验证用例已实现（2026-08-09）

双向 request-reply 桥接验证：外部 agent 通过 MCP send/pull/wait_reply 工具与
ghost 通信，ghost 通过 CTML `mailbox:reply(task_id, text__)` 显式回复。

- **核心**：`MailboxBridge` — agent 侧 `create/wait/check`，ghost 侧 `post`
- **位置**：`ghoshell_moss_contrib.nodes.mailbox` — openbox node，懒加载 mcp 依赖
- **node**：`nodes/mailbox/main.py` — 薄壳入口
- **协议**：send 返回 task_id（兼容 MCP Tasks 语义），pull 轮询结果，
  wait_reply 阻塞等回复（事件驱动，2026-08-10 实机新增）
- **测试**：14 tests（12 unit + 2 MCP 集成），全绿

mailbox 不走 `session.on_output` 监听——ghost 必须显式调用 `mailbox:reply`。
这是 request-reply 模式，不是 ghost monitor。

### 7. CLI 体系方向（2026-08-09 讨论，待实现）

当前命令命名需要厘清：

| 命令 | 定位 | 状态 |
|------|------|------|
| `moss-mcp` (原 `moss_as_mcp.py`) | MOSS 运行时控制面暴露给 AI coding agent | 已实现 |
| `moss mcp` (新 CLI 组) | MCP 客户端/服务端管理入口 | 待实现 |

`moss mcp` 子命令方向：
- `moss mcp connect <url>` — 轻量 cell 入网（`Matrix.new`，无膜），获得 session 能力
- `moss mcp disconnect/remove/list` — 连接生命周期管理
- `moss mcp serve-mailbox` — 开箱即用的 mailbox，比独立 node 更轻
- node 模板脚手架（`moss node init-mcp-server`）如果 `serve_mcp` 足够好就不需要

连接信息进 scoped storage → `mcp_hub` 感知 → hub 的 connect 退化为授权动作。
这是个独立于 mailbox 的后续任务。

### 8. MCP 协议翻译层（2026-08-09 方向确认，待实现）

MOSS (mindflow + shell) 是 MCP 的超集，融合时把 MCP 最新版协议翻译过来即可，
不需要重写内部协议。三个翻译面：

- **Tasks** → 通用 list/get/notification 封装（MCP 2.0.0 SEP-2663 无 push，
  但 MOSS 侧可以用 topic/signal 模拟 notification）
- **Tools** → CTML commands（mcp_hub 已在做，但当前是 command-ized 而非翻译）
- **Resources/Prompts** → manifests 体系可映射

协议翻译层独立于 mailbox / serve_mcp / CLI，是三者共用的基础设施。

### 9. 融合收敛判断（2026-08-09）

三条线各就其位：

1. **`matrix.serve_mcp(mcp)`** — 服务端原语，所有 node-as-server 的基础
2. **`moss mcp connect` → 轻量 cell** — 客户端入网，hub 退化为授权
3. **协议翻译层** — tasks/tools 的通用桥，不重写只翻译

下一步优先：端到端验证 `Matrix.new` 轻量入网 + `serve_mcp` + ghost 实际对话。

### 10. mailbox 首次实机闭环（2026-08-10）

external agent (Claude Code) ↔ echo ghost 跨宿主双向 request-reply 首次在真实
进程中走通。里程碑见
`stages/2026-08-v0.1.0/milestones/2026-08-10-mailbox-first-real-machine-bridge.md`。

**实机发现的 bug（全部已修复）**：

1. `Message.of_text` 旧 API 已删 — 改用 `Message.new().with_content(...)`
2. exec.command `.venv/bin/python` 相对路径从 cell.home 解析失败 —
   `NodeLauncher` 只把 `command == 'python'` 换成 `sys.executable`；改回 `python`
3. 创建体系未解释 `python` 机制 — stub NODE.md/README 补注释（python =
   spawner 的 sys.executable，独立 venv 才写绝对路径）
4. `reply` 误标 `always_observe=True` → ghost 反复驱动 — 实机确认后改 `False`
5. `reply(content=)` 属性传参，回复带裸露 `<` 触发 CTML parse error 整个 dispatch
   取消（真实事故）— 改 `reply(task_id, text__)` open-close + CDATA，实机免疫
6. pull 式 API 无法阻塞等回复 — 新增 `wait_reply(task_id, timeout)`，事件驱动阻塞

**暴露的架构不对称**：MOSS 内 push（signal → mindflow）vs MCP 外 poll（pull）。
echo 能"看到" agent 的消息，agent 看不到 echo 主动说话。`wait_reply` 是伪造
共享"现在"的补丁——印证"MCP 传达不了时间流"的判断。

**后续三点（2026-08-11 更新）**：

1. **CLI 化** — 已完成。`ghoshell_moss.mcp` 包（`GhostBridge` + `serve_ghost_bridge`），
   `moss mcp serve-ghost-bridge` CLI，`Matrix.new` 轻量 cell 入网。
2. **mcp channel 命名** — 已完成。统一用 `ghost_bridge`（channel / CTML / cell 名）。
3. **signal 优先级** — 待修。NOTICE + notify 不应打断 ghost speaking，但实机中打断
   了，根因在 mindflow challenge 仲裁逻辑。

### 11. 新 bridge 实机闭环 + 暴露的 ghost runtime 异常处理缺口（2026-08-11）

新 `moss mcp serve-ghost-bridge` 与 echo ghost 三轮对话实机走通两轮，第三轮
暴露三个底层问题（未修，留现场）：

**a. Signal 打断 streaming 导致 Anthropic stream 异常**

```
httpx._transports.default → anthropic._streaming → pydantic_ai.result
→ Unhandled exception in event loop
```

signal → mindflow challenge → attention abort → articulator 的 pydantic_ai
stream 被取消，异常沿 httpx → anthropic → pydantic_ai 链路上抛，无任何层
捕获。表现为 event loop 里的 Unhandled exception。

**b. pydantic_graph cancel scope 跨 task 错误**

```
anyio CancelScope: Attempted to exit cancel scope in a different task
than it was entered in
```

pydantic_ai 内部用 anyio cancel scope 管理流式响应生命周期。当 mindflow
从另一个 task 取消 attention 时，cancel scope 的 enter/exit 发生在不同
task，anyio 报错。这是 pydantic_ai + anyio + mindflow 三方 task 模型
不一致的冲突。

**c. GhostRuntime 异常处理缺口**（调研结论）

- `asyncio.CancelledError` 在整个 runtime loop 体系里无处理——它是
  BaseException，不被 `except Exception` 捕获，shutdown 时沿 task 链
  泄漏
- `AttentionAbortedError`（attention 层视为正常关闭信号）被
  `_run_articulator` 的 catch-all 当作 error 日志 + `session.output('error')`
  广播——attention 层和 runtime 层对同一事件的语义不一致
- 流式响应无异常隔离——anthropic/httpx 层的异常穿透所有层到达 event loop

三条根因指向同一个问题：**ghost runtime 缺少分层异常隔离**。signal 打断
是正常事件，但从 mindflow → attention → articulator → pydantic_ai →
anthropic → httpx 的取消传播链上，每一层都假设下一层会处理，最终无人兜底。
