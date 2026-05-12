---
created: 2026-05-11
depends: []
description: 实现分形通讯的反向注册模式：子节点通过显式 transport 主动连接到父节点。 包含 transport 参数化、moss-as-fractal
  CLI、debug 回调。
id: zenoh-fractal
milestone: beta
priority: P0
status: in-progress
status_note: FractalServeState 显式监听已完成，待端到端验证两个 workspace
title: Zenoh Fractal — 反向注册与远程联通
updated: '2026-05-11'
---

# Zenoh Fractal — 反向注册与远程联通

## Motivation

MOSS 的跨机器组网场景（如机器人开发板 → Mac）需要**反向注册**模式：
子节点知道父节点地址，主动连接并注册自己，而非依赖多播被动发现。

四种通讯动机的明确：
1. **正向发现** — 环境管理者主动发现、管理子节点 (matrix ↔ apps)
2. **反向注册** — 子节点已知 server 地址，主动注册到对方 (机器人 → Mac)
3. **双向约定** — 双方提前约定地址和协议
4. **桥接** — 组合上述三种模式

Fractal 当前只支持模式 1，机器人场景需要模式 2/3。`transport` 应该是显式的 scheme+address，类似 MCP 的 transport 描述风格。

## Scope (本轮)

| 任务 | 状态 |
|---|---|
| `provide_channel` 单次调用 + RuntimeError | ✅ done |
| `get_virtual_children` proxy 复用 | ✅ done |
| LoggerItf 注入 + 异常日志 | ✅ done |
| 去掉 `_provided_lock` | ✅ done |
| 联通性单测 (2 tests) | ✅ done |
| **transport 参数化** | ✅ done |
| **`moss-as-fractal` CLI** | ✅ done |
| **Hub 按需节点管理 (open_node/close_node)** | ✅ done |
| **liveness key 过滤修复** | ✅ done |
| **debug 回调** (`on_channel_event` → 控制台日志) | ✅ done |
| **FractalServeState — 显式标签页管理** | ✅ done |
| **端到端验证 (两个 workspace)** | 🔲 待人类工程师 debug |

**Out of scope (另开 feature):**
- Fractal 从 `blueprint.matrix` 搬迁到 `blueprint.host`，与 MossRuntime 平级（人类工程师用 PyCharm 迁移）
- 多 transport scheme 支持 (zmq 等)
- `DuplexChannelProvider` 通用事件体系

## Design Index

- 核心实现: `src/ghoshell_moss/host/zenoh_fractal.py`
- 单元测试: `tests/ghoshell_moss/host/test_zenoh_fractal.py`
- 接口定义: `src/ghoshell_moss/core/blueprint/host.py` (Fractal ABC)
- Matrix 集成: `src/ghoshell_moss/host/matrix.py` (fractal() 访问器 — 不自动启动)
- REPL State: `src/ghoshell_moss/host/repl/fractal_serve_state.py` (显式标签页)
- TUI 注册: `src/ghoshell_moss/host/tui_entries/moss_runtime_ui.py`
- CLI 入口: `src/ghoshell_moss/cli/moss_as_fractal.py`
- CLI 注册: `pyproject.toml` → `moss-as-fractal`

## Key Decisions

### 1. 四种通讯动机的定位 (2026-05-11)

Fractal 的核心场景是**模式 2 (反向注册)** + **模式 3 (双向约定)**：
- 机器人开发板 → Mac：反向注册，transport 显式指定
- 本地 apps 组网：已有 Matrix 正向发现，不需要 fractal 重复

`provide_channel` 的 `transport` 不是本地发现用，而是要**告诉子节点往哪连**。

### 2. transport 参数化 — MCP 风格 (2026-05-11)

**决策**: transport 采用 `scheme+address` 格式，类似 MCP 的 transport 描述。
```
zenoh+tcp://192.168.1.100:20770
zenoh+udp://192.168.0.0/24
```
scheme 决定 transport 类型，address 是具体端点。先只支持 zenoh。

**理由**: 机器人场景下地址必须显式给出，不能依赖多播。MCP 风格已经被验证可理解。

### 3. 单次 provide_channel (2026-05-11)

**决策**: 当前只允许注册一个 channel，重复调用抛出 RuntimeError。
**理由**: 多 provider 需要更复杂的资源管理，当前需求不明确。

### 4. 去掉 _provided_lock (2026-05-11)

**决策**: `_provided_future` 只在 event loop 协程内访问，不需要线程锁。

### 5. Proxy 复用 (2026-05-11)

**决策**: `get_virtual_children` 先查缓存，命中复用，避免 runtime 重连。

### 7. FractalServeState — 显式标签页管理 (2026-05-11)

**决策**: Fractal 从自动隐式启动改为显式 REPLState 管理。不再通过 Matrix provider 自动注册，不再在 MossRuntimeImpl 生命周期中自动启动。

**理由**:
- 网络监听端口是有副作用的资源占用，不应每次启动 moss-repl 都自动打开
- Matrix 负责本地环境通讯总线，不应承担 fractal 组网职责
- 用户应明确进入 Fractal Serve 标签页才开始监听

**行为**:
- 进入 FractalServeState → ZenohSessionFractal 启动 + hub channel import
- 离开 FractalServeState → fractal 关闭 + hub channel 清理
- `moss-as-fractal` CLI 独立管理自己的 fractal session，不受此影响

## Implementation Notes

- transport 解析后注入 zenoh session 的 connect 配置，使子节点主动连父
- `moss-as-fractal` 类似 `moss-as-mcp`：启动 runtime → 拿 shell main_channel → fractal.provide_channel
- debug 回调通过 `DuplexChannelProvider.on_channel_event` 注册，moss-as-fractal 用 ClickHandler 打印连接/断开/心跳
- FractalServeState 通过 `__aenter__`/`__aexit__` 管理 fractal 完整生命周期，hub channel 在 enter 时 import 到 shell

## Related

- Depends on: (none)
- Related: `ai-native-feature-tracking`