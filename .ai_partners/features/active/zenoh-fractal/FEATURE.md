---
id: zenoh-fractal
title: Zenoh Fractal — 反向注册与远程联通
status: in-progress
priority: P0
created: 2026-05-11
updated: 2026-05-11
depends: []
milestone: beta
description: >-
  实现分形通讯的反向注册模式：子节点通过显式 transport 主动连接到父节点。
  包含 transport 参数化、moss-as-fractal CLI、debug 回调。
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
| **transport 参数化** (scheme+address, MCP 风格) | 🔲 todo |
| **`moss-as-fractal` CLI** | 🔲 todo |
| **debug 回调** (`on_channel_event` → 控制台日志) | 🔲 todo |

**Out of scope (另开 feature):**
- Fractal 从 `blueprint.matrix` 搬迁到 `blueprint.host`，与 MossRuntime 平级（人类工程师用 PyCharm 迁移）
- 多 transport scheme 支持 (zmq 等)
- `DuplexChannelProvider` 通用事件体系

## Design Index

- 核心实现: `src/ghoshell_moss/host/zenoh_fractal.py`
- 单元测试: `tests/ghoshell_moss/host/test_zenoh_fractal.py`
- 接口定义: `src/ghoshell_moss/core/blueprint/matrix.py` (Fractal ABC)
- CLI 入口: `src/ghoshell_moss/cli/fractal.py` (待创建)

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

### 6. LoggerItf 外部注入 (2026-05-11)

**决策**: `__init__` 接收 `logger: LoggerItf` 必选参数，由上层 IoC 注入。

## Implementation Notes

- transport 解析后注入 zenoh session 的 connect 配置，使子节点主动连父
- `moss-as-fractal` 类似 `moss-as-mcp`：启动 runtime → 拿 shell main_channel → fractal.provide_channel
- debug 回调通过 `DuplexChannelProvider.on_channel_event` 注册，打印连接/断开/心跳

## Related

- Depends on: (none)
- Related: `ai-native-feature-tracking`
