---
id: zmq-hub
title: ZMQ Hub — 动态注册发现
status: completed
priority: P1
created: 2026-05-11
updated: 2026-05-11
depends: []
milestone: beta
description: >-
  重写 zmq_hub.py，用 ROUTER/DEALER 动态注册替代 ZMQHubConfig 静态声明。
  不依赖 Fractal 接口，独立可用的 ZMQ 节点发现机制。
---

# ZMQ Hub — 动态注册发现

## Motivation

alpha `ZMQChannelHub` 的核心缺陷：所有 proxy 地址必须在 `ZMQHubConfig.proxies` 中**静态声明**。
以及 `as_channel()` 有已知 bug (`build.description()()` 调用 str)。

目标: 提供一个独立、自包含的 ZMQ Hub，节点动态注册，自动发现。不绑定 Fractal 接口。

## Scope

| 任务 | 状态 |
|---|---|
| `ZMQHub` 核心实现 (ROUTER registry) | ✅ done |
| `NodeInfo` / `zmq_register` / `zmq_unregister` / `zmq_query` 辅助 | ✅ done |
| `as_channel()` 生成可集成 PyChannel | ✅ done |
| 保留 `ManagedProcess` 不变 | ✅ done |
| 单元测试 (注册/发现/注销/proxy) | ✅ done |

**Out of scope:**
- Fractal 接口实现 (另开 feature)
- 跨机器组网

## Design Index

- 核心实现: `src/ghoshell_moss/bridges/zmq_channel/zmq_hub.py`
- 单元测试: `tests/ghoshell_moss/bridges/zmq_channel/test_zmq_hub.py`
- 设计文档: `.design/2026-05-11-zmq_fractal_architecture.md`

## Key Decisions

### 1. 独立于 Fractal

`ZMQHub` 不实现 `Fractal` 接口。它是一个独立的注册/发现机制，
可被 Fractal 或其他上层抽象使用，也可直接使用。

### 2. ROUTER/DEALER 协议

Hub 绑定 ROUTER，子节点用 DEALER。JSON over ZMQ multipart。
ZMQ ROUTER 天然支持多客户端和 identity 路由。

### 3. IPC 零网络开销

默认 `ipc:///tmp/moss-zmq-hub-{name}.sock`，单机场景不走网络栈。

## Related

- Related: `zenoh-fractal`
- 替代: alpha `ZMQHubConfig` / `ZMQChannelHub`
