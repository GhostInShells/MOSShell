---
title: Topic Ringbuffer
status: draft
priority: P1
created: 2026-06-05
updated: 2026-06-05
depends: []
milestone:
description: >-
  Topic 协议的 ringbuffer 消费抽象 — deque 有界窗口 + poll 流式消费 + values 非破坏性快照，面向监控面板、对话上下文窗口等 "看最新 N 条" 的消费模式。
---

# Topic Ringbuffer

## Motivation

Matrix 体系里有大量可交换数据（监控、日志、传感器状态、对话上下文窗口……）。这些数据的消费模式不是 "全量处理"，而是 **"看最新的 N 条"**——监控面板、TUI 组件、多轮对话上下文都是这个模式。

Topic 协议是这个体系里唯一的强类型广播总线，manifest 声明层已经就绪。但目前 Topic 协议只提供了流式消费（`poll()`），缺少 snapshot 能力。`Session.OutputBuffer` 正是 topic 体系不完整时的独立封装验证——它证明了 bounded buffer + snapshot 模式可行，但绑死在 OutputItem 上，没有上升到 Topic 协议层。

本 feature 在 Topic 协议层提供一个通用的 ringbuffer 消费抽象——基于 `deque(maxlen=N)` 的有界窗口，同时提供流式消费和非破坏性快照。

## Key Decisions

### 1. 新子抽象，不修改 Subscriber

Subscriber 底层是 `janus.Queue`——它是一个管道（put/get），不是缓冲区。`janus.Queue` 没有 peek/values API，强行加 snapshot 要么挖 janus 内部实现，要么旁路维护并行 list，两者都烂。

新抽象与 Subscriber 同级，复用已验证的线程→协程桥接模式，但用 `collections.deque(maxlen=N)` 做自己的存储：
- `deque(maxlen=N)` 自动 evict 最旧元素——不需要手写 ringbuffer index 指针
- `asyncio.Condition` 通知 poll 等待者
- `threading.Lock` 保护 receive/values 之间的竞态

### 2. API 双模式：poll + values

```python
# 流式消费 (destructive, 类似 Subscriber)
topic = await window.poll(timeout=1.0)

# 非破坏性快照 (monitoring/GUI 的核心需求)
latest_n: list[Topic] = window.values()
```

`values()` 返回当前窗口副本——调用方接受它可能已经过时。这和 `SimpleOutputBuffer.values()` 的语义一致。

### 3. 不改 Subscriber 的 maxsize/keep

现有 Subscriber 的 `maxsize` 参数和 `keep` 残留不在此 feature 范围内清理。TopicWindow 是一个独立的新抽象，不依赖 Subscriber 接口变更。

## Implementation Notes

- 存放位置：`ghoshell_moss.core.topic.window`（与 queue_based 同级）
- 存储：`collections.deque(maxlen=N)` — 自动 evict 最旧，零额外复杂度
- 线程安全：`threading.Lock` 保护 deque，`loop.call_soon_threadsafe` 桥接通知
- 通知：`asyncio.Event` + 双检锁模式，避免 clear/wait 竞态（实际比 Condition 更简单可靠）
- 与 Subscriber 的集成：app 层在 subscriber 和 window 之间建立一个 forward 协程即可
- 17 单测通过：覆盖 poll 阻塞/超时/关闭语义、values 快照、maxsize eviction、线程安全、多消费者、含 Subscriber 的全链路集成
