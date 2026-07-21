---
title: MOSS Channel 体系
description: Channel 的能力模型、运行时语义、发现机制。需要理解 Channel 的设计哲学时阅读，具体构建方法走 codex
---

# MOSS Channel 体系

Channel 是 MOSS 架构的能力组织单元。在八层拓扑中，它位于 Logos（CTML 语法）与 Shell（流式调度）之间——Logos 描述"做什么"，Shell 决定"何时做"，Channel 回答"能做**什么**"。

本文档描述 Channel 的概念模型和设计哲学。构建 Channel 的具体方法走 codex 自省，不在此重复。

---

## 1. 核心概念

### 1.1 Channel 是什么

**能力组织单元** —— 类比文件系统：文件系统用目录组织文件，Channel 用树形拓扑组织命令。每个 Channel 封装一组命令、附带指令（instruction）、动态上下文（context_messages）、生命周期钩子。

**上下文窗口组件** —— Channel 在模型每个思维关键帧注入动态上下文。视觉 Channel 提供 capture 命令的同时，通过 context_messages 将当前画面描述注入模型的上下文窗口。

**有状态运行时** —— 区别于无状态的 Tool Calling。Channel 在模型多次调用之间保持运行，持有连接和状态。同通道内命令 FIFO 顺序执行（occupy 语义）。

### 1.2 关键抽象

```bash
moss codex get-interface ghoshell_moss.core.concepts.channel    # Channel 契约
moss codex get-interface ghoshell_moss.core.blueprint.channel_builder   # 构建器
moss codex get-interface ghoshell_moss.core.blueprint.states_channel    # 有状态 Channel
moss codex channeltypes                                                 # Channel 类型索引
```

核心抽象层次：

- **Channel** — 无副作用的声明。`bootstrap()` 产生有副作用的 `ChannelRuntime`
- **ChannelRuntime** — 运行时实例。持有命令、上下文、生命周期
- **Builder** — 装饰器风格构建 API。函数签名即模型接口（Code as Prompt）
- **ChannelState** — 多态运行时状态。不同状态有独立命令集和上下文
- **Command** — 被反射的 Python 函数。docstring 即模型看到的接口描述

### 1.3 一个 Channel 承担的角色

以下维度描述 Channel 在 MOSS 架构中的角色面。每项均有对应的 API——具体签名走 codex，此处只描述概念关系。

**能力创建与组织。** Channel 将 Python async 函数封装为 Command，按树形拓扑组织。子节点可以是本地模块、远程进程、或运行时动态生成的虚拟节点。树形结构支持折叠和渐进披露。

**时序契约。** Channel 声明命令的阻塞语义。Shell 据此调度：同通道内 FIFO，父通道 occupy 期间阻塞子通道，异通道并行。支持抢占控制。

**运行时生命周期。** 完整的 `startup → running → idle → close` 状态流。idle 钩子用于维持姿态、降低帧率等后台行为。

**IoC 依赖注入。** Channel 内的命令通过 IoC 容器获取服务，不硬编码依赖。`moss manifests` 系列命令提供开发时的环境能力清单。

**模型上下文认知。** Channel 通过多种消息类型参与上下文窗口构建：静态接口签名、运行时刷新状态、instruction 注入、每个关键帧的动态上下文。CTML prompt 定义了消费这些消息的协议。

**反身性控制。** Channel 的命令可以修改 Channel 自身——切换状态、增减子通道、更新 instruction。这是 Ghost 反身性的工程基础：能修改自身能力的智能体才可能演进。

**通讯端点。** Channel 是数据在 MOSS 体系中的入口和出口：Signal 上行给 Mindflow，context 注入模型认知窗口，command result 返回给 Shell。传输管线由 Matrix/Session 负责，Channel 是端点。

**远程同构。** 同一个 Channel 接口，可以指向本地函数，也可以指向另一个进程中的 Channel。模型在 CTML 中调用时使用相同的语法，不感知位置差异。详见 Node 体系文档和 Matrix 体系文档。

---

## 2. 运行时语义

### 2.1 Occupy 与并发

- **同通道内**：命令 FIFO 顺序执行。当前命令未完成时，新命令保持 pending
- **父通道 occupy**：父通道有命令执行时，所有子通道的新命令都不会分发
- **异通道间**：并行执行，互不阻塞

`@nonblocking` 标记命令不 occupy，同通道后续命令立刻执行。

### 2.2 Observe 体系

| 机制 | 效果 |
|------|------|
| `always_observe=True` | 结果在下一关键帧展示给模型 |
| `) -> Observe:` 返回值 | 标记观察，不中断并行任务 |
| `raise ObserveError()` | 紧急中断全局，取消一切 |

`always_observe` 的约定：返回"信息"的命令设 True（read、list、query），返回"确认"的设 False（write、delete、start）。

### 2.3 动态性

- **静态信息** — 启动时确定（命令签名、通道描述）
- **动态信息** — 运行时刷新（上下文、新命令、子通道变化）
- **refresh** — 触发 Channel 树的动态更新，新接入的进程能力自动出现

---

## 3. 发现与自解释

### 3.1 开发时发现

```bash
moss codex channeltypes              # 所有 Channel 类型索引
moss codex channeltypes <name>       # 单个 Channel 完整接口
```

索引 `ghoshell_moss.channels` 包下所有模块，读取 docstring 中的类型和状态标记。

### 3.2 运行时发现

```bash
moss manifests channels     # 当前环境中 main channel 的命令树
moss manifests explain      # 所有声明类型的完整清单
```

两者区别：`codex channeltypes` 是开发时视角（有哪些预制能力），`manifests channels` 是运行时视角（当前环境的能力树实际长什么样）。

---

## 4. 与相邻层的关系

```
Logos (CTML 语法)
    ↓ 描述"做什么"
Channel (能力组织)
    ↓ 回答"能做什么"
Shell (流式调度)
    ↓ 决定"何时做"
Matrix (通讯总线)
    ↓ 提供"在哪里做"
```

Channel 向上对接 CTML 的调用语法，向下依赖 Builder 将 Python 函数反射为模型可理解的形式。跨进程时依赖 Matrix 桥接。

---

## 5. 深入探索

| 想了解 | 去这里 |
|--------|--------|
| Channel 构建（Builder、StatefulChannel、生命周期） | `moss codex get-interface ghoshell_moss.core.blueprint.channel_builder` |
| Channel 类型索引 | `moss codex channeltypes` |
| Channel 基础契约 | `moss codex get-interface ghoshell_moss.core.concepts.channel` |
| CTML 如何调用 Channel | `moss ctml read` |
| 跨进程 Channel 桥接 | Matrix 体系文档 |
| Node 中如何使用 Channel | Node 体系文档 |
| 环境中的 Channel 声明 | `moss manifests channels` |
