---
title: Matrix Node 体系
description: Node 是 MOSS 的运行时自迭代单元。需要理解 Node 为什么存在、解决什么问题、与 MCP/Skills 的区别时阅读
---

# Matrix Node 体系

Node 是 MOSS 的**运行时自迭代单元**。它是一个独立进程，通过声明文件被发现，通过 Matrix 总线通讯，被模型在运行时创建、调用、修改、重启。

读完本文应能回答：为什么 MOSS 需要 Node 这个概念？它和 MCP / Skills 有什么区别？它的核心设计决策是什么？

---

## 1. 为什么需要 Node

MOSS 的目标之一是让 Ghost 在运行时演进自己的能力——不是人类预先配置好工具等模型来用，而是模型自己在交互过程中创建新能力、启动新进程、调用、观察、修改。

这个目标暴露了现有方案的缺口：

**MCP** 暴露的是静态 server。工具集在启动前就确定了，模型不能在运行时说"我需要一个新工具"然后创建一个。MCP 解决的是"如何跨进程调用已有工具"，不是"如何在运行时生长新能力"。

**Agent Skill / Tool** 是进程内的函数封装。每次调用走同一进程，没有独立生命周期，不能跨语言或跨依赖环境。一个视觉 Skill 需要 opencv，一个语音 Skill 需要 miniaudio——它们在进程内会冲突。

**微服务网格** 有进程隔离和动态发现，但面向人类运维。它不知道"这个节点应该以什么形式呈现在模型的认知窗口中"。

Node 回答这些问题：

- **运行时创建**：模型可以在对话中让 Ghost 创建新 Node，Matrix 自动发现
- **进程隔离**：每个 Node 是独立进程，独立依赖，崩了不传播
- **膜接口**：Node 通过 Channel 声明自己的能力——Python 函数签名即是接口，模型看不到代码
- **透明调用**：模型在 CTML 中调用远程 Node 的命令，语法和本地完全相同
- **跨机器**：Node 可以在不同机器上运行，通过 zenoh scope 组网，能力自动对模型可见

Node 是一种**有状态有运行时的能力机制**。用它驱动一个机器人、实现一个 GUI——进程运行期间模型可以直接调用，不需要构建中间协议层，不需要像 Skill 那样反复走进程通讯。

---

## 2. 核心设计决策

### 2.1 膜承诺 — Cell 必须提供 Channel

Node 入网的前提是声明自己提供什么能力。这个声明不是文档——是运行时的膜（Channel 接口描述）。Matrix 发现 Node 后，通过这个膜构建 ChannelProxy，模型看到的是一棵统一的 Channel 树。

Node 内部的实现可以是任何东西（opencv 视觉管线、ROS2 机器人控制、pygame 窗口），但膜是统一的——Channel。这保证了发现的通用性。

声明格式通过 codex 自省：

```
moss codex get-interface ghoshell_moss.core.blueprint.cell NodeManifest
```

### 2.2 目录即声明

Node 通过文件系统约定被发现，不需要运行时注册。一个目录 + 一份声明文件 = 一个可被发现的 Node。

```
moss --ai nodes list           # 当前有哪些 Node 可拉起
moss --ai nodes show <path>    # 查看声明原文 + 目录内容
```

声明文件的主体是给 AI 看的 instruction——描述这个 Node 是什么、怎么用。

### 2.3 三面控制 — 同一咽喉，不同入口

Node 的生命周期有三个控制面，共享同一套 spawn 逻辑：

| 控制面 | 入口 | 场景 |
|--------|------|------|
| CLI | `moss nodes run/stop/kill` | 人类调试、脚本 |
| Matrix API | `matrix.run_node(target)` | 父进程以本 Matrix 为治理域拉起子 Node |
| 模型 (CTML) | `nodes:run` / `nodes:stop` 等命令 | Ghost 在运行时自迭代 |

三个面的差异只在"谁发起的"，治理逻辑是一套。

```
moss --ai all-commands --group nodes    # CLI 完整命令
```

### 2.4 进程隔离 — 独立依赖，崩溃不传播

每个 Node 是独立进程。视觉 Node 装 opencv，GUI Node 装 PyQt6，互不污染。一个传感器进程崩了，对话继续——那片叶子枯了，其他枝照常。

依赖声明和启动方式通过 ExecSpec 描述：

```
moss codex get-interface ghoshell_moss.core.blueprint.cell ExecSpec
```

---

## 3. 跨机器组网

Node 不限于同一个 workspace、同一个 project、甚至同一台机器。网络 scope 机制（基于 Eclipse Zenoh）支持跨机器组网。

每台机器有自己的 host 治理本地 Node。A 机器需要 B 机器的能力时，B 的 Node 将 Channel 子树 provide 到网络中，A accept 后挂载到本地 Shell。模型看到的还是那棵 Channel 树。

这是 Plan 9 namespace import/export 哲学做到能力层——跨机器是**组合**（子树挂载），不是参数（对远端 spawn）。没有中心化调度器。

当前内置两种网络配置：

```
moss --ai networks list    # local（单机） / lan（局域网）
```

同 scope 内的 Node 共享 zenoh key namespace，自动发现。外来 Node 默认可见但不自动 accept——需要显式 `mesh:accept` 放行。

---

## 4. 与相邻概念的关系

**Cell** 是抽象概念——Matrix 网络中运行的进程单元。Cell 有两种角色：`host`（中心节点，组织能力供 Ghost 驱动）和 `node`（功能节点）。Node 是 Cell 的 node role 的具体化。

```
moss codex get-interface ghoshell_moss.core.blueprint.cell
```

**Matrix** 是通讯总线。Node 是它上面的节点。Matrix 负责组网、发现、通讯；Node 负责提供能力。两个概念配套设计。

```
moss codex get-interface ghoshell_moss.core.blueprint.matrix
```

**Channel** 是能力的接口形式。Node 通过膜（Channel）暴露能力，模型通过 CTML 调用。Channel 构建方法走 channel_builder。

```
moss codex get-interface ghoshell_moss.core.blueprint.channel_builder
```

**Ghost** 是使用者。Ghost 通过 Shell 看到 Matrix 组织的 Channel 树，在运行时通过 CTML 驱动 Node——包括创建新的和修改已有的。

---

## 5. 最小入口

开发一个 Node 的认知闭环：

| 需要知道 | 入口 |
|----------|------|
| Channel 怎么构建 | `moss codex get-interface ghoshell_moss.core.blueprint.channel_builder` |
| Node 怎么声明 | `moss codex get-interface ghoshell_moss.core.blueprint.cell NodeManifest` |
| Matrix 怎么入网 | `moss codex get-interface ghoshell_moss.core.blueprint.matrix` |
| CTML 怎么调用 | `moss ctml read` |
| CLI 怎么操作 | `moss --ai all-commands --group nodes` |

## 6. 深入

| 想了解 | 入口 |
|--------|------|
| Cell 完整定义（角色、地址、声明、运行时信息） | `moss codex get-interface ghoshell_moss.core.blueprint.cell` |
| Matrix 组网与 host/node 关系 | `moss docs list` 中查看 Matrix 体系文档 |
| Channel 构建完整知识 | `moss codex get-interface ghoshell_moss.core.blueprint.channel_builder` |
| Session 五种通讯原语 | `moss codex get-interface ghoshell_moss.core.blueprint.session` |
| 网络配置（local/lan） | `moss --ai networks list` |
| 项目与 Mode 的 Node 发现路径 | `moss docs list` 中查看相关文档 |
