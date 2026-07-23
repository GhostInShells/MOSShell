---
title: Matrix — 面向 AI 的进程组网与通讯总线
description: Matrix 为什么存在、什么时候需要深入它、关键探索路径。不需要提前全懂——遇到真问题时回来看
---

# Matrix — 面向 AI 的进程组网与通讯总线

Matrix 是 MOSS 的跨进程通讯矩阵。它让分布在独立进程中的能力——GUI 窗口、语音引擎、机器人躯体、视觉感知——以统一的 Channel 树呈现在 Shell 和模型面前。

**读完本文你应该能回答**：为什么 MOSS 需要自己的进程组网方案？什么情况下我需要深入 Matrix API？去哪看什么？

---

## 1. 为什么需要 Matrix

一个直观的场景：

> Ghost 正在和人类对话。它需要同时做三件事：语音输出在播放、摄像头在捕获画面、GUI 在渲染表情。每个都是独立进程——语音崩了不能拖垮 GUI，摄像头需要自己的 opencv 依赖环境。但对模型来说，这三件事应该像调用本地函数一样自然。

传统方案做不到这一点：

- **MCP** 暴露的是静态 server，不具备动态注册能力。AI 不能运行时创建新 server 并立刻使用。
- **ROS2** 解决机器人中间件问题，但它不管理 AI 模型的认知窗口——它不知道"这个节点应该以什么形式呈现在模型面前"。
- **微服务网格** 面向人类运维，不是面向 AI 操作者。

Matrix 回答三个问题：

1. **进程隔离 + 接口统一**：每个能力跑在独立进程中，但从模型视角看是一棵统一的 Channel 树。Cell 崩了，那一枝枯萎。其他枝照常。
2. **AI 在运行时自迭代**：模型创建一个 Cell → Cell 通过 announce 注册自己的膜（Channel 接口）→ Shell 的 Channel 树里自动出现新枝。不需要重启、不需要修改配置、不需要人类介入。
3. **发现即集成**：新进程加入网络后，Matrix 自动感知其存活状态和能力声明。模型不需要知道它跑在哪台机器上。

---

## 2. 什么时候需要深入 Matrix

大部分时候，只需要知道一件事：

```python
# Cell 入口 — 创建 Channel，注册到 Matrix，模型就能调用
if __name__ == "__main__":
    Matrix.discover().run(main)
```

Hello World 级别的 Cell 开发不需要理解 Matrix 内部。`Matrix.discover().run(main)` 是标准入口——发现当前 Cell 身份、管理生命周期。`main` 里调用 `matrix.provide_channel(channel)` 注册能力。到此为止。

但开发一个真正的多进程系统时，模型需要同时面对三样东西：**Matrix API**（怎么跨进程通讯）、**manifests 命令**（每个进程能拿到什么运行时依赖）、**Cell 声明**（怎么声明独立依赖和启动方式）。这三样是配套设计的——Matrix 将共用的运行时依赖屏蔽在环境发现级别的 IoC 下，让跨进程拿到相同组件。`moss manifests --help` 是了解环境能力的第一入口。模型一边看 Matrix 接口，一边用 manifests 查环境，一边开发 Cell——三者共享同一套约定。

这一切的目标：**AI 是第一开发者**。Matrix 的 provide/proxy、manifests 的自解释、Cell 的声明体系——都为模型设计。不需要人类解释"这个服务在哪"、"这个依赖怎么注入"——工具自解释。

**以下情况是信号，告诉你该回来看本文档了**：

- 你需要一个 GUI 进程和一个语音进程**同时运行，互相通讯**
- 你需要将 Cell 部署到**另一台机器**上（开发板、机器人），但希望模型用同样的 CTML 语法控制它
- 你需要**进程崩溃不传播**——一个传感器的故障不能影响对话
- 你需要**不同进程有不同的依赖环境**（视觉需要 opencv，语音需要 miniaudio）
- 你需要理解 **Session 隔离**——同一 workspace 下多个 Ghost 并行运行时如何不互相干扰

遇到这些信号前，只需要知道 Matrix 存在、知道它能跨进程桥接 Channel。遇到之后，带着真问题回来，下面这些概念才会有体感。

---

## 3. 关键概念与探索路径

### 3.1 Matrix 是映射

理解 Matrix 之前，先理解它不是什么：Matrix 不是 mesh（网络拓扑），也不是 mesh 的客户端。**Matrix 实例是"整体在局部的投影"**——蜂巢的整体形状投影到洞穴墙壁上，每个 Cell 进程通过自己墙上的投影感知全局。

这个投影哲学有两个推论：

- **每个 Cell 进程内跑的是同一个 `Matrix.discover()`**，角色由环境变量继承决定。Host 和 Node 不是两种程序，是同一份代码在不同身份下的投影。
- **Cell 的膜（Channel 接口描述）随 announce 广播**——模型不需要先连上才知道对方提供什么。自迭代循环的第一步是"看见存在什么"，不是"猜测然后尝试连接"。

### 3.2 Cell — 组网的最小进程单元

Cell 是 Matrix 网络中独立运行的进程。两种角色：

| 角色 | 身份 | 说明 |
|------|------|------|
| `host` | 主进程，组织所有能力供 Ghost 驱动 | 运行时事实——抢到 listen 端口者为 host |
| `node` | 功能节点，提供具体能力 | 通过声明文件定义，由 host 或 CLI 拉起 |

每个 Cell 有独立地址（`role/name/uid` 三段式）、独立 home 目录、独立生命周期。Cell 通过 **Presence** 宣告自己在网络上的存在（膜 + 活性），通过 **CellNetwork** 观察其他 Cell。两者拆分——宣告是每个 Cell 启动时自带（近乎免费），观察是按需开启（只有 host/Ghost 需要）。

Cell 治理由三个真相域组成，各有两个动词：

| 真相域 | 动词 | 语义 |
|--------|------|------|
| inventory（文件） | create / install | 使之可运行 |
| ledger（所有权） | run / stop | 使之生/死 |
| network（膜） | accept / deny | 使之对我可用/不可用 |

六个动词，代数封闭。list/status/logs 是三个真相域的 join 视图，不是治理。

**探索**：`moss codex get-interface ghoshell_moss.core.blueprint.cell` — Cell、CellPresence、CellNetwork、NodeManifest

### 3.3 通讯总线 — 不止 Channel

Matrix 的通讯层是 **Session**——五种通讯原语共存的共享总线：

| 原语 | 模式 | 用途 |
|------|------|------|
| topic | n×m 广播 | 强类型广播，状态同步 |
| stream | 1→n 有序字节流 | Logos 输出、实时数据 |
| signal | 感知信号 | 驱动 Mindflow 三循环 |
| output | 结构化消息 | 全局事件与状态通知 |
| file | 文件读写 | Session 级别文件共享 |

Channel 跨进程桥接（下节）是其中一种通讯模式——但它构建在 Session 之上，不是 Session 的全部。Cell 入网后，除了 provide channel，还可以 publish event、发 signal、写 stream。模型通过 Ghost 感知到的是一个立体的能力空间，不只是一棵命令树。

**探索**：`moss codex get-interface ghoshell_moss.core.blueprint.session` — Session 抽象与五种原语

### 3.4 Channel 跨进程桥接

跨进程 Channel 通讯的核心模式只有两个动作：

```
进程 A （Provider）              进程 B （Consumer）
───────────────                 ───────────────
matrix.provide_channel(chan)    mesh.channel_proxies()
        │                               │
        ▼                               ▼
   Presence announce               Watcher 发现 + accept
        │                               │
        └─────────── 网络 ─────────────┘
                        │
                 Shell Channel 树中出现远程枝
```

关键：**模型在 CTML 中调用远程 Channel 的语法和本地 Channel 完全相同。** 模型不感知这个 Channel 是在本进程内还是另一台机器上。

Provider 端声明"我有这个能力"，Consumer 端 accept 后创建 proxy。两端可以各自重启，不影响对方。

**探索**：
- `moss codex get-interface ghoshell_moss.core.blueprint.matrix` — `provide_channel()` 与 CellNetwork
- `moss codex get-interface ghoshell_moss.core.concepts.channel` — Channel、ChannelProxy

### 3.5 Session 与 Scope — 并行不串扰

Session scope 是通讯隔离域。同 scope 内的 Cell 共享通讯总线，跨 scope 隔离。这是"并行化身"概念的工程基础——当未来多个 Ghost 化身并行运行时，各自拥有独立的通讯空间，互不污染。

**探索**：`moss codex get-interface ghoshell_moss.core.blueprint.session` — Session 抽象与 scope 语义

### 3.6 跨机器 — channel 分形挂载

当能力分布在多台机器上时——开发板上的传感器、Mac 上的 GUI、远程服务器上的推理节点——不需要一个"跨机器启动"的原语。Matrix 的方案是 **channel 分形挂载**：

每台机器有自己的 host，治理自己机器上的 Cell。A 机器需要 B 机器的能力时，B 将自己的 Channel 子树 provide 到网络中，A accept 后挂载到本地 Shell。模型看到的还是那棵 Channel 树。

这是 Plan 9 namespace import/export 哲学做到能力层——跨机器是**组合**（子树挂载），不是参数（对远端 run_cell）。没有控制平面，没有中心化调度器。

---

## 4. Manifests 与 Matrix 的关系

Manifests 是声明，Matrix 是执行者。

启动时 Matrix 消费 mode manifest（通过 Python import 继承全局基线声明）：遍历 providers → 注入 IoC 容器。遍历 bootstrappers → 后置初始化。遍历 bringup_nodes → 拉起 Cell 子进程。

**声明和实现分离**：开发者（人类和 AI）只在 manifest 文件中写 Python 实例声明，Matrix 负责发现、注入、组网。换一个 transport 实现（zenoh → 未来的其他协议），声明层不需要改动。

**深入**：`moss docs read project-and-mode.md` — Project 与 Mode 体系

---

## 5. 传输协议可替换

Matrix 的抽象层（Cell、Presence/CellNetwork、Session）不绑定具体传输协议。当前默认 transport 可替换——这是架构设计意图，不是临时妥协。未来如果出现更适合 AI OS 场景的通讯协议，切换 transport 不需要改动上层的 Channel 和 Shell。

---

## 6. 深入阅读

| 你想了解 | 去这里 |
|----------|--------|
| Matrix 完整接口 | `moss codex get-interface ghoshell_moss.core.blueprint.matrix` |
| Cell 定义与组网 | `moss codex get-interface ghoshell_moss.core.blueprint.cell` |
| Channel 抽象 | `moss codex get-interface ghoshell_moss.core.concepts.channel` |
| Session 通讯总线 | `moss codex get-interface ghoshell_moss.core.blueprint.session` |
| 架构拓扑中 Matrix 的定位 | `moss docs read architecture-topology.md` |
| Manifests 声明体系 | `moss docs read project-and-mode.md` |

