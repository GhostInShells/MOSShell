---
title: Matrix — 面向 AI 的进程组网与通讯总线
description: Matrix 为什么存在、什么时候需要深入它、关键探索路径。不需要提前全懂——遇到真问题时回来看
---

# Matrix — 面向 AI 的进程组网与通讯总线

Matrix 是 MOSS 的跨进程通讯总线。它让分布在独立进程中的能力——GUI 窗口、语音引擎、机器人躯体、视觉感知——以统一的 Channel 树呈现在 Shell 和模型面前。

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

1. **进程隔离 + 接口统一**：每个能力跑在独立进程中，但从模型视角看是一棵统一的 Channel 树。App 崩了，那一枝枯萎。其他枝照常。
2. **AI 在运行时自迭代**：模型创建一个 App → App 通过 Matrix 注册自己的 Channel → Shell 的 Channel 树里自动出现新枝。不需要重启、不需要修改配置、不需要人类介入。
3. **发现即集成**：新进程加入网络后，Matrix 自动感知其存活状态和能力声明。模型不需要知道它跑在哪台机器上。

---

## 2. 什么时候需要深入 Matrix

大部分时候，只需要知道一件事：

```python
# App 入口 — 创建 Channel，注册到 Matrix，模型就能调用
# main 里就是调用了 matrix.provide_channel(channel)
# 参考 tutorials/L1_hello-world-app.md 的完整例子
if __name__ == "__main__":
    Matrix.discover().run(main)
```

Hello World 级别的 App 开发不需要理解 Matrix 内部。`Matrix.discover().run(main)` 是标准入口——发现当前 Cell 身份、管理生命周期。`main` 里调用 `matrix.provide_channel(channel)` 注册能力。到此为止。

但开发一个真正的多进程 App 时，模型需要同时面对三样东西：**Matrix API**（怎么跨进程通讯）、**manifests 命令**（每个进程能拿到什么运行时依赖）、**App 脚手架**（怎么声明独立依赖和启动方式）。这三样是配套设计的——Matrix 将共用的运行时依赖屏蔽在环境发现级别的 IoC 下，让跨进程拿到相同组件。`moss manifests --help` 是了解环境能力的第一入口。模型一边看 Matrix 接口，一边用 manifests 查环境，一边开发 App——三者共享同一套约定。

这一切的目标：**AI 是第一开发者**。Matrix 的 provide/proxy、manifests 的自解释、App 的脚手架——都为模型设计。不需要人类解释"这个服务在哪"、"这个依赖怎么注入"——工具自解释。

**以下情况是信号，告诉你该回来看本文档了**：

- 你需要一个 GUI 进程和一个语音进程**同时运行，互相通讯**
- 你需要将 App 部署到**另一台机器**上（开发板、机器人），但希望模型用同样的 CTML 语法控制它
- 你需要**进程崩溃不传播**——一个传感器的故障不能影响对话
- 你需要**不同进程有不同的依赖环境**（视觉需要 opencv，语音需要 miniaudio）
- 你需要理解 **Session 隔离**——同一 workspace 下多个 Ghost 并行运行时如何不互相干扰

遇到这些信号前，只需要知道 Matrix 存在、知道它能跨进程桥接 Channel。遇到之后，带着真问题回来，下面这些概念才会有体感。

---

## 3. 关键概念与探索路径

### 3.1 Cell — 网络中的独立节点

Cell 是 Matrix 网络中独立运行的进程。四种类型：

| 类型 | 身份 | 生命周期管理 |
|------|------|-------------|
| `host` | 主进程，持有 Shell 和 Ghost | Host 自己管理 |
| `app` | AI 创建的独立进程 | Circus 子进程，AI 通过 `apps:start/stop` 控制 |
| `fractal` | 外部 Matrix 接入的节点 | 远程自主管理 |
| `script` | 一次性探查脚本 | 用完即退，host 不感知 |

每个 Cell 有独立地址（`{type}/{fullname}`）、独立 workspace、独立日志。Cell 的存活状态被 Matrix 自动感知——加入和离开不需要手动注册。

**探索**：`moss codex get-interface ghoshell_moss.core.blueprint.matrix` — Cell 抽象与地址约定

### 3.2 provide + proxy — Channel 跨进程的同构映射

Matrix 跨进程通讯的核心模式只有两个动作：

```
进程 A （Provider）              进程 B （Consumer）
───────────────                 ───────────────
matrix.provide_channel(chan)    matrix.channel_proxy(address, name)
        │                               │
        ▼                               ▼
   ZenohProvider 注册              ZenohProxyChannel 挂载
        │                               │
        └─────────── Zenoh ─────────────┘
                      │
               Shell Channel 树中出现 apps.<name>
```

关键：**模型在 CTML 中调用远程 Channel 的语法和本地 Channel 完全相同。** `<apps.my_app:greet/>` ——模型不感知这个 Channel 是在本进程内还是另一台机器上。

Provider 端声明"我有这个能力"，Proxy 端声明"我需要那个能力"。Matrix 在中间桥接。两端可以各自重启，不影响对方。

**探索**：
- `moss codex get-interface ghoshell_moss.core.blueprint.matrix` — `provide_channel()` 与 `channel_proxy()` 签名
- `tests/ghoshell_moss/host/test_app_store_channel_proxy.py` — 跨进程 proxy 的测试用法

### 3.3 Session 与 Scope — 并行不串扰

Session scope 是通讯隔离域。同 scope 内的 Cell 共享通讯总线，跨 scope 隔离。这是"并行化身"概念的工程基础——当未来多个 Ghost 化身并行运行时，各自拥有独立的通讯空间，互不污染。

**探索**：`moss codex get-interface ghoshell_moss.core.blueprint.session` — Session 抽象与 scope 语义

### 3.4 Fractal — 跨机器的反向注册

Matrix 解决单机多进程。Fractal 解决多机——开发板上的传感器、Mac 上的 GUI、远程服务器上的推理节点。

核心模式是**反向注册**：子节点知道父节点地址，主动连接并注册自己。Hub（父节点）接受注册，将远程 Channel 桥接到本地 Shell。

现实场景：机器人开发板跑感知 App → 通过 Fractal 注册到 Mac 上的主 Host → Mac 上的 Ghost 用 CTML 直接控制机器人。模型看到的还是那棵 Channel 树。

**探索**：
- `moss codex get-interface ghoshell_moss.core.blueprint.host` — FractalHub 与 FractalNodeProvider
- `.ai_partners/features/workstreams/2026/05/zenoh-fractal/FEATURE.md` — Fractal 的架构决策与端到端验证
- `tests/ghoshell_moss/host/test_zenoh_fractal.py` — 跨机器集成测试

---

## 4. Manifests 与 Matrix 的关系

Manifests 是声明，Matrix 是执行者。

启动时 Matrix 消费 Host 传入的 mode manifest（mode 文件通过 `from MOSS.manifests.xxx import *` 显式继承全局能力）：遍历 providers → `factory(container)` → 注入 IoC。遍历 bootstrappers → `bootstrap(container)` → 后置初始化。遍历 bringup_apps → 拉起子进程。

**声明和实现分离**：开发者（人类和 AI）只在 manifest 文件中写 Python 实例声明，Matrix 负责发现、注入、组网。换一个 transport 实现（zenoh → 未来的其他协议），声明层不需要改动。

**深入**：`moss docs read workspace-and-mode.md` — manifests 声明体系完整论述

---

## 5. 传输协议可替换

Matrix 的抽象层（Cell、provide/proxy、Session）不绑定具体传输协议。当前默认 transport 可替换——这是架构设计意图，不是临时妥协。未来如果出现更适合 AI OS 场景的通讯协议，切换 transport 不需要改动上层的 Channel 和 Shell。

---

## 6. 深入阅读

| 你想了解 | 去这里 |
|----------|--------|
| Matrix 完整接口 | `moss codex get-interface ghoshell_moss.core.blueprint.matrix` |
| Cell 发现与生命周期 | `moss codex get-interface ghoshell_moss.core.blueprint.matrix` — CellType / Cell |
| Channel 跨进程链路 | `tests/ghoshell_moss/host/test_app_store_channel_proxy.py` |
| Fractal 跨机器验证 | `tests/ghoshell_moss/host/test_zenoh_fractal.py` |
| Session scope 隔离 | `moss codex get-interface ghoshell_moss.core.blueprint.session` |
| 架构拓扑中 Matrix 的定位 | `moss docs read architecture-topology.md` — 2.5 节 |
| Manifests 声明体系 | `moss docs read workspace-and-mode.md` |
| Fractal 设计决策 | `.ai_partners/features/workstreams/2026/05/zenoh-fractal/FEATURE.md` |

---

*由 DeepSeek V4 Pro 与人类工程师在 2026-06-03 讨论、设计并撰写。*
