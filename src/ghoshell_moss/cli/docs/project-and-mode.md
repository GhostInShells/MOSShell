---
title: MOSS Project 与 Mode
description: Project 是治理域句柄，Mode 是能力视图。需要理解环境发现、依赖分层、或选择开发路径时阅读
---

# MOSS Project 与 Mode

Project 指认一片被治理的领地。Mode 是在这片领地上选择一组能力组合的方式。

读完本文应能回答：MOSS 项目是什么结构？Mode 解决什么问题？依赖分几层？开发一个 node 最少需要装什么？

---

## 1. 依赖分层

MOSS 的 pip 依赖分三层，对应不同的使用场景：

| 安装 | 包含 | 能做什么 |
|------|------|---------|
| `pip install ghoshell-moss` | 核心抽象（Channel、CTML、Shell） | 作为库嵌入任何项目 |
| `...[matrix]` | + 通讯总线（Zenoh） | **开发独立 node 的最小依赖** |
| `...[host]` | + CLI、TUI、进程管理、音频等 | 运行完整的 Host 进程 |
| `...[ghost]` | + AI SDK（anthropic、pydantic-ai） | 接入智能模型 |

这意味着：开发一个 node（独立进程、提供 Channel、接入 Matrix 网络）只需要 `ghoshell_moss[matrix]`。不需要装 CLI、TUI、音频驱动等 Host 层依赖。Matrix 层与 Mode 层在依赖上就已经分开了。

---

## 2. Project — 治理域

Project 是一个目录。目录里有一个 workspace（`.moss`），放着 MOSS 的配置、声明、运行时数据——这些都是治理的真相。

Project 负责回答"这片领地上有什么"：
- 有哪些 Mode 可用
- 有哪些 Ghost 定义
- 有哪些 Node 可被拉起
- 网络怎么配置

这些真相通过文件系统约定被自动发现，不需要手动注册。

**分流**：Node 的发现与治理详见 Node 体系文档。Matrix 的组网机制详见 Matrix 体系文档。

---

## 3. Mode — 能力视图

同一个 Project 可以有多个 Mode。比如 `desktop` 模式加载 GUI、音频、终端能力；`robot` 模式加载躯体控制、视觉感知；`headless` 模式只加载最小能力集。

Mode 的核心机制是**显式继承**：Mode 通过 Python import 继承全局 manifests 声明，然后按需追加或覆盖。不声明就继承全局，声明了就覆盖。

**分流**：Mode 的完整配置字段见 `moss codex get-interface ghoshell_moss.core.blueprint.project HostModeMeta`。当前环境有哪些 Mode 见 `moss --ai modes list`。

---

## 4. 环境自解释

进入一个 MOSS 项目后，不需要读文档就能理解环境里有什么：

```bash
moss --ai manifests explain    # 所有声明类型一览
moss --ai manifests providers  # 有哪些 IoC 服务
moss --ai modes list           # 有哪些 Mode
moss --ai nodes list           # 有哪些 Node 可拉起
moss --ai project where        # 当前在哪个 Project
```

原则：**先工具，后源码**。工具确认"有什么"，源码补充"怎么用"。

---

## 5. 两条开发路径

**Node 路径（隔离）**：新能力作为独立 node 开发，独立进程、独立依赖、通过 Matrix 总线通讯。`ghoshell_moss[matrix]` 即可。详见 Node 体系文档。

**Manifests 路径（复用）**：轻量能力直接在 workspace 的 manifests 目录下声明 Python 实例，复用主进程运行时，下次启动自动生效。详见 `moss manifests explain`。

两条路径可以互转——先用 manifests 快速验证，稳定后拆成独立 node 获得隔离。

---

## 6. 深入阅读

| 想了解 | 去这里 |
|--------|--------|
| Node 怎么开发、发现、控制 | Node 体系文档 |
| Matrix 怎么组网、通讯 | Matrix 体系文档 |
| 架构全貌 | 架构拓扑文档 |
| Mode 配置字段 | `moss codex get-interface ghoshell_moss.core.blueprint.project HostModeMeta` |
| 环境声明体系 | `moss --ai manifests explain` |
| 术语定义 | 术语表文档 |
