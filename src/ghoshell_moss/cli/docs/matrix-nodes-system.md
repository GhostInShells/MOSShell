---
title: Matrix Node 体系
description: Node 是 Matrix 网络中的功能节点——可被发现、被 CLI 和模型控制、跨进程通讯。需要创建或理解 Node 时阅读
---

# Matrix Node 体系

Node 是 Matrix 网络中提供具体能力的进程单元。它是 Ghost 可插拔的器官——独立进程、独立依赖、Matrix 总线通讯。

读完本文你应该能回答：Node 怎么被系统发现？有哪些方式控制它？Node 之间怎么通讯？开发一个 Node 的最小路径是什么？

---

## 1. Node 在 Matrix 中的位置

Matrix 网络中有两种角色：**host**（中心节点，组织所有能力供 Ghost 驱动）和 **node**（功能节点，提供具体能力）。host 与 node 不是两种程序——它们跑的是同一个 `Matrix.discover()`，角色由环境变量继承决定。

host/node 的相互发现和组网机制见 Matrix 体系文档：

```bash
moss docs read matrix-system.md
```

本文聚焦 node 本身：怎么声明、怎么控制、怎么通讯、怎么开发。

---

## 2. 发现 — 目录即声明

Node 通过目录约定被系统自动发现。一个目录下放一份声明文件，描述"这是什么、怎么启动"，Matrix 的 NodeManager 扫描约定路径后即可发现并拉起。

声明文件承载了 node 的元信息：名称、分类标签、启动命令、是否为单例等。正文是给 AI 看的 instruction。

具体声明格式和字段通过 codex 自省：

```bash
moss codex get-interface ghoshell_moss.core.blueprint.cell NodeManifest
```

声明文件的发现路径由 Mode 配置。当前 Mode 下有哪些 node 可被拉起：

```bash
moss --ai nodes list
```

---

## 3. 控制 — 三面合一

Node 的生命周期有三个控制面，共享同一套治理动词（run/stop），只是入口不同：

| 控制面 | 入口 | 适用场景 |
|--------|------|---------|
| CLI | `moss nodes` 命令组 | 人类调试、脚本 |
| Matrix API | `matrix.run_node(target)` | 父进程以本 Matrix 为治理域拉起子 node |
| 模型 (CTML) | 通过 Channel 暴露的治理命令 | Ghost 在运行时自迭代——创建、启动、调用新能力 |

三个面共享同一个 spawn 咽喉，差异只在"谁发起的"——CLI 进程、host 进程、还是模型通过 channel。治理逻辑是一套。

CLI 命令的完整列表：

```bash
moss --ai all-commands --group nodes
```

Matrix API 入口：

```bash
moss codex get-interface ghoshell_moss.core.blueprint.matrix  # run_node, handled_cells
```

---

## 4. 通讯 — 跨进程、跨环境、跨网络

### 4.1 跨进程：Session 总线

Node 启动后接入 Matrix 网络，自动获得 Session 通讯能力。五种通讯原语——topic、stream、signal、output、file——共享同一条总线。Node 可以向总线发 signal 驱动 Ghost 的注意力、写 stream 输出实时数据、订阅 topic 感知其他 node 的状态变化。

```bash
moss codex get-interface ghoshell_moss.core.blueprint.session
```

### 4.2 提供 Channel 给模型

Node 可以通过 `matrix.provide_channel(channel)` 将自己的能力以 Channel 树形态暴露给模型。模型在 CTML 中调用远程 Channel 的语法和本地完全相同——不感知这个能力跑在哪个进程、哪台机器上。

Channel 构建：

```bash
moss codex get-interface ghoshell_moss.core.blueprint.channel_builder
```

### 4.3 跨网络：scope 对齐即组网

同一台机器上的 node 通过 Session scope 天然互通。跨机器时，每台机器有自己的 host 治理本地 node，A 机器需要 B 机器的能力时，B 将 Channel 子树 provide 到网络中，A accept 后挂载。只要 scope 对齐，不同机器上的 node 就像在同一个网络中。

跨机器是组合同构网络的组合，不是对远端 spawn 的参数。详见 Matrix 体系文档 §3.6。

---

## 5. 开发 — 最小路径

开发一个可被模型发现和调用的 node，只需要理解四个 blueprint：

| 需要知道 | 去哪看 |
|----------|--------|
| Channel 怎么构建（命令定义、上下文注入） | `moss codex get-interface ghoshell_moss.core.blueprint.channel_builder` |
| Matrix 怎么入网（身份、provide、生命周期） | `moss codex get-interface ghoshell_moss.core.blueprint.matrix` |
| Session 怎么通讯（发 signal、写 stream、订阅 topic） | `moss codex get-interface ghoshell_moss.core.blueprint.session` |
| CTML 怎么让模型控制你的 Channel | `moss ctml read` |

这四个 blueprint 是 code as prompt——Python 函数签名就是接口文档。读完就能写出一个标准 node 入口：

```python
from ghoshell_moss.core.blueprint.channel_builder import new_channel
from ghoshell_moss.core.blueprint.matrix import Matrix

async def main(matrix: Matrix):
    # 构建 Channel → 注册到 Matrix → 模型可见
    await matrix.provide_channel(channel)

if __name__ == "__main__":
    Matrix.discover().run(main)
```

node 可以有独立依赖（视觉 node 装 opencv，GUI node 装 PyQt6，互不污染）。依赖隔离机制和启动方式通过 ExecSpec 声明，细节在 NodeManifest 接口中自省。

---

## 6. 深入探索

| 想了解 | 去这里 |
|--------|--------|
| Node 声明格式与字段 | `moss codex get-interface ghoshell_moss.core.blueprint.cell NodeManifest` |
| Node 发现与管理 API | `moss codex get-interface ghoshell_moss.core.blueprint.cell NodeManager` |
| Matrix 组网与 host/node 关系 | `moss docs read matrix-system.md` |
| Channel 构建完整知识 | `moss codex get-interface ghoshell_moss.core.blueprint.channel_builder` |
| Session 五种通讯原语 | `moss codex get-interface ghoshell_moss.core.blueprint.session` |
| CLI 控制面 | `moss --ai all-commands --group nodes` |
