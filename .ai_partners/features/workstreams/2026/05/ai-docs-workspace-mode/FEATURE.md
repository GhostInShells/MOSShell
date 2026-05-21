---
created: 2026-05-21
depends: []
description: 为 docs/ai/ 撰写 workspace 与 moss mode 的系统论述文档，并在过程中修复实现不一致。是架构拓扑文档后最重要的一篇
  docs。
milestone: null
priority: P0
status: in-progress
status_note: 文档初稿完成 + explain CLI 已实现，待人类统一改版 + stub 注释 + what-is-workspace 拆分
title: Ai Docs Workspace & Mode
updated: '2026-05-21'
---

# Ai Docs Workspace & Mode

> Use `moss features set-status ai-docs-workspace-mode <status> -m "note"` to update state.
> See [TOPOLOGY.md](TOPOLOGY.md) for directory layout and [README.md](README.md) for the full convention.

## Motivation

`moss docs ai/` 目前有三篇，拓扑文档中 workspace（2.4 节）仅一段概述，mode 没有独立章节。
`moss how-tos get-moss-design/what-is-workspace.md` 是从设计文档降级为 how-to 的过渡产物，
需要正式拆分为：架构论述进 docs/ai/，操作步骤留在 how-tos（后续重写）。

这篇文档是 MOSS 项目中最重要的架构文档之一——workspace 和 mode 是智能模型进入项目的第一个接触面。

## Key Decisions

### 1. docs 与 how-tos 的边界

- **docs**：系统论述。面向需要理解架构设计的智能模型，回答 what-is / why / 知识探索路径 / 工具与意义。
- **how-tos**：零知识引导。面向没有背景知识的模型（没人预训练 MOSS），带盲从心态 step by step 干活。

`what-is-workspace.md` 在本 feature 完成后删除，内容拆分到两边。

### 2. 智能模型是第一读者

文档围绕"智能模型作为 MOSS 环境的第一开发者和讲解者"来写。三个核心能力诉求：

1. **最小知识理解运行时**：配合工具（manifests / modes / apps），不需要调研复杂生命周期就能建立认知，可以向人类讲解
2. **方便修改**：文件级关注点分离——不同文件治理不同东西。package-module 等发现机制让修改路径可预测。stub 文件需要加头部注释做自解释
3. **开发时/运行时自迭代**：必要的隔离（尤其修改隔离）+ 通过工具反查必要知识

### 3. 贯穿五个用户故事

1. 了解 MOSS 已集成的能力 → 模型通过 manifests / modes / apps 快速理解
2. 开发新功能 → 模型知道如何独立集成（目录拓扑介绍是关键）
3. 知道启动什么 → 模型通过 modes 等命令看一眼就明白
4. 两条开发路径：
   - app → mode 集成（依赖/运行时隔离）
   - src → manifests 集成（依赖/运行时复用）
5. 从最佳实践上手，到必要知识扩展——不全貌到执行，先做事再理解

### 4. 文档结构共识

what-is → why → 用户故事 → 核心机制（自举层/声明层/视图层） → 启动链路 → 知识探索路径 → 相邻概念关联

contracts、providers、nucleus、ghost 等概念各有一句话 + 深入探索路径，不展开但不断链。

### 5. 需要对齐的代码改动

- stub 文件添加头部注释做自解释（本轮任务一部分）
- 文档写作过程中发现的不一致，在 feature 内记录并修复，或新建子任务
- 这是一个长 feature，可以反复对齐

### 6. CLI 重组不影响本文档

CLI 命令归类重组与文档写作并行进行，各自独立。文档引用 Python import path 和概念为主，CLI 命令路径改动后更新"知识入口"部分即可。

## Design Index

- 上游参考：`moss docs ai/architecture-topology.md` 2.4 节
- 待删除：`moss how-tos get-moss-design/what-is-workspace.md`
- 关键源码：
  - `ghoshell_moss.core.blueprint.environment` — Environment 发现与 MossMeta
  - `ghoshell_moss.core.blueprint.manifests` — Manifests 声明体系
  - `ghoshell_moss.host.manifests` — Host 层 manifests 实现
  - `ghoshell_moss.host.impl:Host` — Host 启动链路
  - `ghoshell_moss.host.stubs.workspace` — workspace stub 模板
  - `ghoshell_moss.host.moss_runtime:MossRuntimeImpl` — Runtime 执行层

## Discovered Issues

### 1. `search_channels_from_package` key 不一致

`src/ghoshell_moss/host/manifests/channels.py:28` — channels 以 Python 变量名为键（`found[name] = obj`），而 primitives 以 `Command.name()` 为键。channels 应该也使用 `Channel.name()` 作为键，与 primitives 保持一致。人类工程师确认这是 bug，以为已修复。

### 2. Manifests 自解释体系（已实现）

- `Manifests.explain()` — 基类通用模板 + PackageManifests/MergedManifests 各自组装
- `moss manifests explain` — CLI 唯一真相入口，接受 `--mode`
- 文档 4.2 节引用 CLI 而非硬编码表格

### 3. providers vs contracts 关系（已写入文档 4.2 节）

- providers 服务于 contracts — 添加 Provider 声明 contract 的绑定方式
- `moss manifests contracts` 看"可以拿到什么"（IoC 已绑定列表）
- `moss manifests providers` 看"由谁生产"（工厂声明）

## TODO (不随 commit 带入)

- 人类工程师将统一改一版文档（措辞、节奏、详略）
- stub 文件头部注释（channels.py, primitives.py, configs.py, resources.py, topics.py, providers.py 需要自解释注释）
- `what-is-workspace.md` 需删除并重写为真正的 how-to

## Implementation Notes

- 新文档路径：`src/ghoshell_moss/cli/docs/ai/workspace-and-mode.md`
- 初稿已完成，人类审阅对齐中
- 文档内容直接面向 AI 读者，写中文