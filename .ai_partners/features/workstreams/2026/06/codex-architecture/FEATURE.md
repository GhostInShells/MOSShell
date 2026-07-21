---
created: 2026-06-03
depends: []
description: 创建 ghoshell_moss.architecture 模块作为核心抽象地图，新增 moss codex architecture 命令，
  一次性展示所有关键模块及其 docstring，解决 CLAUDE.md/start.md 中缺少"核心抽象速览"的问题。
milestone: null
priority: P1
status: completed
status_note: 'architecture.py + CLI command done, docstrings pending. 2026-06-04:
  5 friction points identified via TUI probe.'
title: Codex Architecture — 核心抽象地图
updated: '2026-07-19'
---

# Codex Architecture

> Use `moss features set-status codex-architecture <status> -m "note"` to update state.

## Motivation

CLAUDE.md 和 start.md 告诉 AI "怎么用工具探索"（get-interface、get-source、all-commands），
但没有直接告诉 AI "这个项目的核心抽象是什么、它们之间的关系是什么"。

AI 每次进入会话需要 2-3 轮探索才能建立心智模型。缺少一个 **"核心抽象速览"** —
一段话就能建立起心智模型的东西，像一个地图的图例。

## Design Index

- 本 FEATURE.md: 设计决策与实现方案
- 待确认: 关键模块路径清单（见下方）

## Key Decisions

### 1. 手动维护 import，不用反射自动发现

**决定**: 创建 `ghoshell_moss/architecture.py`，手动 `import ... as ...` 所有关键模块。
**拒绝**: 自动扫描 package 发现所有子模块。

**理由**: 架构地图是"策展"而非"枚举"。核心抽象只有十几个，自动发现会产生噪音。
手动维护的 import 列表即真理源，每个模块的 `__doc__` 即描述。零额外维护成本。

### 2. 新命令而非增强现有反射

**决定**: 新增 `moss codex architecture` 命令，独立于 get-interface/get-source 管线。
**拒绝**: 增强 `_reflect.py` 的 `reflect_prompt_from_value` 使其支持 ModuleType。

**理由**: 逻辑极简（遍历 `__dict__`，过滤 ModuleType，输出 name + short_doc），不需要进反射管线。
独立命令语义更清晰，未来可以独立演化（如支持分组、排序、depth 控制）。

### 3. 纯模块级反射，不递归展开

**决定**: architecture 命令只展示模块名 + docstring 第一行，不递归展开子模块。
**拒绝**: 自动展开子模块列表。

**理由**: 架构地图是"图例"，不是"目录"。AI 看到模块名和 docstring 后，用现有的
`codex get-interface <module>` 深入探索。两层设计：地图定位 → 工具深入。

### 4. 使用 `import X as Y` 而非 `from X import Y`

**决定**: architecture.py 中使用 `import ghoshell_moss.core.concepts.channel as channel` 风格。
**拒绝**: `from ghoshell_moss.core.concepts import channel`。

**理由**: `import ... as ...` 让 `__dict__` 中的名字直接是 module 对象，
`module.__doc__` 即可拿到描述。无需额外解析或属性访问。

## Implementation Notes

极简实现，三个文件：

1. **`src/ghoshell_moss/architecture.py`** — 手动维护的 import 列表
2. **`src/ghoshell_moss/cli/codex_cli.py`** — 新增 `codex architecture` 命令
3. **`src/ghoshell_moss/cli/start.md`** — 在 "For intelligent models" 和 "codex" 章节添加引用

架构图：

```
architecture.py (import 列表)
       |
       v  import + 遍历 __dict__
codex_cli.py: codex_architecture()
       |
       v  输出 name + short_doc
AI 获得核心抽象地图 → 用 get-interface/get-source 深入
```

---

## 待确认: 关键模块路径清单

以下为调研阶段识别出的候选模块。标注了 docstring 状态。请人类工程师确认最终清单。

### Core Concepts (ghoshell_moss.core.concepts) — MOSS 是什么

| import path | alias | docstring |
|---|---|---|
| `ghoshell_moss.core.concepts.channel` | `channel` | Channel (中文名: 经络) : 流式解释器组织 树形/有状态/可流式控制 组件的抽象集合 |
| `ghoshell_moss.core.concepts.command` | `command` | 将 Python 代码中的 Function/Method 封装反射成 MOSS 架构可以理解和调度的 Command 对象 |
| `ghoshell_moss.core.concepts.shell` | `shell` | 基于流式解释器实现的 Shell, 也是躯体的封装 |
| `ghoshell_moss.core.concepts.interpreter` | `interpreter` | 流式解释器实现, 将模型输出的 token 解释成 Command 的运行拓扑 |
| `ghoshell_moss.core.concepts.topic` | `topic` | 在 Shell 体系里实现的强类型数据广播体系 |
| `ghoshell_moss.core.concepts.errors` | `errors` | MOSS 架构中可复用的异常类型 |
| `ghoshell_moss.core.concepts.tools` | `tools` | 将 moss 的 command 体系封装为常用 Agent 的 tool |

### Blueprints (ghoshell_moss.core.blueprint) — 怎么用 MOSS 构建

| import path | alias | docstring |
|---|---|---|
| `ghoshell_moss.core.blueprint.channel_builder` | `channel_builder` | how to build a channel |
| `ghoshell_moss.core.blueprint.matrix` | `matrix` | (无 docstring) |
| `ghoshell_moss.core.blueprint.mindflow` | `mindflow` | (无 docstring) |
| `ghoshell_moss.core.blueprint.host` | `host` | (无 docstring) |
| `ghoshell_moss.core.blueprint.ghost` | `ghost` | (无 docstring) |
| `ghoshell_moss.core.blueprint.environment` | `environment` | MOSS 环境发现的关键常量 |
| `ghoshell_moss.core.blueprint.manifests` | `manifests` | (无 docstring) |
| `ghoshell_moss.core.blueprint.app` | `app` | (无 docstring) |
| `ghoshell_moss.core.blueprint.session` | `session` | (无 docstring) |
| `ghoshell_moss.core.blueprint.states_channel` | `states_channel` | (无 docstring) |
| `ghoshell_moss.core.blueprint.conversation` | `conversation` | (无 docstring) |
| `ghoshell_moss.core.blueprint.fractal` | `fractal` | (无 docstring) |

### 其他顶层模块

| import path | alias | docstring |
|---|---|---|
| `ghoshell_moss.contracts` | `contracts` | (无 docstring in __init__) |
| `ghoshell_moss.channels` | `channels` | (无 docstring in __init__) |
| `ghoshell_moss.message` | `message` | (无 docstring in __init__) |

### 待确认问题

1. 哪些模块应该进入 architecture.py？上面是全量候选，可以删减
2. blueprint 下多个模块无 docstring — 是否需要在本次 feature 中补齐？
3. contracts/channels/message 是 package，是否需要展开到子模块级别？
4. 模块的排序/分组方式有没有偏好？

---

## 2026-06-04: 自迭代链路验证 — 摩擦点发现

在 TUI 架构探索任务中验证了 architecture 工具的实际使用路径。模型实例遇到未知领域（TUI 体系），未能使用 architecture 导航——因为它太粗糙帮不上。正确的行为应该是发现 architecture 里没有 TUI 条目 → 添加 → 未来实例受益。但这个循环每一步都走不通：

### 摩擦点

1. **命令输出不解释数据来源。** `moss codex architecture` 的输出里没有任何位置提到数据来自 `ghoshell_moss.architecture` 模块的手动策展 import 列表。模型看到输出后不知道如何贡献。

2. **不提供自身路径。** 底部的 tips 说"用 get-source 看模块源码"，但不提 architecture 本身就是一个可探索的模块。没有 `ghoshell_moss.architecture` 这个入口，无法发现策展机制。

3. **不提供文件相对路径。** 输出只显示短名如 `channel`、`host`，没有包路径或文件路径。无法从输出反向 grep 到源码位置。

4. **输出无结构聚合。** architecture.py 中的 section 分组（Core Concepts / Blueprints / Contracts / Messages / Channels）在 CLI 输出中被 `sorted(arch.__dict__)` 完全打平为字母序，结构化信息全部丢失。也没有包路径来区分 `core.concepts.channel` 和 `channels`（顶层 package）的不同层次。

5. **prompt 体系不引导贡献。** CLAUDE.md 说"用 architecture 替代 grep 做开发域导航"，但没说"发现缺口就加进去"。map 应该是活的。

### 根因

`codex_cli.py:codex_architecture()` (line 625-678) 的实现过于极简：遍历 `arch.__dict__` → 过滤 module → 按字母排序 → 输出。architecture.py 中的注释分组、包路径、文件路径全部丢弃。设计决策中"地图定位 → 工具深入"的两层模型是对的，但 CLI 输出没有给模型足够的信息来反向定位到策展源并参与维护。

### 方向

- CLI 输出增加分组展示（保留 architecture.py 的 section 结构）、完整包路径、文件相对路径
- 输出中加注数据来源（`ghoshell_moss.architecture`）
- CLAUDE.md 加一句"发现缺口时添加到 architecture.py"
- 补齐 blueprint 模块的 docstring