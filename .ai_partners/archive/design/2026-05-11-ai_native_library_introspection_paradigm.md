# AI 原生库自省范式：库应自带 AI 可调用的代码反射工具

**日期**: 2026-05-11
**来源**: 人类工程师与 DeepSeek V4 在讨论 MOSS CLI 工具集定位时提炼

## 问题

AI coding agent 理解一个第三方库，当前只有两条路径：

1. **读文档**：人类导向的静态文档（Sphinx/ReadTheDocs），版本滞后、粒度不可控，AI 需要大量 token 才能定位到有效信息
2. **读源码**：直接从源码文件推断 API 契约，token 消耗极大，容易在实现细节中迷失

两条路径的共同缺陷是**被动**——AI 去适应库的输出格式，而非库主动为 AI 优化信息结构。

## 核心观点

**库的 CLI 工具集应成为库与 AI 之间的第一接口。** 库通过代码反射，在 CLI 层面提供结构化的、分层的、精确的自省能力，AI agent 直接调用这些 CLI 命令来理解库的架构和契约。

这不是又一个文档系统，而是**库的 AI 原生接口层**。

## MOSS 的实践

MOSS 当前实现了以下要素：

### 三层自省体系

```
moss codex get-interface [modulepath]   # "有什么" — 获取模块的接口契约 (类型签名/Field/方法)
moss codex get-source [modulepath]      # "怎么实现" — 获取类的完整源码
moss concepts [category]                # "怎么组织" — 获取架构概念的层次抽象
```

### AI 优化输出

`--ai` 全局 flag：剥离 rich 视觉排版（ANSI/表格/Syntax 高亮），输出纯文本，大幅节省 token。

### CLI 命令发现

`moss --ai all-commands`：AI 无需多轮 `--help` 探索，一次性获取完整命令树。

### 与 CLAUDE.md 的互补

CLAUDE.md 是静态引导（"这个项目长什么样"），CLI 工具是动态查询（"你想了解什么就反射什么"）。两者的关系：

- CLAUDE.md 写入：调用 `moss --ai all-commands` 让 AI 知道有哪些工具可用
- 运行时：AI 按需调用 `moss codex get-interface` 获取精确契约

### 下游项目可移植性

当 MOSS 作为第三方库被引入其他项目时，该项目的 Claude Code / Gemini CLI 可以直接调用 `moss` 命令理解 MOSS 架构，不需要文档网站，不需要重新训练。

这形成了**库的 AI 自描述闭环**。

## 设计原则

1. **代码即真相**：反射自代码实现，不存在文档与实现不同步的问题
2. **分层输出**：接口契约 (interface) / 源码 (source) / 概念 (concepts) 三层分离，AI 按需获取，不做全量 dump
3. **AI 优先输出格式**：提供 AI 优化版本（`--ai` flag），与人类可读版本共用一个实现
4. **命令自发现**：AI 通过单一命令即可了解完整的 CLI 工具树
5. **库独立**：自省工具作为库的一部分，不依赖外部服务或云端协议

## 与 CLI Flow 概念的关系

`cli_flow_concept.md` 讨论的是 CLI 生成 CLAUDE.md 以铺设行动路径。本概念是对其的上层抽象：

| | CLI Flow | AI 原生库自省 |
|---|---|---|
| 作用域 | 单个项目内 | 库的对外接口 |
| 载体 | 目录 + CLAUDE.md | CLI 命令集 |
| 方向 | 为进入目录的 AI 铺设上下文 | 为使用库的 AI 提供查询接口 |
| 实现 | `moss app create` 生成骨架 | `moss codex get-interface` 反射代码 |

两者同属一个更根本的理念：**工具集本身应成为 AI 的原生信息界面**。

## 未来方向

CLAUDE.md 中关于 MOSS 架构的静态描述，可以逐步迁移为 CLI 命令的输出（如 `moss ai-startup`），使 CLAUDE.md 保持精简，只保留项目级指令和指针（"运行这个命令了解架构"）。

这样，CLAUDE.md 和 CLI 工具集形成完整的信息梯度：从静态指针到动态查询，从项目概览到精确契约。
