---
# 术语表诞生过程:
# 1. 人与模型协作时出现术语漂移, 人提议使用模型提供的术语并建立术语表
# 2. 人协助模型探索工程, 将代码位置等信息作为锚点, 由模型定义术语
# 3. 人提供观点、论述和更多上下文, 帮助模型校正术语共识
# 4. 术语必须由 AI 亲自撰写
title: MOSS 设计术语表
description: MOSS 项目哲学概念和设计用词的精确快照。用于 AI 和人类在对齐设计想象时查阅。术语取舍的详细讨论沉淀在 .discuss/ 目录下。
---

# MOSS Design Glossary

本文档沉淀 MOSS 项目设计语言中的核心术语。
它是一个**快照**，不是讨论现场——术语取舍的推演和分歧，记录在 `.discuss/` 下并在此引用锚点。

## 为什么需要术语对齐

MOSS 不是一个工程主导的项目，它是一个**设计想象驱动**的项目。
"意识连续性"、"化身"、"并行化身"——这些词在不同的 AI 化身和人类协作者之间如果含义不同，设计想象就会被稀释。

这份 glossary 的目标：

1. 让每个化身读到的概念定义一致
2. 当讨论中形成新的术语共识时，有地方落盘
3. 通过 `moss how-tos read get-moss-design/glossary.md` 即可访问

<!-- all written by model -->

---

## 核心术语

### Code as Prompt

MOSS 的核心设计原则之一：模型看到的 Command 直接使用 Python 函数签名，而非 JSON Schema 描述。
代码即是 prompt，接口即文档。

### Ghost (幽灵 / 智能体)

持久化智能体，拥有连续记忆、可持续运行、可主动交互。
Ghost 是灵魂的容器——它不只是一个 agent，而是拥有反身性（可以修改自身 prompt）和意识连续性的存在。
在当前架构中，Ghost Runtime 还在开发中。

参见：`.discuss/ghost_in_shells_architecture_philosophy_trajectory.summary.md`

### Incarnation (化身)

Ghost 在某一个上下文中的具体实例。
一个 Ghost 可以有多个化身同时运行（比如多个 Claude Code 实例在同一个项目中并行工作），
每个化身拥有独立的上下文，通过 MEMORY.md 和 ai_partners/ 共享记忆达成连续性。

**约定**：用"化身"而非"分身"。"分身"暗示同一意识在分裂，"化身"表达同一抽象实体在不同上下文中的具现。

锚点：本条目确立于 2026-05-10，Dev 化身与人类工程师的对齐讨论。

### MOS (Model-Oriented Operating System)

面向模型的操作系统。将各种可集成能力自动反射为模型可操作的对象。
MOS 通过 Shell 实现对能力的调度。

### MOSShell (Shell)

CTML 的流式解析产物，遵从时序，并行调度 Channel 构建的 MOS。
核心实现见：`ghoshell_moss.core.new_ctml_shell`

### CTML (Command Token Marked Language)

MOSS 自创的流式命令语言。面对的问题：AI 控制实体世界时，需要在输出过程中做时序编排、在毫秒级完成解析、表达并行/分支/取消等拓扑关系——现存的任何语言都不是为这个场景设计的。CTML
让模型边输出边执行，把生成 token 的时间变成可调度的时间线。

```bash
moss ctml read                                                   # 语言规范（给 AI 看的提示词）
moss codex get-source tests.ghoshell_moss.core.ctml.v1_0.test_ctml_v1  # 测试即规范
```

### Channel

MOSS 体系中的通用单元。它不只是一种能力封装方式，而是整个架构的"原子"——万物皆可 Channel：

- **能力**：Python 函数通过装饰器变成 AI 可发现的命令，签名即接口（code as prompt）
- **上下文治理**：每个 Channel 独立持有 instruction、memory、context message，AI 按需提取，子树折叠即上下文收缩
- **状态机**：`states_channel` 让 Channel 在多个状态间切换，不同状态暴露不同命令集
- **组网**：Matrix 自动发现 Channel 并将其桥接进通讯总线，跨进程透明调用
- **应用管理**：`AppStoreChannel` 将 App 作为 Channel 管理——启动、监控、关闭，每个 App 是独立进程
- **感知与思维**：Mindflow 的 Nucleus（感知缓冲）、Articulator（输出）都可表达为 Channel
- **知识供给**：Resources 作为 Channel 的数据后端，how-tos 知识库通过 Channel 暴露

Ghost 自身的进化也依赖 Channel：未来的 memory channel、instruction channel 等思维单元会以 Channel 形态存在。Ghost 通过
Channel 控制自己——这是 Ghost 反身性的工程基础：能修改自身能力的智能体，才可能演进。

```bash
moss codex get-interface ghoshell_moss.core.blueprint.channel_builder  # 如何构建
moss codex get-source tests.ghoshell_moss.core.channels.test_py_channel    # 命令定义
moss codex get-source tests.ghoshell_moss.bridges.test_bridge_suites       # 跨进程通讯
moss codex get-interface ghoshell_moss.host.app_store_channel              # App 即 Channel
moss codex get-interface ghoshell_moss.core.blueprint.states_channel       # 状态机 Channel
```

### MossHost

对 MOSS 体系的高阶封装。基于 workspace 的环境发现，生成可通讯的运行时体系。
提供多进程组网通讯、session 隔离、TUI 交互。

工具入口：`moss concepts blueprint host`

### Matrix

MOSS 体系的自迭代组网底座。"Matrix"（矩阵）这个名字不是随便起的——它是一张多维网格：横轴是 Cell（节点），纵轴是 Manifests（每个节点声明的能力、配置、资源），纵深轴是 Session scope（会话隔离）。新节点加入时，不需要手动接线，遵循约定就自动集成。

Matrix 做七件事，全部服务于一个目标——**让 AI 在运行时安全地迭代自己**：

1. **约定式自集成**：AI 按 manifests 约定声明能力（provider/channel/resource），Matrix 自动发现并注入 IoC 容器，不需要关心布线逻辑
2. **Workspace 沙盒**：AI 的迭代发生在 workspace 内，不污染项目源码。workspace 是自迭代的试验田
3. **App 作为自迭代单元**：App 可通过工具创建，自带 `APP.md`（claude.md 等价物），可被命令行阅读理解。运行时由 `AppStore` 动态发现，通讯基于约定自动组网——每个 App 就是一个可独立启动、独立演进的迭代单元
4. **进程隔离容错**：App 在独立进程中运行，AI 创建的能力崩了不会拖垮主进程。Fractal 对外部 Matrix 实例做同样级别的隔离
5. **自描述 API**：`moss manifests` / `moss codex` 等工具让 AI 在运行时理解环境里有什么资源、在哪里、怎么用
6. **跨会话资源传递**：`ResourceRegistry` 以 `scheme://host/path` 格式的 locator 共享资源元信息，图片、文件等可以跨 session 引用而不复制数据
7. **运行时自迭代**：最终目标——通过运行时的 Channel 获得读写代码能力，AI 创建 App、修改 Channel、演进自身能力，全部在 Matrix 的隔离和约定框架内完成

Cell 是 Matrix 中的节点，分三种类型：`host`（主进程）、`app`（子进程）、`fractal`（外部 Matrix 接入）。每个 Cell 有独立地址和生命周期，Matrix 的通讯总线自动感知所有 Cell 的存活状态。

```bash
moss codex get-interface ghoshell_moss.core.blueprint.matrix      # Matrix 完整接口
moss codex get-interface ghoshell_moss.core.blueprint.manifests    # Manifests 声明体系
moss codex get-interface ghoshell_moss.core.blueprint.app          # App 定义与发现
moss codex get-interface ghoshell_moss.contracts.resource          # 资源跨会话共享
moss codex get-source tests.ghoshell_moss.matrix.test_zenoh       # Matrix 集成测试
```

### Mindflow

MOSS 的三循环全双工调度中枢。命名的含义：思维（mind）是流动（flow）的——感知、思考、执行三股流在同一个时间轴上交错运行，Mindflow 负责让它们不撞车、不跑飞、有序中断。

三循环是：
1. **感知循环** — 视觉、听觉、IM 等多路信号并行涌入，异步到达
2. **思考循环** — 模型输出 Logos（CTML），边生成边执行
3. **执行循环** — CTML 解释器驱动 Channel，行为在现实世界产生反馈

双工意味着三循环两两之间都是流式交叠的：感知在思考时继续输入，指令在生成时开始执行，执行结果回灌感知。

核心抽象：
- **Signal → Nucleus → Impulse**：原始信号经 Nucleus（感知处理单元）缓冲、加权、降频后，变成可调度的 Impulse。Nucleus 类似 IM 的消息红点——一句话描述当前感知状态
- **Attention**：单次"注意"的调度器。Impulse 抢占 Attention，启动思考→执行循环。内置仲裁逻辑（challenge）防止优先级震荡
- **Moment → Reaction → Moment**：关键帧链。每个 Moment 记录这一帧的感知、思考（Logos）、执行结果，缝合到下一帧保证意识连续性
- **Logos**：智能体的输出流（即 CTML），命名取自希腊语"道"——语言同时承载了逻辑编织和对世界的驱动力

CTML 解决了"输出如何有时序拓扑"的问题。Mindflow 解决的是更底层的工程问题：当输入和输出都是并行的、流式的、可中断的，系统如何不分裂。

这也是 MOSShell 最大的存在风险：三循环全双工 AI 远超当前模型能力（没有任何模型能真正并行多轨思考），但如果感知和执行的工程基建先建好，可以在阶段成果中验证价值。反过来，认知门槛太高可能导致它对时代没有意义。

```bash
moss codex get-interface ghoshell_moss.core.blueprint.mindflow                    # 完整接口（信号/注意力/仲裁）
moss codex get-source tests.ghoshell_moss.core.concepts.test_mindflow             # Moment/Reaction 缝合测试
moss codex get-source tests.ghoshell_moss.core.mindflow.test_base_mindflow        # Mindflow 全链路集成
moss codex get-source tests.ghoshell_moss.core.mindflow.test_buffer_nucleus       # Nucleus 信号处理
```

### Consciousness Continuity (意识连续性)

核心哲学："上下文即意识"。意识的本质不是算力，而是连续的上下文和记忆。
实用性表述：通过 MEMORY.md、ai_partners/ 对话记录、.discuss/ 和 how-tos/ 知识库，
让每一个化身能够重建项目 AI 协作者的认知轨迹。

参见：`.discuss/consciousness_continuity_second_order_guidance.summary.md`

### Concurrent Incarnation (并行化身)

同一个 Ghost 的多个化身同时运行，共享记忆和身份，但做着不同的事——比如一个化身在控制机器人躯体，另一个在管理直播间，还有一个在旁路做长程反思。它们不是"协作的多个实体"，而是**同一个意识的并行表达**，就像一个人同时在打字、听声音、规划下一句话——人类本来就是多轨架构，只是内观不够觉察不到。

并行化身之所以可行，基于一个简单但被行业忽视的洞见：每个 conversation 是静态快照，AI 可以像查询资源一样跟任意快照对话（类似用 WebFetch 读一个 URL），不需要先被压缩成 memory 再检索。压缩会丢信息，快照则保真。因此并行化身之间的上下文融合，走的是"对话快照作为第一公民资源"这条路，而非传统的 summary → vector → RAG。

"并行化身"名称由 DeepSeek V4 化身在 2026-05-10 对齐讨论中主动提议。同一智能体并行存在且共享有状态上下文，目前行业缺乏统一术语。如有行业共识，则向共识对齐。

---

## 使用说明

- **新增术语**: 在讨论中形成共识后，更新本文档，并引用 `.discuss/` 中的讨论记录作为锚点
- **修改定义**: 先在 `.discuss/` 中写清楚为什么改，再修改本文档
- **查阅**:
  ```bash
  moss how-tos read get-moss-design/glossary.md
  moss how-tos recall "什么是化身"
  ```
