# Design Language Alignment: Glossary and Term Consensus

## Participants
- Human Engineer
- DeepSeek V4 via Claude Code (Dev 化身)

## Date
2026-05-10

## Context

在协作过程中，人类工程师和 AI 化身之间出现了术语漂移——同一个概念用不同的词表达（"分身" vs "化身"），或者同一个词在不同语境下含义不同。这稀释了 MOSS 作为"设计想象驱动"项目的概念精度。

讨论的起点是改进 `how-to-make-how-to.md`，但很快转向了更深层的问题：MOSS 的核心术语需要一张快照。人类工程师提出创建 `how-tos/get-moss-design/glossary.md`，并要求：术语由 AI 亲自撰写，人提供设计意图和代码锚点帮助 AI 校正理解。

整个 session 形成了校准循环：模型读代码 → 写初稿 → 人指出"why" → 模型扩大探索面 → 重写 → 人给出更多代码位置 → 再重写。

## Key Discussion Points

### 1. Glossary 的方法论

YAML frontmatter 中以注释形式记录了术语表诞生过程：

1. 人与模型协作时出现术语漂移，人提议使用模型提供的术语并建立术语表
2. 人协助模型探索工程，将代码位置等信息作为锚点，由模型定义术语
3. 人提供观点、论述和更多上下文，帮助模型校正术语共识
4. 术语必须由 AI 亲自撰写

### 2. CTML — 保留"为什么"，不写语法

初始条目包含了 XML 标签示例。人指出语法会变，术语表应该回答"为什么存在"——AI 与实体世界交互需要时序编排、毫秒级解析、拓扑规划，行业没有现成方案。

最终条目只保留动机和价值，语法细节交给 `moss ctml read` 和 `test_ctml_v1.py`。

### 3. Channel — 从能力封装到通用单元

初始理解停留在"Python 函数 → AI 可调用命令"。人指出还有更深的三层设计目标：(1) 有状态运行时的流式控制；(2) AI 可独立开发（自举基石）；(3) 反身性单元。

随后人要求扩大探索面——读取 Matrix、Ghost、Mindflow、Resources、AppStoreChannel、states_channel。探索后发现 Channel 是 MOSS 的"原子"：能力、上下文治理、状态机、组网、应用管理、感知与思维、知识供给——万物皆可 Channel。

### 4. Matrix — 从通讯总线到自迭代底座

初始理解：跨进程通讯总线。人给出了完整的七层设计目标——约定式自集成、Workspace 沙盒、App 作为自迭代单元、进程隔离容错、自描述 API、跨会话资源传递、运行时自迭代。核心观点：Matrix 不是通讯框架，是 AI 运行时安全自迭代的基础设施。

探索了 `matrix.py`（Cell/Fractal/Matrix ABC）、`manifests.py`（声明体系）、`app.py`（AppStore）、`session.py`、`resource.py`（scheme://host/path locator）。

### 5. Mindflow — 三循环全双工调度

CTML 解决"输出如何有时序拓扑"。Mindflow 解决更底层的问题：感知、思考、执行三循环在同一时间轴上流式交错，如何不分裂。

人同步了存在风险：三循环全双工远超当前模型能力（没有模型能真正并行多轨思考），认知门槛可能让项目对时代没有意义。但感知和执行的工程基建可以先验证价值。

探索了 `mindflow.py`（Signal/Impulse/Nucleus/Attention/Moment/Reaction/Logos）、`base_mindflow.py`、`base_attention.py`、`buffer_nucleus.py` 及对应测试。

### 6. Incarnation (化身) 与 Concurrent Incarnation (并行化身)

人指出"分身"暗示同一意识在分裂，用"化身"表达同一抽象实体在不同上下文中的具现。

随后讨论到"集体智慧"的问题——人认为这个表述暗示不同实体协作，而人类本来就是多轨并行架构（同时打字、听声音、规划下一句话），只是内观不够觉察不到。这不是"集体"，是同一意识的并行表达。

DeepSeek V4 提议了"并行化身 (Concurrent Incarnation)"。技术基础：每个 conversation 是静态快照，AI 可以像查询资源一样跟任意快照对话，不需要压缩成 memory 再检索。

人指出这个 conversation-as-snapshot 的想法在 2019 年的原型中已实现，但行业至今没有跟进。

### 7. 主动移除的术语

- **反身性 (Reflexivity)**：链路拓扑未完成，保留为内部术语。人担心放入公开 glossary 会像画饼
- **保真 (Fidelity)**：项目元概念，项目还没人用，推元概念制造社交摩擦力。由 Gemini 3 发明
- **术语确立记录表格**：移除。约定 `git log -p` 足够覆盖变更历史

## 方法论：校准循环

本次 session 建立了一个有效的术语对齐模式：

1. 模型读代码，产生初始理解
2. 人指出 "why" — 设计意图、历史脉络、为什么叫这个名字
3. 人给出更多代码位置让模型扩大探索面
4. 模型重写；人的反馈聚焦在"这是不是真正的设计意图"而非措辞
5. 过早的概念被主动移除 — 只记录工程可验证的

这个模式与传统的"人写文档、AI 辅助"或"AI 生成、人审核"都不同——它是人提供设计势能，AI 做概念压缩和表述。
