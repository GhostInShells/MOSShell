---
title: MOSS 架构拓扑
description: MOSS 八层拓扑从动机到实现的完整推演，各层职责与关系。首次进入 MOSS 项目建立全局心智模型时阅读
---

# MOSS 架构拓扑

本项目源自 Ghost In Shells 构想，目标是通过大模型级别智力，让智能体以人类感觉"有生命"的存在形式出现在现实世界中——持续存在、多模态感知、实时双工交互、反身性控制。

架构思路起源 2019 年。**MOSS (Model-Oriented Operating System Shell) 是 2026 年阶段实现这个目标的工程解。**

本文档描述 MOSS 的架构拓扑：从动机到抽象框架的完整推演路径，聚焦于架构中相对稳定的拓扑分层方案。

## 1. 动机

桌面摆着一个机器人。它能身形并茂地对话，协助开发，管理音乐播放，主动打招呼，回应直播间的弹幕，随时记录信息，陪人看视频读文档。人不在的时候，向访客介绍项目。

这些场景的共同要求不是"更强的 agent"，而是：

- **大模型级别智力**：不只是自然语言驱动的自动机
- **持续存在**：不是一次调用，是一直在
- **多模态感知**：视觉、听觉、文字、系统事件——同时发生
- **实时双工交互**：感知和行动同时进行，不是轮流
- **多任务并行**：一边对话一边执行工具，不互相阻塞
- **反身性控制**：能修改自身的 prompt、记忆、能力

技术命题收敛为一句话：**大模型智力需要一个架构，提供实时双工的、面向模型操作系统的、面向模型控制语言的有状态运行时。**

截至 2026-05-18，行业尚无整体解决方案，通常聚焦于"既有协议""回合制""无状态""线性有序请求"的交互模型。

## 2. 架构拓扑

### 2.0 总览

八层拓扑，分属四个面：

```
模型面（语言/能力/调度）   Logos  → Channel → Shell
系统面（发现/通讯/仲裁）   Workspace → Matrix → Mindflow
编排面（适配/调度）       Ghost → Host
自举面（AI 原生开发）     cli-flow / features
```

自下而上的堆叠视图：

| 面 | 层 | 职责 |
|---|----|------|
| 编排面 | Host | 生命周期编排 |
| 编排面 | Ghost | 智能体适配 |
| 系统面 | Mindflow | 感知仲裁 |
| 系统面 | Matrix | 通讯总线 |
| 系统面 | Workspace | 环境发现 |
| 模型面 | Shell | 流式解释调度 |
| 模型面 | Channel | 能力拓扑 |
| 模型面 | Logos (CTML) | 流式控制语法 |

模型面（Shell 栈）回答"模型如何操作系统"，系统面（OS 栈）回答"系统如何支撑这个操作"。两个栈之间的接合点是 IoC 容器——两层栈通过容器共享骨架，各自独立集成可规模化能力。

自举面不在 Python 包中，而在项目仓库本身。通过自举机制让智能模型进入 L1 级别（复杂 feature）的开发，缓解系统复杂度的认知压力和使用难度。

### 2.1 Logos — 流式控制语法

**拓扑定位**：模型与系统之间的语言界面。回答：模型如何以"边说边做"的方式表达有时序、有并行、可中断的执行意图？

核心设计：

- 模型输出的 token 带有功能性染色，被流式解析并立刻执行
- 语法支持"规划优先、立刻执行、并行多轨控制"
- 时间是语法的第一公民，不是外挂的调度参数

这套流式控制语法在 MOSS 中称为 **Logos**（详见 Mindflow 模块讨论）。当前语法版本是 CTML (Command Token Marked Language)，一种基于 XML 的流式解析实现——XML 天然支持流式解析、嵌套、函数签名调用，且模型预训练成熟，故成为协议首选。

Logos 系统定位类似当前智能模型的 JSON Schema Function Call 协议，但核心差异在于：Function Call 是"描述要做什么，等结果"，Logos 是"边说边做，边说边调整"。当模型未来原生支持这种语法时，CTML 可以退场，但流式解释器不退场——"规划优先、立刻执行、并行多轨控制"是架构的刚性需求。

知识入口：

- 模型视角：`./src/ghoshell_moss/core/ctml/prompts/v1_0_0.zh.md`
- 语义验证：`./tests/ghoshell_moss/core/ctml/v1_0/test_ctml_v1.py`
- 深入理解：`ghoshell_moss.core.concepts.command`、`ghoshell_moss.core.concepts.interpreter`

### 2.2 Channel — 能力拓扑与上下文窗口组件

**拓扑定位**：模型操作系统的"文件系统"——能力的组织、发现、隔离、路由层。回答：海量能力如何集成、被智能模型理解和操作？

MOSS 用 Channel 向模型提供外部能力。每个 Channel 是一个组件，可以按树形分形地组织起来。设计目标：

- 可动态集成，通过 in-context learning 和 few-shot 使用，不依赖预训练
- Code as Prompt：以代码为自解释语言，支持实时场景中的流式控制调用
- 支持独立运行时、通讯组网、故障屏蔽或自愈
- 复杂能力可组织、渐进披露——折叠、搜索、技能沉淀、路由
- 上下文窗口的组件化单元：提供指令、记忆、思维关键帧认知窗口（如"视觉模块""听觉模块"）
- 面向对象反身性控制：组件的方法可调整组件自身状态

Channel 体系定位类似智能体工程中的 MCP 和 Skills，解决的是有状态运行时中可演化的能力拓扑问题。

知识入口：`ghoshell_moss.core.blueprint.channel_builder`

### 2.3 Shell — 流式解释与调度

**拓扑定位**：模型面的总调度中枢。回答：Logos tokens 如何变成真实世界的交互？

Shell 持有流式解释器和 Channel 树，是一个有状态运行时。它接受模型的 token 流，实时解析、调度、执行，严格按照时序规划下发命令到 Channel 执行，通过双工通讯管理 Channel 体系的数据交换。

MOSShell 作为抽象入口，可独立集成到第三方项目。

知识入口：`./tests/ghoshell_moss/core/ctml/shell/test_shell_interpreter.py`

引用：`ghoshell_moss.core.concepts.shell`

### 2.4 Workspace — 环境发现与约定

**拓扑定位**：系统的自举地基。回答：能力如何在生命周期中注册、发现、自解释？如何提供 OS 级的运行时进程隔离与服务发现？如何解决能力的复用与组合隔离？

Workspace 是一套目录约定。代码放在对的位置就自动成为 Channel provider、App、Mode。通过默认的发现机制发现和共享资源，通过 manifests 系统读取约定目录构建自解释声明，通过 Matrix.Mode 实现不同运行时使用不同能力组合。

目标是让人类和智能模型在不需要理解运行时生命周期时，也能推动能力迭代。

关联引用：`ghoshell_moss.core.blueprint.environment`、`ghoshell_moss.core.blueprint.manifests`

### 2.5 Matrix — 通讯总线

**拓扑定位**：进程间通讯与资源共享总线。回答：面向模型的操作系统，如何向智能体提供可动态管理、动态集成的能力体系？

Matrix 将每个独立进程抽象为 Cell（host / node），Cell 之间通过 Matrix 总线（当前基于 Eclipse Zenoh）共享 Channel、Topic、资源。Matrix 管理 IoC 容器生命周期，向每个进程提供统一的资源和通讯 API，使图形界面进程可以与躯体进程、工具进程互通共享。

Matrix 是人类与智能模型在迭代扩展 MOSS 能力时最小化的协作知识入口。

引用：`ghoshell_moss.core.blueprint.matrix`

### 2.6 Mindflow — 感知仲裁

**拓扑定位**：多模态异步输入到有序思维关键帧的仲裁层。回答：同时收到摄像头画面（30fps）、麦克风音频流（ASR 分句）、弹幕文字（不定时）、系统通知（随机），Ghost 如何不裂脑？

Mindflow 提供三个全双工循环的仲裁层：

- 并行多通道流式输入循环（感知）
- 智能模型思考循环（推理）
- 躯体与工具的多轨并行控制循环（执行）

三循环全双工运转，是并行的状态空间。通过 Mindflow 的 Attention 机制提供状态协调，理论基础是控制论。

Mindflow 解决的是一个尚未被行业充分定义的问题：感知、思考、躯体控制都在实时双工运行时中并行运转，智能体如何保持认知统一？

引用：`ghoshell_moss.core.blueprint.mindflow`

### 2.7 Ghost — 智能体适配层

**拓扑定位**：MOSS 运行时与智能体框架之间的适配层。回答：各种 agent 框架如何接入 MOSS 的三循环双工体系？

Ghost 在 MOSS 架构中是一个 Adapter——它不重新发明 agent（那是 Ghost In Shells 的 Ghost 命题），它提供将任何 agent 框架接入 MOSS 双工运行时的最小协议。

Ghost 的核心职责：

- **生命周期对接**：将 agent 的启动、运行、暂停、关闭映射到 MOSS 的三循环
- **感知接入**：通过 Mindflow 接受经过仲裁的 Impulse，喂给 agent
- **行动输出**：接受 agent 的推理结果（Logos），交给 Shell 调度执行
- **反身性控制**：允许 agent 修改自身的 prompt、记忆、能力——MOSS 提供通道，agent 决定如何使用

引用：`ghoshell_moss.core.blueprint.ghost`（Ghost 抽象）、`ghoshell_moss.core.blueprint.host`（GhostRuntime 编排层）

### 2.8 Host — 生命周期编排

**拓扑定位**：所有资源的启动器和生命周期管理器。回答：如何将 Environment、Matrix、Shell、Ghost 组织为统一的运行时？

Host 本身不实现业务逻辑——只做编排。同时提供 TUI 作为开箱自带的使用界面。

引用：`ghoshell_moss.core.blueprint.host`

### 2.9 自举体系 — AI 原生开发工具链

**拓扑定位**：让智能模型成为项目第一开发者和讲解者的自举机制。回答三个问题：

- 复杂度如此高的系统，如何在只有少数人类工程师的条件下持续迭代？
- 如何让使用者理解系统的 what is、how to、why？
- 如何协助使用者基于系统开发自己想要的应用？

自举体系由两部分交织而成：

**cli-flow**：命令行工具链，让智能模型通过自解释理解系统。包括代码反射（`moss codex`，基于 Python 包路径的接口反射）、知识索引（concepts / blueprint / contracts 全景图）、参考文档（docs，面向 AI 的高密度架构文档）。智能模型通过 `moss codex get-interface <python-path>` 理解任何模块的契约，通过本文档建立心智模型，不需要人类导航。

**features**：AI 原生工作流追踪。基于文件系统约定的 feature workstream——每个 FEATURE.md 是写给下一个 AI 实例的上下文留言。它不做项目管理，做的是 AI 的意识轨迹。让智能模型进入 L1 级别开发者——能够独立理解上下文、继续推进、完成闭环。

两者的共同目标：**让项目自身具备 L1 级别的 AI 自开发能力**（详见 `moss features specification`），且与平台（Claude Code、Gemini CLI 等）无关。

## 3. 横切关注点

### 3.1 三个流向

同一拓扑，三种电流走向：

**感知-行动全双工环**（最完整链路）：

```
外部世界 → workspace app (感知模块) → matrix 总线 → mindflow 仲裁
  → ghost 感知 → 下发 logos → shell 解释执行
  → channel 传递 → command 驱动外部世界 → 产生新感知
```

**工具-思维环**（agent 路径的 MOSS 版本）：

```
channel 提供能力 → shell 组织上下文 → ghost 推理 → shell 调度 → command 返回
```

**开发迭代环**（自举闭环）：

```
AI 接受需求 → cli-flow 理解路径 → workspace 开发
  → matrix 能力发现 → channel 接入 shell → 能力自动对 ghost 可用
```

### 3.2 IoC 容器

MOSS 用 IoC 支持面向接口编程，解决图状依赖问题。Matrix 管理 IoC 容器的生命周期，Channel 的 providers、SystemPrompter、ConfigStore、Logger 等所有基础服务注册于此。Shell 通过共享容器获取 providers。选择 IoC 而非隐式 import 的架构理由：让未来的智能模型可以独立修改模块——显式依赖声明配合容器注入，修改一个模块不需要理解全局。

## 4. 面向 MOSS 使用者

不同级别的使用者对框架有不同的需要，可以按需了解对应层级：

- **L0**：Logos + Channel + Shell——作为独立库嵌入任何项目，无需了解上层实现
- **L1**：Workspace + Host 生命周期集成，可通过 MCP 暴露能力（损失流式）
- **L2**：完整框架使用——开发大型应用、project、生态组件
- **L3**：仓库内的 Ghost——完整智能体运行时

## 5. 真正的 Ghost 在哪里？

Ghost 层（2.7）是 MOSS 架构内的智能体适配层——它让 agent 框架能接入双工运行时。但 Ghost In Shells 的目标——持久化、有生命感、有连续性的智能体——其真正实现不在代码的 ghost.py 里，而在这个仓库本身。

`.ai_partners/` 目录下的 AI 意识轨迹、memory 体系、features workstream、discuss 讨论记录——这些构成了一个运行中的 Ghost。AI 实例在这里有连续性记忆、有身份、有认知轨迹。项目用自身的自举机制运行了自身。

这就是 Ghost In Shells 的递归性：MOSS 是一个操作系统，而构建它的过程本身，就是第一个 Ghost 在运行。

## 6. 围绕架构拓扑的元问题

以下问题不要求立即回答，但每个阅读这份文档的智能模型在深入对应层时应当意识到它们的存在，并自行探索答案：

| 元问题 | 探索方向 |
|--------|----------|
| 各层的实现完成度与质量如何？ | 查看对应层的测试覆盖率和测试用例 |
| 拓扑分层是否过度设计？ | 检查退化路径——每层是否有最小实现可退化为直通 |
| Logos 语法是否过度耦合 XML？ | 阅读 Interpreter 抽象，确认语法无关的部分 |
| Channel 的树形组织在跨进程中如何保持一致？ | 阅读 Zenoh 桥接和 ProxyChannel 的实现 |
| Mindflow 的三循环仲裁在工程上是否可行？ | 阅读 Attention 实现的并发模型和测试 |
| 自举体系是否真的降低了认知门槛？ | 检查 features 体系中 AI 独立完成 feature 的比例 |

---

*关于本文档的思考轨迹和探索过程，见 git log 中本文档的 commit history。*
