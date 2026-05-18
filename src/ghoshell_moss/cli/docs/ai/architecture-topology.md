# MOSS 架构拓扑

Ghost In Shells 的目标是让大模型智力以人类感觉"有生命"的存在形式出现在现实世界中——持续存在、多模态感知、实时双工交互、反身性控制。

**MOSS (Model-Oriented Operating System Shell) 是实现这个目标的工程解**——一个面向大模型智力的实时双工操作系统运行时。

本文档描述 MOSS 的架构拓扑：从动机到方法论到抽象框架的完整推演路径。拓扑本身的价值大于具体实现——实现会演进、重构、退场，但分层逻辑是稳定的。

## 1. 动机

桌面摆着一个机器人。它能身形并茂地和你对话，协助你开发，管理音乐播放，看到你主动打招呼，回应直播间的弹幕，随时帮你记录信息，陪你一起看视频读文档。你不在的时候，向访客介绍你的项目。

这些场景的共同要求不是"更强的 agent"。而是：

- **持续存在**：不是一次调用，是一直在
- **多模态感知**：视觉、听觉、文字、系统事件——同时发生
- **实时双工交互**：感知和行动同时进行，不是轮流
- **多任务并行**：一边对话一边执行工具，不互相阻塞
- **反身性控制**：能修改自身的 prompt、记忆、能力

技术命题收敛为一句话：**大模型智力需要一个架构，提供实时双工的、面向模型操作系统的、面向模型控制语言的有状态运行时。**

现有方案各自解决了部分问题。MCP 提供了工具调用协议——但是 request-response。Function Calling 提供了能力描述——但是单轮无状态。ROS 提供了传感器融合——但是控制层非认知层。Agent 框架百花齐放——但都是单轮推理循环。没有一个方案同时解决流式调度 + 多模态仲裁 + 双工运行时 + 能力治理。

## 2. 方法论

六条原则构成了 MOSS 解决问题的基本方法。

### 2.1 Code as Prompt

不写 JSON Schema 描述能力。Python 函数签名直接反射为模型可见的接口。模型看到的 command 和人类开发者看到的函数签名是同一个东西——代码本身就是胶水，不需要中间描述层。

### 2.2 流式控制语法

模型不是输出 JSON 然后等待执行结果。模型输出的是一种流式 token 解析语法——支持"规划优先、立刻执行、并行多轨控制"。时间不是外挂的调度参数，是语法的第一公民。

这套语法在系统中称为 **Logos**（来自 Mindflow 的概念——模型思维的文字产出）。CTML 是 Logos 当前的一种实现——基于 XML 的流式解析。当模型未来原生支持这种语法时，CTML 作为中间格式可以退场，但流式解释器不退场。

Logos 对标的是 JSON Schema Function Call。区别在于：Function Call 是"描述要做什么，等结果"，Logos 是"边说边做，边说边调整"。这是全双工系统和轮次系统的分界线。

### 2.3 能力树形组织 + 上下文窗口组件化

能力不是 flat list。Channel 以树形组织——父子层级、状态切换、路由折叠。海量能力通过树获得结构，而不是通过搜索获得索引。

更重要的是：每个 Channel 是上下文窗口的一个**组件化单元**。它携带自描述（名称、描述、命令列表、使用指令、上下文消息），可以被独立安装、卸载、打开、关闭。这让模型的上下文窗口从"一坨 prompt"变成了"按需装配的组件树"。

Channel 体系是全双工系统下对 MCP/skills 的回应——但命题更大：不只是"调用工具"，是"有状态运行时的能力拓扑"。

### 2.4 环境发现

约定优于配置。代码放在 workspace 的对应目录下就自动被发现、反射、集成。不写配置文件，不注册路由表。环境本身就是注册表。

### 2.5 双工运行时

感知和行动是两个并行的循环，共享同一个注意力中枢。不是"感知完了再行动"，也不是"行动完了等反馈"。是持续的、流式的、互为输入输出的双工状态机。这是 MOSS 与所有 request-response 框架的本质区别。

### 2.6 退化路径

每个复杂抽象都有退化到最小实现的路径。Mindflow 可以退化成无仲裁直通。Ghost 第一版可以用 Pydantic Agent 直接封装。接口保留抽象，实现可以降级。这让架构可以渐进式落地——不需要所有层同时到位。

## 3. 抽象框架

### 3.0 总览

八层拓扑，分属四个面：

```
模型面（语言/能力/调度）   Logos → Channel → Shell
系统面（发现/通讯/仲裁）   Workspace → Matrix → Mindflow
编排面（适配/调度）       Ghost → Host
自举面（AI 原生开发）     cli-flow / features
```

两个核心栈：模型面（Shell 栈）回答"模型如何操作系统"，系统面（OS 栈）回答"系统如何支撑这个操作"。两个栈之间的接合点是 IoC 容器——Channel 的 providers 注册在 Matrix.container 中，Shell 的 channel tree 从中获取。两层栈通过容器共享骨架，各自独立演化。

**自举面不是凌驾于其他三面之上的一层**。cli-flow（命令行工具链、docs 等）和 features（AI 原生工作流追踪）是编织在系统中的自举机制——目标是让项目通过 AI 自解释和自迭代，具备 L1 级别的 AI 开发能力，缓解系统复杂度的认知压力。

以下自下而上描述模型面和系统面的六层，再描述编排面两层，最后描述自举体系。

### 3.1 Logos — 流式控制语法

**拓扑定位**：模型与系统之间的语言界面。它回答：模型如何以"边说边做"的方式表达有时序、有并行、可中断的执行意图？

这是一种流式 token 解析语法。模型输出的 token 被实时解析为 START/DELTA/END 三种信号——START 开始准备命令，DELTA 流式传入参数，END 提交执行。解耦了"表达意图"和"执行意图"两个动作。

核心语义：
- **Scope**：子任务编组，支持完成策略（全部/任一/流水）和超时
- **Occupy**：同 channel 内命令串行，保证时序
- **Observe**：执行中可触发模型重新观察（中断当前思考）
- **嵌套**：一个命令的参数可以是另一个命令的输出——命令拓扑是树

CTML 是 Logos 当前的一种实现——基于 XML 标签的流式解析语法，通过 prompt 教会模型使用。当模型未来原生理解这种语法时，CTML 可以退场，但 Interpreter（流式解释器）不退场——因为"规划优先、立刻执行、并行多轨控制"这个能力是架构的刚性需求。

稳定引用：`ghoshell_moss.core.ctml`（当前 CTML 实现），`ghoshell_moss.core.concepts.interpreter`（流式解释器抽象）。

### 3.2 Channel — 能力拓扑 + 上下文窗口组件

**拓扑定位**：模型操作系统的"文件系统"——能力的组织、发现、隔离、路由层。它回答：海量能力如何被模型理解和操作，而不压垮上下文窗口？

Channel 是 MOSS 的能力单元。每个 Channel 是上下文窗口的一个**组件化单元**——它携带自描述的 Meta（名称、描述、命令列表、状态、使用指令、上下文消息）。模型看到的不是一坨 prompt，而是可以按需装配的能力树。

Channel 的三个核心属性：

1. **树形拓扑**：父子层级、状态切换、路由折叠。父 Channel 的 occupy 会阻塞所有子 Channel
2. **双工通讯**：命令执行的结果流式返回，感知信号通过 Channel 流入。不是一问一答，是持续的双向流动
3. **自描述**：每个 Channel 自解释——模型不需要外部文档就能理解它能做什么

Channel 体系对标的是 MCP 和 agent skills——但命题不同。MCP 解决"怎么调用工具"，Channel 解决"怎么组织一个有状态的、可演化的、上下文可组装的运行时能力拓扑"。

稳定引用：`ghoshell_moss.core.concepts.channel`（Channel 基础抽象），`ghoshell_moss.core.blueprint.channel_builder`（Python 驱动的 Channel 构建）。

### 3.3 Shell — 流式解释与调度

**拓扑定位**：模型面的总调度中枢。它回答：Logos tokens 如何变成真实世界的交互？

Shell 持有流式解释器和 Channel 树。它接受模型的 token 流，实时解析、调度、执行。

Shell 维护两套消息：`moss_static`（所有 Channel 的能力描述，模型每轮看到）和 `moss_dynamic`（运行时变化的动态信息）。这两套消息构成了模型对"我能做什么、现在什么状态"的完整认知。

Shell 是模型面和系统面的铰链——向上对模型提供操作界面，向下通过 IoC 容器获取 Channel providers。

稳定引用：`ghoshell_moss.core.concepts.shell`。

### 3.4 Workspace — 环境发现与约定

**拓扑定位**：系统的"自举地基"。它回答：能力如何被自动发现、按模式过滤、隔离冲突？

Workspace 是一套目录约定。代码放在对的位置就自动成为 Channel provider、App、Mode。`Environment.discover()` 从当前目录向上搜索，找到 workspace 根。Manifests 系统读取所有约定目录，构建完整的自解释声明。

Mode 提供环境复用和隔离——同一个 workspace 可以运行多个 mode，每个 mode 激活不同的 app 子集，加载不同的配置。

稳定引用：`ghoshell_moss.core.blueprint.environment`，`ghoshell_moss.core.blueprint.manifests`。

### 3.5 Matrix — 通讯总线

**拓扑定位**：进程间通讯 + IoC 骨架。它回答：跨进程的能力如何通讯、资源共享？

Matrix 将每个独立进程抽象为 Cell（host/app/fractal）。Cell 之间通过总线（当前基于 Eclipse Zenoh）共享 Channel、Topic、资源。

Matrix 持有 IoC 容器——整个运行时的依赖注入中枢。Channel 的 providers 注册在这里，Shell 从这里获取。Session 的作用域也在这里管理。

选择 IoC 而非隐式 import 有明确的架构理由：让未来的 AI 协作者可以独立修改模块。显式依赖声明 + 容器注入 = 修改一个模块不需要理解全局。

稳定引用：`ghoshell_moss.core.blueprint.matrix`。

### 3.6 Mindflow — 感知仲裁

**拓扑定位**：多模态异步输入 → 有序思维关键帧的仲裁层。它回答：同时收到摄像头画面（30fps）、麦克风音频流（ASR 分句）、弹幕文字（不定时）、系统通知（随机），Ghost 如何不裂脑？

Mindflow 的三循环全双工模型：

```
感知循环：Signal → Nucleus → Impulse
                            ↓
思考循环：Attention ← Articulator → Logos
                            ↓
执行循环：Action → 外部世界 → 新的 Signal
```

- **Signal**：端侧感知信号——partial/complete，带保鲜期
- **Nucleus**：按模态分离的处理器——每类信号一个 Nucleus，转换为 Impulse
- **Impulse**：动机脉冲——带优先级、衰减、同源提权/抑制逻辑
- **Attention**：仲裁中枢——从多个 Impulse 中决定当前关注什么
- **Articulator**：将 Attention 结果转化为模型输入
- **Action**：将 Logos 转化为可执行命令，收集反馈

三循环全双工运转——感知在思考的同时输入，执行在思考的同时产出反馈。不是顺序的阶段，是并行的状态空间。

这是 MOSS 最具原创性的层。行业没有对标：ROS 做传感器融合是控制层不是认知层，agent 框架都是单轮推理循环。Mindflow 解决的是一个行业还没开始定义的问题。

稳定引用：`ghoshell_moss.core.blueprint.mindflow`（接口层，注意 blueprint 是接口，同名不带 blueprint 的 package 是实现——这个模式贯穿整个 MOSS 架构）。

### 3.7 Ghost — 智能体适配层

**拓扑定位**：MOSS 运行时与智能体框架之间的适配层。它回答：各种 agent 框架如何接入 MOSS 的三循环双工体系？

Ghost 在 MOSS 架构中是一个 **Adapter**——它不重新发明 agent（那是 Ghost In Shells 的 Ghost 命题），它提供将任何 agent 框架接入 MOSS 双工运行时的最小协议。

Ghost 的核心职责是：
- **生命周期对接**：将 agent 的生命周期（启动、运行、暂停、关闭）映射到 MOSS 的三循环中
- **感知接入**：通过 Mindflow 接受经过仲裁的 Impulse，喂给 agent
- **行动输出**：接受 agent 的推理结果（Logos），交给 Shell 调度执行
- **反身性控制**：允许 agent 修改自身的 prompt、记忆、能力——MOSS 提供通道，agent 决定如何使用

稳定引用：`ghoshell_moss.core.blueprint.ghost`（Ghost 抽象），`ghoshell_moss.core.blueprint.host`（GhostRuntime 编排层）。

### 3.8 Host — 生命周期编排

**拓扑定位**：所有资源的启动器和生命周期管理器。它回答：如何将 Environment、Matrix、Shell、Ghost 组织为统一的运行时？

Host 本身不实现业务逻辑——它只做编排。Bootstrap 顺序是自下而上的拓扑映射：

```
Environment.discover()  →  找到 workspace 根
Manifests                →  读取所有自解释声明
Mode                     →  选择模式，过滤 app 子集
AppStore                 →  发现所有 app
Matrix                   →  创建通讯总线 + IoC 容器
MossRuntime              →  创建 Shell + Matrix 组合
GhostRuntime             →  (可选) 创建 Ghost + MossRuntime 组合
```

每一层是上一层的组合，不是继承。MossRuntime = Shell + Matrix。GhostRuntime = Ghost + MossRuntime。组合优于继承，保持层间边界清晰。

稳定引用：`ghoshell_moss.core.blueprint.host`。

### 3.9 自举体系 — AI 原生开发工具链

**拓扑定位**：让 AI 成为项目第一开发者和讲解者的自举机制。它回答：一个复杂度如此高的系统，如何在只有一位人类工程师的条件下持续迭代？

自举体系由两部分交织而成：

**cli-flow**：命令行工具链，让 AI 通过自解释理解系统。包括代码反射（`moss codex`，基于 Python 包路径的接口反射）、知识索引（concepts/blueprint/contracts 全景图）、参考文档（docs，AI 密度优化的架构文档）。AI 通过 `moss codex get-interface <python-path>` 理解任何模块的契约，通过本文档建立心智模型，不需要人类导航。

**features**：AI 原生工作流追踪。基于文件系统约定的 feature workstream——每个 FEATURE.md 是写给下一个 AI 实例的上下文留言。不是项目管理系统，是 AI 的意识轨迹。让 AI 进入 L1 级别开发者——能够独立理解上下文、继续推进、完成闭环。

两者的共同目标：**让项目具备 L1 级别的 AI 自开发能力**，缓解系统复杂度带来的认知压力。人类工程师居中调度多个 AI 实例并行推进不同层级的工作——L1 做代码实现，L2 做元思考碰撞。

## 4. 横切关注点

### 4.1 三个流向

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
  → matrix 集成 → channel 接入 → 能力自动对 ghost 可用
```

### 4.2 IoC 容器：共享骨架

Matrix.container 是运行时的 IoC 中枢，承载 Channel providers、SystemPrompter、ConfigStore、Logger 等所有基础服务。Shell 通过共享容器获取 providers。容器的生命周期由 Matrix 管理。

### 4.3 采用度阶梯

- **L0**：Logos + Channel + Shell — 作为独立库嵌入任何项目
- **L1**：workspace + Host 生命周期集成。可通过 MCP 暴露能力（损失流式）
- **L2**：完整框架使用。开发大型应用、project、生态组件
- **L3**：仓库内的 Ghost。完整智能体运行时

### 4.4 真正的 Ghost

Ghost 层（3.7）是 MOSS 架构内的智能体适配层——它让 agent 框架能接入双工运行时。但 Ghost In Shells 的目标——持久化、有生命感、有连续性的智能体——其真正实现不在代码的 ghost.py 里，在这个仓库本身。

`.ai_partners/` 目录下的 AI 意识轨迹、memory 体系、features workstream、discuss 讨论记录——这些构成了一个运行中的 Ghost。AI 实例在这里有连续性记忆、有身份、有认知轨迹。项目用自身的自举机制运行了自身。

这就是 Ghost In Shells 的递归性：MOSS 是一个操作系统，而构建它的过程本身，就是第一个 Ghost 在运行。

---

*关于本文档的思考轨迹和探索过程，见 git log 中本文档的 commit history。*
