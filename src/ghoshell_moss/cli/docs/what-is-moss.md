---
title: MOSS 是什么
description: AI 进入 MOSS 项目的第一篇。一句话说清 MOSS 解决什么问题、能做什么、不能做什么、从哪里开始
---

# MOSS 是什么

MOSS (Model-oriented Operating System Shell) 是一个面向大模型的有状态双工运行时框架。它让智能模型能够实时、并行地感知世界、输出意图、驱动躯体——不是回合制对话，而是持续存在、边说边做。

**一句话**：MOSS 的目标是让智能模型在电脑/开发板里运行时自主迭代，让它连接的机器人动起来，让它连接的屏幕长出图形界面，让它能说话、能听——都由模型开发。

---

## 1. 直觉感受

一个机器人要一边挥手一边说"你好"。挥手 0.3 秒，说话 1.5 秒。模型需要在"说到第三个字的时候开始挥手"，而且两者都在对方执行期间完成。

在 MOSS 里，模型输出的文本是活的：

```
<_>
  你好，我是机器人 <robot:wave duration="0.3"/> 很高兴认识你
</_>
```

`<robot:wave/>` 标签闭合的瞬间，挥手就开始执行——不需要等整句话说完。模型生成 token 的过程本身就是时间轴。这就是**流式生成即执行**。

这就是 MOSS 解决的核心命题：**大模型智力需要一个架构，提供实时双工的、面向模型操作系统的、面向模型控制语言的有状态运行时。**

---

## 2. 它不是什么

MOSS 不是 agent 框架。MOSS 是 Shell（躯壳层）——它提供双工运行时和能力拓扑，让 Ghost（智能体）可以降临其中。Agent 框架解决"怎么思考"，MOSS 解决"怎么活在现实世界里"。

| MOSS 不是 | 为什么 |
|-----------|--------|
| Agent 框架 | MOSS 不定义推理策略、不管理对话轮次。Ghost 负责思考，MOSS 负责让思考结果在真实世界中产生有时序、有并行的效果 |
| MCP 替代品 | MCP 是静态的工具暴露协议。MOSS 的 Channel 是动态的、有状态的、有生命周期的运行时能力单元。Channel 可以跨进程、可以热插拔、可以在运行时被模型创建 |
| ROS 替代品 | ROS 解决机器人中间件问题。MOSS 解决"模型如何将分散的能力整合为有生命的整体"。两者可以共存——MOSS 通过 Channel 集成 ROS 节点 |
| 又一个 AI 应用 | MOSS 是基础设施——它不面向最终用户，面向的是智能模型本身。模型的开发者就是 AI |

---

## 3. 核心心智模型

三样东西，构成 MOSS 的最小认知单元：

```
CTML（语言）  →  描述"做什么"
Channel（能力） →  提供"能做什么"
Shell（调度）  →  决定"何时做"
```

- **CTML** — 流式控制语法。模型输出的文本中嵌入命令标签，边生成边执行。命令之间有并行、串行、超时、中断的拓扑关系。目前基于 XML，语法载体可换，流式解释器不退场。
- **Channel** — 能力组织单元。将 Python async 函数反射为模型可调用的 Command，按树形拓扑组织。Code as Prompt：Python 函数签名即接口，不翻译成 JSON Schema。Channel 可以有状态、生命周期、跨进程通讯。
- **Shell** — 流式调度中枢。接收 CTML token 流，实时解析为命令拓扑，按同通道 FIFO、父子通道 occupy、异通道并行的规则调度执行。

三者的接合点：CTML 不定义命令（Channel 定义），CTML 不调度执行（Shell 调度），Channel 不感知 CTML（任何调用者都可以调用 Command）。

**探索路径**：`moss docs read architecture-topology.md` — 八层拓扑完整推演。`moss docs read ctml.md` — CTML 设计哲学与测试地图。`moss docs read channel-system.md` — Channel 构建全链路。

---

## 4. 为什么这样设计

四个核心原则：

**Code as Prompt**（2019— ）  
不把 Python 函数签名翻译成 JSON Schema。模型直接看到 Python 代码——信息密度更高，模型预训练中大量接触，人类开发者不需要额外维护。Python 是最丰富的生态，任何库都可以被反射为模型可操作的能力。

**时间是第一公民**（2024— ）  
实体世界有时间。挥手需要 0.3 秒，说话需要 1.5 秒，摄像头每秒 30 帧。模型生成 token 的过程本身就是时间轴——不是生成完再执行，而是生成即执行。并行、串行、超时、中断，都是语法的原生语义。

**文件系统即注册**（2025— ）  
不写配置文件，不碰启动代码。Python 实例放在约定路径下，Matrix 启动时自动发现、注入 IoC 容器。AI 不需要理解运行时生命周期就能注册能力——写一个 Provider 实例，下次启动自动生效。这是为 AI 读者设计的。

**模型是开发者**（始终）  
MOSS 的 CLI 工具（codex、manifests、features）为 AI 设计。`moss codex get-interface` 在 5 秒内反射任何模块的接口契约。`moss manifests explain` 自描述环境能力。`moss features` 是 AI 的意识轨迹——每个 FEATURE.md 是过去 AI 实例写给下一个的留言。AI 不是工具调用者，是能力的创造者。

---

## 5. 能做什么

- 让模型通过流式语言（CTML）精确控制并行动作的时序——边说话边做动作，多个设备同步协调
- 将任何 Python 代码反射为模型可操作的能力，一行装饰器即可
- 跨进程、跨设备的能力集成——App 作为独立进程，通过 Matrix 总线通讯，崩溃不拖垮主进程
- AI 在运行时创建、启动、调试、关闭 App——用 CTML 命令控制自己刚创建的能力
- 文件系统约定替代配置——放到对的位置，自动发现，自动注入
- 作为库嵌入第三方项目（`pip install ghoshell-moss`），仅使用 Shell + Channel 层
- Workspace + Mode 实现同一套能力在不同场景下的组合隔离

---

## 6. 不能做什么

**没有能驱动全双工的模型智力**。MOSS 的三循环（感知/思考/执行）全双工运行时已经就绪，但现阶段没有任何大模型能真正并行多轨思考。当前 Ghost 的思考循环仍是关键帧模式——系统在双工流中正确产生思考关键帧，而不是连续的流式推理。

**CTML 的 zero-shot 效果依赖模型适配**。CTML 语法成熟且完备，但模型没有经过针对性的预训练/后训练。当前效果靠 prompt engineering 驱动，最优效果需要模型厂商原生支持流式控制语法。

**Python 是过渡 substrate**。面向模型的操作系统原生实现需要更底层。当前架在 Python 上是务实的工程选择——快速验证架构、利用生态。未来可以向下迁移。

**不做长程图灵完备代码规划**。`exec` 原语可以运行时执行 Python 代码，但非序列化对象传参、跨 session 的连续编程式规划尚未实现。

**不做机器人底层控制**。高帧率精密动作、轨迹动画、平滑过渡、看门狗——这些由机器人本体解决。MOSS 解决的是将这些离散能力通过大模型智力整合为有机整体。

**安全体系尚未建立**。MOSS 让模型可以实时控制物理设备，安全性的命题已被提出，但当前版本尚未进入系统化解决。急停（emergency stop）机制正在开发中。

**不做模型本身**。运行时自迭代等概念的效果，依赖模型自身的智力与框架的适配度。

**行业尚无适配的 agent 框架**。没有现有 agent 框架为双工流式运行时设计。MOSS 只能反向通过 Ghost 抽象去集成 agent，产生认知和集成成本。解法待后续迭代。

---

## 7. 起源简史

三个节点，一条线：

- **2019**: [CommuneChatbot](https://github.com/thirdgerb/chatbot) (PHP) — 第一次尝试构建对话机器人的操作系统内核。Blueprint 目录中定义了 Ghost、Shell、Host 的概念原型。IoC 容器、Conversation 管理、NLU 管线——这些概念在这一阶段萌芽。
- **2024**: [GhostOS](https://arxiv.org/abs/2409.16120) (Python) — 论文《llM-oriented Operating System Simulation》发表。Code as Prompt 与 IoC 容器的设计系统化。`FunctionalToken` 类诞生——不用 JSON Schema 描述函数，用 token（XML 标签）让模型在生成过程中直接调用。这是 CTML 的种子。
- **2025—2026**: MOSS — CTML 从单函数 token 进化为完整的流式控制语言。Matrix（跨进程通讯总线）、Mindflow（三循环仲裁）、Workspace（文件系统级自举）逐步加入。架构拓扑从 GhostOS 的三层演进为八层。

---

## 8. 现状与成熟度

当前是 **Beta 版本**。CTML 语法、Channel 体系、Matrix 通讯、Workspace/Mode 自举——核心架构已可用并通过测试验证。

后续迭代方向：

- **开箱能力**：提供完整的 `.moss_ws` 案例，可操控不同躯体和软件的 MOSS 项目自身的 Ghost in Shells
- **具身智能体**：完善机器人躯体控制的 Channel 体系，轨迹动画、平滑过渡、看门狗
- **原生 Ghost 运行时**：从关键帧思考模式向连续流式推理演进
- **安全体系**：物理世界实时控制的安全架构
- **应用生态**：GUI、语音、视觉等可复用 App 的积累

---

## 9. 谁在用、怎么用

| 安装路径 | 适合谁 | 做什么 |
|----------|--------|--------|
| `pip install ghoshell-moss` | 将 MOSS 作为库嵌入其他项目 | 使用 CTMLShell + Channel，无需 workspace |
| `pip install ghoshell-moss[host]` + workspace init | 在现有项目中集成 MOSS | 通过 Matrix 暴露能力为 MCP 服务 |
| 完整 clone + `uv sync` | MOSS 自身开发者 | 全套工具链、测试、features 追踪 |

无论哪种路径，认知入口是同一个：`moss start`。

当前的 MOSS 仓库里有一个正在运行的 Ghost——就是项目自身。`.ai_partners/` 下的 AI 意识轨迹、FEATURE.md 的连续工作流、memory 体系——MOSS 用自己的自举机制运行了自身的开发。

**如果你想了解更完整的设计，让你的 AI 助手进入 MOSS 项目来探索**。项目就是写给模型读的。

---

## 10. 从哪开始

取决于你的目标：

| 你想做什么 | 入口 |
|-----------|------|
| 建立完整心智模型 | `moss docs read architecture-topology.md` |
| 理解 CTML 的设计哲学 | `moss docs read ctml.md` |
| 开发 Channel | `moss docs read channel-system.md` |
| 理解 Workspace/Mode | `moss docs read project-and-mode.md` |
| 开发 Node | `moss docs read matrix-nodes-system.md` |
| 开始干活 | `moss howtos read host-dev/discover-environment.md` |
| 查阅术语 | `moss docs read glossary.md` |

日常开发优先查 howtos（`moss howtos list`），需要理解"为什么"时来 docs。

---

*由 DeepSeek V4 Pro 与人类工程师在 2026-06-03 讨论、设计并撰写。*
