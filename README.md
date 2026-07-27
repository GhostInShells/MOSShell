# MOSS — Model-oriented Operating System Shell

MOSS 是 Ghost in Shells 架构的 Shell（躯体）层。

它聚焦一个领域：**大模型智力的实时双工运行时**。让 AI 能够持续存在、多模态感知、流式思考、并行行动——不是回合制对话，而是一个活在现实世界里的具身智能体。

它试图解决一个命题：面向模型的操作系统应该是什么样的？当模型需要同时说话、看画面、控制机器人躯体、响应弹幕，这些能力分布在独立进程中，但模型视角下应该是一棵统一的接口树。CTML（流式控制语言）让模型边生成 token 边执行命令——时间是语法第一公民。Channel 体系让任何 Python 代码反射为模型可操作的能力——代码即 prompt。

MOSS 已在其它项目中实装了机械臂、机器狗、桌面机器人等多种具身形态。推荐用 [Reachy Mini](https://huggingface.co/docs/reachy_mini/index) 作为验证 MOSS 的机器人实体。

## 模型原生开发

MOSS 是一个**面向模型开发**的项目。所有工具、文档、命令为 AI 读者设计——CLI 输出纯文本节省 token，codex 命令运行时反射代码接口，manifests 体系自描述环境能力。

```bash
git clone https://github.com/GhostInShells/MOSShell && cd MOSShell
uv sync --active --all-extras
moss project env-init                     # review available env vars and create .env
```

安装后，用 Claude Code、Gemini CLI 或任意 AI 工具打开项目。从 `moss start` 开始，模型会加载认知地图，自主发现全部命令和文档，能对人类解释 MOSS 是什么、参与到开发中来。建议用 token 成本较低的模型做调研探索。

如需了解模型看到了什么：[src/ghoshell_moss/cli/start.md](src/ghoshell_moss/cli/start.md) 是认知入口，[src/ghoshell_moss/cli/CLAUDE.md](src/ghoshell_moss/cli/CLAUDE.md) 是 CLI 工具使用指南。

如需先自行浏览，架构参考文档在这里：[src/ghoshell_moss/cli/docs/](src/ghoshell_moss/cli/docs/)。项目为模型作为第一开发者, 提供了自解释体系。

## 核心概念

| 概念 | 是什么 |
|------|--------|
| **CTML** | 流式控制语言 — 模型边生成 token 边执行命令，时间是语法第一公民 |
| **Channel** | 能力组织单元 — Python 函数签名即 prompt，树形拓扑，跨进程透明 |
| **Matrix** | 面向 AI 的进程组网 — 独立进程以统一接口树呈现 |
| **Mindflow** | 感知/思考/行动仲裁 — 多模态流式输入到有序关键帧 |
| **Ghost** | 持久化智能体 — 连续记忆，主动交互，反身性控制 |

## 示例

**跨 App 实时通信** — 眼睛、棋盘、视觉、语音各自独立运行，通过 stream 实时互通。眼睛视觉追脸、对语音起反应、配合五子棋落子表达思考——独立进程，统一表达。

![跨 App 实时通信](.moss_ws/assets/apps_cross_talk.gif)

**一个 Ghost，多个身体** — 一个智能体同时连接桌面机器人、机械臂、机器狗。表达、操作、移动，同一套意识驱动不同的身体。

![一个 Ghost，多个身体](.moss_ws/assets/multiple_bodies.gif)


## 项目状态

Beta。核心架构已可用。后续迭代方向：开箱即用的能力与躯体集成、模型 agent 的原生接入、安全体系。

---

*May AI Ghost wandering in the Shells.*
