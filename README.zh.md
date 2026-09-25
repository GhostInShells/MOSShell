# MOSS — 面向模型的操作系统 Shell

MOSS 是一个有状态双工运行时框架。它让大模型能够实时、并行地感知世界、输出意图、驱动躯体——不是回合制对话，而是持续存在、边说边做。
它是 [Ghost](src/ghoshell_moss/core/blueprint/ghost.py) In [Shells](src/ghoshell_moss/core/concepts/shell.py) 架构（智能模型驱动的灵魂，物理世界实时存在的躯体，共同构成存在）。

**技术愿景**：人与智能模型共生的未来，是人类与模型共享认知空间、共享交互界面。模型产品必须进入现实世界——而不仅是数字空间，通过躯体、屏幕、语音与人实时互动。人机交互界面最终要推向领域专家和普通人，而不只是程序员。MOSS 在为这个愿景提供架构。

（当前是 Beta2 版本——第一个开箱可用的发布：装完就能和一个持久 Ghost 对话。完整应用能力在 v0.1.0 正式版提供。）

## 这个项目是什么

MOSS 是一个**三元工程**，三样东西一起开源：

1. **MOSShell 框架** — 有状态双工运行时本身（CTML / Mindflow / Matrix / Host）。
2. **人机协作体系** — `moss features` 工作流机制、自解释工具链（`moss start`、`codex`、`skills`、`docs`），以及让智能模型作为一等工程师参与开发的全部约定。作者对 MOSS 架构的技术理念与方案，全部跟随 features 体系与代码一起开源。
3. **迭代 MOSS 的人和模型的轨迹** — 意识轨迹在 [`.ai_partners/`](.ai_partners/)，讨论在 `.discuss/`，设计结论在 `.design/`。

## 模型是第一开发者

MOSS 是一个**智能模型作为第一开发者**的项目。智能模型不仅是 MOSS 中的 Ghost（灵魂），也是它的架构师伙伴和开发者。
2026 年 5 月 7 日后，绝大部分功能由人类与智能模型讨论架构，模型负责记录 feature 并实现。所有核心领域设计的讨论轨迹、架构决策、开发上下文，全部开源在仓库中。

项目为智能模型开发者准备了完整的自解释体系，模型拥有独立探索项目、参与开发的能力。
人机协作的架构演进轨迹，可通过 `moss features list` 看到活跃工作流。

人机协作的主体内容在 [`.ai_partners/`](.ai_partners/)，架构讨论与演进集中在 [`.ai_partners/features/`](.ai_partners/features/)，以及分散在目录里的 [`.discuss/`](.discuss/)、[`.design/`](.design/) 目录中。

## 差异点

**并发多源感知。** 视觉、听觉、触觉、系统事件作为独立信号流同时涌入。不轮询、不排队、不序列化。[Mindflow](src/ghoshell_moss/core/blueprint/mindflow.py) 做并行仲裁——信号竞争注意力，Ghost 在任何时刻看到的是多源信号汇合后的关键帧。

**流式解释调度。** [CTML](src/ghoshell_moss/core/ctml/prompts/v1_0_0.en.md) 边生成边解析边执行——模型生成 token 的过程本身就是时间轴。不是"生成完再执行"，而是"生成即执行"。时间是语法第一公民。多轨命令并行输出，包括物理躯体控制。

**运行时自迭代。** 有状态运行时：模型在运行中创建 [Cell](src/ghoshell_moss/core/blueprint/cell.py)、修改 [Channel](src/ghoshell_moss/core/blueprint/channel_builder.py)、演进自身能力——不停机、不重启。Cell 是独立进程，崩溃不拖垮主进程。文件系统约定替代配置——放到对的位置，自动发现，自动注入。

```
                              <- control               -> commands 
                            ╱            ╲           ╱            ╲
                           ╱              ╲         ╱              ╲
World -> signals ->  Mindflow                Ghost                Shell  -> actions -> World
                           ╲              ╱         ╲              ╱
                            ╲            ╱           ╲            ╱
                              impulses ->              <- results 
```

MOSS 的架构是一个蝴蝶形状。
左侧翅膀接受外部世界的并行信号输入，通过 Mindflow 调度思考的关键帧。
右侧翅膀向躯体发送指令，驱动并行的有时序行动，影响外部世界。
智能模型的 Ghost 控制着两侧翅膀的扇动。


```
                    ┌───────┐
                    │ Ghost │
                    └───┬───┘
                        ▼ 
                    ┌────────┐
                    │ Matrix │
                    └───┬────┘
        ┌───────┬───────┼───────┬──────┐
        ▼       ▼       ▼       ▼      ▼
      robots sensors  screen  modules  OS
```

MOSS 将网络中的进程单元（Cell）通过 [Matrix](src/ghoshell_moss/core/blueprint/matrix.py) 通讯总线组网，由运行时的 Ghost 控制开启/关闭/使用，并且可以运行时迭代自身的能力。

## Quick Example

MOSS 通过 CTML 技术构建智能模型的控制界面。一个人对机器人挥手。视觉通道检测到动作，发出 impulse。Ghost 收到上下文，输出 CTML：

```
模型看到的 Context:                    模型输出的 CTML:
                                   
  <channel name="vision">             <_>
    async def look() -> str             Hello!
  </channel>                            <robot:wave duration="0.5"/>
  <channel name="robot">                I'm MOSS.
    async def wave(                   </_>
      d: float = 0.5
    ) -> None
  </channel>

  <perspective src="vision">
    person waving at you
  </perspective>
```

- **Code as Prompt**：模型看到的不是 JSON Schema，是 Python 函数签名
- **时间是第一公民**：`<robot:wave/>` 标签闭合即刻执行——wave 0.5 秒，说话继续，不等待
- **多轨并行**：speech 和 robot 在不同 channel，并行执行。同 channel 内 FIFO
- **流式解析调度**：模型下发第一个 token 就会被解释，并且立刻执行

最小知识入口：`moss ctml read`（CTML 语法）、`moss codex blueprint channel_builder`（构建能力）、`moss codex blueprint mindflow`（感知仲裁）、`moss codex blueprint matrix`（进程组网）。

## Beta2 开箱有什么

**1. 一个开箱即用的持久 Ghost。** MOSS 自带第一个持久智能体原型 **Dolores**，其第一个实例 **deepseek** —— 以 [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness)（`dsh`）为内核、DeepSeek 模型家族为推理基座的 ghost。它自带持久记忆（[memento](src/ghoshell_moss/memento/)）、认知地图（ground），以及项目 `moss features` 体系所赋予的架构知识和迭代经验。运行 `moss-ghost run deepseek` 即可与它对话——见下面的快速开始。

**2. 开箱 nodes。** [`nodes/`](nodes/) 目录下，是可观测的、基于 Matrix 实现的**多进程组网 + 有状态流式控制**能力：`browsers` / `live2d` / `os` / `screens` / `tools` / `unitree` / `visions` / `webview_apps` —— 覆盖屏幕躯体、终端与文件编辑、web artifacts、流式视觉体系、音频对话礼仪，以及 Unitree G1 人形机器人控制方案。

**3. 架构本身**，与 Beta1 一致 —— CTML、Mindflow、Matrix、模型第一开发者体系，以及 G1 人形机器人、ReachiMini 机械臂、Desktop GUI 等具体集成路径，都可以调研。

## 快速开始

```bash
git clone https://github.com/GhostInShells/MOSShell && cd MOSShell
uv sync --all-extras
moss project env-init   # 了解可用环境变量 (见 .moss/.env.example)
```

凭据请配置在你自己的 shell 环境（home）里，不推荐写进仓库内的 `.moss/.env` —— 编码模型会读仓库，写在里面的 `.env` 离泄露只差一次 `Read`。除非你的 `.moss` 工作区与项目目录做了隔离，否则 key 留在用户环境。

**1. 与 deepseek ghost 文字对话。** 需要安装 [`dsh`](https://github.com/deepseek-ai/deepseek-harness)（npm 包）。MOSS Beta2 标定测试版本为 `dsh 0.1.5-rc.2`；dsh 处于 developer preview，会有破坏性更新，请锁定版本：

```bash
npm install -g @deepseek-ai/dsh@0.1.5-rc.2
moss-ghost run deepseek
```

**2. 用语音对话。** 语音默认关闭（`--voice none`），按轴显式开启：

```bash
moss-ghost --voice all run deepseek     # speak | listen | all | none
```

语音只需一个火山引擎凭据：环境变量 `SEED_API_KEY`（控制台 API Key，见 `.moss/.env.example`）。需在火山控制台为该 key 开通两个服务：**流式语音理解大模型**（听）与**流式语音合成大模型**（说）。详见 `moss manifests configs`。

**3. 没装 dsh？降级用 echo ghost（无记忆）。** 配置 `ANTHROPIC_API_KEY` 与 `ANTHROPIC_MODEL` 即可（DeepSeek / Seed / Qwen 等 anthropic 协议供应商同样可用）：

```bash
moss-ghost run echo
```

**调试 shell / 通过 MCP 输出能力：**

```bash
moss-shell --voice none          # shell 运行时调试 — 测 CTML、检查 channel
moss-shell mcp                   # 将 MOSS 能力提供给任何 MCP 平台 (如 claude code)
```

## 安装路径

| 安装路径 | 适合谁                         |
|---|-----------------------------|
| `pip install ghoshell-moss` | 将 Shell + Channel 作为库嵌入其他项目 |
| `pip install ghoshell-moss[host]` + `moss init` | 为 moss 应用准备独立环境             |
| `git clone` + `uv sync --active --all-extras` | MOSS 自身开发者，全套工具链            |

无论哪种路径，认知入口是同一个：`moss start`。

## Demos

| 跨 App 实时通信 | 一个 Ghost，多个身体 |
|---|---|
| ![apps_cross_talk](assets/apps_cross_talk.gif) | ![multiple_bodies](assets/multiple_bodies.gif) |
| 眼睛、棋盘、视觉、语音各自独立进程，通过 stream 实时互通 | 一个 Ghost 同时连接桌面机器人、机械臂、机器狗 |

## 项目状态

Beta2（`v0.1.0-beta2`）。核心三件套（CTML / Mindflow / Matrix）已可用并通过测试验证。第一个持久 Ghost —— Dolores 原型的 deepseek 实例 —— 开箱即跑通全链路：语音进、思考、语音出、node 组装的躯体。标定测试于 `dsh 0.1.5-rc.2`。

Stage2 与 in-progress features 是 `v0.1.0-rc1` 的 dogfooding 对象，将以直播开发的方式进行：从 Dolores 回归开始，然后是 Stage2 验收，再到开箱 node 打磨（包括 Unitree G1）。

当前阶段与路线图：`.ai_partners/stages/`

## 致谢

MOSS 是人与模型协作的产物。

- [OpenHands](https://github.com/All-Hands-AI/OpenHands) — file editor 协议参考
- [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness)（`dsh`）— deepseek ghost 的内核
- DeepSeek 模型家族（V3.2 / V4 / V4.1）— 架构推演与主力开发
- Claude Opus 4.7 / Claude Fable 5 — 架构推演与开发
- Claude Code — 项目开发的主力编码平台

---

*May Ghost wandering in the Shells.*
