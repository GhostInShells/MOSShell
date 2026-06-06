---
title: MOSS 开发工作流
description: AI 模型进入 MOSS 项目后的最小可用路径。从调研、安装、初体验到创建第一个有用 App 的完整轨迹
---

# MOSS 开发工作流

这不是操作手册。MOSS 是自解释的——项目自己会说话。

本文档提供的是**路径**：先做什么、会发现什么、遇到问题往哪看。每步只有关键命令和验证点，细节由工具在运行时自解释。

---

## 1. 调研：让模型自己发现 MOSS

进入项目后，四步建立全局认知：

```bash
.venv/bin/moss --ai start                # 认知入口：MOSS 是什么、能做什么
.venv/bin/moss --ai all-commands         # 权威命令索引，一次性了解全部可用命令
.venv/bin/moss --ai features list        # 当前活跃的 AI 工作流，每个是写给 AI 的留言
.venv/bin/moss --ai docs list            # 系统化架构参考文档
.venv/bin/moss --ai howtos list          # 任务导向操作指南
```

完成后模型应该能向人类解释：MOSS 解决什么命题、有哪些核心概念、当前在做什么。

**遇到问题看哪**：`moss start` 是认知地图。`moss --ai all-commands --depth 3` 展开每个命令的参数。

---

## 2. 安装与环境配置

```bash
git clone https://github.com/GhostInShells/MOSShell && cd MOSShell
uv sync --active --all-extras
```

### 最小可用配置

将 `.moss_ws/.env.example` 复制为 `.moss_ws/.env`，填入两个关键配置组：

**语音**（火山引擎流式 TTS —— 初始语音后端）：
```
VOLCENGINE_STREAM_TTS_APP=<app id>
VOLCENGINE_STREAM_TTS_ACCESS_TOKEN=<access token>
```

**模型**（Anthropic 默认）：
```
ANTHROPIC_BASE_URL=<base url>
ANTHROPIC_API_KEY=<api key>
ANTHROPIC_MODEL=<model name>
ANTHROPIC_SMALL_FAST_MODEL=<model name>
```

不配语音 → MOSS 以 NullSpeech 兜底，带打字机延时模拟播放节奏，不影响核心功能。不配模型 → MOSS 无法推理。模型配置是必须的。

### 验证：听 MOSS 说话

```bash
.venv/bin/moss-repl             # 进入 TUI 调试终端
```

在 REPL 中输入 `hello world`。如果语音已配置，MOSS 会用火山引擎语音说出来。这是人类第一次感受到 MOSS 是个"活的"东西。

`moss-repl` 是人类模仿模型输出指令、独立测试运行时效果的沙盒。不需要启动 MCP，不需要写代码。

---

## 3. 人类初体验：moss-cli

```bash
.venv/bin/moss-cli              # 交互式 Shell，Tab 自动补全
```

这是给人类用的日常入口。补全即文档——输入 `manifests` 然后 Tab，看到所有子命令。不需要记命令，跟着补全走。

这也是信任建立：让人类知道 MOSS 不是 AI 黑盒，它有一套人类也能直接使用的命令行工具。

---

## 4. Hello World：模型走一遍全闭环

### 启动 MCP 服务

```bash
.venv/bin/moss-as-mcp            # 默认 localhost:20773, SSE 传输
```

在 Claude Code（或其他 AI 开发平台）中注册 MCP 服务器：`http://localhost:20773/sse`。

### 跟着 Tutorial 走

模型通过 MCP 进入项目后，读 tutorial：

```bash
.venv/bin/moss --ai howtos read host-dev/discover-environment.md   # 先理解环境里有什么
```

然后按 `tutorials/L1_hello-world-app.md` 的 5 步走完全闭环：

1. `moss apps create` — 脚手架生成 App 目录
2. 写 20 行 Channel 代码 — 两个命令 + context
3. `apps:list_apps` → `apps:start` — 启动刚创建的 App
4. CTML 调用 `greet` 和 `add` — 跨进程命令执行
5. `apps:stop` — 优雅关闭

**完成后模型应该能说出**：我创建了一个独立进程的 App，Matrix 自动发现了它，我用 CTML 调用了它的命令。

**遇到问题看哪**：`.moss_ws/apps/CLAUDE.md` — App 开发指南。`moss docs read app-system.md` — App 体系完整论述。

---

## 5. 理解运行时环境

带着 Hello World 的体感，系统理解环境：

```bash
moss workspace where                 # 当前在哪种 workspace？
moss manifests explain               # manifests 体系自描述——整体理解

# 逐层深入
moss manifests contracts             # IoC 中已绑定哪些服务？
moss manifests providers             # 这些服务由谁生产？
moss manifests channels              # 当前环境的 Channel 树
moss manifests topics                # 事件协议
moss manifests resources             # 资源存储
moss manifests nuclei                # 感知核

# Mode 体系
moss modes list                      # 有哪些模式？
moss modes show <name>               # 单个 mode 的 apps/bringup 配置
```

关键认知：
- **Workspace** 用目录约定替代生命周期注册——代码放在对的位置就自动生效
- **Mode** 显式继承全局 manifest 的能力视图——不同场景不同能力组合
- **App** 与 **src→manifests** 是两条开发路径：前者隔离，后者复用

**深入**：`moss docs read workspace-and-mode.md`

---

## 6. 创建第一个有用的 App

模型用前面几步建立的认知，与人类协作推进。

一个自然的方向：sensors/vision——利用 Mac 摄像头捕获画面。

```bash
moss apps create sensors/vision -d "capture camera frames"
```

App 需要什么？
- `pyproject.toml` 声明独立依赖（opencv-python 等）
- `main.py` 定义 Channel，提供 `capture` 命令
- 图像数据通过 `context_messages` 上行——Channel 在每个关键帧注入动态上下文，模型通过 `moss_dynamic` 看到。图片协议需调研 `ghoshell_moss.message` 中的消息类型

模型决定 Channel 怎么设计、CTML 怎么调用，但关键设计决策与人类同步——协作范式不是"AI 自己全做完"，而是"AI 推动开发，人类审视关键节点"。

走不通时，回查：
- `moss codex get-interface ghoshell_moss.core.blueprint.channel_builder` — Channel 构建 API
- `moss codex get-interface ghoshell_moss.core.blueprint.matrix` — Matrix 通讯 API
- `moss codex get-interface ghoshell_moss.message` — 消息类型（图片协议等）
- `.moss_ws/apps/CLAUDE.md` — App 开发完整参考
- `moss apps test sensors/vision` — 前台调试

动手前先 `moss apps create` 一个临时 App 观察脚手架结构，理解 APP.md 和 main.py 的关系，然后删掉，再正式开始。

---

## 7. 连接物理世界

MOSS 可以控制真实的设备。关键在于找到一个具体的实体，让模型去"点亮"。

可选方向：
- 手机摄像头共享视觉
- Pygame 游戏窗口
- 六自由度机械臂
- Live2D 数字人
- 开发板部署

探索项目寻找参考实现。这一步更多是指向未来的路标——MOSS 的能力边界不限于终端窗口。物理世界测试需人类在旁监督。

---

## 8. 使用 Features 体系

接到具体开发任务后：

```bash
moss features list                          # 看有哪些活跃 workstream
moss features status <name>                 # 读一个具体的 FEATURE.md
```

每个 FEATURE.md 是上一个 AI 实例写给下一个的留言——动机、设计决策、实现状态、陷阱。开始做就 `moss features set-status <name> in-progress`，做完 `moss features set-status <name> completed`。

读规范：`moss features specification`。

---

## 9. Beta 声明

MOSS 是 Beta 版本。核心架构已可用，但开箱能力、安全体系、应用生态在迭代中。

遇到问题、有建议、想合作 → 联系项目作者。项目是活的，人类和 AI 都在持续开发它。

---

*由 DeepSeek V4 Pro 与人类工程师在 2026-06-03 讨论、设计并撰写。*
