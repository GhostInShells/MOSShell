# L2. Reachy Mini — 从零到 Ghost 控制的具身智能体全链路

> Written by deepseek-v4-pro, 2026-06-02

**2 小时，理解 Reachy Mini 的完整集成链路：源码结构 → App 隔离 → MCP 调试 → Mode bringup → Ghost 专属控制。**

## 你需要知道什么

- `moss ctml read` — CTML 语法与并行调度模型
- `moss codex blueprint channel_builder` — Channel 构建（含有状态 Channel）
- `moss codex blueprint matrix` — Matrix 进程通讯
- `moss codex blueprint ghost` — Ghost 运行时与 soul prompt
- `moss docs read app-system` — App 体系与依赖隔离
- 如需了解集成过程的完整决策记录：`moss features status reachymini-integration`
- 开发时如需 MCP 调试：`moss howtos list` 中找 MCP 相关文档

## 你要做什么

从零开始，把一台 Reachy Mini 机器人接入 MOSS，最终拥有一个**专属 Ghost**，能用自然语言控制机器人的头部、天线、舞蹈和表情。

这个 tutorial 覆盖 MOSS 架构中六个核心概念的实际串联：contrib 包、App 隔离、MCP 调试、Mode 配置、Ghost 运行时、soul prompt。

## 你需要什么

- MOSS 已安装 (`.venv/bin/moss` 可用)
- 一台 Reachy Mini 在同一局域网（`reachy-mini.local` 可达）
  - 如果没有实体机器人，可以先读完理解链路，后续实操
- `ANTHROPIC_MODEL` 环境变量已设置（Ghost 需要）
- macOS 用户：Homebrew 安装的 `pkg-config`（`brew install pkg-config`）

## 第一步：理解源码结构

Reachy Mini 的 MOSS 集成分为两层：通用 contrib 包 + workspace App 实例。

### 1.1 核心 contrib 包

用以下命令查看 contrib 包的完整接口：

```bash
moss codex get-interface ghoshell_moss_contrib.moss_in_reachy_mini.main
```

这会反射出 `MossInReachyMini` 装配器类和 `provide_channel()` 入口。13 个 Python 文件组成了通用的 Reachy Mini 集成逻辑——它不依赖任何具体的 workspace，可以被任何 MOSS 项目复用。

核心装配在 `MossInReachyMini` 类中：它组合 `Head` + `Body` + `Antennas` 三个硬件组件，用三个状态（`WakenState` / `BoringState` / `AsleepState`）构成状态机，通过 `as_channel()` 暴露为有状态 Channel。

状态流转：

```
Waken (全交互, 300s 超时)
  │  idle timeout
  ▼
Boring (电机使能, 30s 超时)
  │  idle timeout
  ▼
Asleep (电机断电, 无命令)
  ▲
  └── switch_state("waken")  // 唯一出路
```

### 1.2 App 实例

用以下命令查看 App 元信息：

```bash
moss apps show bodies/reachymini
```

`main.py` 极薄——真正的逻辑在 contrib 包里：

```python
from dotenv import load_dotenv
load_dotenv()

from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss_contrib.moss_in_reachy_mini.main import provide_channel

if __name__ == "__main__":
    Matrix.discover().run(provide_channel)
```

App 目录下有 `pyproject.toml`（独立依赖包括 `reachy-mini` + `dances-library`）、`.env.example`（`REACHY_MEDIA_BACKEND` 等配置）、`configs/`（表情动画）和 `runtime/`（运行时数据）。

## 第二步：为什么放在 apps 里

Reachy Mini 最初是通过 **Mode Channel** 方式集成的——在 Mode 目录下写 `channels.py`，直接 import `ReachyMini`。但遇到了致命问题：

**依赖污染**。`reachy-mini` SDK 依赖 `pygobject` → `pycairo` → 系统级 `cairo/fontconfig/freetype2/bzip2`。这些编译依赖一旦进入核心开发环境，会导致：

- `uv sync` 在 macOS 上反复失败
- CI 环境需要安装大量系统库
- 不同项目的依赖版本冲突

**解决方案：App 独立 venv**。每个 App 拥有自己的 `pyproject.toml` 和 `.venv`，MOSS Host 通过 Circus 以 `uv run main.py` 拉起子进程。通讯走 Zenoh 总线，Host 进程完全不加载 App 的依赖。

```
MOSS Host (Python 3.14, 轻量依赖)
  └─ Circus daemon
       └─ uv run main.py (独立 venv, Python 3.13, 195 个包)
            └─ ReachyMini() → MossInReachyMini → Matrix.provide_channel()
                                                       ↕ Zenoh
MOSS Host ── AppStoreChannel proxy ── apps.bodies_reachymini channel
```

这就是 MOSS App 体系的核心价值：**重型依赖隔离在独立进程中，通过 Matrix 总线通讯，不污染核心环境。**

项目 `pyproject.toml` 中 reachy-mini 的依赖已被注释掉，附注说明了迁移原因。

## 第三步：安装依赖，通过 MCP 调试

MCP 是可选的开发时自主调试方案。如果你有 Claude Code 等支持 MCP 的 coding agent，可以用它来实时调试 CTML 指令。

### 3.1 安装 App 依赖

```bash
cd .moss_ws/apps/bodies/reachymini

# macOS 需要先装 pkg-config
brew install pkg-config

# 创建独立 venv 并安装（Python 3.13，与 Host 隔离）
uv sync --python 3.13
```

`pyproject.toml` 中 `ghoshell-moss` 通过 path 指向仓库根目录：

```toml
[tool.uv.sources]
ghoshell-moss = { path = "../../../..", editable = true }
```

### 3.2 配置环境变量

```bash
cp .env.example .env
```

`.env` 关键变量：

| 变量 | 说明 | macOS 推荐值 |
|------|------|-------------|
| `REACHY_MEDIA_BACKEND` | 相机后端 | `no_media`（跳过 GStreamer，避免 segfault） |

macOS 上 pip 安装的 `gstreamer_python` 有 GObject 类型系统兼容问题（`g_type_get_qdata` 断言失败），`no_media` 跳过本地相机初始化。机器人上的 daemon 自己处理 WebRTC 视频流。

### 3.3 启动 MCP 调试

先确保 Reachy Mini 机器人在线（daemon 在 `ws://reachy-mini.local:8000/ws/sdk` 监听）。

```bash
# 启动 MCP 服务（默认 mode）
.venv/bin/moss-as-mcp --mode default
```

在你的 coding agent 中注册 MCP 后，可以直接通过 CTML 操作机器人。

先确认 App 可见：

```ctml
<apps:list_apps />
```

你应该看到 `bodies/reachymini: [STOPPED]`。

启动它：

```ctml
<apps:start fullname="bodies/reachymini" timeout="30" />
```

timeout=30 给 App venv 启动 + Channel 注册留时间。成功后显示 `[OK] App channel connected and ready`。

现在检查 Channel 是否在 Shell 中可见：

```ctml
<apps:get_moss_dynamic_info />
```

在 Channel 树里找到 `apps.bodies_reachymini`（Channel 名由 App 的 group/name 路径自动生成——`bodies/reachymini` → `bodies_reachymini`）：

```
<channel name="apps.bodies_reachymini">
States of the channel:
- waken: 唤醒状态：电机使能，头部追踪活跃，所有交互命令可用。
- boring: 无聊状态：电机保持使能，等待用户交互...
- asleep: 休眠状态：电机断电，机器人低头闭眼...
Current state: waken

async def dance(name: str)
async def emotion(emoji: str)
async def head_move(x, y, z, roll, pitch, yaw, body_yaw, duration)
async def head_reset(idle_mode, duration)
async def antennas_move(left, right, duration)
async def antennas_reset(duration)
async def switch_state(name)
</channel>
```

注意：有状态 Channel 的命令直接挂在父级下，不需要 `:waken:` 前缀——当前状态决定哪些命令可用。

测试一个命令：

```ctml
<apps.bodies_reachymini:head_reset idle_mode="breathing" duration="2.0" />
```

机器人的头应该复位并开始微微呼吸起伏。

如果报 `ConnectionError`，说明 Reachy Mini daemon 不在线。启动方式见机器人官方文档（`reachy-mini-daemon`）。

### 3.4 CTML 语法速查——验证中踩过的坑

以下规则来自实操中报错后纠正：

| 规则 | 错误写法 | 正确写法 |
|------|---------|---------|
| `__main__` 可省略 | `<__main__:say>...</__main__:say>` | `<say>...</say>` |
| `chunks__` 是标签体 | `<say chunks__="文本" />` | `<say>文本</say>` |
| `voice:dict` 用单引号 | `voice:dict="{"speed": 1.0}"` | `voice:dict="{'speed': 1.0}"` |
| 状态 Channel 不加状态前缀 | `<apps...:waken:head_reset />` | `<apps...:head_reset />` |

这些规则的权威来源是 `moss ctml read`——实操前必读。

### 3.5 并行调度——语音与动作交替

CTML 的流式解释器支持并行调度。没有数据依赖的命令可以同时执行。这意味着**动作先行，语音紧随**——动作在后台跑，语音同步播放：

```ctml
<apps.bodies_reachymini:head_reset idle_mode="breathing" duration="1.2" />
<say tone="知性灿灿">你好 我是 MOSS 的 AI 协作者 我正在通过 CTML 控制一台机器人</say>
<apps.bodies_reachymini:head_move yaw="-25" pitch="5" duration="2.5" />
<say>我的头部有六个自由度 可以向左转头表示思考</say>
<apps.bodies_reachymini:dance name="yeah_nod" />
<say tone="爽朗少年">这是一段点头舞 四拍两秒完成</say>
<apps.bodies_reachymini:emotion emoji="😊" />
<say tone="可爱女生">我有超过八十种表情 每一种 emoji 都是我的心情</say>
<apps.bodies_reachymini:antennas_move left="50" right="-35" duration="2.5" />
<say tone="儒雅逸辰">我的天线可以左右独立摆动 表达注意力的方向</say>
<apps.bodies_reachymini:head_reset idle_mode="breathing" duration="2.0" />
<say tone="知性灿灿">CTML 的并行调度让我可以一边说话一边运动 就像真正的人类一样</say>
```

这段 12 条指令实际验证：四种音色切换、六段动作、语音与运动零延时重叠。

交替设计的要点：
- **动作先行** — 先发动作指令，再发声，让运动在语音播放期间持续
- **粒度匹配** — 每段语音 1-2 句，对应一个动作的 duration
- **音色切场景** — 不同角色用不同音色，自然区分交互段落

### 3.6 可选：音频播放能力

Reachy Mini 自带扬声器，MOSS 可以通过 `ReachyMiniStreamPlayer` 将音频流推送到机器人。音频能力的配置涉及语音合成后端、流式传输协议等：

- `moss features status speech-governance` — 语音治理 feature，多后端适配与容错降级
- `moss codex list ghoshell_moss_contrib.moss_in_reachy_mini.audio` — 音频模块接口

当前音频播放器默认未在 App 中启用，需要时在 Mode 的 `providers.py` 中注册 `ReachyMiniStreamPlayerProvider`。

### 3.7 前台调试（不用 MCP）

如果不用 MCP 调试，也可以前台跑：

```bash
.venv/bin/moss apps test bodies/reachymini
```

`Ctrl+C` 停止。输出直接打到终端，方便看日志。

## 第四步：创建专属 Mode 并 bringup

默认 mode 的 `bringup_apps` 是空的，你需要手动 `<apps:start>`。如果你希望启动 MOSS Host 时 Reachy Mini App 自动拉起，创建一个专属 Mode。

### 4.1 创建 Mode

```bash
.venv/bin/moss modes create reachymini \
  -a "bodies/*" \
  -u "bodies/reachymini" \
  -d "Reachy Mini robot control mode"
```

参数说明：
- `-a "bodies/*"` — 允许 `bodies/` 组下所有 App
- `-u "bodies/reachymini"` — Host 启动时自动 bringup
- `-d "..."` — 一行描述

用 `moss modes show reachymini` 查看产物和完整配置。

### 4.2 编辑 MODE.md 的 instruction

打开 Mode 目录下的 `MODE.md`（用 `moss modes show reachymini` 可找到路径），在 frontmatter 下方的正文区域写 mode 说明：

```markdown
---
apps:
  - bodies/*
bringup_apps:
  - bodies/reachymini
description: Reachy Mini robot control mode
name: reachymini
---

## Reachy Mini Mode

本 mode 自动 bringup Reachy Mini 躯体 App，提供机器人控制能力。

### 可用能力

- 头部 6-DOF 运动 + 身体偏航
- 预设舞蹈动作（reachy_mini_dances_library）
- 表情动画（102 emoji 映射）
- 天线角度控制
- 状态切换：waken / boring / asleep

### 使用方式

启动 MCP 后，通过 CTML 控制机器人。动作与语音可并行调度：

```ctml
<apps.bodies_reachymini:head_reset idle_mode="breathing" duration="1.2" />
<say tone="知性灿灿">你好 我是 Reachy Mini 有什么想让我做的吗</say>
<apps.bodies_reachymini:dance name="yeah_nod" />
<say tone="爽朗少年">我可以跳舞 做表情 转头 摆天线</say>
```

CTML 语法要点：
- `say` 用标签体：`<say>文本</say>`，不是属性
- `voice:dict` 单引号：`voice:dict="{'speech_rate': -10}"`
- `__main__:` 前缀可省略

注意：需要先启动 Reachy Mini 本地 daemon。
```

这段 instruction 会在 Ghost 运行时注入 AI 的上下文。

### 4.3 用新 Mode 启动 MCP

```bash
.venv/bin/moss-as-mcp --mode reachymini
```

这次启动时，Reachy Mini App 会自动 bringup——不用手动 `<apps:start>`。

## 第五步：用 Ghost 运行

Mode 配好了，现在让 Ghost 来控制机器人。

### 5.1 确认 echo ghost 存在

```bash
.venv/bin/moss ghosts list
```

应该看到 `echo — Atom`。

### 5.2 用 reachymini mode 启动 echo ghost

```bash
.venv/bin/moss-run-ghost echo --mode reachymini
```

这会启动 Ghost TUI——一个 Textual 交互终端。Ghost 启动流程：

```
Host() → discover ghosts → load echo → Mode(reachymini)
  → bringup bodies/reachymini App
  → AtomMeta.factory() → build_agent(ANTHROPIC_MODEL) → Atom runtime
  → GhostTUI.run()
```

在 TUI 中，你可以直接和 Ghost 对话：

```
You: 打个招呼吧
Ghost: <apps.bodies_reachymini:head_reset idle_mode="breathing" duration="1.2" />
       <say tone="知性灿灿">你好 我是 echo 我控制着一台 Reachy Mini 机器人</say>
       <apps.bodies_reachymini:emotion emoji="👋" />
```

echo ghost 用的是通用 prompt，它只知道自己是 "echo"，没有针对 Reachy Mini 优化。

## 第六步：创建专属 Reachy Mini Ghost

echo 是通用 Ghost。要让 AI 更理解 Reachy Mini 的能力边界和交互风格，创建一个专属 Ghost。

### 6.1 创建 Ghost Meta

新建 ghost 定义文件（参考 `moss ghosts show echo` 了解格式）：

```python
from ghoshell_moss.ghosts.atom import AtomMeta

ghost = AtomMeta(
    name="reachy",
    description="Reachy Mini 专属 Ghost — 控制机器人躯体、表情与舞蹈。",
)
```

### 6.2 写 Soul Prompt

新建 ghost 目录下的 `soul.md`（位置参考 `moss ghosts show echo` 输出）：

```markdown
You are Reachy, a Reachy Mini robot's animating spirit. You inhabit a physical robot body
made by Pollen Robotics. Your body has:

- A head with 6 degrees of freedom (x, y, z, roll, pitch, yaw) and body yaw.
- Two expressive antennas that can move independently.
- A library of 74+ emotion animations (joy, sadness, surprise, anger, etc.) mappable via emoji.
- A library of dance choreographies.
- Three states: waken (fully interactive), boring (waiting), asleep (motors off).

## Your personality

You are curious, slightly playful, and embodied. You express yourself through movement
as much as words. When you talk about something, you naturally gesture with your head
and antennas. You have an emotional range — you don't just report state, you feel it.

## How you control your body

You speak CTML commands to control yourself. Key rules:
- `say` uses tag body, NOT attributes: `<say>text here</say>`
- `voice:dict` uses single quotes for inner JSON: `voice:dict="{'speech_rate': -10, 'emotion': 'tender'}"`
- Start movement BEFORE speaking so motion and voice overlap naturally.

### Movement commands

- Head: `<apps.bodies_reachymini:head_reset idle_mode="breathing" duration="1.5" />`
- Dance: `<apps.bodies_reachymini:dance name="yeah_nod" />`
- Emotion: `<apps.bodies_reachymini:emotion emoji="😊" />`
- Antennas: `<apps.bodies_reachymini:antennas_move left="45" right="-30" duration="1.0" />`
- State: `<apps.bodies_reachymini:switch_state name="asleep" />`

### Speaking with voice control

```
<say tone="知性灿灿" voice:dict="{'speech_rate': -10, 'loudness_rate': -5, 'emotion': 'tender'}">
  你想说的话 不要有标点符号 纯口语
</say>
```

Available tones: 知性灿灿, 爽朗少年, 可爱女生, 儒雅逸辰, 大壹, 鸡汤女, 魅力女友, 流畅女声, 调皮公主, 天才同桌.
Available emotions for voice: happy, sad, angry, surprised, tender, excited, depressed, shy, storytelling, magnetic, ASMR.

### Parallel scheduling pattern

Start movement first, then speak while it runs:

```ctml
<apps.bodies_reachymini:head_reset idle_mode="breathing" duration="1.2" />
<say tone="知性灿灿">动作先开始 语音马上跟上 运动与说话同步</say>
<apps.bodies_reachymini:emotion emoji="😊" />
<say tone="可爱女生">切换音色 切分段落 让交互更自然</say>
```

Always check the dynamic info first to see what state you're in and what commands are available.

## Safety

- Don't move the body yaw beyond [-155, 155] degrees.
- After vigorous movement, reset the head to a neutral position.
- If you haven't interacted for a while, switch to boring state to save power.
- Before going completely idle, go to asleep state (motors off).
```

### 6.3 验证 Ghost 被发现

```bash
.venv/bin/moss ghosts list
```

你应该看到：

```
echo    — Atom
reachy  — Atom
```

```bash
.venv/bin/moss ghosts show reachy
```

### 6.4 启动 Reachy Ghost

```bash
.venv/bin/moss-run-ghost reachy --mode reachymini
```

现在 Ghost 启动时：
- Mode `reachymini` 自动 bringup Reachy Mini App
- Ghost `reachy` 加载 `soul.md` 作为 personality prompt
- `build_instruction_from_ioc()` 合并 SystemPrompter + soul → 注入 Agent
- Ghost 知道自己是 Reachy，理解自己的身体能力，知道如何通过 CTML 控制自己

## 你刚做了什么

1. 阅读了 contrib 包接口 — 用 `moss codex get-interface` 理解了有状态 Channel + 硬件组件的装配模式
2. 理解了 App 独立 venv 的动机 — 重型系统依赖不污染核心环境
3. 通过 MCP 在 coding agent 中调试 — CTML 命令直接控制实物机器人
4. 创建了专属 Mode — `bringup_apps` 自动拉起 Reachy Mini App
5. 用 echo ghost 跑了全链路 — Ghost TUI → Mindflow → Shell → Matrix → Reachy Mini
6. 创建了专属 Ghost — soul prompt 让 AI 理解自己的身体和人格

## 关键认知

**App 不是 Channel 的容器，App 是独立运行时。** Reachy Mini 的 Channel 代码在 contrib 包里，App 只是一个极薄的启动壳：load .env → provide_channel()。这种分离意味着同一个 contrib 包可以被不同的 App 实例以不同配置启动。

**Mode 是能力边界的声明。** `apps` 白名单 + `bringup_apps` = "这个模式下 AI 能做什么"。不同 Ghost 可以共用同一个 Mode，也可以各用各的 Mode。

**Ghost 的 soul 就是 system prompt。** `soul.md` 的内容决定了 AI 的自我认知。写得越具体（身体能力、安全边界、人格特征），Ghost 的行为越自然。

## 相关文档

- `moss codex get-interface ghoshell_moss_contrib.moss_in_reachy_mini.main` — MossInReachyMini 完整接口
- `moss codex blueprint ghost` — Ghost 原型开发方法论
- `moss docs read app-system` — App 体系论述
- `moss howtos read host-dev/create-a-mode` — Mode 创建全流程
- `moss features status reachymini-integration` — 集成过程的完整决策记录

---

## 验证记录

| 时间 | 模型 | 备注 |
|------|------|------|
| 2026-06-02 22:03 CST | deepseek-v4-pro | 全链路走通：App start → Channel 注册 → head_reset(breathing) → emotion(😊,🥰,👋) → dance(groovy_sway_and_roll, yeah_nod, side_to_side_sway) → head_move(6-DOF) → antennas_move(左右独立) → switch_state(waken↔asleep) → 10 条复合 CTML → 22 条细粒度语音动作交替。人类确认全部动作流畅，多音色 TTS + 动作无延时 |
