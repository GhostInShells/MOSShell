---
title: Ghost Live2D Avatar
status: draft
# priority: importance within the current stage (iteration cycle) — not development urgency
priority: P1
created: 2026-09-13
updated: 2026-09-15
depends: []
milestone: beta-release
description: >-
  给 ghost 一个可交互的 live2d 看伴娘躯体：一个 live2d node 做驱动，按目录发现模型包，
  能用 cdi3.json 自动映射出 channel，也允许单个模型包用显式 channel 目录覆盖。
  模型资产不外分发（gitignore），背板大图可换。
---

# Ghost Live2D Avatar

> Use `moss features set-status ghost-live2d-avatar <status> -m "note"` to update state.
> See [TOPOLOGY.md](TOPOLOGY.md) for directory layout and [README.md](README.md) for the full convention.

## Motivation

dolores ghost 即将完成，ghost 需要一个**可交互的虚拟形象** —— 不是给人看的贴图，而是
ghost 能通过 CTML 驱动眼动、唇动、表情、肢体，并被人看见的躯体。

MOSS 里已有半份实现：`nodes/live2d/miku/`（alpha）用 `live2d-py`（native OpenGL）暴露了
`body / head / eye / eyebrow / mouth / arm / elbow / leg / necktie / expression / motions`
十来个 `PyChannel`，`mouth.py:24` 已经证明参数化驱动可行（`SetParameterValue("ParamMouthOpenY")`）。
但它有三处不可继承：

1. **渲染走 native**（`live2d-py` + OpenGL + glfw）—— 平台绑死，且 web 侧本就更容易被 ws 单流控制。
2. **channel 是 per-model 硬编码的 Python** —— 每换一个模型就要重写一遍 channel。
3. 模型是 Miku（经典 IP，不适用）；README 自承重构未完成，无 `main.py` / `NODE.md` / `pyproject.toml`。

本 feature 把 alpha 重构为 beta1 独立 node，核心命题是**驱动与模型解耦**：
驱动是代码（随仓库分发），模型是数据（本地持有、不分发），两者的粘合面是**约定**而不是代码。

## Design Index

- Key design documents: `design/`
- Key discussion records: `discuss/`

## Key Decisions

### KD1. 模型按目录发现；两级解析，显式 channel 目录优先，不做合并

`models/<pack>/` 下每个子目录是一个模型包。驱动启动时枚举目录，对每个包做两级解析：

- **有显式 channel 目录** → 用它，完全接管该包的 channel 面；
- **没有** → 走自动映射，从模型包自带的 manifest 生成 channel。

**已决（2026-09-13）**：显式与自动**不合并**。要合并就得定义优先级、冲突消解、谁覆盖谁——
而写显式 channel 的人本意就是"这个模型我要完全控制"。合并只会让两边都不可预测，还引入
双驱动的可能（同一参数被两条路径写）。**显式即接管。**

代价：写显式 channel 等于重新引入 per-model 代码，这正是 KD1 想避免的东西。接受——
它是 escape hatch，不是默认路径。自动映射必须能覆盖大多数标准 Cubism 模型，显式留给
参数结构异常的少数。

### KD2. 自动映射的数据源 = 模型包自带的 `model3.json` + `cdi3.json`

不需要额外描述文件：Cubism 模型包自己就带齐了元数据。实测（`Live2D/CubismWebSamples`）：

- `Hiyori.model3.json` → `Groups: [LipSync → ParamMouthOpenY, EyeBlink → ParamEyeLOpen/ParamEyeROpen]`、
  `Motions: {Idle:[9], TapBody:[1]}`、`HitAreas: [Body]`、无 `Expressions`
- `Hiyori.cdi3.json` → 每个参数形如 `{"Id":"ParamAngleX","GroupId":"ParamGroupFace","Name":"角度 X"}`，
  70 个参数按 `GroupId` 恰好聚成 12 组：`Face / Eyes / Eyeballs / Brows / Mouth / Body / Arms / Sway×5`

**`GroupId` 就是现成的 channel 树**。映射规则：`cdi3` 参数按 `GroupId` 分组 → 每个组一个 channel，
组内参数成为该 channel 的控制面；`model3.json` 的 `Motions` / `Expressions` → 动作与表情 channel；
`HitAreas` → 点击交互面；`Groups` 里的 `LipSync` / `EyeBlink` 声明 → 唇动与眨眼的参数绑定。

### KD3. 参数噪声用共享规则过滤，不引入第三个 per-model 配置文件

Hiyori 的 70 个参数里有 28 个是 `Param_Angle_Rotation_<n>_ArtMesh<id>` —— 旋转变形器自动生成的
中间参数，对模型驱动是纯噪声。过滤规则（`Param_Angle_Rotation_\d+_ArtMesh\d+` 等）写死在
映射器里，作为**共享规则**。

**否决**：上一轮讨论中提出的 per-model `channel-map.yaml`（白名单 / 改名 / 降噪）。它与
"显式 channel 目录"职责重叠 = 三层机制。KD1 已给出一层覆盖机制，降噪又是所有模型共通的，
放进映射器即可。**两层 > 三层。**

同理，`cdi3` 的 `ParameterGroups` 名字是日文（`顔 / 目 / 眉毛 / 口 / 胴体 / 腕 / 髪揺れ`、
Hiyori 是 `顔 / 目 / 目玉 / 眉 / 口 / 体 / 腕 / 揺れ`）。映射器内置一张**共享的 group-id → 英文名**
表（`ParamGroupFace→face`、`ParamGroupMouth→mouth` …），未知 group 回退到原名。这仍是共享的，
不是 per-model 的。

### KD4. 唇形等由我们自己做；但仍从 manifest 读声明的绑定参数名

唇动不依赖 SDK 的自动 lip-sync：我们从音频（`speech_channel` / TTS）取音量包络，按切片
（约 20ms）驱动嘴部参数。眼动、表情、肢体同理，全部由 ghost 侧决定。

但**参数名不从代码里写死**：`model3.json` 的 `Groups.LipSync` 已经声明了该模型用哪个参数
做唇形（Hiyori / Haru 都是 `ParamMouthOpenY`），`EyeBlink` 声明了眨眼参数。映射器读取这些声明，
驱动层按声明写。

**硬约束**：既然自己做，就不能让 SDK 也驱动同一参数 —— 双驱动会让唇形抖动。映射器把
`LipSync` / `EyeBlink` 声明的参数标记为"ghost 独占"，SDK 侧不喂。

### KD5. 启动 argument 指定默认模型；运行期可 switch_model

`NODE.md` 的 `exec.args` 传启动参数指定默认模型包。因为 KD1 的目录发现已经枚举了全部包，
再暴露一个运行期 `switch_model(name)` 命令几乎是零成本 —— ghost 可以中途换身体。

**原则（沿用 vision-first-class KD 的同一条）**：argument = "我是谁 / 绑在哪"，env = "我怎么行为"。
模型名是身份 → 走 argument。

### KD6. `background` 是页面自己的背板层，与 screen node 的 background 槽同名不同物

需求只有一个：`background/` 目录里能放大图，网页能设置背景。定位是 live2d 页面的**背板图层**
（一张大图铺在模型后面），不是 screen node 的 background 槽（那是"跨布局持久的窗口槽"）。

**术语警告**：两者同名，极易混淆。本 feature 文档与代码里一律称 **backdrop**（背板），
"background" 仅在与 human 对话时使用。backdrop 图不走 live2d 资产目录，避免模型包与背板耦合。

### KD7. 模型包与 Cubism Core 都不可分发 —— 这不是规避，是 license 的硬要求

Live2D 条款明写 **"may not redistribute all or part of the material to third parties"**：
模型包不能进 MOSS 仓库。同一条也适用于 `live2dcubismcore.min.js`（Core 是专有库，不在 GitHub 上，
只随官方 SDK 包分发）。所以 gitignore 必须覆盖**两类**：`models/*` 与 `vendor/live2dcubismcore.min.js`。

`INSTALL.md` 声明两步拉取（步骤本身就是授权条款的执行）：

1. `git clone --depth 1 https://github.com/Live2D/CubismWebSamples.git` → 取 `Samples/Resources/<Pack>` 到 `models/`
2. 官网下载 Cubism SDK for Web → 拷 `Core/live2dcubismcore.min.js` 到 `vendor/`

`CubismWebFramework` 是开源的（Live2D Open Software License），可随仓库或走 npm。

### KD8. 命令有时间轨迹 —— driver 持有时间，不 fire-and-forget

旁路验证暴露了 alpha 版的最大缺口：所有命令 fire-and-forget，微秒级返回，而身体要动
1.6–8.6 秒。于是 channel 认为"动作早完了"、待机立刻回来、同轨命令根本没有时间感——
"时间第一公民"在 avatar 上名存实亡。

已决（2026-09-15）：参数命令占默认缓动时长（0.3s），动作命令占 `motion3.json` 的
`Meta.Duration`（可用 `hold` 覆盖延长），结束在 `finally` 里复原。动作文件全部
`Loop:True`（实测 hiyori 10 个 motion），所以"结束"是 driver 自己计时后发
`clear_motion` 停掉，**不靠页面回报** —— 页面只服从帧，是纯执行器。

代价：同轨命令会串行（两个 face 参数先后 0.3s+0.3s），异轨仍并行。这是特性不是缺陷。

### KD9. 待机是 driver 仲裁的待机循环，跑在 build.idle

最初用 `build.idle` 做待机，但内核 bug 让它失效：`build.idle` 只在**该 channel 自身**
收到命令时才取消，子 channel 命令不取消父 idle（`_tree_channel_runtime.py` 的
`is_self_task` gate，已查证），而动作命令在 `motions` 子 channel 下，播动作时父 idle
不退出。于是待机一度跑在 `build.running` 里的永续仲裁循环。

内核修正后（`fix(runtime): child command now clears parent idle`），blocking 命令（含
子命令）都会取消父 idle，于是待机回到 `build.idle` 生命周期：无 blocking 命令时进入，
新命令到达取消。循环内仍保留空闲超过 `idle.delay`（默认 3s，可配）才进待机、说话
（唇动采样）与点按（on_tap 后台 play）让位 —— 后两者不是命令，内核看不到，由
`speaking` / `_foreground` 在循环里额外让位。部件级 idle（眨眼/呼吸）是 SDK 原生，
由 `idle.parts` 配置开关。

副作用（顺带修复）：blocking 参数命令现在也打断待机，参数命令不再被 idle 动作曲线
立即吃掉；但待机回来后动作曲线仍覆写它驱动的参数（见 CLAUDE.md 已知问题）。

### KD10. 唇动/眨眼是框架默认能力，不开成命令

旁路验证发现 `lip_sync` 开/关命令不该出现在默认命令面——唇动是框架默认行为，模型
不需要知道它的开关；把它暴露成命令只会诱导模型去手动控嘴，制造双驱动。

已决：自动映射的命令面**不注册** `lip_sync`。手动控嘴仍自动关唇动、`reset()` 仍恢复
（`avatar.set_lip_sync` 保留为内部能力，供 `channel.py` 作者按需用）。

### KD11. 人设 / 音色 / per-group instruction / idle 配置都进 AVATAR.md

`AVATAR.md`（frontmatter markdown，同 NODE.md 惯例）承载形象的可分发文本面：

- `name` / `description` / `voice` —— 一句冷人设 + 推荐音色，进 root instruction
- `groups.<slug>.instruction` —— 覆盖某 group 子 channel 的 instruction，缺省用自动冷描述
- `idle.delay` / `idle.loop` / `idle.parts.blink|breath` —— 待机配置

**instruction 要冷、准确**：散文人设不进 instruction（留给 channel.py 作者/人类），
instruction 只放冷事实 + 一句人设 + 一句音色 + 一条 `<say>` 前后顺序的硬约束。

### 实测基线更正（2026-09-15）

KD2 的实测表来自 CubismWebSamples 的**免费 Hiyori**（70 参数/12 组）。本地实际可用的
是 **hiyori_pro**（`live2d-py-test` / `kalidokit` 里的 t10/t11）：42 非噪声参数、8 组
（face/eye/eyeball/brow/mouth/body/arm/move）、无 Expressions、动作组是
Idle·Flick·FlickDown·FlickUp·Tap·Tap@Body·Flick@Body。映射器在异构包上照样成立。

### KD12. animation 轨迹编程 — 纯代码编排动作

CTML 里用 `<wait><motions:tap/><face:angle_x/></wait>` 拼轨迹，等价于把一段编排固化成
一条命令。给 avatar 一个"写代码"的出口：`avatars/<name>/animations.py` 里每个
`async def` 是一条动画，函数体的 await 序列就是时间轨迹（KD8 的另一面）。

已决（2026-09-15）：复用三个现成件，不新造抽象：

- `codex.compiler.Compiler` 编译文件源码，`local_injections` 注入 `get_avatar() -> Avatar`
  与 `asyncio` —— 函数体是纯代码，无需 import；
- `channel_builder.new_command` 反射协程函数 → command（签名即接口）；
- `ChannelModule`（`states_channel.py`）打包命令集，`with_module` 挂主 channel，
  同名覆盖 = 热更新；`reload_animations` 命令触发（编译失败抛错、保留上一版）。

绑定在主 channel 根轨（不是子 channel）—— 一条动画可能跨多个 group。

## Implementation Notes

### 实测资产（2026-09-13，来自官方仓库 `develop` 分支）

| | Hiyori | Haru |
|---|---|---|
| 动作 | Idle×9 + TapBody×1 | Idle×2 + **TapBody×4（带 4 个 .wav）** |
| **表情** | **0（无 .exp3.json）** | **F01–F08** |
| HitArea | 仅 Body | Head + Body |
| 参数 | 70（含 28 个噪声） | 42，干净 |
| 面部词汇 | 基础 | 多 `ParamTere`(害羞) `ParamTear`(泪) `ParamFaceForm` `ParamEyeForm` `ParamEyeBallForm` |
| 肢体 | `ParamArmLA/RA/LB/RB` `ParamHandL/R/LB/RB` **`ParamLeg`** `ParamShoulder` | `ParamArmLA/RA/LB/RB` `ParamHandChangeR/HandAngleR/HandDhangeL/HandAngleL` |
| 头发/衣物 | `HairAhoge/Front/Back` `SideupRibbon` `Ribbon` `Skirt` `Skirt2` | `HairFront/Side/Back` `ParamScarf` |

**默认 pack 取 Haru + Hiyori**：同仓库、同授权，Haru 提供表情与触摸动作，Hiyori 提供更多的
Idle 变化与 `ParamLeg`（更接近全身）。两个一起也正好验证映射器不是只对单一模型成立的。

共同点：`ParamArm*` / `ParamHand*` 都在 —— **肢体可纯代码驱动**；物理（头发衣物摆动）、
pose、眨眼、唇动参数全套自带。

### 可复用 alpha 资产

`nodes/live2d/miku/miku_channels/` 的参数词表（`ParamMouthOpenY`、`ParamAngleX/Y/Z` 等）
与 `motions.py` 的 `open_close()` 补间思路可继承；模型资产（MIKU）与 native 渲染栈不继承。

### 渲染 / 驱动选型（倾向，未锁定）

渲染层倾向**官方 Cubism Web SDK + 薄 wrapper**：无 Pixi 依赖，眨眼 / 物理 / pose 由
`model3.json` 声明 + framework 直接处理，参数全可控。需要鼠标跟随 / hit-test 时可退到
`pixi-live2d-display`（Kalidokit 的 live2d 模板即用它）。参考实现：Open-LLM-VTuber
（Python 后端 WS 发 `{type:"audio", volumes:[...], expressions:[...]}` → 前端按切片驱动嘴部参数）。

**Kalidokit 不可作依赖**（2022 起官方废弃、单人维护）；且它的 Pose/Hand 求解器输出的是
VRM 3D 骨骼旋转，**不是** Live2D 手臂参数 —— "骨骼动画驱动 Live2D 肢体"不是它现成能做的。
可借鉴的只有它的中间结构 `{eye:{l,r}, mouth:{x,y,shape:{A,E,I,O,U}}, head:{x,y,z}, brow, pupil}`，
作为 ghost 侧"动作意图"的规范形态。

## Open Questions

- 显式 channel 目录的格式：yaml 声明式，还是 py 可编程？（倾向 yaml —— 声明式才符合"约定"）
- 启动 argument 的命名与默认值；默认 pack 是 Haru、Hiyori 还是让 INSTALL 决定"有什么用什么"。
- backdrop 由 ws 设置还是 URL query 设置；大图的尺寸/格式边界。
- 是否/如何挂进 screen node 的 background 槽（跨 node 依赖，与 text_blocks "不依赖 screen-node"
  的独立性相反）—— 还是 node 自持 PySide 窗口（`QWebEngineView`）。
- 自动映射对非标准模型的失败面：`cdi3.json` 缺失时的降级策略（无分组 → 单一扁平 channel？）。

## Out of Scope（不在本 feature，另立）

- **骨骼动捕 / 表情捕捉输入**（MediaPipe / Kalidokit 路线）：本 feature 只做"ghost 驱动躯体"的
  驱动面，不做人类动作捕捉。若未来要人驱动，另立。
- **模型二次编辑**（改色 / 改服装）：被 Hiyori 条款禁止（"No changes of any kind to the design"），
  且需要 Cubism Editor，不是运行时能力。
- **模型资产进入仓库分发**：被 license 禁止（见 KD7），永久排除。
