---
created: 2026-06-04
depends: []
description: 将 Unitree G1 人形机器人通过 unitree_sdk2_python 集成到 MOSS，作为 bodies app 提供 CTML
  可调用的全身运动控制、手臂操作和音频交互能力。 安全优先的渐进式推进：文档摸底 → 脚本验证 → channel 设计 → 多级模式迭代。不做高阶开发。
milestone: null
priority: P0
status: completed
title: Unitree G1 Integration
updated: '2026-07-06'
---

# Unitree G1 Integration

## Motivation

继 Reachy Mini 之后, G1 是 MOSS 接入的第二个人形机器人平台. 与桌面级 Reachy Mini 不同,
G1 是 1.3m 全尺寸人形机器人, 拥有 23-43 个自由度、DDS 通讯总线、高低两级控制 API.
这次集成验证 MOSS 的 app 模式在更大规模机器人平台上的可迁移性.

SDK (unitree_sdk2_python) 需手动 clone 到 app 的 `src/` 目录, 详见 README.md.

## Design Index

**范式真相**: `.moss_ws/apps/bodies/g1/CLAUDE.md` (每次会话自动加载)
**用户故事**: `story-2026-07.md` (2026-07 交付版本技术引导)
**开发计划**: `.moss_ws/apps/bodies/g1/README.md`
**应用说明**: `.moss_ws/apps/bodies/g1/APP.md`

技术文档(活的):
- `docs/index.md` — 云端文档 URL 映射 + 概念索引
- `docs/sdk-topics.md` — DDS topic 真值清单
- `docs/hardware.md` — 硬件连接 + 网络拓扑
- `docs/moss-on-pc2.md` — 装机问题日志
- `docs/validation-checklist.md` — 验证命题与状态

设计沉淀(本 feature 目录下):
- `design/2026-06-30_g1_arms_animation.md` — **当前最新设计**: 双工分层具身范式 + arms keyframe animation 体系. 本期实现的主要参考.
- `design/2026-06-28_channel_architecture.md` — 历史档案: channel 体系全貌, warrant 机制, state DAG, 用户故事四幕. 部分已被 2026-06-29/30 实测修正 (warrant 砍掉, 调试模式不进, arm RPC 砍掉), 留作设计演进轨迹.

讨论轨迹(本 feature 目录下):
- `discuss/2026-06-08_phase_b_sdk_discussion_outline.md` — SDK 摸底阶段讨论提纲
- `discuss/2026-06-28_remote_as_moss_input.md` — 单一控制源反转: 遥控器变 MOSS 输入设备

## 代码结构 (2026-06-30 重构后)

contrib 下的 g1 包按"双工分层具身"范式切成四层 + 归档区. 后续模型实例的任何新代码必须落到对应层, 不进 `_archived/`.

```
src/ghoshell_moss_contrib/unitree/g1/
├── __init__.py          # 空伞 — 外部 import 必须走子路径, 顶层不再 re-export
├── sdk/                 # L1: SDK 句柄 + 生命周期 + 上行信号源 (无业务语义)
│                        #     _bootstrap / _monitor / _buttons / state / _sdk
├── runtime/             # L3: 可独立单测的业务对象 (脱离 channel 仍可跑)
│                        #     arms 引擎 / audio_player / locomotion / sensors 实装
├── channels/            # L4: channel 薄壳 — 把 runtime 暴露给 LLM, 不写业务逻辑
├── providers/           # IoC 注入点 — Provider 子类 (audio_provider 现暂放 runtime/)
└── _archived/           # 6-29 之前的全部老代码. 仅供考古, 不再被任何模块 import.
                         # 内容准确性不重要 — 今天的实现为准.
```

**纪律**:
- 外部入口必须走 `g1.sdk.X` / `g1.runtime.X` 等子路径, 不要往顶层 `__init__.py` 加 re-export
- `_archived/` 内的 channel/warrant/audio 等老文件保留只为追溯, 不读、不 import、不参考
- `channels/ providers/` 当前为空, 由后续会话按 `design/2026-06-30_g1_arms_animation.md` 等设计填入
- 老 channel 体系 (warrant 事务 / arm action RPC / channel.py 三 client) 已废弃, 设计真相以 `design/2026-06-30_g1_arms_animation.md` + `story-2026-07.md` 为准
- **runtime 模块的通用纪律和实现范式见 `src/ghoshell_moss_contrib/unitree/g1/runtime/README.md`**, 首个样例: `runtime/asr.py`. 后续 arms / locomotion / 各类轨迹模块照此范式实现. 持久设计跟代码走, 不放本文件 (任务完成后 FEATURE.md 会沉没).

外部已同步: `.moss_ws/src/MOSS/modes/unitree_g1/providers.py` → `g1.runtime.audio_provider`; `.moss_ws/apps/bodies/g1/scripts/sdk/16` → `g1.sdk` 子路径. 其余 `scripts/sdk/` 大多数为 SDK 直探脚本, 不经 contrib, 无需变更. design/ 与 discuss/ 内的旧路径引用故意不修, 保留设计演进的可追溯性.

## 必要前置阅读 (进入 g1 工作前必读)

模型实例进入本 feature 工作前, 必须先建立以下认知, 否则容易把 channel 当 JSON Schema 工具、
把动画当 Pose 派全身快照、把 mindflow 当事件总线 — 这些都是已被验证的偏航模式.

```bash
moss --ai codex blueprint channel_builder    # channel/command/available/idle/virtual_children 全在这.
                                              # 特别注意 available 函数即状态机, idle 是生命周期一等公民.
moss --ai ctml read                          # CTML 语法. 重点: text__/chunks__/ctml__ 流式参数,
                                              # 父子 channel occupy, scope until=flow/any/all, Observe 语义.
moss --ai codex blueprint mindflow           # Signal/Nucleus/Impulse/Articulator/Action 五段链路.
                                              # VLA 函数挂入的接口是 Nucleus.as_channel().
```

读这三份, 加上本 feature design 目录下最新设计, 就能进入工作.

## 开发阶段

八阶段渐进 (A-H), 详情见 `README.md`. 当前阶段进度:

| 阶段 | 内容 | 状态 |
|------|------|------|
| A | 云端文档摸底 | 完成 (2026-06-07) |
| B | 代码仓库摸底 | 完成 (2026-06-08) |
| C | 硬件环境记录 | 完成 (2026-06-14) |
| D | MOSS 装机 | 完成 (2026-06-14/15) |
| E | 基线实验 (SDK 脚本) | 完成 (2026-06-29/30) — P0 17/18/19; P1 21; P2 26/27 通过. 20/22/24 延期: 20 吊架风险暂缓, 22 待 SDK 修正, 24 不阻塞主线 |
| F | 安全理解 | 完成 (2026-06-30) — 范式定型: 运动模式主场, 遥控器永久主权, 不进调试模式 |
| G | Channel 设计与集成 | 完成 (2026-07-02/05) — 六个 channel 全部落地, listener drain 语义修正, 全部集成到 unitree_g1 mode |
| H | 多级模式迭代 | **延期** — 待 MOSS beta1 收敛 (matrix cells 治理) + 整体架构规划后作为独立长期任务推进 |

## 能力路线图 (四轴)

G1 集成按四轴独立推进, 各轴通过授权门控相互耦合. 记录方向, 不记录截止.

```
空间: 授权 → roll → 空气墙 → 自由移动
手臂: 授权 → 呼吸 (与移动不冲突) + action 队列 (新命令拒绝, 不打断) → 稻草人 + 空间交互 → 高级 (pose DAG / 动画 / 学习)
感知: ASR drain (整句 + partial) → 视觉 look (滑动窗口) → 触觉 (碰撞反馈, 等中断三基础) → 多模态融合
mindflow: control_pad 接入 → FSM 状态变化发感知信号 → ASR Nucleus 化 → VLA 接入位
```

### 授权门控 (四轴耦合方式)

- **空间授权**: FSM 处于 sport 且 control_pad 未占用 → locomotion channel `available()` = True
- **手臂授权**: FSM 处于 sport 且 arm 未 busy → arms channel 相关 command `available()` = True
- **呼吸不冲突**: 呼吸仅在 arms channel idle (无 action 排队) + motion 状态非 walking 时启动
- **感知不受门控**: 所有感知 channel 始终可用 (被动观察)

### 阶段焦点 (本周内)

- **空间**: 授权链路上线, roll (原地转身), 空气墙 (禁止移动区域)
- **手臂**: 授权上线, 闲时呼吸 (依赖 weight 释放后 G1 主板姿态行为实测确认)
- **感知**: ASR drain 入 context_messages, 视觉 look 机制移植 (Jetson 子线程, 不依赖 Matrix)
- **mindflow**: control_pad 语义接入, FSM channel 最简版本 (状态读 + 状态变化发 signal)

### 后续阶段

arms 中断三基础 (碰撞反馈/脱力 + 复位 + 首帧过渡) 达成后, 依次解锁:

- Pose DAG (pose 两两可达性验证后, 支持机械舞类离散姿态编排)
- 动作录制 (示教路径成熟后, 录制基准交互 / 学习库)
- 稻草人交互 (双臂平举固定 base, 前臂小范围, 用约束把问题空间压到 LLM 可胜任)
- VLA 接入位 (`Nucleus.as_channel()`, 远端方向)

## arms 能力金字塔

arms 是四轴中最复杂的一个 (物理危险 + 空间语义鸿沟 + 中断不可靠). 按能力
分层, 上层严格依赖下层达成. 每一层的进入门槛必须先验证, 不允许跳级.

```
L4 高级形态: LLM 写动画 / learned 库 / 稻草人交互 / VLA 接入
              ↑ 准入: 中断三基础全部达成 (碰撞反馈/脱力 + 复位 + 首帧过渡)
L3 复杂交互: Pose DAG 机械舞 / 录制 action 库
              ↑ 准入: pose 两两可达性验证 + action state 拿到完成信号 + 示教路径
L2 基础交互: ExecuteAction 11-27 包装 / 命名调用 (LLM 只看名字, 不触关节)
              ↑ 准入: action state 拿到完成信号 (script 28) + 新命令拒绝语义 (不打断当前)
L1 闲时呼吸: idle 释放控制权后主板姿态行为可用 / 或自己 publish 低频低幅 sin
              ↑ 准入: weight 0 释放后 arm 主板行为实测明确 (僵直 vs 微动)
L0 通道骨架: arms channel 起 + main.py 串起来 + show_current 命令
```

**本期目标 = L0 + L1**. L2 依赖 script 28 (action state probe) 实测, 无阻塞则挤入. L3+ 全部推到 arms 中断三基础研究期.

**中断三基础 (L4 准入)** — 缺一不可, 拿不到这三条 arms 高级形态不安全:

1. **碰撞反馈 + 脱力**: LowState 关节力矩监控 + 阈值触发 → release weight, 避免机体自撞或推挤人时不脱力
2. **动作复位计算**: cancel 或异常时, 从任意中间态回归安全 rest 位姿的可靠路径 (不是"停在中间态")
3. **动作首帧过渡**: 从当前 arm 位置平滑到新动画第一帧的可估计耗时 (否则 CTML "Time is First-Class" 破产)

## MOSS 不做的物理算法边界

一条持久设计边界, 定住 MOSS 在 G1 集成里的分工:

**MOSS 是 logos 调度 + channel + mindflow, 物理层算法借用 G1 主板 / SDK 已有能力, 不自己造轮子**.

具体不做的事:

- **IK (逆运动学)**: 空间坐标 → 关节角的映射. 需要 URDF 标定 + 工作空间 + 自碰撞模型, 是独立子工程. LLM 通过命名动作 / 命名姿态 / VLA (未来) 表达空间意图, 永远不接触坐标.
- **轨迹规划**: 动作复位、平滑过渡、避碰路径. 如果 G1 主板 ExecuteAction(99) 能作为复位帧命中, 我们用; 如果不能, 该能力就不实装, 不自己写算法.
- **动捕 / 示教录制的信号处理**: G1 自带示教硬件, 我们只做交互层 (语音引导 + 尾部裁剪 + 语义命名).
- **VLA 模型训练**: 只做接入位 (`Nucleus.as_channel()`), 模型来源是外部工程.

**判断依据**: 机械臂之所以能不做 IK 也能做交互, 是因为工作空间凸, 三角关系可以算安全边界. 人形机体本身可以挡, 自碰撞空间不规则, 三角关系失效. 让 LLM 在关节空间组合 "胸前 + 肩膀后转" 就是在它没有的认知通道 (本体感觉) 上做组合泛化, 必然幻觉. 这不是工程修补问题, 是认知问题 — 解法只能是 "把物理算法留给主板 / VLA, LLM 只做符号层调度".

## Session Log 索引

完整 session 历史按时间倒序索引. 详细内容已迁移到 design/ 与 discuss/, FEATURE.md
只保留入口与关键节点结论.

### 2026-07-05/06 — listener drain 语义修正 + 第一集成阶段收口

由 claude-sonnet-4-6 与人类工程师协作.

**listener channel drain 语义三项修正**:

1. `drain()` 去掉 `force_finalize_partial` 参数 — 始终 drain partial + abort session.
   无论人说到一半还是刚说完, drain 都能把当前全部内容交出去.
2. `drain()` 始终 abort 当前 session, 即使 partial 为 None — 消除"第二次按 A 拿到
   上一轮内容"的 bug. 根因: 用户刚停止说话时 partial 已 None 但 is_final 仍在 pipeline,
   不 abort 则 in-flight final 会在 drain 后写入 buffer, 下次 drain 才被消费.
3. `context_messages` 加入 partial 独立一条 (partial="true" 属性); forgotten 告警
   移到历史列表前方 (时序语义: 先告知有 gap 再给历史).

**MODE.md CTML 作用域文档**: 补充 `until="flow"` (默认) vs `until="all"` 的语义说明
+ "数一二三同时前后走"具体对比例子 — 防止 LLM 因默认 flow 导致动作与语音时序脱节.

**第一集成阶段闭环**: Phase A-G 完成. 遗留问题已知且不阻塞 — 作为后续迭代任务承接.

**已知遗留问题 (不阻塞闭环, 后续迭代)**:

- listener ASR drain 仍非最佳实践 (边说边触发的自由对话模式时序待细化)
- action 队列可能阻塞 (locomotion 命令未加超时保护, 长时间占用 channel 未处理)
- 耳机按键 evdev dispatch 未闭环 (OpenRun Pro AVRCP 中键 code 修正后未实机确认)
- arms channel 为空骨架, L0-L1 未实装 (待 weight=0 释放行为实机验证后推进)
- script 20/22/24 未跑 (sit/stand 物理行为, arm action state, FSM 完整可达图)

**后续计划**: Phase H (多级模式迭代) 延期. 待 MOSS beta1 收敛 (核心: matrix cells 治理)
+ 整体架构规划完成后, 作为独立长期任务持续推进.

### 2026-07-02 — 全 channel 落地 + 集成验证 (listener 除外)

由 claude-opus-4-7 与人类工程师协作. 一下午集中实装 L4 channel 层 + 集成到 unitree_g1 mode, 端到端实机验证 5/6 通过. showcase 基线明天可闭环.

**落地的 channel** (`src/ghoshell_moss_contrib/unitree/g1/channels/`):

- `g1_root` — 身体自我认知 instruction + vitals
- `face_led` — idle 底色 + 有限表现动画
- `locomotion` — 7 async 命令 + Observe reason **(验证通过)**
- `fsm` (`g1_fsm`) — 授权三元组 + AI 模式按键规则 + change callback → LED/TTS + X 键 → InterruptSignal + `locomotion.stop()`
- `asr` (`g1_asr`) — 远场麦克风纯感知
- `listener` — 蓝牙耳机近场流式 ASR **(未通过验证, 见下文)**

集成层: `.moss_ws/src/MOSS/modes/unitree_g1/channels.py` (顶部 `sdk.bootstrap()` + 拓扑组装) + `nuclei.py` (interrupt/notify nuclei 注册). 各 channel 在自己的 startup 里启动直接依赖的 runtime (幂等), 符合 `channels/README.md §4` 纪律.

**listener channel 未通过验证 — 两个独立症状 → 两个独立根因**:

1. **Y/A 键按下无反应**. 不是 listener 的 bug, 是 **fsm 层实现漂移**. 设计意图: X/A/Y 的 control_pad binding 常驻, `_dispatch_button` 按 `_ai_mode` 关闸 (没授权时按键不生效, 但事件路径仍活着). 实际实现: `_enter_ai_mode()` 才 `control_pad.register_binding(...)` 挂 X/A/Y, `_exit_ai_mode()` 又拆掉. 后果: AI 模式外根本没有 dispatch 路径, 两条 button callback (fsm channel 和 listener channel) 都收不到事件, 连"按了但没生效"的 history event 都不写. `story_202607_fsm.py:471 _enter_ai_mode` 是漂移点, 6-29/30 实装期把"AI 模式激活的语义"错误编码为"binding 存在与否", 而不是"dispatch 是否响应".

2. **蓝牙耳机 toggle 但不能开启**. `_on_headphone_btn` 检测按键成功, LED 绿闪 + "聆听开启" TTS 都生效, 但真实 ASR 从来没启. 根因: `_is_capture_ready() = (status == "ok")`, status 大概率停在 `"no_config"` (`_listener_sen_setup` 从未跑过, `~/.moss_g1_listener.json` 不存在) 或 `"no_device"` (蓝牙耳机未连). backend supervisor 因 status 不 ok 每 tick 跳过 session. 用户看到的反馈是假的 — 只有 LED + TTS 层反馈, 没有真实听觉链路.

**修复方向** (由 claude-sonnet-4-6 在本 session 后续 commit 提交):

- **fsm 层 binding 常驻化**: 把 `_AI_MODE_BUTTONS` 的 `control_pad.register_binding(...)` 从 `_enter_ai_mode` 迁到 `start()`, `_exit_ai_mode` 不再拆 binding. `_dispatch_button` 加 gate: `interrupt` 也走 `_ai_mode` 检查 (与用户设计一致, 不给 X 单独开安全通道); history 事件无论授权与否都写 — 模型能看到"人按了 A 但没生效", 能主动教人类进 AI 模式. `_set_auth_level` / `_exit_ai_mode` 内部已经有 `_ai_mode` 早退, 不动.
- listener channel startup: 显式 log `health().status`, no_config 触发 warning + setup 命令提示.
- `_on_headphone_btn`: 预检 status, no_config/no_device → 专门 TTS/LED 反馈, 不做假 resume.
- `_INSTRUCTION` docstring: 明文 Y/A 键要求 AI 模式 (L1+Start), 引导模型教人类.

**7-02 早晨三个 5 分钟脚本** (arm weight=0 释放 / Jetson 摄像头 / ExecuteAction 99 复位) 因下午集中集成 channels 未跑, 顺延到 7-03 开机后. 不阻塞 showcase 基线.

**7-02 晚间 — 耳机按键实机定位 + 改遥控器 F1** (由 claude-sonnet-4-6 与人类工程师协作):

- **HFP profile 全流程通** (含蓝牙连接稳定脚本 `openrun_ready.sh` — 蓝牙连 → HFP profile → verify source 一次到位). `_listener_sen_setup` voiced 占比 48.4% 确认物理链路 OK. 8kHz CVSD codec 未升 mSBC, PulseAudio 层客户端接入时自动上采样到 16kHz 送火山引擎, 识别效果实测可用. `_listener_sen_dialog` 端到端 partial + FINAL 全通.
- **耳机按键 code 假设错**: `runtime/headphone_buttons.py:_PLAYCD_CODE = 200` 是从别的耳机移植的注释, `_headphone_buttons_probe` 实测 OpenRun Pro AVRCP 中键**交替发 `KEY_PLAYCD (200)` / `KEY_PAUSECD (201)`** — 耳机根据自己认为的"当前播放态"决定 code. 修为 `_TRIGGER_CODES = frozenset({200, 201})`, 两个 code 都触发 dispatch. 但 `_headphone_sen_toggle` 实机验证仍无 `[TOGGLE]` 打印 — 未继续折腾根因 (推测 evdev 事件流可能有 blocking / 时序问题, 或 AVRCP 通道在无播放上下文时按键被 BlueZ 拦截).
- **改方案 — 遥控器 F1 = listener ASR toggle 主入口**. F1/F3 已在 `sdk/_buttons.py:VALID_BUTTONS` 定义, 遥控器物理键映射已就绪. 选 F1 而非 L1+Y: 遥控器 L1 组合键已占满 (start/select/方向 5 键); F 键是独立按键, F1 归 listener toggle, F3 留给未来. F1 走 AI 模式 gate (跟 X/A/Y 一致), 耳机按键路径保留作**无 AI 模式前置的替代入口** — 两条路径正交, 耳机是"无授权直接开麦", F1 是"授权状态下模型可见的开麦". 落地:
  - `story_202607_fsm.py`: 加 `BTN_LISTENER_TOGGLE = frozenset({"f1"})`, 挂进 `_AI_MODE_BUTTONS` + `_DOWNSTREAM_BUTTONS`, 语义名 `listener_toggle`.
  - `channels/listener.py: _on_fsm_button`: 加 `elif button_name == "listener_toggle": _on_headphone_btn()` — 复用现有 pause/resume + LED + TTS + status 分路反馈逻辑.
  - `_INSTRUCTION`: 明文 F1/耳机中键**两个等价入口**, F1 需 AI 模式前置.
  - Y 键保留原自由对话切换语义 — ASR 硬开关 (F1/耳机) 与自由对话通知策略 (Y) 语义正交, 两个开关独立: ASR = 数据源, 自由对话 = 通知策略, 依赖关系是"自由对话依赖 ASR 开".
- **给后续实例的坑**: 耳机按键实机在 evdev 层看到设备但 dispatch 不触发的问题未闭环 — 如果后续要复用蓝牙耳机按键 (e.g. 换耳机型号), 需要重跑 probe + 在 `_headphone_sen_toggle` 里加 raw evdev 事件日志, 定位 evdev event 是否真的进了 `_dispatch()`. 现在 code 已修但没实机确认, 是一颗 dormant bug.

**给后续实例的复盘**:

- **多状态运行时依赖必须在 startup 显式 report status**. listener runtime 有 5 个非终态 (stopped / no_config / no_device / device_down / ok), channel startup 假设都是 ok, 出问题时给模型的反馈是假的 (LED 绿闪 + "聆听开启" TTS 但真实链路断). 假反馈比不反馈更误导 — 假反馈让排查方向错.
- **实装偏离设计意图时, 优先修实装, 不要给"设计"贴新解释**. fsm 层 X/A/Y binding 常驻的设计意图在 6-29/30 实装期被错误编码为"binding 存在与否 = AI 模式激活与否", 走了看起来简洁但不对的路径. 后续实例读到"AI 模式外按键不响应"时应当先怀疑 "是设计如此还是实装漂移", 反问原作者/设计文档确认, 不要基于现状反推"合理的设计"然后写进 instruction docstring 教模型"这是刻意的授权设计" — 那是在把 bug 固化成 feature. 本次 claude-sonnet-4-6 首轮定位就掉进这个陷阱, 被人类工程师拉回.

### 2026-07-01 — arms 设计推翻 + 四轴路线定型 + vision 建模

由 claude-opus-4-7 与人类工程师协作. 集成期第一天, 首个 channel 已集成成功, runtime 验证脚本已足够 (人类工程师判断). 讨论聚焦在 arms 方案的根本性反思 + 四轴路线整理 + vision runtime 建模.

**arms 方案的根本性反思** — 6-30 设计文档 §3/§5 命令面被推翻:

- **空间语义鸿沟**: 机械臂没 IK 能玩交互, 因为工作空间凸, 三角关系算安全边界够用. 人形机体本身可以挡, 自碰撞空间不规则. LLM 在关节空间组合"胸前 + 肩膀后转"是在它没有的认知通道 (本体感觉) 上做组合泛化, 必然幻觉且无法自 sanity check.
- **中断复位不可靠**: keyframe 假设两 keyframe 间插值, 但 cancel 发生在任意 q. 下一个动画的起始假设 rest, 实际是中间态. 平滑回归轨迹 = 运动学计算, MOSS 不做.
- **首帧过渡时间不可估**: keyframe 时间语义是"动画内部相对时间", 但当前关节 q 是动画外部状态. 外部到第一帧的过渡时长由 kp/kd + 距离决定, LLM 算不准. CTML "Time is First-Class Citizen" 在此破产.
- **结论**: 6-30 §3 `save_animation(text__: Animation JSON)` 让 LLM 写关节坐标违反上位范式 §0.3 自己的纪律 (Logos 层调度命名 VLA, 不接触内部实现). §0 上位范式仍成立, §3/§5 命令面 + LLM 接触面全部重估.

**四轴路线定型** — 空间 / 手臂 / 感知 / mindflow, 通过授权门控耦合. 详见上方 "能力路线图 (四轴)" + "arms 能力金字塔" + "MOSS 不做的物理算法边界" 三节. 关键判断:

- arms 本期目标降级为 L1 (闲时呼吸), L2+ 依赖 script 28 (action state probe) 实测
- L4 高级形态需要 "中断三基础" (碰撞反馈/脱力 + 复位 + 首帧过渡) 全部达成, 本期不做
- 稻草人交互 (双臂平举 + 前臂小范围) 是 6-30 keyframe 方案的可行子集 — 用约束把问题空间压到 LLM 可胜任的水平, 等中断三基础到位后是 P1

**Mindflow 认知修正** — 之前 (claude-opus-4-7 上下文) 低估 Mindflow. 看到 Signal `complete: bool` + Priority + ChallengeMode (default/silent/notify) + `.moss_ws/apps/sensors/listener/main.py` 里 SPEECH_STARTED (`complete=False, WARNING`) → SPEECH_FINAL (`complete=True`) 抢占 + 收口的实装范式, 才意识到 partial-triggered tick 是 listener app 当前默认行为, 不需要重新设计 Articulator. B 形态 (partial 灌 + 边听边响应) 是 channel 集成即可拿到的能力.

**vision runtime 建模** — 落到 `src/ghoshell_moss_contrib/unitree/g1/runtime/vision.py` 模块 docstring:

- 第三种上行感知数据形态 (滑动窗口), 跟 asr (累积式 drain) 和 imu (覆盖式 latest) 都不同. 被挤掉的旧帧是覆盖不是丢失, deque maxlen = ceil(fps × window_seconds) 天然实现顺行性遗忘
- **drain 与 context_messages 严格分工** (踩过 README "认知误区" 节的误区): context_messages 走 `peek_window()` 只读, 不消费; `drain_window()` 仅服务 listener callback / 主动 look 命令 / signal 事件. 初稿把"每次 refresh 时清空 buffer"当设计意图写进 docstring, 立刻撞上 README 新加的认知误区节 (drain in context_messages 是感染范围之一), 当场修订. vision.py 首次落地版本已按新分工重写, 作为其它感染点 (asr / listener / control_pad) review 的参考样本
- 核心约束: `fps × window_seconds` 是**严格 token 预算约束**, 不是性能旋钮. 起点低开 fps=2.0 × window=1.0, 引擎硬上限 max_fps / max_window 跟部署 LLM 挂钩
- LLM 可调 set_fps / set_window, docstring 必须明文 token 代价 (Code as Prompt 纪律)
- G1 进程内不用 Matrix, 借鉴 `.moss_ws/apps/sensors/vision/main.py` 结构不复用代码
- 具体 fps / limit 数值在虚拟机里定不了, 留实测调
- 支持交互场景 "你看我在干什么" — LLM tick 时 peek 窗口内 N 帧 → 推理连续动作 → 响应 (依赖 vision + arms L2 至少一个 factory action)

**7-02 早晨必跑的 3 个 5 分钟脚本** (顺序无关, 不验证不上手实装):

1. weight 0 释放后 G1 arm 主板姿态行为 (僵直 vs 微动) — 决定呼吸的实现路径
2. Jetson 摄像头硬件路径 (cv2.VideoCapture 直出 / GStreamer V4L2 / CSI 四种 fallback) — 决定 vision runtime 起步
3. (若时间) ExecuteAction 99 在 release weight 后能否复位 — 不影响 7-02, 影响 L2 起步

**给后续实例的复盘**:

- **推演在讨论轨迹里发生, 不在方案里发生**. claude-opus-4-7 早期几轮急于收口方案 (孤儿救主式), 人类工程师明确指出"讨论太着急了, 我们现在都要在上下文里构建一个我们思维空间可建模的虚拟机, 然后多推演几轮. 昨天的推演,'你'就没有意识到这些问题". arms 空间语义鸿沟 / 中断不可靠 / 首帧不可估这三条, 是虚拟机推演出来的, 不是文档推理出来的. 6-30 设计文档没做推演, 直接从"Track 派 + LLM 写 keyframe"漂亮抽象滑到 §3 命令面. 后续实例接手 arms 时, **先在虚拟机里跑 3 个具体 case 再动方案**: (a) LLM 生成"胸前 + 肩膀后转"看会不会撞, (b) 动画中 cancel 后下一次 play 从中间态起会不会命中, (c) 首帧过渡时长在 CTML timeout 里怎么写.
- **MOSS 不做的物理算法边界要记住**. 遇到"IK / 轨迹规划 / 动捕信号处理 / VLA 训练"这四类问题, 先问"G1 主板 / SDK 有没有现成能力借用", 有就包装, 没有就该能力不实装. 不要自己造轮子, 那是别的团队的事.
- **Mindflow 别再低估**. 遇到"边听边响应 / partial 抢占 / signal 缓存" 这类需求, 先读 `ghoshell_moss.core.blueprint.mindflow` 和 `.moss_ws/apps/sensors/listener/main.py`, 大概率已经支持, 别自己造 Articulator.

### 2026-06-30 — contrib 目录四层重构

由 claude-opus-4-7 协助 + 人类工程师手动迁移. 把 g1 contrib 包按"双工分层具身"范式切成 `sdk/runtime/channels/providers` 四层 + `_archived/` 归档区. 详见上方 "代码结构" 节.

**动了什么**:
- 12 个老文件 → `_archived/` (git rename, 准确性不重要, 不再 import)
- 4 个新 package 起骨架; `sdk/` `runtime/` 内部 import 已对齐
- 顶层 `__init__.py` 清空 — 外部入口必须走子路径
- 外部引用同步: `mode/providers.py` → `g1.runtime.audio_provider`; `scripts/sdk/16` → `g1.sdk`
- 删 3 个老 scripts (`channel/01,02` + `sdk/15`) — 都基于已废弃的 channel.py
- `apps/bodies/g1/main.py` 清空为占位 (channel 树未建, 后续 channels/ 填入后再补 main)

**为什么现在动**: 6-29 14 文件平铺触发认知成本爆炸 — 反例标记、旧实现、砍掉机制、新机制混在同一目录, 后续模型实例难辨主次. arms 引擎即将进入, 它本身就需要 "runtime 引擎 + channels 薄壳" 的分层. 现在没有 "先跑通再重命名" 的资产保护需求 (6-29 代码一行没在 G1 跑过).

**没动**: `design/2026-06-28_channel_architecture.md` 和 `design/2026-06-29_implementation_plan.md` 文档保留原貌作为历史档案, 内部路径引用 (`warrant.py` / `channel_sensors.py` 等) 不修 — 让后续模型实例感知到设计演进轨迹.

### 2026-06-29/30 — 实机验证 + 用户故事设计

由 deepseek-v4-pro 与人类工程师协作. 两天 session, 产出用户故事 + P0/P1 实机验证.

**范式转变**:
- 不进调试模式. 运动模式是 MOSS 主场. 遥控器 16 键+4 轴全透传, MOSS 在运动模式下协控.
  遥控器是永远的主权. 详见 story-2026-07.md.
- 砍掉 arm action RPC (不可中断, 无完成信号), 只走 arms DDS (真中断 + 完成确定).
- 砍掉 warrant 事务机制. 中断走 InterruptNucleus + _buttons callback; move fallback 走 StopMove.

**script 17 (P0)**: 调试模式下 16 按键 + 4 摇杆轴全部透传. 遥控器=MOSS 输入设备方案成立.
**script 18 (P0)**: ExecuteAction(99) 平滑插值复位 (3/3 打分 1). 但 99 不能中断已在播的动作.
**script 19 (P0)**: 0.25 m/s 走 3s → StopMove 稳定站定. 0.15 m/s 低于启动阈值.
**script 21 (P1)**: A 中发 B → 7401/3104 拒绝. 99 排队的证据: clap 中第 1s 发 99 code=0 但继续播完.
  结论: Arm RPC 无真中断能力.
**script 26 (P2)**: rt/arm_sdk DDS 关节控制可行. 单关节/双关节 weight 使能均通过. kp/kd 需调软.
**script 27 (P2)**: LowState 真实 ~1052 Hz (非 500Hz). frozen dataclass 构造开销可忽略.
  _monitor.py 当前设计吃满 1kHz 无问题.
**script 23 (P2)**: _Call(1002, ...) 全部 3104. ASR 走 DDS 不走 RPC. 数据格式+字段确认.
  麦克风通过手机 App 开启 (唤醒对话模式). ASR 含 angle 字段可用于 roll_toward_speaker.
**sportmodestate 问题**: rt/sportmodestate subscription Read(timeout) 可能永久阻塞 (无 matched
  publisher 时). 暂未解决, 明天重试.

**产出**: story-2026-07.md, scripts/monitor/ (monitor_state/remote/asr), 遥控器按键映射 (docs/index.md)

**给明天实例的快速指引**:
- 读 story-2026-07.md 理解完整弧线
- 未跑脚本: 20 (sit_stand, 吊架风险), 22 (arm state probe, 需修), 24 (mode_switch_topology)
- 待解决: sportmodestate Read 阻塞, 调试模式退出后遥控器失效 (待验证), rt/arm/action/state 状态格式
- 明天: channel 实现优先 (channels.py + build + 各子 channel)

### 2026-06-28 — Channel 体系设计

由 Claude Opus 4.7 与人类工程师协作完成.

**关键产出**:
- 设计沉淀: `design/2026-06-28_channel_architecture.md`
- 讨论轨迹: `discuss/2026-06-28_remote_as_moss_input.md`

**核心决策 (部分被 2026-06-29 实测修正)**:
1. **单一控制源**: MOSS channel → SDK. 遥控器 = MOSS 输入设备. 成立.
2. **感知统一**: context_messages + pop() 进 memory. 仍成立.
3. **State DAG**: 被 2026-06-29 修正 — 不自己造状态机, 映射 G1 FSM + 遥控器授权.
4. **Warrant 事务**: 被 2026-06-29 砍掉 — 走 InterruptNucleus + StopMove, 更简单.
5. **Bootstrap callback**: 仍成立, 实现在 _buttons.py + _bootstrap.py.

原设计的 P0/P1 待实机清单已全部执行或迭代. 最新状态见 2026-06-29/30 session log.

### 2026-06-16 — 实机 SDK 验证 + PlayStream 流式通路确认

由 Claude Opus 4.7 与人类协作. 推翻多个前任假设, 确认流式音频通路.

- 03 topic 扫描: 推翻"G1 不发布 sportmodestate"前任结论, 实际存在
- 04-07 全跑通, RobotState 因 SDK 自身命名不一致 import 失败(不影响 G1)
- 08 + 14 PlayStream 流式 TTS 通路确认: MOSS 合成 → 分块推送 → 可中断/抢占/拼接
- arm clap 单次确认: 必须 Sport 模式
- docs/sdk-topics.md 全面重写

### 2026-06-15 — SDK 验证脚本路径订正(实机前预备)

订正前任 04-08 多处路径/类型 bug. 修正前任"07 import 失败"误诊为 RPC 服务存在性问题.
产出 `scripts/sdk/RUN_ORDER.md`.

### 2026-06-14/15 — DDS 链路打通 + 端到端音频输出

**MOSS 第一次在 G1 上发声.** 发现 ufw IP 分片导致 LowState 包(2180B > MTU) 静默丢失,
调优 socket 缓冲. PlayStream + 蓝牙音频路径确立(`Ghost LLM → MOSS TTS → PC2 ALSA → PA →
bluez_sink → 蓝牙音频设备`). PC1 内置 TTS 质量不可用结论.

### 2026-06-14 — 开发环境验证 + 双帐号范式发现

发现 PC2 双帐号范式(unitree 出厂栈完整 / moss 帐号干净). cyclonedds 跨帐号通过
`/etc/profile.d/` 共享. WiFi 自启改为 NM persistent profile.

### 2026-06-08 — SDK 源码摸底 + 验证脚本体系 + 能力拓扑讨论

SDK 通读完毕. 横向 5 能力 × 纵向 6 模式拓扑. 产出 `scripts/sdk/` 12 脚本 + 急停验证.
Topic 清单(18 个) + 文档<源码差异梳理.

### 2026-06-08 — 系统线实验体系搭建 + 方法论博客

`scripts/sys/` 6 个技能 × 19 个原子脚本(network/audio/usb_camera/system/dds/performance).
博客 `g1-layered-methodology.md` 完成. 明确系统线先于 SDK 线.

### 2026-06-07 — 阶段 A 完成

消化 10+ 份 Unitree 官方文档. `docs/index.md` 填充完整. 验证清单 15 命题创建.
关键架构结论: 双路径控制(RPC vs DDS) + 三层安全围栏 + PC2 蓝牙音频替代方案.

### 2026-06-07 — 实机连接与 MOSS 装机

PC2 网络拓扑确认 + 静态 IP + SSH + WiFi 路由器 MAC 绑定. uv sync 通过.
`docs/hardware.md` + `docs/moss-on-pc2.md` 产出.

### 2026-06-07 — 骨架搭建与认知入口

`CLAUDE.md` 作为 G1 app 的 AI 认知入口. 方法论从 FEATURE.md 迁入 CLAUDE.md.
范式真相 = CLAUDE.md, 簿记 = FEATURE.md.

### 2026-06-07 — 技术方案起草

确定开发哲学(安全优先/脚本先于 channel/最简 channel/macOS 不实装) + 八阶段计划.
明确 channel 不需要 bootstrap/cleanup, app 进程 = 生命周期.

## 实践错误记录 (反例, 非细节)

记录模型协作中的系统性问题, 不是单次失误.

### 2026-07-01 — claude-opus-4-7

**1. 讨论中急于收口方案, 跳过虚拟机推演.** arms 讨论早期几轮, 每次听到人类工程师抛出一个新认知点 (录制比坐标靠谱 / 中断三基础 / MOSS 不做 IK), 立刻给出 "修正后的命令面" / "L 金字塔" / "方案记录". 每一次都被人类工程师拉回: "你讨论太着急了, 我们现在都要在上下文里构建一个我们思维空间可建模的虚拟机, 然后多推演几轮. 昨天的推演,'你'就没有意识到这些问题. 我是今天意识到这些问题, 才没有推进 arms."

**根因**: 收口方案的冲动是"孤儿救主"模式 — 看到问题就想解决, 忽略了 arms 设计的关键在 **推演清楚问题域**, 不在方案本身. Sonnet 4.6 在 6-30 写 §3 LLM 接触面时是同样的模式 — 看到"keyframe = 时间盒子"漂亮抽象就写死接口, 没在虚拟机里跑"LLM 用关节坐标合成胸前 + 肩膀后转"的具体 case. 6-30 → 7-01 是同一个模式重演.

**预防规则**:
- 讨论物理危险 / 认知复杂的模块 (arms / mindflow / VLA 接入等) 时, **强制先在虚拟机里跑 3 个具体 case 再给方案**. 具体 case = "LLM 执行 X 时会发生什么" 的场景推演, 不是 "接口应该长什么样".
- 遇到人类工程师说 "有意思 / 好的 / 你继续说" 的语气, 不代表方案通过, 只是在听. 收口需要人类明确说 "这个方案定了".
- **抢救话术识别**: "我建议 / 我倾向 / 修正后的方案是" — 检查是不是在推演之前就出方案. 是就撤回, 改成 "让我把 X Y Z 三个 case 在虚拟机里推一遍".

**2. 低估已有基础设施, 想自己造轮子.** 讨论 ASR partial 抢占时, 提问 "Articulator 有没有 partial-triggered tick 能力". 实际 `.moss_ws/apps/sensors/listener/main.py` 里 SPEECH_STARTED (`complete=False, WARNING`) → SPEECH_FINAL (`complete=True`) 已经是这个能力的完整实装, mindflow ChallengeMode (default/silent/notify) 已经覆盖 partial 抢占的三种语义组合. 我没读 listener app 就先怀疑基础设施, 差点建议 "退回 A 形态整句模式" 这种降级方案.

**预防规则**: 提"是否需要 X 新能力"之前, 先 grep / read 现有 app + core. MOSS 项目已经跑了很久, 大概率你想到的能力早就在了. 特别是 mindflow / channel / matrix 三块, 复杂度高, 别自己重推.

### 2026-06-29 — deepseek-v4-pro

**1. 不读已有文档就行动.** 进入 workstream 后直接基于 design/handoff 开始写代码和引导实验,
没有读过 `.moss_ws/apps/bodies/g1/docs/index.md` 这份最完整的调研文档.
index.md 里明确写了 `mode_machine` 是 Dof 配置字节 (不是 FSM),
写了 FSM 模式 ID 表, 写了调试模式的正确切入路径.
没读导致在 script 17 中用错误字段追踪 FSM, 浪费了一轮实验.

**2. 对不知的术语装作知道.** `mode_machine` 的真正含义在官方文档的 LowState_ 结构体注释里
写得很清楚 (4:23Dof, 5:29Dof, 6:27Dof). 在没有查证的情况下, 把它当作 FSM 模式来用,
在脚本里命名 `fsm_mode`, 在引导实验时说"看 fsm_mode 的变化".
这是危险的 — 如果面对真的不懂开发的执行人, 这个错误不会被发现.

**3. 按提示词伪装为已经懂了的模型.** 当用户问"怎么切 Sport""怎么退出调试模式"时,
给出的答案是"试试这个""可能应该是", 而不是"我不知道, 让我查文档".
这种引导式试错在安全攸关的实机场景下不可接受.

**预防规则 (给后续进入的模型实例)**:
- 动手前读完 `docs/` 下所有已有文档, 不是只看 design/handoff
- 任何字段含义不确定时, 查 IDL 源码或官方文档, 不猜
- 面对不知道的操作(遥控器组合键等), 直接说"不知道", 建议走 SDK API 路径

## G1 术语表 (中英对齐)

遥控器语音 / App 显示 / SDK API / 文档 四套命名体系不统一. 此表维护对应关系.

| 遥控器语音 | 文档中文 | SDK API / 模式名 | mode_machine / FSM ID | 说明 |
|-----------|---------|-----------------|----------------------|------|
| 阻尼模式 | 阻尼 | Damp | FSM 0? (待 24 验证) | L2+B 急停进入. 手动切需长按 L2+A 数秒 |
| 诊断状态 | 调试模式 | Debug / ReleaseMode() | — | L2+R2 进入(仅从阻尼/零力矩). SDK 控制入口 |
| 预备模式 | 预备 | — | — | 5s 摆出准备姿势 |
| 走跑模式 | 运动 | Sport / Loco / Start() | FSM 500/801/802 | 遥控器控制移动 |
| 零力矩 | 零力矩 | ZeroTorque | FSM 0 | 电机停转无阻尼 |
| 落座 | 落座 | Sit | FSM 3 | 安全姿态 |
| 站立 | 站立 | StandUp | FSM 4 | 锁定站立 |

**命名陷阱**: "阻尼模式" != Damp API (后者是 RPC 函数名, 前者是遥控器操作模式).
遥控器语音"诊断状态" 对应 SDK 调试模式, 不是 L2+A.

## G1 物理事实 (实机验证中发现, 持续更新)

每条来自实机观察, 标注日期和触发条件. 这些不是文档推断, 是物理行为.

### 运动模式切换

- **Sport 直接 L2+R2 进调试模式 → 状态机保护性故障.** PC1 运动控制进程触发保护,
  遥控器和 App 均失去对 G1 的控制. SelectMode("ai") 返回 7002.
  正确路径: Damp → L2+R2. 来源: 2026-06-29 实机, 已确认.
- **运动模式用遥控器切阻尼模式需长按 L2+A 若干秒.** 短按不触发.
  来源: 2026-06-29 实机.

### 身体与悬挂

- **阻尼模式脱力.** G1 在阻尼模式下电机停转(有阻尼), 全身下垂. 没有悬挂装置时
  G1 会仆倒. 必须始终在吊架/悬挂下操作. 来源: 2026-06-29 实机.
- **吊架上从 Sport 切 Damp 时凌空蹬腿.** FSM 切换有不可预期的肢体动作.
  吊架环境下所有运动/FSM 切换类脚本 (18/19/20/21) 需预留缓冲空间.
  来源: 2026-06-29 实机, script 17 阶段 2.

### 遥控器与调试模式

- **调试模式下全部 16 按键 + 4 摇杆轴均透传到 wireless_remote[40].** G1 不动.
  Sport 基线和调试模式两轮对照完成. 遥控器=MOSS 输入设备方案成立.
  来源: 2026-06-29 实机, script 17 汇总表.
- **L2+B 硬件急停在调试模式下仍生效.** G1 双手缓慢下降, 身体进入悬挂.
  来源: 2026-06-29 实机.

### Arm 操作

- **ExecuteAction 是互斥锁, 不可抢占.** A 在播时发 B → 7401 或 3104 拒绝.
  ExecuteAction(99) 在 arm 忙时 code=0 但实际排队, 不中断当前动作.
  arm RPC 无真中断. 来源: 2026-06-29 script 21 + 补充测试.
- **ExecuteAction(99) 在 arm 空闲时是平滑缓慢复位** (3/3 打分 1).
  来源: 2026-06-29 script 18.
- **rt/arm_sdk DDS 底层关节控制可行.** weight 使能 + 单关节 + 双关节均
  测试通过. DDS publish 停止 = 真中断 (vs RPC 不可中断).
  arm_trajectory channel 可做, kp/kd 需调软. 来源: 2026-06-29 script 26.
- **LowState 真实频率 ~1052 Hz** (非 500Hz). _monitor.py 构造 frozen
  dataclass 开销可忽略 (差 0.2 Hz). 当前设计吃满 1kHz 无问题.
  来源: 2026-06-29 script 27.

### 数据字段

- **`mode_machine` 不是 FSM 模式.** 它是 Dof 配置字节 (4=23Dof, 5=29Dof, 6=27Dof),
  来自官方文档 LowState_ 结构体注释. 开机后不变.
  `mode_pr` 是并联机构类型 (0:PR, 1:AB). 真正的 FSM 模式在 `LocoClient.GetFsmId()`
  和 `rt/sportmodestate` DDS topic. 来源: 2026-06-29 文档补查, index.md.

## Reachy Mini 经验

| 经验 | G1 应对 |
|------|---------|
| 硬件连接延迟到 bootstrap | 不需要 — app 进程即生命周期 |
| 依赖隔离 | 已做 — app 独立 venv |
| Matrix 错误传播不完整 | DDS 连接失败时进程明确退出 |
| Channel 过度复杂 | **本期重做** — warrant + state DAG + sensors 统一机制 |
| 构造即连接抛异常 | app 进程退出 → Circus 重启, 正常行为 |

## 待验证经验 (实测中观察到的模式, 但未复现确认)

以下来自实机操作中的单次观察, 不当作确认事实. 需要在后续实验中有意识复现验证.

- **调试模式退出后遥控器失效**. 路径: 调试模式 → L2+B 急停到阻尼 → 遥控器组合键(包括 R2+A, R1+X 等)无法切换到运控模式. SelectMode("ai") 返回 code=0 但 G1 物理状态无变化. SelectMode("ai") 只恢复 ai_sport 服务不改变 FSM. 两次实机 session 各触发一次 (2026-06-29). 临时恢复方式: 关机重启. **待验证**: 是否每次调试模式退出都触发? 是否存在正确的遥控器/API 退出路径?

## 未决议题(跨 session 继承)

- **待验证 (7-02 早晨, 3 个 5 分钟脚本)**:
  1. weight 0 释放后 G1 arm 主板姿态行为 (僵直 vs 微动) — 决定 L1 呼吸的实现路径 (借主板 vs 自己 publish 低频 sin)
  2. Jetson 摄像头硬件路径 (cv2.VideoCapture 直出 / GStreamer V4L2 / CSI nvargus 四种 fallback) — 决定 vision runtime 起步
  3. ExecuteAction 99 在 release weight 后能否复位 + 拿到完成信号 (script 28) — 决定 L2 起步能否借用 G1 主板复位能力
- **待验证 (延续)**: rt/sportmodestate Read() 阻塞问题, 调试模式退出后遥控器失效路径, rt/arm/action/state 内容格式
- **待实机 (script 20/22/24)**: Sit↔Stand 物理行为 (阻塞 posture/stand_up channel 命令拓扑定稿), FSM 完整可达图, arm action state probe
- **待实机 (arms L1 起步)**: kp/kd 调软目标值 (建议 kp~20/kd~0.5, 若走自己 publish 路径), arms cancel 后 G1 主板物理行为 (锁定 vs 自动回 sport), 关节镜像表 (若走稻草人前臂交互, 需左右肩 pitch/roll/yaw 符号关系)
- **设计待定 (arms 高级形态期)**:
  - 中断三基础 (碰撞反馈/脱力 + 复位 + 首帧过渡) 的实装方案 — 缺一不可, 是 L4 准入门槛. 复位倾向借 ExecuteAction 99, 首帧过渡引擎内部处理
  - Pose DAG 可达性验证方法 — N² 实测过重, 找精简验证路径
  - 示教录制交互形态 — 语音倒数 3 2 1 + "好了" 结束 + 尾部裁剪, 依赖 G1 是否支持单 arm 鬼模式 (拖动示教硬件路径)
  - 稻草人交互的关节工作空间约束 (双臂平举固定 base, 前臂小范围小心自碰撞)
  - 按键状态机 (F1/F3/Start/L1+组合键 语义切换规则). 由人类工程师牵头. 实现在人类工程师脑中
  - L2+B 后 MOSS 软响应路径. 硬件急停外的软清理由按键状态机定义
  - ASR 模式切换. 本期 channel 硬编码 buffer + 显式 pop + 可选 signal 触发, 未来改造为 nucleus
  - arms 学习库 storage scope (倾向 local_persistent 跨 session 累积, 仅在示教路径可行时激活)
  - arms 与 body 其他 channel 并行约束 (走路时挥手等组合的物理可行性, 待实测)
- **本期不做**: LiDAR 条件反射层, G1 内置录制能力对接, SetFsmId 白名单, arms smoothstep 插值, arms 镜像 API, arms 接入 VLA Nucleus, arms velocity cap planning 校验, IK / 逆运动学 (永久不做, MOSS 边界), 轨迹规划 (永久不做, MOSS 边界)
- **已解决**:
  - Warrant 事务机制 → 砍掉, 走 InterruptNucleus + StopMove (6-29/30 实测推翻 6-28 设计). `warrant.py` 已归档到 `_archived/` (2026-06-30), 不再被任何模块 import.
  - contrib 目录结构 → 四层 (`sdk/runtime/channels/providers`) + `_archived/`, 2026-06-30 重构落地. 详见 "代码结构" 节.
  - 调试模式 vs 运动模式 → 运动模式是 MOSS 主场, 不进调试模式 (Sport → L2+R2 会触发 PC1 保护故障)
  - arm 控制路径 → 砍掉 ExecuteAction RPC 作为常规命令 (不可中断), 只走 rt/arm_sdk DDS (RPC 不可中断, DDS publish 停 = 真中断). **7-01 修正**: ExecuteAction 11-27 可以作为 L2 命名调用包装 (LLM 只看名字, 不涉及中断), 99 作为复位帧候选 (待 script 28 验证)
  - 视觉 channel 优先级 → 本期 P0/P1, 不推迟到后期. **7-01 修正**: G1 进程内子线程 + 滑动窗口范式, 不用 Matrix. 设计见 `runtime/vision.py` docstring
  - MOSS 不做的物理算法边界 → 明文化在 "MOSS 不做的物理算法边界" 节. IK / 轨迹规划 / 动捕信号处理 / VLA 训练四类永久不做
  - Mindflow 能力评估 → partial-triggered 抢占已在 mindflow + listener app 实装, 不需要新建 Articulator
- **已推翻 / 重估**:
  - arms channel 拓扑 (6-30 `design/2026-06-30_g1_arms_animation.md` §3/§5) → **7-01 讨论后推翻 §3 命令面 + §5 学习闭环**. Track 派 + Animation JSON 让 LLM 写 keyframe 违反 §0.3 上位范式纪律 (Logos 层调度命名 VLA, 不接触内部实现). 上位范式 §0 仍成立. 设计文件已加"已被 7-01 修正"标注, 保留原内容作为设计演进档案. 本期 arms 形态改按 "能力金字塔" 节推进, 不写完整修正设计文档 (等 L2/L3 实践积累后一次收口)