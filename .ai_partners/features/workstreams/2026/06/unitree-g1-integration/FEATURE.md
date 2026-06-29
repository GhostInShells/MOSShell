---
title: Unitree G1 Integration
status: in-progress
priority: P0
created: 2026-06-04
updated: 2026-06-30
depends: []
milestone:
description: >-
  将 Unitree G1 人形机器人通过 unitree_sdk2_python 集成到 MOSS，作为 bodies app 提供 CTML 可调用的全身运动控制、手臂操作和音频交互能力。
  安全优先的渐进式推进：文档摸底 → 脚本验证 → channel 设计 → 多级模式迭代。不做高阶开发。
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
- `design/2026-06-28_channel_architecture.md` — **当前最新设计**: channel 体系全貌, warrant 机制, state DAG, 用户故事四幕

讨论轨迹(本 feature 目录下):
- `discuss/2026-06-08_phase_b_sdk_discussion_outline.md` — SDK 摸底阶段讨论提纲
- `discuss/2026-06-28_remote_as_moss_input.md` — 单一控制源反转: 遥控器变 MOSS 输入设备

## 开发阶段

八阶段渐进 (A-H), 详情见 `README.md`. 当前阶段进度:

| 阶段 | 内容 | 状态 |
|------|------|------|
| A | 云端文档摸底 | 完成 (2026-06-07) |
| B | 代码仓库摸底 | 完成 (2026-06-08) |
| C | 硬件环境记录 | 完成 (2026-06-14) |
| D | MOSS 装机 | 完成 (2026-06-14/15) |
| E | 基线实验 (SDK 脚本) | **进行中** — P0 17/18/19 通过; P1 21 通过; P2 26/27 通过; 20/22 待实机 |
| F | 安全理解 | **进行中** — 范式转为运动模式主场; 遥控器主权; 不进调试模式 |
| G | Channel 设计 | **进行中** — 2026-07-01 进入实现; story-2026-07.md 已落 |
| H | 多级模式迭代 | 未开始 |

## Session Log 索引

完整 session 历史按时间倒序索引. 详细内容已迁移到 design/ 与 discuss/, FEATURE.md
只保留入口与关键节点结论.

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

- **待验证 (明天)**: rt/sportmodestate Read() 阻塞问题, 调试模式退出后遥控器失效路径, rt/arm/action/state 内容格式
- **待实机 (script 20/22/24)**: Sit↔Stand 物理行为, FSM 完整可达图, arm action state prob
- **本期不做**: LiDAR 条件反射层, 录制能力, SetFsmId
- **已解决**: 授权键分配 (F1/F3/Start/L1+组合键, 见 story-2026-07.md); Warrant 机制 (砍掉, 走 InterruptNucleus); L2+B 响应模型 (硬件急停 + soft cleanup via signal)
