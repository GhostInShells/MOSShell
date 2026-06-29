---
title: Unitree G1 Integration
status: in-progress
priority: P0
created: 2026-06-04
updated: 2026-06-28
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
| E | 基线实验 (SDK 脚本) | **进行中** — 03-08/13/14 已通过, 09/15-19 待实机 |
| F | 安全理解 | **进行中** — channel_architecture 已涵盖 |
| G | Channel 设计 | **进行中** — 2026-06-28 体系蓝图已落 design/ |
| H | 多级模式迭代 | 未开始 — 等地基脚本通过 |

## Session Log 索引

完整 session 历史按时间倒序索引. 详细内容已迁移到 design/ 与 discuss/, FEATURE.md
只保留入口与关键节点结论.

### 2026-06-28 — Channel 体系设计

由 Claude Opus 4.7 与人类工程师协作完成. 本 session 重写了 g1 集成的整套机制层架构.

**关键产出**:
- 设计沉淀: `design/2026-06-28_channel_architecture.md`
- 讨论轨迹: `discuss/2026-06-28_remote_as_moss_input.md`

**核心范式决策**:
1. **单一控制源**: 物理指令唯一通过 MOSS channel → SDK. 遥控器在调试/AI 模式下被 G1 主板
   忽略(除 L2+B), MOSS 把遥控器字节捡起来当作自己的人机协作输入设备
2. **感知统一**: 所有感知进 context_messages, `pop()` 命令显式进 memory
3. **State DAG**: channel 状态单向拓扑, 不用 StatefulChannel, 用 available_fn + virtual_children 映射
4. **Warrant 事务**: 危险命令统一封装, 三回调 race(正常完成/中断信号/state 失效)
5. **Bootstrap callback**: 线程安全注册遥控器按键 callback, 跟 channel 生命周期对齐

**下一步**:
1. 实机跑 17/18/19 三条 P0 地基验证脚本(见下方"待实机清单"). 这三条决定整套设计是否成立.
2. 17 通过 → 整理遥控器物理按键到 warrant scope/state DAG 边的绑定表
3. 18 通过 → arm channel 进入 warrant 包装, 19 通过 → move 同
4. 任何一条不通过 → 回炉重设计
5. P1 三条(20/21/22)在 P0 通过后排进下一轮

**待实机清单(P0, 阻塞性)**:
- `scripts/sdk/17_remote_keys_passthrough.py` — 调试/AI 模式下按键 + 摇杆是否仍透传 wireless_remote
- `scripts/sdk/18_arm_release_behavior.py` — ExecuteAction(99) 物理行为
- `scripts/sdk/19_loco_stopmove_under_motion.py` — move 中 SetVelocity(0,0,0) 是否站定

**待实机清单(P1, 补完用户故事 + channel 实现)**:
- `scripts/sdk/20_sit_stand_cycle.py` — Sit↔Stand SDK 可达性, 706 双向性, 用户故事幕三可行性
- `scripts/sdk/21_arm_action_interruption.py` — Action A 中发 B(非 99) 的行为: 覆盖/排队/拒绝
- `scripts/sdk/22_arm_action_state_probe.py` — rt/arm/action/state 内容, 决定 arm 命令 await 实现路径

详细执行顺序见 `scripts/sdk/RUN_ORDER.md`.

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

## Reachy Mini 经验

| 经验 | G1 应对 |
|------|---------|
| 硬件连接延迟到 bootstrap | 不需要 — app 进程即生命周期 |
| 依赖隔离 | 已做 — app 独立 venv |
| Matrix 错误传播不完整 | DDS 连接失败时进程明确退出 |
| Channel 过度复杂 | **本期重做** — warrant + state DAG + sensors 统一机制 |
| 构造即连接抛异常 | app 进程退出 → Circus 重启, 正常行为 |

## 未决议题(跨 session 继承)

- SetFsmId 白名单拦截
- L2+B 后 MOSS 响应模型(2026-06-28 部分回答: 软清理 + signal 通知 ghost)
- 条件反射层归属(LiDAR 避障, 本期不做)
- Warrant 最终命名(暂用 warrant, 等实现时定)
- 录制能力: SDK 是否暴露, 22 实测后决策
- 授权键的物理分配(17 实测可用键集合后再定)
