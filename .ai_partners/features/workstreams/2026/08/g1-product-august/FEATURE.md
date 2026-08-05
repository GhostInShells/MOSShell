---
title: G1 Product August
status: in-progress
priority: P0
created: 2026-08-05
updated: 2026-08-05
depends:
  - node-migration
milestone: v0.1.0
description: >-
  g1 产品化的 8月版本。unitree-g1-integration (completed 2026-07-06) 之后的独立演进 —
  单 feature 大架构 + 子任务文档治理。8月核心目标全在交互 (可学习 pose + 上肢平滑
  切换)。与其他能力 (voice / screen) 的关联只作验收目标, 不作
  开发任务。开发任务仅围绕 g1。
---

# G1 Product August

> 人类架构师 + deepseek-v4-flash。v0.1.0 STAGE.md 高危线 "G1 development needs
> restart as a new feature" 的落点。本 FEATURE.md 是**核心索引 + 状态管理** —
> 大过程由 `subtasks/` 子任务文档逐个推进治理, 本文件承载产品叙事、关键决策、
> 能力路线图与子任务状态。

## Motivation

unitree-g1-integration 完成了"集成验证" (Phase A-G): 范式定型 (双工分层具身、
遥控器永久主权、运动模式主场)、6 channel 落地、showcase 基线 5/6。但那是
**验证可行性**, 不是**产品**。产品化的差距是: 没有在真机上跑通过完整的、可演示
的 Ghost 身体闭环, 交互是碎片化的, 遗留问题 (锁阻塞、arms 空骨架、listener
未闭环) 拦在路上。

本 feature 的决策是 **"8月能做多少"**: 单 feature 大架构承载 g1 的独立演进,
不做功能切分。核心判断 — 8月的重心全部在**交互**上, 把一台"活着"的、可交互的
G1 做出来。

**产品叙事**: g1 是 MOSS 具身能力的代表 — 在真机上跑通一台"活着"的、可交互的
G1, 是可演示、可扩展的人形机器人身体产品。8月的重心全在交互上, 把交互做好。

## 治理结构

- **单 feature 大架构**: g1 是一个独立演进, 不拆多个 features。膨胀到要严肃分割时,
  complete 当前、新建下一个 (子任务文档随之收敛)。
- **子任务文档**: 大过程由 `subtasks/` 逐个文档治理。**子文档不预建**, 推进到该任务时
  创建。本 FEATURE.md 是核心索引和状态管理。
- **验收目标 ≠ 开发任务**: 与其他能力的关联 (voice 多渠道 / screen 流式 GUI /
  演示载体) 一律作为验收目标, 不开发它们。开发任务仅围绕 g1。
- **nodes 并行哲学**: MOSS 的并行能力天然不互相依赖 (nodes 架构最强点)。g1 的
  各能力线 (感知 / 控制 / 交互) 与外部能力各自演进, 不构成阻塞。

## 8月范围 (P0/P1)

**P0 — 交付闭环 (核心)**:
- node 迁移 (category 级) — `nodes/bodies/g1/` 多 node + node 启动替代 mode
- 锁问题修复 — 移动命令稳定阻塞 5-6s 的根因定位与修复 (交互流畅度前提)
- idle 体系 — 身体呼吸感, 让 g1 "活着"
- arms 交互 — 两实验 → 命名 pose + 上肢平滑切换 (8月最重要交互点)
- passenger 模式 — 遥控器主权下 g1 仍流畅交互
- 视觉实装 — 视频线 + context messages
- g1 × 流式 GUI 互动 — 交付 P0 目标

**P1 — 能力推进**:
- 高阶模式 + 高阶 action — MOSS 控制下第一次翻跟头 (安全验证后)
- 弱网看门狗 — 遥控器急停之外的软保障
- 雷达感知验证 — 雷达/vision/slam → context messages 的可能性
- speech 输出 node 化 — 语音输出成为可选 channel

**远期 (8月明确不做)**: 跟随模式 (避障 + 跟随信号外设, 手机拉着 g1 走)。

## arms 核心命题 (最重要的认知修正)

历史 workstream 的 `design/2026-06-30_g1_arms_animation.md` 与 7-01 推翻
("中断三基础") 是 claude-opus-4-7 的推演, **不继承其悲观结论**。人类架构师
明确指出: 记录有预判倾向, 一直在误导后续模型。arms 设计以 **SDK 调研 + 实机
实验** 为准。等结论验证后再与 opus 复盘, 不预先改写其记录。

**两个关键实验定 arms 命运**:
1. **可打断的手臂平滑复位** — 复位过程随时可被打断转向新目标, 且平滑。
2. **柔性运动** — 低 kp/kd 下的柔顺动作, 碰撞能让步。

**痛点**: 复位过程不可中断 → 时序逻辑难写 (动作队列无法"新命令抢占复位")。
人类架构师有优化招数, 方法多得很。

**解决路径树** (由 SDK 调研结论分叉):
- SDK action 机制若**可打断** → 平滑复位是算法问题, 可解。双臂在固定躯体上的
  运动算法不难; 中间躯干问题用幅度约定约束; 必要时做 IK。
- 若**不可打断** → action 复位 + 后续 arms 命令碰到 action 未释放直接返回失败。
  pose 是安全的 (命名 pose 不依赖打断, 如碇司令对话姿势)。
- **退行策略**: 提前录制手臂 action + 计时预先算好, 手臂交互从"短帧动作"变
  "长对话"。方法多得很。

**安全是第一约束**: 状态控制保证安全, 远比动作灵活性重要。

## 历史继承 (unitree-g1-integration)

继承的范式真相 (不推翻, 是地基):
- **双工分层具身**: `sdk/runtime/channels/providers` 四层 + `_archived/` 归档区
- **四轴能力路线图**: 空间 / 手臂 / 感知 / mindflow, 通过授权门控耦合
- **遥控器永久主权**: 遥控器是永远的控制主权, 运动模式是 MOSS 主场, 不进调试模式
- **MOSS 不做的物理算法边界**: IK / 轨迹规划 / 动捕信号处理 / VLA 训练永久不做,
  物理层算法借用 G1 主板 / SDK 已有能力
- **感知纪律**: context_messages peek/drain 双面, "听不见"必须是可感知的事实

## 子任务索引 (状态管理)

> 子任务文档在推进到该任务时创建于 `subtasks/`, 逐个推进, 不预建。

| # | 子任务 | 状态 | 说明 |
|---|--------|------|------|
| 01 | SDK action 机制调研 | 待启动 | 纯读代码; 产出 arms 可打断性结论, 定 arms 时序设计 |
| 02 | node 迁移 (category 级) | 待启动 | `nodes/bodies/g1/` 多 node + node 启动替代 mode |
| 03 | 锁问题修复 | 待启动 | 移动命令 5-6s 阻塞根因定位与修复 |
| 04 | idle 体系 | 待启动 | 身体呼吸感 |
| 05 | arms 交互 | 待启动 | 两实验 → 命名 pose + 平滑切换 |
| 06 | 视觉实装 | 待启动 | 视频线 + context messages |
| 07 | passenger 模式 | 待启动 | 遥控器主权下的流畅交互 |
| 08 | 高阶 action | 待启动 | 翻跟头链路 (P1) |
| 09 | 弱网看门狗 | 待启动 | (P1) |
| 10 | 雷达验证 | 待启动 | (P1) |
| 11 | g1 × 流式 GUI | 待启动 | 验收目标落地: 交付 P0 |
| 12 | speech 输出 node 化 | 待启动 | (P1) |

## 验收目标 (不开发, 只验收)

以下能力的**开发**不在本 feature 内, g1 与它们的**互动/协作**是验收标准:

- **voice 多渠道**: moss 原生支持多语音输入源并存。g1 的 listener/asr channel
  保留 (作为 g1 渠道之一), `sensors/voice` 是另一个渠道。g1 语音输入不完整但可用,
  不阻塞交互。
- **screen 流式 GUI**: 交付 P0 "g1 + 流式 GUI 可以互动"。screen node 由
  `screen-node` workstream 开发, g1 侧只做互动形态。
- **可演示性 (demoable)**: g1 的交互能力可独立演示。演示工具 / 载体本身是其他路径。

## Design Index

- 历史 workstream: `unitree-g1-integration` (2026-06/07) — 范式真相、四轴、
  能力金字塔、物理事实表。**arms 段落以本 feature 的修正为准, 不继承其中断三基础**。
- 参考治理先例: `matrix-cell-governance` (§ 契约段落), `ghost-runtime-safemode`
  (Key Decisions + Rounds)。
- 迁移基础: `node-migration` — nodes 体系、NODE.md、category 目录先例
  (`nodes/screens/` + `qt_screen`)。
- 语音关联: `voice-input-state-machine` — voice node 四层状态机。
- 讨论轨迹: `discuss/` (本 feature 下, 随设计碰撞沉淀)。
- 子任务文档: `subtasks/` (逐个推进时创建)。

## Key Decisions

1. **单 feature 大架构, 子任务文档治理**。参考 matrix-cell-governance 与
   ghost-runtime-safemode。g1 是独立演进; 膨胀到要严肃分割时 complete 当前新建。
2. **验收目标 ≠ 开发任务**。与其他能力的关联只做验收, 开发任务仅围绕 g1。
   被拒: 把 voice / screen / 社区 node 的开发并进本 feature。
3. **arms 以 SDK 调研 + 实机实验为准**。不继承 opus 的"中断三基础"悲观结论
   (记录有预判倾向)。核心命题 = 可打断平滑复位 + 柔性运动。
4. **8月核心 = 交互**。可学习 pose + 上肢平滑切换是最重要交互点; idle 呼吸感是
   "活着"的产品质感; passenger 模式保证遥控器主权下仍可交互。
5. **node 启动替代 mode 注册 channel**。g1 从 `unitree_g1 mode` 切到
   `moss nodes run`, 装配逻辑迁入 node 的 main.py。

## Implementation Notes

- **第一批推进**: 子任务 01 (SDK action 机制调研) — 纯读代码, 不依赖真机,
  产出 arms 时序设计的依据。随后 02 (node 迁移)。
- **SDK 位置**: `.moss_ws/apps/bodies/g1/src/unitree_sdk2_python/` (gitignored,
  手动 clone)。macOS 上读源码规划, PC2 实机验证。
- **遗留承接**: 移动命令阻塞 (03)、arms 空骨架 (05)、listener 未闭环 (验收关联)、
  耳机按键 evdev dormant bug — 均从历史 workstream 继承。
- **FEATURE.md 更新纪律**: 每个子任务启动/完成时更新状态表; 关键决策编辑本节;
  完成时 `set-status completed` 随代码 commit。
