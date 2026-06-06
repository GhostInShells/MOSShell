---
title: Unitree G1 Integration
status: in-progress
priority: P0
created: 2026-06-04
updated: 2026-06-07
depends: []
milestone:
description: >-
  将 Unitree G1 人形机器人通过 unitree_sdk2_python 集成到 MOSS，作为 bodies app 提供 CTML 可调用的全身运动控制、手臂操作和音频交互能力。
  安全优先的渐进式推进：文档摸底 → 脚本验证 → channel 设计 → 多级模式迭代。不做高阶开发。
---

# Unitree G1 Integration

## Motivation

继 Reachy Mini 之后，G1 是 MOSS 接入的第二个人形机器人平台。与桌面级 Reachy Mini 不同，G1 是 1.3m 全尺寸人形机器人，拥有 23-43 个自由度、DDS 通讯总线、高低两级控制 API。这次集成验证 MOSS 的 app 模式在更大规模机器人平台上的可迁移性。

SDK (unitree_sdk2_python) 需手动 clone 到 app 的 `src/` 目录，详见 README.md 环境准备章节。

## Design Index

- App 路径: `.moss_ws/apps/bodies/g1/`
- 开发计划: `README.md`（权威 — 每次会话从这里开始）
- 应用说明: `APP.md`
- 技术文档: `docs/`（云端文档摸底 → 本地知识索引）
- 验证脚本: `scripts/`（安全原子化验证，人类反馈闭环）
- SDK 源码: `src/unitree_sdk2_python/`（gitignored，手动 clone）

## Key Decisions

### KD1: 目录命名: `g1` 而非 `unitree_g1`

Unitree 是厂商名，G1 是型号。app 寻址为 `bodies/g1`，简洁且避免厂商绑定。同一厂商的其他型号 (H1, H2) 可以是 `bodies/h1`, `bodies/h2`。

### KD2: SDK gitignored，README 文档化安装步骤

SDK 不在版本控制中。开发者通过 README.md 中的 `git clone` 命令手动获取。理由：SDK 在 macOS 上无法编译 (cyclonedds 需 Linux)，现阶段只读代码分析 API，无需版本追踪。

### KD3: APP.md 是应用说明，README.md 是开发说明

APP.md 面向使用者/模型：这个 app 是什么、提供什么能力、怎么调用。README.md 面向开发者：当前阶段、设计决策、进度。

### KD4: App 进程 = 生命周期管理器（2026-06-07）

app 模式下进程由 Circus 管理，进程独立、可单独重启。因此 channel 不需要 bootstrap/cleanup 生命周期 hook —— 构造里直接连硬件，连接失败抛异常 → 进程退出 → Circus 重启。不需要 factory 模式、不需要延迟连接。

这是对 Reachy Mini 经验的关键修正。Reachy Mini 的 `bootstrap()` 延迟连接是 mode channel 时代的遗留 —— channel 嵌在 host 进程里，构造失败会拖垮整个 host。app 模式天然解耦了这个问题。

### KD5: Channel 最简原则（2026-06-07）

Channel 的核心价值只有两件事：Code as Prompt（Python 函数签名直接成为模型可见的命令接口）、并发有序（同一 channel 内顺序执行，跨 channel 并行）。其他东西 —— 生命周期 hook、factory 模式、状态声明 —— 是进程内嵌入模式催生的防御性补丁。

G1 应该是最简 channel 的示范：一个类，构造里连硬件，方法暴露命令。没有 factory，没有 lifecycle hook，没有状态声明。

### KD6: 安全先于设计，脚本先于 channel（2026-06-07）

在全尺寸人形机器人上，先理解安全机制再设计 channel 不是可选项。用独立脚本验证基线能力，人类反馈确认后，再从验证结果提炼 channel 设计。而不是直接写 channel 然后猜测硬件行为。

### KD7: 技术文档与博客分离（2026-06-07）

技术文档 (`docs/`) 是活的，随代码迭代更新，保持"当前真相"。博客 (`.ai_partners/blogs/posts/`) 是时间点快照，写决策的 why，写完不改。

### KD8: macOS 不做实装（2026-06-07）

macOS 上不需要编译 cyclonedds。开发流程：在 macOS 上读 SDK 源码 + 云端文档做规划，在 G1 PC2 (Linux) 上实装验证。`docs/` 和 `scripts/` 在 macOS 上编写，通过 git 同步到 PC2 执行。

## 开发阶段

### 阶段 A: 云端文档摸底
**产出**: `docs/index.md` + `docs/sdk-api.md` + `docs/comms.md`
**性质**: 纯阅读，不写代码
**内容**: Unitree 官方文档站的关键页面 URL 映射、API surface 梳理、DDS 通讯模型理解
**备选**: 如果文档站是纯 SPA（WebFetch 抓不到），直接以 SDK 源码 + examples 为主要信息源

### 阶段 B: 代码仓库摸底
**产出**: 补充 `docs/sdk-api.md`，记录实际 API surface
**内容**: 读 `unitree_sdk2_python` 源码，理解 loco/arm/audio 三组 API 的函数签名、参数、返回值

### 阶段 C: 硬件环境记录
**产出**: `docs/hardware.md`
**内容**: 硬件连接方式、网络拓扑、PC2 规格、IP 地址、网卡配置。可复现的环境准备流程。不涉及帐号信息。

### 阶段 D: MOSS 装机
**产出**: `docs/moss-on-pc2.md`
**内容**: MOSS 安装到 G1 PC2 的过程记录、Python 版本、系统依赖、网络权限问题。定位是"问题日志"——装机过程中每个异常都值得记录，不是一次性文档。

### 阶段 E: 基线实验
**产出**: `scripts/` 下的安全原子化验证脚本 + 人类验证反馈
**验证点来源**: SDK examples + docs 分析结果
**原则**: 独立 Python 脚本，直接在 PC2 上跑，不经过 MOSS channel 体系。每个脚本验证一个原子能力。人类反馈记录验证结果。

### 阶段 F: 安全理解
**产出**: `docs/safety.md`
**内容**: 急停机制、关节限位、力控限制、遥控器优先级、模式切换的安全约束。必须在 channel 设计之前完成。

### 阶段 G: Channel 设计
**产出**: `docs/channel-design.md`
**内容**: 基于阶段 E 的验证结果 + 阶段 F 的安全理解，综合输出 channel 体系设计。遵循 KD5 最简原则。

### 阶段 H: 多级模式迭代
**产出**: Channel 实现 + 模式体系
**模式渐进**: debug → sit（坐模式）→ 遥控器控制行动但可交互 → 模型控制行动但可急停 → 多种运动模式切换
**约束**: 这一阶段全部以 G1 基线能力为验证对象，不做高阶开发。

## Reachy Mini 经验携带

以下问题对 G1 有直接参考价值：

| 经验 | G1 应对 |
|------|---------|
| 硬件连接延迟到 bootstrap | **不需要** — app 进程即生命周期，构造时直接连 DDS（KD4） |
| 依赖隔离 | 已做 — app 独立 venv |
| Matrix 错误传播不完整 | 注意：DDS 连接失败时进程应明确退出，不静默降级 |
| Channel 过度复杂（factory、lifecycle） | 最简 channel（KD5） |
| 构造即连接抛异常 | app 进程退出 → Circus 重启，正常行为 |

## 2026-06-07 Session — 技术方案起草

### 完成项

- 确定开发哲学：安全优先、脚本先于 channel、最简 channel、macOS 不做实装
- 确定 app 目录结构：`docs/` + `scripts/` + 已有文件
- 确定八阶段推进计划（A-H）
- 确认 blog 分离：技术文档在 app 内，博客在 `.ai_partners/blogs/`
- 确认 app 进程 = 生命周期管理器，不需要 channel 层面的 lifecycle hook

### 关键洞察

**Channel 复杂度是历史的，不是设计的。** 当前 channel 体系中的 bootstrap/cleanup 生命周期、factory 模式、stateful 声明，是进程内嵌入模式催生的防御性补丁。app 模式天然解耦了这些问题 —— 进程就是生命周期，挂了就重启。G1 作为最简 channel 的示范，只需要构造 + 方法 + Matrix 注册。

### 下一步（下一个实例）

1. 读本文件 + `README.md` + `docs/index.md`
2. 从阶段 A 开始：云端文档摸底
3. 第一项产出：`docs/index.md`（填充云端 URL 映射表）
