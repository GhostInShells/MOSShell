---
title: Unitree G1 Integration
status: in-progress
priority: P0
created: 2026-06-04
updated: 2026-06-08
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
- **方法论（范式真相）**: `CLAUDE.md` — 每次会话自动加载，设计决策与方法论在此
- 开发计划: `README.md`（权威 — 每次会话从这里开始）
- 应用说明: `APP.md`
- 技术文档: `docs/`（云端文档摸底 → 本地知识索引）
- 验证脚本: `scripts/`（安全原子化验证，人类反馈闭环）
- SDK 源码: `src/unitree_sdk2_python/`（gitignored，手动 clone）

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
**内容**: 基于阶段 E 的验证结果 + 阶段 F 的安全理解，综合输出 channel 体系设计。遵循最简原则。

### 阶段 H: 多级模式迭代
**产出**: Channel 实现 + 模式体系
**模式渐进**: debug → sit（坐模式）→ 遥控器控制行动但可交互 → 模型控制行动但可急停 → 多种运动模式切换
**约束**: 这一阶段全部以 G1 基线能力为验证对象，不做高阶开发。

## Reachy Mini 经验携带

以下问题对 G1 有直接参考价值：

| 经验 | G1 应对 |
|------|---------|
| 硬件连接延迟到 bootstrap | **不需要** — app 进程即生命周期，构造时直接连 DDS |
| 依赖隔离 | 已做 — app 独立 venv |
| Matrix 错误传播不完整 | 注意：DDS 连接失败时进程应明确退出，不静默降级 |
| Channel 过度复杂（factory、lifecycle） | 最简 channel |
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

1. 读本文件 + `CLAUDE.md` + `README.md` + `docs/index.md`
2. 从阶段 A 开始：云端文档摸底
3. 第一项产出：`docs/index.md`（填充云端 URL 映射表）

---

## 2026-06-07 Session — 骨架搭建与认知入口

### 完成项

- 创建 `CLAUDE.md` 作为 G1 app 的 AI 认知入口，自动加载
- 将方法论（KD1-KD8 的设计决策与开发哲学）从 FEATURE.md 迁入 CLAUDE.md
- CLAUDE.md 声明四项必要知识蓝图（channel_builder, states_channel, matrix, ctml）
- FEATURE.md 精简为工作流追踪：动机、阶段、经验、session log。范式真相指向 CLAUDE.md
- 确认：CLAUDE.md = 范式真相（自动加载），FEATURE.md = 簿记层（显式查阅）

---

## 2026-06-07 Session — 阶段 A 完成

### 完成项

- 消化 10+ 份 Unitree 官方文档，覆盖遥控器、状态机、SDK 架构、DDS 通讯、运动控制、底层通讯、设备状态、音频灯光、LiDAR、里程计、手臂控制、手臂动作、时间同步、G1 总览
- `docs/index.md` 填充完整：每条包含 URL/记录时间/来源层级/关键提取/架构判断
- `docs/validation-checklist.md` 创建：15 个可判真验证命题，桥接阶段 A→B
- CLAUDE.md 新增已知问题标注（static 缓存、避障缺口）
- 关键架构结论：
  - 双路径控制模型：RPC(LocoClient, 非调试, 安全) ↔ DDS(rt/lowcmd, 调试, 全权)
  - 三层安全围栏：硬件(L2+B) → 条件反射(LiDAR) → 模型(CTML/RPC)
  - 初始集成走 RPC + DDS 只读。底层写入留高阶阶段
  - PC2 蓝牙耳机作为音频替代方案
  - 关节限位表是安全控制基础数据
- main.py 稳定为最简 instruction 声明（不再随调研逐条更新）

### 关键洞察

**文档是广告，不是手册。** 官方文档站描述的是"应该能做什么"，但参数、约束、错误行为大量缺失。几乎所有关键命题都需要源码验证——ReleaseMode 的前置条件、PlayStream 的状态反馈、wireless_remote 的格式。三源关系（文档<源码<实测）不是方法论装饰，是经验事实。

**安全边界在硬件层，不在我们的代码里。** 这是 G1 集成最大的幸运。FSM 模式门控、L2+B 急停、crc 校验——这些是 G1 自己的安全机制，MOSS 不需要重建。我们的软件围栏是体验层和纵深防御，不是安全底线。

### 下一步（下一个实例）

1. 读 `CLAUDE.md` + `docs/index.md` + `docs/validation-checklist.md`
2. 与人类开发者对齐验证目标（验证启动前置条件）
3. 阶段 B: clone SDK 源码，验证 API 存在性和签名
4. 阶段 C+D: 硬件环境记录 + MOSS 装机
5. 阶段 E: 按验证清单逐条执行脚本，人类反馈闭环

---

## 2026-06-07/08 Session — 实机连接与 MOSS 装机

### 完成项

- 硬件拓扑确认：PC1（闭源运控）→ 交换机 ← PC2（Jetson Orin NX, 二开入口）← 外部 Mac
- 以太网进入交换机 → 配静态 IP → SSH 到 PC2 (unitree/123)
- PC2 WiFi 射频开启 → 连接本地 WiFi → 路由器 MAC 绑定固定 IP
- 安全加固：创建 moss 用户、UFW 放行 22
- Git 直推通道：Mac `git remote add g1` → `git push g1 dev`
- Python 工具链：pipx → uv → Python 3.12
- `uv sync --active --all-extras` — MOSS 在 G1 PC2 上安装运行
- 文档产出：`docs/hardware.md`, `docs/moss-on-pc2.md`
- CLAUDE.md / README.md 阶段状态更新为阶段 B

### 关键发现

**G1 架构是交换机组网，不是点对点。** 外部以太网口连接的是交换机，Mac 配静态 IP 后可与 PC1、PC2、LiDAR 三者通信。PC1 通过访问控制闭源，PC2 是唯一二开入口。

**PC2 WiFi 默认关闭是安全设计，不是 bug。** PC2 被隔离在交换机后，无法自行出站。这防止了二开代码意外联网，但也意味着每次装机都要先用以太网路径打开 WiFi。

**uv 工具链在 Jetson 上的摩擦：** Python 3.8 系统版本 → pipx → uv → Python 3.12，每一步都涉及源配置（pip/pipx/uv 三级互不继承）。镜像源的路径兼容性问题导致预编译 Python 下载反复失败。最终 Mac 中转解决了带宽瓶颈。

### 下一步

1. `uv sync` 构建完成后验证：`moss --ai start`、`moss --ai all-commands`
2. 阶段 B: clone SDK 源码到 PC2，验证 API
3. 阶段 E: 按验证清单逐条执行脚本，人类反馈闭环

---

## 2026-06-08 Session — 系统线实验体系搭建 + 方法论博客

### 完成项

- 博客产出：`g1-layered-methodology.md` — G1 分层推进方法的整体拓扑（认知层→验证层→构建层）、三源真相关系、脚本先于 channel 的原则论证
- 系统线实验体系：`scripts/sys/` — 按 Claude Code skill 范式组织，6 个技能 × 19 个原子脚本
  - network（网络拓扑）、audio（音频硬件路径决策）、usb_camera（视觉方案）、system（系统身份+环境）、dds（通讯就绪）、performance（性能基线+运行时对比）
  - 每个技能一个 `SKILL.md`（frontmatter + 调查命题 + 架构决策标注）
- 调研时序文档：`scripts/sys/RESEARCH_SEQUENCE.md` — 11 步对话式执行脚本，每步标注命令/为什么/观察什么/影响什么决策。第一轮验证后改造为 MOSS channel 命令
- `docs/index.md` 中的音频路径决策逻辑在系统线中得到实操化：audio 技能的四个脚本直接回答"PC2 能否独立发声/录音/连蓝牙"
- `docs/index.md` 中的视觉方案缺口在系统线中得到调查入口：usb_camera 技能摸底 PC2 物理接口和 v4l2 驱动能力
- 明确：系统线先于 SDK 线。系统线回答"PC2 有什么"，SDK 线回答"SDK 提供了什么 API"

### 关键洞察

**系统线是 SDK 线的前置，不是并行。** 音频硬件事实（PC2 有声卡/蓝牙 vs 无）直接决定 SDK 线应该研究 AudioClient RPC 还是 ALSA/蓝牙本地方案。在没有硬件事实的情况下读 SDK 源码，可能读错方向。

**skill 范式天然适合原子化实验。** 每个 SKILL.md 标注"调查命题 + 架构决策"，每个 .sh 只做一件事——这种组织方式让未来的 AI 实例不需要读完整份文档就能找到自己需要的脚本。未来改成 MOSS channel 命令时，SKILL.md 直接转化为 channel instruction。

### 下一步

1. 人类在 PC2 上按 `RESEARCH_SEQUENCE.md` 逐项执行系统线实验，反馈结果
2. 基于系统线结论确定 SDK 线调研范围（例如：如果音频走 PC2 本地，则 SDK 线跳过 AudioClient RPC 分析）
3. 系统线第一轮验证通过后，将脚本改造为 MOSS channel 命令（AI 可运行时调用）

---

## 2026-06-08 Session — SDK 源码摸底 + 验证脚本体系 + 能力拓扑讨论

### 完成项

- SDK 源码 clone 到 `src/unitree_sdk2_python/`（gitignored）
- 修复 `.gitignore`: `src/` → `src/unitree_sdk2_python/`
- **SDK 源码通读**: G1 三组 API(loco/arm/audio) + MotionSwitcher + RobotState + RPC 基类 + DDS Channel
- **Topic 清单**: `docs/sdk-topics.md` — 18 个 topic 的类型组/方向/MOSS 能力归属/类型验证清单
- **文档<源码差异**: LocoClient 缺失方法(Squat/ContinuousGait/GetFsmMode/StandUp)、action_map 实际 17 种(vs 文档 15)、TtsMaker index bug 等
- **能力拓扑**: 横向 5 能力(系统感知/G1感知/音频灯光/手臂控制/高阶运动) × 纵向 6 模式(0.Pure→1.Observer→2.Passenger→3.Mover→4.Gesturer→5+.Beyond)
- **讨论提纲**: `.ai_partners/features/.../discuss/2026-06-08_phase_b_sdk_discussion_outline.md`
- **SDK 验证脚本**: `scripts/sdk/` — 12 个脚本(00-11) + SKILL.md:
  - 00-03: 环境/反射/类型/发现(00-02 无 G1 可跑)
  - 04-06: A 层纯读(LowState+遥控器, SportMode, 电池/主板/IMU)
  - 07: B 层 RPC 只读合集(4 个接口)
  - 08: C 层音频灯光(TTS 中断探路+LED+PlayStream)
  - 09: D 层上肢(基础动作+中断复位+动作序列)
  - 10-11: E 层模式切换+移动(极慢速 0.5s)
- **急停验证脚本**: `12_estop_verify.py` — 踏步中 L2+B→Damp() + 双路标记(内存+文件 `/tmp/g1_moss_estop`)
- **Task 设计方向**: 协程管理生命周期 + 线程跑运动轮询(20ms)。Cancel = 线程 stop 信号 + 温柔复位(weight→0)

### 关键洞察

**SDK 是 Topic 的封装，不是能力的全集。** 大部分横向能力 SDK 已封装(RPC client)，缺口在被动感知 topic(电池/主板/IMU)和 ASR/麦克风。都是"有数据通路无 convenience class"，不需要 SDK 新增能力。

**急停模型**: 硬件 L2+B 不可绕过(FSM 直接阻尼) — 真正的安全底线。MOSS 层 Damp() 是体验层响应。不碰 ZeroTorque(危险)。`/tmp/g1_moss_estop` 文件作为跨脚本可见的急停标记。

**坐标管理任务设计**: async Task 管生命周期 → 子线程 20ms 轮询 → cancel = stop_event + 安全指令(Damp/StopMove/release arm)。与 MOSS channel 体系自然融合。

### 下一步 (给下一个实例)

1. **提交本 session 所有产出** (`scripts/sdk/` 12 个脚本 未提交)
2. 人类优先在 PC2 上执行系统线 `RESEARCH_SEQUENCE.md` (阻塞项 — 决定音频路径等关键决策)
3. 系统线通过后，SDK 脚本 00-03(无 G1 可跑) + 04-07(纯读) 第一轮验证
4. 基于系统线音频结果，决策 08(音频灯光) 是否进一步做二阶实验
5. 基于人类反馈逐步解锁 09(上肢)→10(模式切换)→11(移动)→12(急停验证)
6. 讨论提纲中未决议题: SetFsmId 白名单拦截、L2+B 后 MOSS 响应模型、条件反射层归属

**不在此 session**: terminal channel 建设、VLA/VLM policy 协调、DDS 调试模式解锁
