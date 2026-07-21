# G1 Phase B 讨论提纲 — SDK 源码摸底后的决策框架

**上下文**: 2026-06-08 会话，阶段 B SDK 源码摸底完成。已通读 G1 三组 API 源码 + 5 个关键 example + MotionSwitcher/RobotStateClient + RPC 基础设施。Topic 清单已落地 `docs/sdk-topics.md`。

---

## 一、能力拓扑：横向解耦 × 纵向集成

### 横向能力（独立模块，判断互斥关系）

| # | 能力 | 数据源 | 可否独立 | 互斥约束 |
|---|------|--------|---------|---------|
| 1 | 系统感知 | Linux `/proc`/`/sys`/`psutil` — G1 无关 | 是 | 无 |
| 2 | G1 感知 | DDS 订阅 (`rt/lf/lowstate`, `rt/lf/bmsstate`, `rt/odommodestate` 等) | 是 | 需要 DDS 环境就绪 |
| 3 | 音频/灯光 | AudioClient RPC + `rt/audio_msg` DDS + UDP 组播 (麦克风) | 是 | LED 和 TTS 互不干扰；ASR 和 TTS 可能资源竞争 |
| 4 | 手臂控制 | G1ArmActionClient RPC (预设) + `rt/arm_sdk` DDS (自定义) + `rt/dex3/*/cmd` (灵巧手) | 是 | `g1_arm_example` 服务与 rt/arm_sdk 互斥 |
| 5 | 高阶运动 | LocoClient RPC (非调试) + `rt/lowcmd` DDS (调试) | 是 | RPC 与 DDS 路径互斥（调试/非调试模式二选一）|

前三项纯读/无副作用。后两项有副作用，需要安全围栏。

### 纵向模式（横向能力的组合授权）

模式是能力的**组合开关**，决定哪个横向能力在什么安全约束下可用。

**0. Pure MOSS on PC2**  
- 能力: 系统感知(1)  
- 约束: 无 G1 连接。MOSS 在 PC2 上运行但不接触机器人  
- 前置: MOSS 装机完成（阶段 D）

**1. Observer — 感知对接**  
- 能力: 系统感知(1) + G1 感知(2)  
- 约束: 只读 DDS 订阅。零副作用  
- 前置: DDS 环境就绪 + `ChannelSubscriber` 验证  
- 用途: 状态监控、遥控器读取、数据记录

**2. Passenger — 感知 + 语音**  
- 能力: 系统感知(1) + G1 感知(2) + 音频/灯光(3)  
- 约束: 无运动副作用。可在运动模式下运行（遥控器控制身体，MOSS 做感知+对话）  
- 前置: Observer 就绪 + 音频路径确认（系统线步骤 6-8）

**3. Mover — 高阶运控矩阵**  
- 能力: 1 + 2 + 3 + 高阶运动(5)  

| 象限 | 非调试模式 | 调试模式 |
|------|-----------|---------|
| 坐姿 + 上肢 | LocoClient (Sit) + G1ArmActionClient (17 预设) | 不适用 |
| 全身运动 | LocoClient RPC (Move/StandUp/Damp) — 有内置运控兜底 | `rt/lowcmd` DDS — 无安全网 |

- 约束: 非调试模式优先。调试模式留高阶阶段  
- 安全: L2+B 硬件急停始终有效；channel 层对 SetFsmId 做白名单拦截

**4. Gesturer — 高阶运控 + 上肢**  
- 能力: 1 + 2 + 3 + 4 + 5（全横向能力）  
- 约束: 手臂控制选路径 — RPC 预设动作 vs 自定义 DDS 关键帧。两者服务互斥  
- 安全: 手臂控制与身体运控的模式互斥检查

**5+. Beyond** — MOSS 不架在 SDK 上，架在 VLA/VLM/Policy 协调状态机。第一阶段不做。

### 第一阶段范围: 0→1→2→3(非调试象限)→4(RPC预设)

讨论点:
1. 模式 3 的"坐姿 + 上肢"和"全身运动"是否应是两个独立模式？还是同一个模式的不同子状态？
2. 模式切换的安全前置条件 — 例如从 Passenger(2) 切到 Mover(3)，需要人类确认还是遥控器确认？
3. 模式 4 的手臂 DDS 自定义关键帧是否第一阶段完全不碰？

---

## 二、横向能力 × SDK/Topic 覆盖

| 横向能力 | SDK 已封装 | Topic 层需自建 | 缺口 |
|---------|-----------|---------------|------|
| 系统感知 | 无 | 无（不走 DDS） | 无 — Linux 层自给 |
| G1 感知 | `ChannelSubscriber` + LowState_/SportModeState_ | BmsState_, MainBoardState_, IMUState_ (里程计/机身IMU) — 需自建订阅 | 类型路径待 PC2 import 验证 |
| 音频/灯光 | AudioClient (TTS/LED/音量/PlayStream) | `rt/audio_msg` (ASR) + UDP 组播 (麦克风) — 无封装 | ASR 被动可收，麦克风需裸 socket |
| 手臂控制 | G1ArmActionClient (17 预设) | `rt/arm_sdk` + `rt/dex3/*/cmd` — 有 example 无封装 class | 自定义关键帧需自建 |
| 高阶运动 | LocoClient RPC — 模式切换/Move/WaveHand | `rt/lowcmd` — 有完整 example 无封装 class | CRC 校验必须 |

**判断**: Topic 层缺口都是"有数据通路、无 convenience 封装"——不需要 SDK 新增能力。MOSS channel 子线程跑 DDS 回调 → 写 asyncio 队列 → command 读取。

讨论点:
1. 未封装的 topic 订阅（BmsState_ 等），是直接在每个需要它们的 channel 内裸写 `ChannelSubscriber`，还是先写一个统一的 G1 感知层 channel？
2. `docs/sdk-topics.md` 中标注的类型路径需要在 PC2 上 import 验证。这是 SDK 线脚本 02 的内容

---

## 三、文档 < 源码 差异

| 文档说 | 源码实 | 影响 |
|--------|--------|------|
| `Squat()` 存在 | 无。`Squat2StandUp()` = SetFsmId(706) | 命名不匹配，需在 sdk-api.md 标注 |
| `ContinuousGait(bool)` 存在 | 无。持续移动用 `Move(vx,vy,vyaw, True)` (duration=864000s) | 功能等效 |
| `GetFsmMode()`/`GetFsmId()` | API 已注册但无封装方法 | 需自行封装或从 SportModeState DDS 读 |
| `StandUp()` 存在 | 无。`Start()` = SetFsmId(500) | 命名映射 |
| 预设动作 15 种 | action_map 含 17 种 | x-ray, right heart, reject, left kiss 未在文档描述 |
| AudioClient 有 ASR | API ID 1002 已注册但无封装方法 | ASR 不可控（与文档吻合） |
| `PlayStream` 流式推送 | `_CallRequestWithParamAndBin` 支持二进制 | PCM 推送能力存在 |
| `LedControl` 间隔 >200ms | 无源码强制 | channel 层需做间隔限制 |
| TtsMaker index | `self.tts_index += self.tts_index` — 永远为 0 | Bug 但不影响功能 |

讨论点:
1. 差异合并入 `docs/sdk-api.md` 还是 `docs/index.md`？
2. 缺失封装（GetFsmId/GetFsmMode）MOSS 侧是否需要补充？

---

## 四、音频路径交叉判断

系统线步骤 6-8 的结果直接决定：

| 系统线结果 | → 模式 2 (Passenger) 音频方案 |
|-----------|---------------------------|
| PC2 有声卡 + 可播放 | 音频走 PC2 本地 ALSA/PulseAudio。AudioClient 仅用于 LED |
| PC2 有蓝牙 + 可连设备 | 同上，蓝牙方案 |
| 两者皆无 | 音频唯一路径: AudioClient RPC。需验证 TtsMaker 耗时、PlayStream 状态回调、PlayStop 取消 |
| PC2 无录音设备 | 语音输入: G1 四麦阵列 UDP 组播 (239.168.123.161:5555) |

讨论点:
1. 音频路径是否等系统线出结果再定 SDK 线脚本范围？
2. 如果 PC2 有声卡，AudioClient RPC 是否可以完全跳过不调研？

---

## 五、安全围栏设计

源码确认的安全事实：
1. LocoClient.SetFsmId() 可直接切模式到 Damp(1)/ZeroTorque(0)/Sit(3)/Start(500)
2. wireless_remote[40] 解析格式已确认 — MOSS 能读到 L2+B
3. LowCmd_ 需 crc (utils/crc.py) — 不在第一阶段使用
4. MotorCmd_.mode 逐电机使能 — 不在第一阶段使用

讨论点:
1. channel 层是否对 SetFsmId 做白名单？禁止模型直接调 ZeroTorque(0)？
2. L2+B 检测后 MOSS 的响应: 立即 Damp() 还是标记状态等模型处理？
3. 条件反射层（LiDAR 近场 → StopMove）— channel 内部还是独立进程？

---

## 六、下一步行动

| 行动 | 阻塞/并行 | 说明 |
|------|----------|------|
| 系统线执行 | B | 人类在 PC2 跑 RESEARCH_SEQUENCE.md 11 步 |
| `docs/sdk-api.md` 编写 | P | AI 整理源码摸底结果（素材已齐） |
| SDK 线脚本 | B→P | 依赖系统线确认 cyclonedds 环境 |
| Topic 类型路径验证 | B→P | 依赖 PC2 import — 并入 SDK 线脚本 02 |
| `docs/index.md` 差异更新 | P | 文档<源码差异表合并 |

讨论点:
1. 这次会话是否先把 `docs/sdk-api.md` 产出？还是等系统线反馈后再定范围？
2. 纵向模式 0-4 的定义是否写入 `docs/channel-design.md` 还是单独文档？

---

*提纲由 DeepSeek V4 Pro 在 2026-06-08 会话中准备，等待人类工程师输入。*
