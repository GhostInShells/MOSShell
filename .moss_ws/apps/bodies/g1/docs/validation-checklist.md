# G1 验证清单

阶段 A 文档摸底产出。每个命题来自文档，待源码分析 + 实测验证。

**验证启动前置条件**: 执行任何验证脚本前，需与人类开发者重新对齐验证目标。文档结论可能因固件版本/硬件配置变化而失效。

**验证顺序**: 源码先验证 API 存在性和签名(阶段 B)，实测再验证运行时行为(阶段 E)。

## 控制路径

| # | 命题 | 来源文档 | 验证方法 |
|---|------|---------|---------|
| C1 | LocoClient RPC 和 rt/lowstate DDS 订阅可同时工作 | sport_services + basic_services | 源码: 确认无互斥锁。实测: 同时调 Move() 和订阅 rt/lowstate |
| C2 | rt/arm_sdk 控制手臂时 LocoClient 可控制下肢 | arm_control_routine | 实测: 锁定站立模式，同时发 arm_sdk 和 Move() |
| C3 | ReleaseMode() 是否需要先进入阻尼/零力矩 | motion_switcher + remote_control | 源码: 查 ReleaseMode 实现。实测: 运动模式下直接调 ReleaseMode() |
| C4 | SelectMode("ai") 从调试模式恢复到运控的延迟 | motion_switcher | 实测: ReleaseMode → SelectMode("ai")，计时 |
| C5 | `g1_arm_example` 关闭后 rt/arm_sdk 是否立即可用 | arm_action + robot_state | 实测: ServiceSwitch 关闭后立即发 arm_sdk 指令 |

## 状态感知

| # | 命题 | 来源文档 | 验证方法 |
|---|------|---------|---------|
| S1 | wireless_remote[40] 的字节格式和 L2+B 编码 | basic_services | 源码: 查 IDL 或解析例程。实测: 按键对照 |
| S2 | SportModeState.fsm_mode 从 1 切回 0 的典型延迟 | sport_services | 实测: 统计运动→静态切换时间 |
| S3 | rt/lowstate.tick 的精度是否稳定 1ms | basic_services | 实测: 订阅 1000 tick，统计间隔分布 |

## 音频

| # | 命题 | 来源文档 | 验证方法 |
|---|------|---------|---------|
| A1 | play_state DDS 通知是否可靠标识播放完成（0=停止） | VuiClient_Service | 实测: PlayStream 后等 play_state 变化 |
| A2 | PlayStop() 是否能中断正在播放的 TTS | VuiClient_Service | 实测: TtsMaker 长文本，中途 PlayStop() |
| A3 | PlayStream 同 stream_id 续播：是否队列追加还是覆盖 | VuiClient_Service | 实测: 两次 PlayStream 同 stream_id，观察行为 |
| A4 | PC2 蓝牙是否可用（Jetson Orin NX 硬件确认） | about_G1 | 硬测: PC2 上 scan 蓝牙设备，连耳机 |

## LiDAR 与感知

| # | 命题 | 来源文档 | 验证方法 |
|---|------|---------|---------|
| L1 | rt/utlidar/cloud_livox_mid360 DDS vs Livox SDK2 直连的延迟差异 | lidar_services + lidar_Instructions | 实测: 两种路径同时收点云，对比时间戳 |
| L2 | LiDAR 噪点 tag 过滤后有效点云密度是否满足条件反射需求 | lidar_Instructions | 实测: 统计 10Hz x N 帧的有效点比例 |

## 关节与安全

| # | 命题 | 来源文档 | 验证方法 |
|---|------|---------|---------|
| J1 | 文档中的关节限位表与实际 LowState_ 反馈是否一致 | about_G1 + basic_services | 实测: 读取 motor_state[].q，对比限位范围 |
| J2 | LowCmd_.crc 校验失败时的行为（静默丢弃 vs 错误返回） | basic_services | 源码: 查 crc 处理逻辑 |

---

**已验证项** (待更新):

| # | 结论 | 验证人 | 日期 |
|---|------|--------|------|
| — | — | — | — |
