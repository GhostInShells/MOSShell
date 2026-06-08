# G1 DDS Topic 清单

完整 topic 列表。每个 topic 标注 MOSS 横向能力归属和 IDL 类型。

来源: 官方文档 + 人类工程师确认。每条待源码验证类型路径。

## Topic 总表

| Topic | 类型组 | 消息类型 | 方向 | 频率 | MOSS 能力 | 说明 |
|-------|--------|---------|------|------|----------|------|
| `rt/lowstate` | hg | LowState_ | 读 | 高频 | G1 感知 | IMU + 电机状态(35) + 遥控器 + crc |
| `rt/lf/lowstate` | hg | LowState_ | 读 | 低频 | G1 感知 | 同上低频版，推荐 MOSS 订阅此以减少带宽 |
| `rt/lowcmd` | hg | LowCmd_ | 写 | — | 高阶运动(DDS) | 关节级控制。调试模式下可用。需 crc |
| `rt/arm_sdk` | hg | LowCmd_ | 写 | — | 手臂控制(DDS) | 上肢+腰(12-28) + weight(29)。非调试运控模式下可用 |
| `rt/dex3/left/state` | hg | HandState_ | 读 | 高频 | G1 感知 | 左灵巧手反馈 |
| `rt/lf/dex3/left/state` | hg | HandState_ | 读 | 低频 | G1 感知 | 同上低频版 |
| `rt/dex3/left/cmd` | hg | HandCmd_ | 写 | — | 手臂控制(DDS) | 控制左灵巧手 |
| `rt/dex3/right/state` | hg | HandState_ | 读 | 高频 | G1 感知 | 右灵巧手反馈 |
| `rt/lf/dex3/right/state` | hg | HandState_ | 读 | 低频 | G1 感知 | 同上低频版 |
| `rt/dex3/right/cmd` | hg | HandCmd_ | 写 | — | 手臂控制(DDS) | 控制右灵巧手 |
| `rt/odommodestate` | go2 | IMUState_ | 读 | 高频 | G1 感知 | 里程计(位置/姿态/速度) |
| `rt/lf/odommodestate` | go2 | IMUState_ | 读 | 低频 | G1 感知 | 同上低频版 |
| `rt/lf/bmsstate` | hg | BmsState_ | 读 | 低频 | G1 感知 | 电池状态(电压/电流/百分比/温度) |
| `rt/lf/mainboardstate` | hg | MainBoardState_ | 读 | 低频 | G1 感知 | 主板反馈信息 |
| `rt/secondary_imu` | hg | IMUState_ | 读 | 高频 | G1 感知 | 机身 IMU (独立于 LowState_) |
| `rt/lf/secondary_imu` | hg | IMUState_ | 读 | 低频 | G1 感知 | 同上低频版 |
| `rt/sportmodestate` | go2? | SportModeState_ | 读 | ? | G1 感知 | 运动模式状态(fsm_id + fsm_mode + task_id) |
| `rt/audio_msg` | ? | String_ (JSON) | 读 | 事件 | 音频 | ASR 识别结果 |
| `rt/utlidar/cloud_livox_mid360` | ? | PointCloud2_ | 读 | 10Hz | G1 感知 | LiDAR 点云 |
| `rt/utlidar/imu_livox_mid360` | ? | Imu_ | 读 | 200Hz | G1 感知 | LiDAR IMU |

## 类型组对照

| 类型组 | IDL 路径 | 适用机器人 |
|--------|---------|-----------|
| hg | `unitree_hg.msg.dds_` | G1, H1-2 |
| go2 | `unitree_go.msg.dds_` | Go2, B2, H1 (部分与 G1 共享) |

## MOSS 能力 × Topic 映射

| MOSS 横向能力 | 订阅 Topic | 发布 Topic | 说明 |
|---------------|-----------|-----------|------|
| 系统感知 | — | — | 不经过 DDS，Linux 系统层 |
| G1 感知 | `rt/lf/lowstate`, `rt/lf/bmsstate`, `rt/lf/odommodestate`, `rt/lf/mainboardstate`, `rt/lf/secondary_imu`, `rt/sportmodestate` | — | 全部只读，优先低频 |
| 音频/灯光 | `rt/audio_msg`(ASR) | — | 播放走 AudioClient RPC |
| 手臂控制(RPC) | — | — | 走 G1ArmActionClient RPC |
| 手臂控制(DDS) | — | `rt/arm_sdk`, `rt/dex3/*/cmd` | 调试模式或运控模式下可用 |
| 高阶运动(RPC) | — | — | 走 LocoClient RPC |
| 高阶运动(DDS) | — | `rt/lowcmd` | 调试模式。高阶阶段 |

## 类型验证清单

以下 IDL 类型路径需在 PC2 上 import 验证：

- [ ] `unitree_hg.msg.dds_.LowState_`
- [ ] `unitree_hg.msg.dds_.LowCmd_`
- [ ] `unitree_hg.msg.dds_.HandState_`
- [ ] `unitree_hg.msg.dds_.HandCmd_`
- [ ] `unitree_hg.msg.dds_.BmsState_`
- [ ] `unitree_hg.msg.dds_.MainBoardState_`
- [ ] `unitree_hg.msg.dds_.IMUState_`
- [ ] `unitree_go.msg.dds_.IMUState_`
- [ ] `unitree_go.msg.dds_.SportModeState_`

---

*编写: DeepSeek V4 Pro, 2026-06-08。来源: 人类工程师提供的 Topic 清单 + SDK 源码摸底。*
