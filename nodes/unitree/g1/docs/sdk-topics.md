# G1 DDS Topic 清单

完整 topic 列表。每个 topic 标注 MOSS 横向能力归属和 IDL 类型。

来源: 官方文档 + 人类工程师确认。每条待源码验证类型路径。

验证状态: 2026-06-16 实机 topic 扫描确认以下 topic。前任硬编码清单中的类型错误已订正。

## Topic 总表

| Topic | 类型组 | 消息类型 | 方向 | MOSS 能力 | 验证 | 说明 |
|-------|--------|---------|------|----------|------|------|
| `rt/lowstate` | hg | LowState_ | 读 | G1 感知 | OK 06-16 | IMU + 电机(35槽,G1占0-28,29-34保留) + 遥控器 |
| `rt/lf/lowstate` | hg | LowState_ | 读 | G1 感知 | OK 06-16 | 同上低频版 |
| `rt/lowstate_doubleimu` | hg | LowState_ | 读 | G1 感知 | 存在 06-16 | 双 IMU 版 LowState |
| `rt/lowcmd` | hg | LowCmd_ | 写 | 高阶运动(DDS) | — | 关节级控制。调试模式。需 crc |
| `rt/arm_sdk` | hg | LowCmd_ | 写 | 手臂控制(DDS) | 存在 06-16 | 上肢+腰(12-28)。非调试运控模式可用 |
| `rt/loco_sdk` | hg | — | 写 | 运控(DDS) | 存在 06-16 | SDK 直通运控通道 |
| `rt/dex3/left/state` | hg | HandState_ | 读 | G1 感知 | 存在 06-16 | 左灵巧手反馈 |
| `rt/dex3/right/state` | hg | HandState_ | 读 | G1 感知 | 存在 06-16 | 右灵巧手反馈 |
| `rt/dex3/left/cmd` | hg | HandCmd_ | 写 | 手臂控制(DDS) | — | 控制左灵巧手 |
| `rt/dex3/right/cmd` | hg | HandCmd_ | 写 | 手臂控制(DDS) | — | 控制右灵巧手 |
| `rt/odommodestate` | go2 | **SportModeState_** | 读 | G1 感知 | OK 06-16 | **订正: 非 IMUState_**。里程计+身体高度+足力+速度 |
| `rt/lf/odommodestate` | go2 | SportModeState_ | 读 | G1 感知 | 存在 06-16 | 同上低频版 |
| `rt/sportmodestate` | go2 | SportModeState_ | 读 | G1 感知 | **OK 06-16** | **推翻前任"不发布"结论**。运动模式+步态 |
| `rt/lf/sportmodestate` | go2 | SportModeState_ | 读 | G1 感知 | 存在 06-16 | 同上低频版 |
| `rt/lf/bmsstate` | hg | BmsState_ | 读 | G1 感知 | OK 06-16 | 电池(SOC/电压/电流/温度/循环) |
| `rt/lf/battery_alarm` | hg | — | 读 | G1 感知 | 存在 06-16 | 电池告警 |
| `rt/lf/mainboardstate` | hg | MainBoardState_ | 读 | G1 感知 | OK 06-16 | 主板(温度/风扇/电压) |
| `rt/secondary_imu` | hg | IMUState_ | 读 | G1 感知 | 存在 06-16 | 机身 IMU |
| `rt/lf/secondary_imu` | hg | IMUState_ | 读 | G1 感知 | OK 06-16 | 同上低频版 |
| `rt/wirelesscontroller` | — | — | 读 | G1 感知 | 存在 06-16 | 遥控器原始数据 |
| `rt/arm/action/state` | — | — | 读 | 手臂控制 | 存在 06-16 | 手臂动作执行状态反馈 |
| `rt/audio_msg` | — | String_ (JSON) | 读 | 音频 | 存在 06-16 | ASR 识别结果 |
| `rt/audio_msg/filter` | — | String_ (JSON) | 读 | 音频 | 存在 06-16 | ASR 滤波后输出 |
| `rt/utlidar/cloud_livox_mid360` | — | PointCloud2_ | 读 | G1 感知 | 存在 06-16 | LiDAR 点云 |
| `rt/utlidar/imu_livox_mid360` | — | Imu_ | 读 | G1 感知 | 存在 06-16 | LiDAR IMU |
| `rt/utlidar/range_info` | — | — | 读 | G1 感知 | 存在 06-16 | LiDAR 测距信息 |

### RPC 服务 topic (api 请求/响应对)

| Service | Request Topic | Response Topic | 验证 |
|---------|--------------|----------------|------|
| robot_state | `rt/api/robot_state/request` | `rt/api/robot_state/response` | 服务存在，Python client 不可用 (b2 模块缺 rpc.client_internal) |
| sport | `rt/api/sport/request` | `rt/api/sport/response` | 存在 |
| arm | `rt/api/arm/request` | `rt/api/arm/response` | OK 06-16 (GetActionList 返回 23 项) |
| audiohub | `rt/api/audiohub/request` | `rt/api/audiohub/response` | 存在 |
| voice | `rt/api/voice/request` | `rt/api/voice/response` | 存在 |
| motion_switcher | `rt/api/motion_switcher/request` | `rt/api/motion_switcher/response` | OK 06-16 (CheckMode="ai") |
| loco | `rt/api/loco/request` | `rt/api/loco/response` | 存在 |
| config | `rt/api/config/request` | `rt/api/config/response` | 存在 |

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
