---
arguments: ''
description: ''
executable: uv
respawn: false
script: main.py
workers: 1
---

Unitree G1 人形机器人身体控制通道。

## 能力

- **运动控制**: 全身关节运动、步态控制、平衡管理
- **手臂操作**: 5/7 自由度手臂运动规划与执行
- **音频交互**: 机器人端音频播放
- **状态感知**: 实时关节状态、IMU 数据、电池信息

## CTML 调用

启动后通过 Matrix 总线注册为 `apps.bodies_g1`，典型调用：

<apps:start fullname="bodies/g1" />
<bodies_g1:stand />
<bodies_g1:move_arm joint="right" x="0.5" y="0.0" z="0.3" />
<bodies_g1:say text="你好" />

## 依赖

- Unitree SDK2 Python (需手动 clone 到 src/，见 README.md)
- CycloneDDS (系统级依赖，当前仅 Linux 支持)
- 与 G1 机器人的网络连接 (DDS over Ethernet)