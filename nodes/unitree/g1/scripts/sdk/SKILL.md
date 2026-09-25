---
name: g1-sdk-verification
description: G1 SDK 验证脚本 — import 检查、RPC 反射、DDS 类型内省、topic 发现、交互验证。在 PC2 上本地执行，需要 cyclonedds 和 unitree_sdk2py 可 import。
---

# G1 SDK 验证

回答两个问题：
1. SDK 提供了什么 API，和文档是否一致？（00-03）
2. 这些 API 在实际 G1 上的行为是什么？（04-11）

## 技能索引

| # | 脚本 | 层 | 需要 G1 | 风险 | 说明 |
|---|------|---|--------|------|------|
| 00 | `import_verify.py` | 环境 | 否 | 零 | 所有 SDK 模块 import 检查 |
| 01 | `rpc_client_reflect.py` | 反射 | 否 | 零 | inspect RPC client API surface |
| 02 | `message_types.py` | 反射 | 否 | 零 | dir() DDS 消息类型字段 |
| 03 | `topic_discover.py` | 发现 | 是 | 零 | cyclonedds CLI wrapper — 真扫描 |
| 04 | `lowstate_sub.py` | A: 纯读 | 是 | 零 | LowState + 遥控器解析 (rt/lowstate 高频) |
| 05 | `sportmode_sub.py` | A: 纯读 | 是 | 零 | SportModeState 发布探测 (短超时) |
| 06 | `battery_sub.py` | A: 纯读 | 是 | 零 | bms/mainboard/imu/odom 多候选 topic 探测 |
| 07 | `rpc_readonly.py` | B: RPC 只读 | 是 | 零 | MotionSwitcher/Arm/Audio/RobotState (RS 可能 G1 不可用) |
| 08 | `audio_led.py` | C: 音频灯光 | 是 | 低 | 音量/LED/TTS + TTS 中断探路 |
| 09 | `arm_preset.py` | D: 上肢 | 是 | 中 | 挥手 + 中断复位 + 动作序列 |
| 10 | `loco_mode.py` | E: 模式切换 | 是 | 高 | Damp→Sit→Start + 安全确认 |
| 11 | `loco_move.py` | E: 移动 | 是 | 高 | 0.5s 极慢速前进/横移/旋转 |

## 执行顺序

```bash
# 第一轮: 无 G1 也可跑 (macOS/PC2)
python 00_import_verify.py
python 01_rpc_client_reflect.py
python 02_message_types.py

# 第二轮: G1 开机后，零风险
python 03_topic_discover.py <nic>
python 04_lowstate_sub.py <nic>
python 05_sportmode_sub.py <nic>
python 06_battery_sub.py <nic>
python 07_rpc_readonly.py <nic>

# 第三轮: 低风险交互
python 08_audio_led.py <nic>

# 第四轮: 逐层解锁 (需人类在场 + 清场)
python 09_arm_preset.py <nic>     # 坐姿下手动确认
python 10_loco_mode.py <nic>      # 模式切换安全确认
python 11_loco_move.py <nic>      # 移动 2m 清场 + 遥控急停
```

## 前置

```bash
cd .moss_ws/apps/bodies/g1
source .venv/bin/activate
python 00_import_verify.py  # 必须先通过
```

实机测试 2 小时窗口的逐步执行顺序见 `RUN_ORDER.md` (同目录)。
