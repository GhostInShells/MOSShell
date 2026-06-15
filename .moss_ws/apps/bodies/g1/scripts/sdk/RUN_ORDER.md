# RUN_ORDER — 实机测试 2 小时窗口执行顺序

生成时间: 2026-06-15
上一次实机 session: 2026-06-14/15 (DDS 链路打通 + 端到端音频输出)

## 本轮窗口目标 (按价值排序)

1. **订正 topic 真值清单** — 前任 docs/sdk-topics.md 部分项未实测，本轮 03+06 给出真实命中名
2. **确认 G1 RPC 服务存在性** — RobotStateClient 是否在 G1 bus 上 (前任结论模糊)
3. **SportModeState 发布探测** — 推翻或确认"G1 不发"的前任结论
4. **急停延迟实测** — 改用 rt/lowstate (高频) 后的真实 L2+B 感知时间
5. **TTS/LED/PlayStream 兜底确认** — 上次已通，本次确认无回归即可

## 改动一览 (本轮 session)

| 脚本 | 关键变更 |
|------|---------|
| 03 | 硬编码清单 → cyclonedds CLI wrapper (真扫描) |
| 04 | rt/lf/lowstate → rt/lowstate (canonical) + MOTOR_NAMES 注释订正 |
| 05 | 50s 阻塞订阅 → 3 次 2s 短超时探测 |
| 06 | odom 类型错 (IMUState_ → SportModeState_) + 多 topic 候选探测 |
| 07 | RobotStateClient 顺序后置 + 3s 超时 + 独立 try (前任 5s 阻塞被误诊 import 失败) |
| 08 | GetVolume 返回 dict → 用 `_vol_value()` 解包 (前任直接 dict 当 int 传) |
| 12 | rt/lf/lowstate → rt/lowstate (急停延迟测量必须用高频) |
| 09/10/11/13/14 | 未改 — review 后未发现致命路径错 |

## 执行顺序

**前置 (1 分钟)**
```bash
cd .moss_ws/apps/bodies/g1
source .venv/bin/activate
python scripts/sdk/00_import_verify.py   # 全 OK 才往下
```

**阶段一: 无副作用探测 (10-15 分钟)**
```bash
# 真实 topic 清单
python scripts/sdk/03_topic_discover.py rt/    # 过滤 rt/ 前缀

# 被动订阅 — 验证类型 + topic 候选
python scripts/sdk/04_lowstate_sub.py eth0     # 20 帧自动停
python scripts/sdk/05_sportmode_sub.py eth0    # 3 × 2s 探测, ~10s 结束
python scripts/sdk/06_battery_sub.py eth0      # 4 类 × 2-3 候选, ~30s 结束

# RPC 只读
python scripts/sdk/07_rpc_readonly.py eth0     # 4 项, RobotState 放最后
```

**阶段二: 音频灯光 (5-10 分钟)**
```bash
# 注意: G1 内置 TTS 质量不可用 (前任 6/15 结论)
# 本步骤仅验证 RPC 接口可达 + PlayStream 通路
python scripts/sdk/08_audio_led.py eth0
```

**阶段三: 上肢 + 第一个 channel 案例 (20-30 分钟，需 G1 落座)**

人类确认 G1 已 Sit 模式 + 手臂周围空：
```bash
python scripts/sdk/09_arm_preset.py eth0       # 基础动作 + 中断复位 + 序列
python scripts/sdk/15_channel_action.py eth0   # ← 第一个 channel + SDK 链路案例
```

`15_channel_action.py` 是本轮关键产出：通过 `chan.bootstrap()` 让 g1_channel 独立
运行，按 test_py_channel 范式直接 `runtime.execute_command(...)`，验证
channel layer 的 CTML 入口契约 + SDK 调用链路全通。完成后可进入 Matrix.provide_channel
接 MCP 阶段。

**阶段四: 全身运动 (剩余时间)**

人类确认场地空旷 + 遥控器在手：
```bash
python scripts/sdk/10_loco_mode.py eth0        # Damp → Sit → Start
python scripts/sdk/11_loco_move.py eth0        # 极慢速移动 4 项
python scripts/sdk/12_estop_verify.py eth0     # 踏步中 L2+B 急停延迟
```

## 每个脚本的"该看什么"

| 脚本 | 看什么 |
|------|-------|
| 03 | 真实 topic 清单。重点: `rt/lowstate` 是否独立存在? `rt/sportmodestate` 是否有? |
| 04 | 遥控器解析是否对? motor_state 29-34 是否都 0? imu.rpy 合理性? |
| 05 | "G1 不发布" 是否成立? 收到任何一帧 → 推翻前任结论 |
| 06 | 4 类各命中哪个候选 topic? 全 FAIL 的项需 03 的真清单订正 |
| 07 | RobotState 是 FAIL 还是 OK? 是 FAIL 的话从可用 RPC 清单删掉 |
| 08 | GetVolume 解析后是否 = int? LED 收尾是否变黑 (不留白色)? |
| 12 | t_detect 延迟 < 100ms 算优秀 / 100-300ms 可接受 / >300ms 需排查 |

## 失败兜底

- **DDS 通讯不通**: ufw IP 分片问题已修 (前任 6/15)。重连后跑 04 看是否能收数据
- **某脚本 hang**: Ctrl+C，跳过，记录到 FEATURE.md "已知问题"
- **cyclonedds CLI 不存在**: 03 会指引 source /etc/profile.d/cyclonedds.sh

## 时间余量分配

理想 2 小时:
- 0:00-0:15 阶段一 (无副作用)
- 0:15-0:30 阶段二 (音频)
- 0:30-1:00 阶段三 (上肢)
- 1:00-1:45 阶段四 (运动 + 急停)
- 1:45-2:00 结论梳理 + FEATURE.md 追加 session log

如果阶段一/二快: 把多出来的时间给运动 + 急停的多次验证 (一次单点不可靠)。