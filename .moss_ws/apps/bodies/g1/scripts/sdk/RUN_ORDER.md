# RUN_ORDER — 实机测试 2 小时窗口执行顺序

生成时间: 2026-06-15
**最后更新: 2026-06-29** (追加 P2 全套脚本 23-27)
上一次实机 session: 2026-06-16 (PlayStream 流式通路确认)

---

## 2026-06-29 追加: P2 — 设计补完用实验

P2 不阻塞 channel 体系第一波实现, 但每条都对应一个未来 channel/能力的可行性.

| # | 脚本 | 命题 | 决定 |
|---|------|------|------|
| 23 | `23_asr_api_probe.py` | _Call(1002,...) 调用约定盲探 + ASR DDS topic 扫描 | asr sensor channel 的实现路径 |
| 24 | `24_mode_switch_topology.py` | FSM 模式完整可达图 + 各边时长 | state DAG 具体边的 SDK 调用选择 |
| 25 | `25_recording_capability_probe.py` | G1 内置录制是否暴露(topic + 协作探测 + RPC 试探) | recording channel 是接 SDK 还是自造 |
| 26 | `26_arm_sdk_dds_joints_write.py` | rt/arm_sdk 底层写关节角是否可用 + 与 Sport 共存 | 自造手臂轨迹动画的基础可行性 |
| 27 | `27_lowstate_sample_rate.py` | LowState 真实频率 + monitor 处理上限 | state.py 监控参数 + sensors 采样配置 |

---

## 2026-06-29: P1 — 状态切换与 action 语义补完

| # | 脚本 | 命题 | 决定 |
|---|------|------|------|
| 20 | `20_sit_stand_cycle.py` | Sit→Stand SDK 可达性, 706 是否双向, 时长, 是否需要 Start 才进 Sport | 用户故事幕三"自己站起来"是否可行, state DAG 形态 |
| 21 | `21_arm_action_interruption.py` | Action A 播放中发 Action B(非 99) — 覆盖/排队/拒绝 | arm channel 是否需要 "in_progress" 跟踪, CTML 时序对齐 |
| 22 | `22_arm_action_state_probe.py` | rt/arm/action/state topic 内容(in_progress / done?) | arm 命令的 await 实现 — topic 等待 vs 关节速度趋零 vs 预估 sleep |

---

## 2026-06-28: Channel 体系地基验证 (P0, 阻塞)

本轮 channel 体系全套设计已落 `design/2026-06-28_channel_architecture.md`.
该设计的地基由三个脚本验证 — **任何一条不通过, 设计要回炉**.

| # | 脚本 | 命题 | 决定 |
|---|------|------|------|
| 17 | `17_remote_keys_passthrough.py` | 调试/AI 模式下非 L2+B 按键 + 摇杆是否仍透传 wireless_remote | "遥控器=MOSS 输入设备"方案能否成立 |
| 18 | `18_arm_release_behavior.py` | ExecuteAction(99) 物理行为(缓慢复位/突变/脱控) | arm 类是否可进 warrant 事务 |
| 19 | `19_loco_stopmove_under_motion.py` | move 中 SetVelocity(0,0,0) 是否站定不仆 | move 类 warrant fallback 是否可信 |

---

## 推荐执行顺序 (本次窗口)

按风险递增 + 依赖关系排. P0 必跑, P1 补完用户故事, P2 解锁未来 channel.

| 优先级 | 阶段 | 时长估 | 脚本 | 风险 | 备注 |
|--------|------|--------|------|------|------|
| | 0 | 5 min | 04 / 05 基线确认 | 无 | 跳过如果今天 G1 已确认状态正常 |
| **P0** | 1 | 30 min | **17** remote_keys | 无 | 纯按键观察 |
| **P0** | 2 | 20 min | **19** loco_stopmove | 中 | 慢速移动 |
| **P0** | 3 | 20 min | **18** arm_release | 高 | 周围 1m 无物 |
| **P1** | 4 | 15 min | **22** arm_state_probe | 中 | 探测 topic + face wave |
| **P1** | 5 | 20 min | **20** sit_stand_cycle | 中 | 前后 2m 缓冲 |
| **P1** | 6 | 20 min | **21** arm_interruption | 高 | Action 组合 |
| **P2** | 7 | 30 min | **23** asr_api_probe | 无 | 盲探 _Call(1002), 你说几句话 |
| **P2** | 8 | 25 min | **24** mode_switch_topology | 中 | 多次 FSM 切换 |
| **P2** | 9 | 30 min | **25** recording_probe | 低 | 协作探测, 手机配合 |
| **P2** | 10 | 25 min | **26** arm_sdk_dds_write | 高 | 底层 DDS 写关节, kp 偏硬 |
| **P2** | 11 | 5 min | **27** lowstate_rate | 无 | 测频率, 后台跑 |

P0 必跑 (1.5h). P1 强烈推荐 (1h). P2 选做 (~2h, 全跑超过单窗口).

**建议分两次实机**:
- 第一次窗口: 04/05 + P0(17,19,18) + P1(22,20,21) ≈ 2h
- 第二次窗口: P2(23,24,25,26,27) ≈ 2h

```bash
cd .moss_ws/apps/bodies/g1
source .venv/bin/activate

# 第一次窗口 (P0 + P1)
python scripts/sdk/17_remote_keys_passthrough.py eth0
python scripts/sdk/19_loco_stopmove_under_motion.py eth0
python scripts/sdk/18_arm_release_behavior.py eth0
python scripts/sdk/22_arm_action_state_probe.py eth0
python scripts/sdk/20_sit_stand_cycle.py eth0
python scripts/sdk/21_arm_action_interruption.py eth0

# 第二次窗口 (P2)
python scripts/sdk/23_asr_api_probe.py eth0
python scripts/sdk/24_mode_switch_topology.py eth0
python scripts/sdk/25_recording_capability_probe.py eth0
python scripts/sdk/26_arm_sdk_dds_joints_write.py eth0
python scripts/sdk/27_lowstate_sample_rate.py eth0
```

**反馈给模型实例 (依次)**:
- 17 → 按键 → warrant scope / state DAG 边的分配
- 18 → arm 是否进 warrant
- 19 → move warrant fallback 是否需要减速曲线
- 20 → state DAG 完整边定义 + 用户故事时序
- 21 → arm command 并发语义
- 22 → arm 命令 await 的实现路径
- 23 → asr sensor channel 实现
- 24 → state DAG 全图
- 25 → recording channel 路径
- 26 → arm_trajectory channel 可行性
- 27 → state.py monitor 参数

---

## 历史: 2026-06-15/16 windowed work (已完成, 仅供参考)

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