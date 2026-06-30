# Runtime Verification Checklist

2026-06-30 实机验证索引。每个条目的验证结论进入对应脚本 docstring 或模块 docstring，本文件做进度索引。

**快速跳转**: 失败 → [已知风险](#已知风险与fallback) | SOP → [README.md 实机验证流程](./README.md) Step 1-6

## 前置条件

- [ ] PC2 SSH 可达，cd 到仓库根目录
- [ ] `.venv` 就绪 (`uv sync`)
- [ ] `UNITREE_G1_SDK_PATH` 环境变量已设
- [ ] `scripts/sdk/00_import_verify.py` 通过 (sdk 链路 OK)

---

## Phase 0: Import 探伤 (PC2, 无需 G1 开机)

**目标**: 确认全部 runtime 模块 + 验证脚本可在 PC2 上 `import` 通过。SDK 顶层 import 会触发 `unitree_sdk2py` 加载，这是最快的失败信号。

| # | 条目 | 命令 | P | 结果 |
|---|------|------|---|------|
| 0.1 | sdk 链路 | `python -c "from ghoshell_moss_contrib.unitree.g1.sdk import bootstrap; print('OK')"` | P0 | PASS (macOS) |
| 0.2 | 全部 runtime 模块 import | `python -c "from ghoshell_moss_contrib.unitree.g1.runtime import asr, listener, control_pad, motion, imu, arm_joints, system_info, locomotion, led, audio, audio_player, audio_provider; print('OK')"` | P0 | PASS (macOS) |
| 0.3 | 全部 tes/sen 脚本 import | 逐脚本 `python -m ghoshell_moss_contrib.unitree.g1.runtime._control_pad_tes_001...` | P0 | 见 Phase 1 (6/6 PASS)

**失败即阻塞** — Phase 0 不通则后续全部无法进行。先修 import。

---

## Phase 1: TES 单测 (PC2, 无需 G1 开机)

**目标**: 验证每个模块的纯逻辑契约 (start/stop 幂等、ring buffer 语义、listener 隔离、debounce 等)。不依赖 DDS/RPC 真实数据源。

### control_pad (最完善, 6 个 tes)

| # | 脚本 | 验证什么 | P | 结果 |
|---|------|----------|---|------|
| 1.1 | `_control_pad_tes_001_register_unregister.py` | listener 注册/注销 + handle 去重 | P0 | PASS |
| 1.2 | `_control_pad_tes_002_exact_match.py` | binding 精确匹配 + 不匹配不触发 | P0 | PASS (修: 跨 case debounce 残留加 sleep) |
| 1.3 | `_control_pad_tes_003_debounce_per_binding.py` | per-binding debounce 独立计时 | P0 | PASS |
| 1.4 | `_control_pad_tes_004_fallthrough_global_debounce.py` | fallthrough + global debounce | P0 | PASS |
| 1.5 | `_control_pad_tes_005_listener_exception_isolation.py` | listener 异常不炸 reader 线程 | P0 | PASS (修: `c` → `start` 非法按键名) |
| 1.6 | `_control_pad_tes_006_ringbuffer_forgotten_count.py` | ring buffer forgotten 计数正确 | P0 | PASS |

### 其他模块 (待补单测)

| # | 模块 | 应验证 | P | 结果 |
|---|------|--------|---|------|
| 1.7 | asr | start/stop 幂等 (无 DDS 连接时 graceful degrade) | P1 | |
| 1.8 | listener | start/stop 幂等 + config 解析 | P1 | |
| 1.9 | motion | start/stop 幂等 + ring buffer forgotten | P1 | |
| 1.10 | imu | start/stop 幂等 + ring buffer forgotten | P1 | |
| 1.11 | arm_joints | start/stop 幂等 + ring buffer forgotten | P1 | |
| 1.12 | system_info | stateless query 不抛异常 | P1 | |
| 1.13 | locomotion | start/stop_runtime 幂等 + version 递增 | P1 | |
| 1.14 | led | start/stop 幂等 + 轨道优先级覆盖契约 | P1 | |
| 1.15 | audio | start/stop 幂等 + volume 范围 check | P1 | |

**注意**: P1 条目多数没有现成 tes 脚本。最小验证: `python -c "from module import start, stop; start(); assert is_running(); stop()"` — 一行命令即可。

---

## Phase 2: SEN 上行感知 (PC2 + G1 开机, 低风险)

**目标**: 验证 G1 真实数据流进入 runtime 模块后，drain/peek_latest/listener/health 等接口行为正确。纯被动读取，不控制 G1 物理动作。

### 2.1 system_info (stateless, 最安全)

| # | 脚本 | 验证什么 | P | 结果 |
|---|------|----------|---|------|
| 2.1 | `_system_info_sen_read.py` | 电池/主板状态 stateless read，无 daemon 无 ring buffer | P0 | PASS |

<details>
<summary>实测样本 (2026-07-01, G1 开机 Sport 模式)</summary>

```json
{"battery_soc":86,"battery_soh":99,"battery_voltage":52.46,"battery_current":-2.42,
 "battery_temperature_max":32,"battery_cycle":8,"board_temp":50,"fan_running":false}
```
- **修复**: `_monitor.py` 字段名对齐 IDL (5 处) + bmsvoltage 取 max + current mA→A。
</details>

### 2.2 motion (FSM 快照)

| # | 脚本 | 验证什么 | P | 结果 |
|---|------|----------|---|------|
| 2.2 | `_motion_sen_fsm_transitions.py` | FSM 模式快照 + 切换事件轨迹 (2Hz RPC) | P0 | PASS |

<details>
<summary>实测: WalkRun(801) → Damp(1) transition 检测正确. 数据源 RPC 7001.</summary>

- **已知权衡**: 2Hz 轮询 = 最坏 500ms 延迟。对 Damp 安全（G1 不吃 Move），对 Sport→启用 move channel 有感知滞后。等 channel 状态机设计敲定后决定是否调频率或换检测路径 (sportmodestate DDS 固件升级后可能可用)。
</details>

### 2.3 imu

| # | 脚本 | 验证什么 | P | 结果 |
|---|------|----------|---|------|
| 2.3 | `_imu_sen_listen_and_drain.py` | rpy/gyro/accel 2Hz 采样 + drain | P0 | PASS |
| 2.4 | `_arm_joints_sen_listen_and_drain.py` | 双臂 10 关节 rad 快照 + 2Hz 采样 | P0 | PASS |
| 2.5 | `_control_pad_sen_listen_and_drain.py` | 遥控器 16 键 + 4 轴实时透传 | P0 | PASS |

**前置**: G1 在运动模式 Sport。需要人按遥控器按键。

### 2.6 asr

| # | 脚本 | 验证什么 | P | 结果 |
|---|------|----------|---|------|
| 2.6 | `_asr_sen_listen_and_drain.py` | G1 内置远场 ASR 整句 VAD | P0 | PASS |

<details>
<summary>实测: drain 正常, angle/spkr_id 始终 0 (固件不启用). emotion/language 字段有效.</summary>
</details>

**前置**: 手机 App 开启唤醒对话模式 (麦克风开启)。需要人对 G1 说话。

### 2.7 listener

| # | 脚本 | 验证什么 | P | 结果 |
|---|------|----------|---|------|
| 2.7 | `_listener_sen_setup.py` | 蓝牙耳机首次配置 (生成 asr config) | P0 | PASS |

<details>
<summary>实测: Shokz OpenRun Pro 骨传导, HFP profile(headset_head_unit), 16000Hz 1ch, voiced 32.4%.</summary>
</details>
| 2.8 | `_listener_sen_dialog.py` | 端到端流式 ASR dialog | P0 | BLOCKED (火山 key 过期, ws 403. 通路确认 OK) |

**前置**: 2.7 通过 + 火山引擎 ASR key 有效。

---

## Phase 3: SEN 动作执行 (PC2 + G1 开机 + 物理动作)

**目标**: 验证 runtime 模块发出 SDK 命令后 G1 物理响应正确。涉及 G1 躯体运动，需要安全确认。

### 3.1 locomotion (最高风险)

| # | 脚本 | 验证什么 | P | 结果 |
|---|------|----------|---|------|
| 3.1 | `_locomotion_sen_basic.py` | 前后/横移/转身 + version 互斥 + Observe reason | P0 | f/b/s 通过; l/r 0.15 不动 (阈值 bug, 已修 V_LATERAL→0.25); 抢占待明天 `_locomotion_tes_preempt.py` |

**物理事实**: 0.15 m/s 低于横移启动阈值, V_LATERAL 修正为 0.25. 前进/后退 0.25 m/s 正常. 转身 V_YAW 低/中/高 均工作正常. Duration 精确, version 递增正确.

### 3.2 led

| # | 脚本 | 验证什么 | P | 结果 |
|---|------|----------|---|------|
| 3.2 | `_led_sen_factory_showcase.py` | 三轨道优先级 + 4 easing + 20Hz daemon | P1 | PASS |

<details>
<summary>实测: solid/blink/breath/rainbow/pulse/off 全部正常. 物理事实: 20fps 肉眼可见帧间, 头壳蓝底绿色不显. event blink red 抢占 bg breath blue 后正确恢复. LED driver sdk_call_count=3420 无异常. 退出后残留蓝色 (没在 stop 前 off).</summary>
</details>

### 3.3 audio

| # | 脚本 | 验证什么 | P | 结果 |
|---|------|----------|---|------|
| 3.3 | `_audio_sen_speak_and_play.py` | TTS + PCM 流播放 + 音量控制 | P1 | PASS |

<details>
<summary>实测: TTS 中文女声(speaker_id=0)可打断, 英文(speaker_id=1)同样可打断. :tone PCM 流播放/cancel 正常. vol 0-100 有效.</summary>
</details>

---

## Phase 4: 辅助工具验证

| # | 条目 | 验证什么 | P | 结果 |
|---|------|----------|---|------|
| 4.1 | `_headphone_buttons_probe.py` | 蓝牙耳机按键事件发现 (为后续 headphone_buttons 模块准备数据) | P2 | |
| 4.2 | `audio_player.py` + `audio_provider.py` | StreamAudioPlayer 与 speech pipeline 集成 | P2 | |

---

## Phase 5: README 待实测清单 (标定/优化类, 本期可选)

这些是 README.md "待实测/待回填" 节的内容，不阻塞 runtime 模块验证通过，但影响后续 channel 层使用质量。详见 [README.md](./README.md) 对应条目。

| # | 条目 | P | 状态 |
|---|------|---|------|
| 5.1 | asr: DDS sub Close 后能否 Init 重启 | P2 | |
| 5.2 | asr: filter topic `rt/audio_msg/filter` 是否并入 | P2 | |
| 5.3 | asr: `angle` 字段正值方向标定 | P2 | 实机: angle 始终 0 (固件不启用声源定位). |
| 5.4 | listener: 蓝牙 HFP 实际采样率 | P2 | 实机: Shokz OpenRun Pro 16000Hz 1ch. |
| 5.5 | listener: 蓝牙断连→重连→capture 自动恢复 | P2 | |
| 5.6 | listener: `drain(force_finalize_partial=True)` 后 session 干净度 | P2 | |
| 5.7 | motion: Dance/Debug 等模式 FSM ID 实测确认 | P2 | 官方 ID 表已对齐: 0=ZeroTorque,1=Damp,3=Sit,4=Stand,500=Regular,801=WalkRun. |
| 5.8 | imu: roll/pitch 零位 + 正方向坐标系标定 | P2 | |
| 5.9 | imu: yaw 漂移速率实测 | P2 | |
| 5.10 | imu: 静止折叠阈值合理性 | P2 | |
| 5.11 | arm_joints: 10 关节 rad 零位 + 正方向标定 | P2 | |
| 5.12 | arm_joints: `_HISTORY_DELTA_THRESHOLD` 合理性 | P2 | |
| 5.13 | system_info: 温度阈值实机正常范围 | P2 | 实机: SOC=86% 板温=50°C (Sport 模式). |
| 5.14 | audio: TtsMaker 是否可被 PlayStop 中断 | P2 | 实机: **可打断** (连续 speak 会抢占). |
| 5.15 | audio: speaker_id 音色标定 | P2 | 实机: 0=中文女声, 1=英文. 仅两个值. |
| 5.16 | audio: Volume 有效区间标定 (0-9 vs 0-100) | P2 | 实机: 0-100 (文档确认). |
| 5.17 | audio: TTS 时长估算公式标定 | P2 | |
| 5.18 | locomotion: V_YAW low/medium/high 实测角速度 | P1 | |
| 5.19 | locomotion: Move keepalive 重发是否必要 | P1 | |
| 5.20 | locomotion: 抢占切换平滑度 | P1 | 待明天 `_locomotion_tes_preempt.py`. |
| 5.21 | locomotion: 强停后 G1 物理状态 | P1 | 实机: stop → 立即 Stand idle, 无抽搐. |
| 5.22 | led: LedControl 调用率上限 (30/50 fps) | P2 | |
| 5.23 | led: HSV easing vs LINEAR 视觉对比 | P2 | |
| 5.24 | led: breath 默认参数自然度 | P2 | |
| 5.25 | led: SDK 抽离后 driver 线程改 sdk 新路径 | P2 | |

---

## 已知风险与 Fallback

| 风险 | 影响 | 状态 | Fallback |
|------|------|------|----------|
| `rt/sportmodestate` Read 永久阻塞 | motion 模块无法获取 FSM 状态 | **已绕过** | 改用 `LocoClient._Call(7001)` RPC (GetFsmId). 官方 ID 表对齐. |
| odommodestate mode 字段不变 | 无法区分 Damp/Sport | **已绕过** | RPC 7001 正确返回 0=Damp, 1=Stand, 801=WalkRun 等 |
| G1 横移 0.15 m/s 不动 | l/r 命令无效 | **已修复** | V_LATERAL 改为 0.25 m/s. 0.5 太猛. |
| 火山引擎 ASR key 过期 | listener dialog 无法验证 | **阻塞** | 等 key 续期. ws 通路已确认 OK. |
| 蓝牙 HFP profile 每次重启回退 | 重连后 mic 不可用 | **有方案** | 脚本 `connect_open_run.sh` 一键重连+切 profile. |
| asr angle/speaker_id 固件不启用 | 无法做声源定位/说话人识别 | **记录** | 暂不依赖. 影响 story-2026-07 roll_toward_speaker 功能. |

---

