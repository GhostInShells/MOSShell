# G1 Channel 体系架构

创建: 2026-06-28
作者: Claude Opus 4.7 + 人类工程师

## 背景

G1 集成进入阶段 G (Channel 设计). 第一版 channel.py 是 SDK 命令的薄包装,
被人类工程师评为"乱成麻花, 实机不可用". 本设计重做整个体系, 以"机制优先于实现"
为主轴: 先把 bootstrap / 授权事务 / 感知机制 / 状态管理 等机制层敲定,
具体 channel 实现退化为"写几个 SDK 调用 + 声明 scope".

## 范式决策

### 1. 单一控制源 + 遥控器作为 MOSS 输入

传统形态: 遥控器 → G1 主板 → 机体; SDK → G1 主板 → 机体. 两源争抢, 靠
"调试模式互斥" + L2+B 硬件急停兜底.

本设计: 调试/AI 模式下, G1 主板对遥控器只保留 L2+B 监控, 其他按键/摇杆字节
仍透传到 LowState.wireless_remote[40] 但不驱动机体. **MOSS 把这堆空闲字节
捡起来当作自己的人机协作输入设备.**

- 物理控制源唯一: MOSS channel → SDK → 机体
- 遥控器输入语义全部由 MOSS 自定义: 双击 X = 切 channel state, 长按 Y = 给 ghost 发 signal, 等等
- L2+B 硬件级 Damp 仍然生效, 同时 MOSS 也读 wireless_remote 看到 L2+B → 同步做软清理(取消队列 + 通知 ghost). 两层冗余

**前置假设**(P0 实机验证项): 调试/AI 模式下, 非 L2+B 的所有按键 bit + 摇杆轴
确实仍透传到 wireless_remote[40]. 如果不成立, 整个遥控器输入通道方案要重做.

### 2. 感知统一进 context_messages, 显式 pop 进 memory

所有感知 channel(轨迹 / 里程计 / ASR / 关节 / 被动 action / 视觉) 的输出
默认只进 context_messages — 过期即忘, 不占 memory.

每个感知 channel 提供 `pop()` 命令: 模型调用 → 当前快照作为 command result
返回 → 进 memory. 这把"什么值得记住"的决策权交给模型.

context_messages 滑动窗口大小由模型通过 channel 命令配置. 不同感知有不同 window_size.

### 3. 控制类用 state DAG 管理, 不用 StatefulChannel

state 切换是单向 DAG, 不是来回 FSM. 进 mobile_low 后想回 seated 需要人类显式
操作(遥控器按键), 不是模型自决.

不依赖 `StatefulChannel`. 用两个机制组合:
- `command.available()` 函数动态返回 True/False, 实现命令级闸门
- state 变更 → 重新计算 channel 的 virtual_children() 映射 → 暴露面变化

State DAG 节点示例(对应用户故事四幕):
- `seated` — 默认. 感知全开, 控制只有 audio/led/volume/music
- `seated_arms` — 叠加: arm 预设(低危) + 手臂轨迹动画
- `mobile_low` — 叠加: posture 切换权(单向 sit→start→sport) + move(低速) + arm 预设(高危)
- `mobile_unrestricted` — 叠加: move 解除速度限制 + 危险预设 action

**DAG 边带授权要求**. 每条边声明"授权键". 模型请求切换 → channel 进入
"等待授权"窗口 → 人按对应键 → input signal 通知模型 → state 实际切换.

### 4. Warrant 事务机制(命名待最终确认, 暂用此名)

控制类危险命令的统一封装. 用法:

```python
async with bootstrap.warrant("locomotion") as w:
    await w.run(
        client.SetVelocity(vx, vy, vyaw, duration),
        fallback=lambda: client.SetVelocity(0, 0, 0),
    )
```

语义:
- `warrant(scope)` 申请进入一个授权事务作用域
- `w.run(coro, fallback=...)` 在事务保护下执行
- 三个回调 race:
  1. coro 正常完成 → commit, fallback **不执行**
  2. scope 对应 abort 信号触发(遥控器急停键、外部 cancel) → coro cancel + fallback
  3. 当前 state 变更(失效当前事务) → coro cancel + fallback
- 中断时, coro 发出的物理信号视为"未发生" — 对 SetVelocity 而言, RPC 还没到达 G1 之前 cancel 等于这条 move 从未存在

**Scope 跟 channel 解耦.** scope 是"物理授权通道", 一个 scope 可以覆盖多个 channel
的命令, 一个 channel 也可以分多个 scope. 开发者按物理风险类别命名: `locomotion` /
`arm_motion` / `high_risk` 等. 遥控器按键绑定到 scope 维度的急停信号, 一键管一组.

**不可信 fallback 的处理**: arm 类的 fallback 是 ExecuteAction(99) = "release arm",
其物理行为(缓慢复位 vs 瞬间脱控)是 P0 验证项. 验证通过前不允许把 arm 命令包进 warrant.

### 5. Bootstrap 回调架构

bootstrap 暴露线程安全的按键 callback 注册接口:

```python
def register_button_callback(key: str, cb: Callable[[], None]) -> CallbackHandle: ...
def unregister_button_callback(handle: CallbackHandle) -> None: ...
```

cyclonedds reader 线程接到 LowState → 解析按键事件 → 调注册的 cb. cb 内部用
`loop.call_soon_threadsafe()` 把 signal 推到 channel 的 event loop. 按键处理逻辑
全部在 channel event loop 里跑, 零锁.

注册/反注册在 `channel.startup` / `channel.close` 里管理 — 跟 channel 生命周期对齐.

## Channel 体系蓝图

### 顶层 (main → bodies_g1)

- `sensors/` — 无副作用感知子树
- `audio` — 音量
- `led` — 灯光多帧编排
- `music` — 音乐文件播放(后续, 本期不做)
- `posture` — 模式切换(state DAG 的驱动)
- `move` — 移动控制
- `arm` — 手臂预设动作
- `arm_trajectory` — 手臂轨迹动画(后做, 工程量最大)
- `recording` — 录制 + 回放(取决于 SDK 是否暴露, 若不暴露则跟轨迹动画合并)

### sensors 子树

每个 sensor 都遵循统一模式: `open(window_size=N)` / `close()` / `pop()`.
默认展开哪些由 sensors 配置决定; 其余需要 `open` 开启变为 virtual channel.

| Sensor | window 单位 | 数据形态 | 备注 |
|--------|-------------|----------|------|
| trajectory | 关键帧 1Hz | 最近 N 帧位置 | passenger 模式下也记录 |
| odometry | 关键帧 1Hz | 里程计变化 | |
| asr | 句 | 最近 N 句 + speaker_id + 方位 | 不发 signal, 只进 context |
| joints | 帧 | 当前快照(可配置展示哪些关节) | 默认: 肩/肘/腰 |
| actions | 条 | 最近 N 个被动 action | passenger 模式可见 G1 自主行为 |
| vision | 帧 | 最近 N 帧 + look(seconds, fps) | 脸部摄像头, 不进 memory |

### 控制类的 state 隶属

| Channel | seated | seated_arms | mobile_low | mobile_unrestricted |
|---------|--------|-------------|------------|---------------------|
| audio/led/volume | ✓ | ✓ | ✓ | ✓ |
| arm (低危) |  | ✓ | ✓ | ✓ |
| arm (高危) |  |  | ✓ | ✓ |
| arm_trajectory |  | ✓ | ✓ | ✓ |
| posture (sit→start→sport) |  |  | ✓ | ✓ |
| posture (sport→sit) |  |  |  | 需人类显式 |
| move (低速 0-0.2 m/s) |  |  | ✓ | ✓ |
| move (高速) |  |  |  | ✓ |
| 危险预设(后空翻等) |  |  |  | ✓ |

## 用户故事四幕

### 幕一: 坐姿开机

进入态 `seated`. 感知全开(可读), 控制仅 audio/led/volume/audio-say.

Ghost 入场动作:
1. `audio.set_volume(70)` — 音量推到能听清
2. 读 `battery().soc` 报状态, 必要时建议接电源
3. `sensors.vision.open(fps=2, window=3)`
4. `sensors.asr.open(window=5)`
5. audio 说出"我醒了 + 当前状态摘要 + 你想做什么"

idle 行为: 低幅度 LED 呼吸, 提示"我在听".

### 幕二: 授权双手 → `seated_arms`

人按双手授权键 → input signal 到 ghost.

Ghost 反应:
1. 确认 "收到, 手部授权"
2. `arm.list_actions()` + pop 进 memory
3. 说话时穿插小动作("face wave"等)
4. 不自己授权站立 — 等下一个授权键

### 幕三: 授权高级运动 → `mobile_low`

人按运动授权键. Ghost:
1. 看 `motion().fsm_mode` 确认当前是 Sit
2. 说"我要站起来了, 请确认场地空旷" — 给反悔窗口
3. 等几秒 → `posture.squat_to_stand_up()`
4. 起立后 `trajectory.pop()` 记起点
5. 低速移动 + 配合手势挥手
6. 持续看 IMU rpy. 异常 → 主动 `move.stop()` 并询问

约束: ghost **不能**自己回 seated. 想坐下要说出来, 等人按对应键.

### 幕四: 清场 → `mobile_unrestricted`

人清场后按无限制授权键. Ghost 显著降低自主性:
1. 复述将要做的事("走 1m 正方形 + 后空翻"), 要求人退到 3m 外, 按确认键
2. 等确认 input signal. 不收到不动
3. 走正方形(4 段 move + 4 段 vyaw 转向)
4. 完成后 `move.stop()`, 再宣布后空翻, 留 5 秒反悔窗口
5. 后空翻后立刻看 IMU. |roll| 或 |pitch| > 0.3rad → 主动报"我可能没站稳"

## 实验脚本清单

按"先验证地基, 再造具体 channel"排:

| # | 脚本 | 命题 | 决定什么 | 阻塞性 |
|---|------|------|----------|--------|
| P0 (地基) | | | | |
| 17 | remote_keys_passthrough | 调试/AI 模式下每个按键 bit + 摇杆轴是否仍透传 wireless_remote | 遥控器=MOSS 输入设备方案能否成立 | 阻塞 warrant + state DAG 授权 |
| 18 | arm_release_behavior | ExecuteAction(99) 是缓慢复位 / 瞬间脱控 / 撤销 | arm 类 fallback 是否可信 | 阻塞 arm channel + warrant arm scope |
| 19 | loco_stopmove_under_motion | move 中 SetVelocity(0,0,0) 是否站定不仆 | move 类 fallback 是否可信 | 阻塞 move channel + warrant locomotion scope |
| P1 (补完用户故事) | | | | |
| 20 | sit_stand_cycle | Sit→Stand SDK 可达性, 706 双向, Start→Sport 路径, 时长 | 用户故事幕三"自己站起来"可行性, state DAG 形态 | 阻塞 posture channel + 用户故事时序 |
| 21 | arm_action_interruption | Action A 播放中发 B(非 99) — 覆盖/排队/拒绝 | arm command 并发语义, channel 是否需排队 | 阻塞 arm channel await 实现 |
| 22 | arm_action_state_probe | rt/arm/action/state topic 内容 — 是否含 in_progress/done | arm 命令 await 走 topic 等待 vs 关节速度 vs sleep | 阻塞 arm command CTML 时序对齐 |
| P2 (解锁未来 channel) | | | | |
| 23 | asr_api_probe | _Call(1002,...) 调用约定盲探 + ASR DDS topic 扫描 | asr sensor channel 的实现路径 | 阻塞 asr sensor 实装 |
| 24 | mode_switch_topology | FSM 模式完整可达图 + 各边时长 | state DAG 具体边的 SDK 调用选择 | 完善 state DAG, 不影响第一波 |
| 25 | recording_capability_probe | G1 内置录制是否暴露(topic + 协作探测 + RPC 试探) | recording channel 接 SDK 还是自造 | 阻塞 recording channel |
| 26 | arm_sdk_dds_joints_write | rt/arm_sdk 底层写关节角是否可用 + 与 Sport 共存 | 自造手臂轨迹动画的基础可行性 | 阻塞 arm_trajectory channel |
| 27 | lowstate_sample_rate | LowState 真实频率 + monitor 处理上限 | state.py 监控参数 + sensors 采样配置 | 参数级, 不阻塞 |

## 未决议题

继承前 session:
- SetFsmId 白名单拦截
- L2+B 后 MOSS 响应模型(本设计已部分回答: 软清理 + signal 通知 ghost)
- 条件反射层归属(LiDAR 避障, 本期不做)

本次新增:
- **Warrant 最终命名**: warrant / grant / permit / transaction / 其他
- **录制能力**: 24 实测前不能决定接 SDK 还是自造
- **授权键的物理分配**: 哪些键对应哪些 state 边/scope 急停. 17 实测可用键集合后再分配

## SDK 接口的已知与未知 (2026-06-29 补完)

下表是 channel 实现期必须知道的事实, 区分"已知 / 推断 / 待 P0-P1 验证".

### Arm 控制接口

| 维度 | 当前事实 | 来源 |
|------|----------|------|
| 高层接口形态 | ExecuteAction(action_id: int) — 输入是整数 ID, 不是关节角或坐标 | SDK 源码 |
| 高层物理速度 | G1 主板预编排, 出厂调校保守 — **不会"定死 90度 1秒打飞人"** | 推断 + 09 脚本实测 clap 安全 |
| 低层接口形态 | rt/arm_sdk DDS 写 motor_cmd[joint].q/dq/tau/kp/kd, 50Hz 控制循环 + 自写线性插值 | SDK example arm7_sdk_dds |
| RPC 阻塞性 | 立即返回(RPC 应答即 return), 物理动作在 G1 内部播 | SDK 源码 |
| 模式约束 | 仅 Sport (mode_machine=6) 可执行 | 2026-06-16 实测 |
| 中断语义(99 release) | release arm 可打断当前动作(09 脚本测试 2 成功) | 09 脚本 |
| 99 物理行为 | **未实测** — 缓慢插值 / 瞬间脱控 / 撤销控制 | 待 18 验证 |
| 非 99 action 中断行为 | **未实测** — A 播放中发 B 是覆盖/排队/拒绝 | 待 21 验证 |
| 完成感知接口 | SDK 不提供. 三条路: rt/arm/action/state(待 22 验证) / 关节速度趋零(精, 复杂) / 预估 sleep(粗) | 推断 |
| 碰撞反射 | SDK 不提供专门接口. LowState.motor_state[i].tau_est 可读力矩, 自做需 MOSS 层 | SDK 全检 |
| 23-DoF 手腕能力 | 只有 wrist_roll(绕前臂自转), 没有 pitch/yaw | SDK G1JointIndex 注释 |

### Loco / 状态切换接口

| 维度 | 当前事实 | 来源 |
|------|----------|------|
| 控制源单一性 | 调试/AI 模式下 G1 主板忽略遥控器(除 L2+B), MOSS 是唯一物理控制源 | docs/index.md |
| StopMove 实现 | = SetVelocity(0,0,0), 不切 FSM, Sport 平衡控制器接管 | SDK 源码 |
| StopMove 物理行为 | **未实测** — 是否真站定不仆, 高速时冲程? | 待 19 验证 |
| Damp 危险性 | = SetFsmId(1) → 缓慢瘫坐. **永久封禁作为命令** | SDK 源码 + 文档 |
| ZeroTorque 危险性 | = SetFsmId(0) → 全身脱力仆街. **永久封禁** | SDK 源码 + 文档 |
| Sit→Stand 接口 | Squat2StandUp() = SetFsmId(706) | SDK 源码 |
| 706 双向性 | **SDK 源码 Squat2StandUp 和 StandUp2Squat 都 = 706** — 双向还是单向? | 待 20 验证 |
| Sit→Sport 路径 | **未实测** — 直接 Start(500) 还是必须先 706 再 500? | 待 20 验证 |
| 各阶段时长 | **未实测** — Sit→Stand 估 3-5s, Stand→Sit 估同 | 待 20 验证 |
| 模式间合法切换 | 部分已知: 调试模式入口仅 Damp/ZeroTorque + L2+R2. 其他边未系统测 | 部分待 23 验证 |

### 遥控器接口

| 维度 | 当前事实 | 来源 |
|------|----------|------|
| wireless_remote[40] 布局 | bytes[2-3]=按键, bytes[4-7,8-11,12-15,20-23]=四摇杆 float32 | 04 脚本解析对照通过 |
| Sport 模式下透传 | 按键 + 摇杆都正常透传, 解析与物理操作一致 | 2026-06-16 实测 |
| 调试/AI 模式下透传 | **未实测** — 这是 17 的核心命题 | 待 17 验证 |
| L2+B 急停 | 任何模式都生效(硬件级), 不可绕过 | 文档 + 04 脚本 |
| 按键组合的 G1 内部消化 | L2+R2 进调试 / L2+A 进 Damp 等组合键被 G1 消化, 单按 L2 / R2 是否仍报 bit? | 待 17 验证 |

### 录制 / 回放

| 维度 | 当前事实 | 来源 |
|------|----------|------|
| SDK API | 不存在(grep 全检, example 0 个录制相关) | SDK 全检 2026-06-28 |
| PC1 手机蓝牙录制 | **存在**(人类工程师 2026-06-28 提及, 录制开启在 PC1 蓝牙连手机做) | 人类口述 |
| 录制文件可否拷出 | **未实测** — 路径未知, 格式未知 | 待 24 验证 |
| 自造路径 | rt/arm_sdk DDS 写关节角时间序列 + LowState 高频订阅采样, 完全自做 | 推断 |

### 给下个实例的快速指引

读完这份表 + 实验脚本清单, 下一个实例进 g1 工作时:

1. 检查实验脚本结果有没有反馈到这张表(看"已知"列是否增加, "待 N 验证"列是否减少)
2. 如果某些"待 X 验证"未做或翻车, 对应 channel 不要写, 或加显式 disclaimer
3. 这张表是 channel 实现的真值源 — 比 SDK 源码 + 文档更可信, 因为它包含实机验证
