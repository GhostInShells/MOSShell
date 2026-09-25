# G1 云端文档索引

Unitree 官方文档站 (https://support.unitree.com/home/zh/G1_developer) 的本地映射。

**三源关系**: 文档 < 源码 < 实测。文档告诉我们"应该是什么"，源码验证"实际是什么"，实测确认"运行时是什么"。记录时标注每条信息的来源层级。

**记录格式**: 每条记录包含 URL、记录时间、来源层级（文档/源码/实测）、关键提取。

---

## 文档映射表

### 遥控器与状态机

| 项目 | 内容 |
|------|------|
| URL | https://support.unitree.com/home/zh/G1_developer/remote_control |
| 记录时间 | 2026-06-07 |
| 来源层级 | 文档（待源码验证） |
| 文档更新时间 | 2026-05-06 |

**遥控器组合键映射 (2026-06-29 实机确认)**:

| 操作 | 目标模式 | 备注 |
|------|---------|------|
| L2 + Y | 零力矩 | 电机停转无阻尼 |
| L2 + B | 阻尼 (Damp) | 硬件急停, 任何模式生效 |
| L2 + 上 | 锁定站立 | FSM 4 |
| L2 + 左 (长按) | 落座 (Sit) | FSM 3 |
| L2 + A (长按) | 蹲↔站 切换 | FSM 706, 平衡控制 |
| L2 + X | 躺→站 | FSM 702 |
| R1 + X | 常规运控 (单腰) | FSM 500 |
| R1 + Y | 常规运控 (三腰) | FSM 501 |
| R2 + A | 走跑运控 | FSM 801/802 |
| R2 + 下 | 低速 | 走跑运控调速 |
| R2 + 上 | 高速 | 走跑运控调速 |
| R1 + B | 舞蹈运控 | FSM 待确认 |
| R2 + B | 越障运控 | FSM 待确认 |
| L2 + R2 | 调试/诊断模式 | 仅从零力矩或阻尼进入 |

**G1 状态机模式**:

| 模式 | LED | 说明 | MOSS 相关度 |
|------|-----|------|------------|
| 零力矩 | 紫 | 电机停转，无阻尼，可自由摆动 | 可进入调试的入口 |
| 阻尼 | 橙 | 电机停转，有阻尼。**L2+B 全局急停，调试模式下仍有效** | 安全底线 |
| 预备 | 深蓝 | 5s 内摆出准备姿势，无平衡控制 | 低 |
| 下蹲 | — | 5s 内摆出下蹲姿势，无平衡控制 | 低 |
| 落座 | 绿 | 5s 内摆出落座姿势，无平衡控制 | 安全姿态 |
| 运动 | 蓝 | 遥控器控制移动 | **SDK 不可控** |
| 持续走路 | — | 始终踏步 | **SDK 不可控** |
| 站立 | — | 摇杆零指令时停止踏步 | **SDK 不可控** |
| 调试 | 黄 | **SDK 开发模式。停止运动控制程序发指令，避免冲突** | **MOSS 唯一入口** |
| 异常 | 红 | 错误状态 | — |

**关键发现**:

1. **SDK 控制仅在调试模式可用。** 文档明确："在需要使用 SDK 进行开发调试时，请务必确保 G1 已经进入调试模式……以停止运动控制程序发送指令，这样可以避免潜在的指令冲突问题。" 
   - 这意味着 MOSS 无法在运动模式下与遥控器共存——遥控器和 SDK 是互斥的控制源。
   - **待源码/实测验证**: 调试模式下 SDK 的控制范围——是只能读写状态，还是可以发运动指令？文档说"停止运动控制程序发送指令"暗示 SDK 接管运动控制。

2. **L2+B 是硬件级急停底线。** 即使在调试模式下，遥控器 L2+B 始终有效，进入阻尼状态。这是 G1 安全模型的基石——遥控器永远是第一公民。

3. **调试模式入口受限。** 只能从零力矩或阻尼模式通过 L2+R2 进入。不能从运动模式直接切换。

4. **LED 颜色可作为视觉确认。** 调试模式=黄灯，阻尼=橙灯。MOSS 运行时可利用视觉通道做模式交叉验证。

**对 MOSS 集成的影响 — Passenger/Pilot 双模架构** (2026-06-07):

安全边界不在 MOSS 代码里，在 G1 硬件状态机里。非调试模式下 SDK 运动指令根本发不出去——这是硬件围栏，不是 channel 校验。

**Passenger 路径**（G1 非调试模式）：
- 遥控器控制运动，MOSS 读状态 + 音频交互 + 视觉
- channel 只暴露感知类命令，运动命令不可用
- 模型是观察者和对话者，不是控制者

**Pilot 路径**（G1 调试模式）：
- 人类通过遥控器将 G1 切到调试模式（阻尼 → L2+R2）
- MOSS 检测到调试模式后，channel 动态暴露运控命令
- L2+B 始终有效——遥控器的急停权不可绕过

**两层围栏**：
- 硬件层（G1 状态机）：不可绕过的物理围栏。非调试模式 SDK 运动指令直接无效
- 软件层（channel 前置校验）：提前报错 + 动态暴露命令。体验层，不是安全机制

**渐进路径**：运动模式（遥控器控制）→ 落座 → 阻尼 → L2+R2 调试模式 → MOSS 接管。遥控器始终握着急停权。

---

### G1 总览

| 项目 | 内容 |
|------|------|
| URL | https://support.unitree.com/home/zh/G1_developer/about_G1 |
| 记录时间 | 2026-06-07 |

**硬件关键参数**:
- PC2: Jetson Orin NX, 192.168.123.164, **WiFi 6 + 蓝牙 5.2** → PC2 蓝牙耳机方案可行
- PC1: 运控专用(不开放), 192.168.123.161 (NTP)
- LiDAR: 192.168.123.120 (独立 IP)
- 电池: 13 串锂电 9000mAh, ~2h 续航
- G1: 23 Dof (mode_machine=4), G1-EDU: 23~43 Dof (mode_machine=5)
- 膝关节扭矩: 90N.m (G1) / 120N.m (G1-EDU)
- 手臂负载: ~2kg / ~3kg

**关节序号与限位**（关键安全数据，LowCmd_ motor_cmd 索引对应）:
| 索引 | 关节 | 限位(rad) |
|------|------|----------|
| 0-5 | 左腿 (6) | 见文档 |
| 6-11 | 右腿 (6) | 见文档 |
| 12-14 | 腰部 (1-3) | 见文档 |
| 15-21 | 左臂 (7, 29Dof) | 见文档 |
| 22-28 | 右臂 (7, 29Dof) | 见文档 |
| 29 | 权重(kNotUsedJoint) | [0.0, 1.0] |

完整限位表见源文档。MOSS 侧需在 channel 层对所有 MotorCmd_.q 做限位裁剪。

**Thor 背包**: Jetson T5000, 128GB 统一内存, 2070 TFLOPS (FP4)。大模型部署选项，需单独采购。

### 调试说明

| URL | https://support.unitree.com/home/zh/G1_developer/debugging_specification |
|------|------|
| 记录时间 | 2026-06-07 |

基本连接说明。**人类工程师方案**: 不按官方建议走 User PC → 以太网 → PC1。直接在 PC2 上建 MOSS，WiFi 组网，脚本验证完成后远程 fractal 调试。更简洁且利用已有硬件。

### 运动控制 (Loco)

| 本地摘要 | 云端 URL | 状态 |
|----------|----------|------|
| [待填充] | [待发现] | ⬜ 未读 |

### 手臂操作 (Arm)

| 本地摘要 | 云端 URL | 状态 |
|----------|----------|------|
| [待填充] | [待发现] | ⬜ 未读 |

### 音频 (Audio)

| 本地摘要 | 云端 URL | 状态 |
|----------|----------|------|
| [待填充] | [待发现] | ⬜ 未读 |

### SDK 架构与 DDS 通讯

| 项目 | 内容 |
|------|------|
| URL | https://support.unitree.com/home/zh/G1_developer/dds_services_interface |
| 记录时间 | 2026-06-07 |
| 来源层级 | 文档（待源码验证） |

**三层通讯架构**:

1. **DDS 层**（底层）：OMG 分布式通信规范，发布/订阅模型，QoS 策略保障实时性。G1 内部进程间和外部进程间统一使用 DDS。
2. **Channel 层**（SDK 封装）：ChannelFactory → ChannelPublisher/ChannelSubscriber。对 DDS Topic 的 OOP 封装，提供 Send/Recv 语义。
3. **RPC 层**（SDK 封装）：基于 DDS Topic 实现的请求/响应机制。Service Client 模式，支持超时（默认 1s）、租约、错误码体系。

**ChannelFactory**（单例，使用前必须 Init）:
- `Init(domainId, networkInterface, enableSharedMemory)` — **G1 外部开发时 `enableSharedMemory=false`**
- `CreateSendChannel<MSG>(name)` → ChannelPublisher
- `CreateRecvChannel<MSG>(name, callback, queuelen)` → ChannelSubscriber
- **macOS 注意**: ChannelFactory 依赖 cyclonedds，macOS 上无法编译 Init。源码分析优先。

**RPC Client 通用接口**: `Init()`, `SetTimeout(seconds)`, `WaitLeaseApplied()`（阻塞至获取租约）, `GetApiVersion()`, `GetServerApiVersion()`

**RPC 错误码体系**（部分）:
- 3001 未知错误，3102 请求发送错误，3103 API 未注册，3104 请求超时，3105 请求响应数据不匹配，3106 响应数据无效
- 3202 服务端内部错误，3203 API 服务端未实现，3204 API 参数错误，3205 请求被拒绝

**G1 核心 Topic 列表**（`hg` = humanoid G1, `go2` = 机器狗共享, `lf` = 低频模式）:

| Topic | 类型组 | 消息类型 | 方向 | 说明 |
|-------|--------|---------|------|------|
| `rt/lowstate` | hg | LowState_ | 读 | 底层反馈：IMU、电机状态 |
| `rt/lf/lowstate` | hg | LowState_ | 读 | 同上-低频模式 |
| `rt/lowcmd` | hg | LowCmd_ | **写** | **底层控制命令 — MOSS 运控入口** |
| `rt/dex3/left/state` | hg | HandState_ | 读 | 左灵巧手反馈 |
| `rt/dex3/right/state` | hg | HandState_ | 读 | 右灵巧手反馈 |
| `rt/dex3/left/cmd` | hg | HandCmd_ | 写 | 控制左灵巧手 |
| `rt/dex3/right/cmd` | hg | HandCmd_ | 写 | 控制右灵巧手 |
| `rt/lf/dex3/left/state` | hg | HandState_ | 读 | 左灵巧手-低频 |
| `rt/lf/dex3/right/state` | hg | HandState_ | 读 | 右灵巧手-低频 |
| `rt/lf/bmsstate` | hg | BmsState_ | 读 | 电池反馈 |
| `rt/lf/mainboardstate` | hg | MainBoardState_ | 读 | 主板反馈 |
| `rt/odommodestate` | go2 | IMUState_ | 读 | 里程计 |
| `rt/lf/odommodestate` | go2 | IMUState_ | 读 | 里程计-低频 |
| `rt/secondary_imu` | hg | IMUState_ | 读 | 机身 IMU |
| `rt/lf/secondary_imu` | hg | IMUState_ | 读 | 机身 IMU-低频 |

**关键推断**（待源码验证）:
- `rt/lowcmd` 是 MOSS 的运动控制写入点。需要理解 LowCmd_ 的消息结构（关节角度/力矩/目标位置等）
- `rt/lowstate` 是 MOSS 的状态读取点。LowState_ 包含 IMU + 电机反馈
- 灵巧手（dex3）有独立的读写 topic，与身体运控分离
- "低频模式" topic 的存在说明 G1 有带宽管理机制。`rt/lowstate` 高频 vs `rt/lf/lowstate` 低频

### 高层运动服务接口

| 项目 | 内容 |
|------|------|
| URL | https://support.unitree.com/home/zh/G1_developer/sport_services_interface |
| 记录时间 | 2026-06-07 |
| 来源层级 | 文档（待源码验证） |

**核心发现 — 修正之前的 Passenger/Pilot 假设**:

> "高层运动服务依赖于内置运控，进入调试模式后内置运控完全退出，高层运动服务失效。"

这意味着：
- **LocoClient RPC (`Sit()`, `Move()`, `StandUp()` 等) 在非调试模式下可用，在调试模式下失效**
- **`rt/lowcmd` DDS 在调试模式下可用，在非调试模式下会与内置运控冲突**
- 之前假设"非调试=MOSS 旁观，调试=MOSS 控制"是错的——实际上是两条不同的控制路径

**修正后的控制模型**:

| 维度 | RPC 路径（非调试） | DDS 路径（调试） |
|------|-------------------|------------------|
| 控制接口 | LocoClient RPC | rt/lowcmd DDS pub/sub |
| 内置运控 | 运行中，处理平衡/步态 | 完全退出 |
| 安全性 | 高 — 内置运控在兜底 | 低 — 无内置安全网 |
| 控制粒度 | 高层（Sit/StandUp/Move） | 底层（关节级 LowCmd_） |
| MOSS 角色 | 发指令者，内置运控执行 | 完全接管，自己实现运动 |
| 适用场景 | 初始集成、安全优先 | 高级开发、自定义运动 |

**LocoClient RPC API**（非调试模式可用）:

普通接口:
| 函数 | 说明 |
|------|------|
| `Damp()` | 进入阻尼模式 |
| `Start()` | 进入主运控 |
| `Squat()` | 进入下蹲 |
| `Sit()` | 进入落座 |
| `StandUp()` | 进入站立 |
| `ZeroTorque()` | 进入零力矩 |
| `Move(vx, vy, vyaw)` | 速度指令，默认 1s |
| `Move(vx, vy, vyaw, continuous)` | 速度指令，可选持续 |
| `StopMove()` | 速度置零 |
| `BalanceStand()` | 平衡站立 |
| `ContinuousGait(bool)` | 改变 Move() 持续性 |

专家接口:
| 函数 | 说明 |
|------|------|
| `GetFsmId(fsm_id)` | 获取当前 FSM 模式 ID |
| `SetFsmId(fsm_id)` | **设置 FSM 模式 — 可直接切换模式** |
| `GetFsmMode(fsm_mode)` | 0=站立, 1=移动 |
| `SetVelocity(vx, vy, omega, duration)` | 带持续时间的速度指令 |
| `SetSpeedMode(mode)` | 走跑速度模式 0-3 (1.0~3.0 m/s) |

**FSM 模式 ID 表**:

| ID | 模式 | 平衡控制 |
|----|------|---------|
| 0 | 零力矩 | 无 |
| 1 | 阻尼 | 无 |
| 2 | 位控下蹲 | 无 |
| 3 | 位控落座 | 无 |
| 4 | 锁定站立 | 无 |
| 500 | 常规运控 | 有 |
| 501 | 常规运控-3Dof-waist | 有 |
| 702 | 躺起 | — |
| 706 | 平衡下蹲/蹲起 | 有 |
| 801/802 | 走跑运控 | 有 |

**DDS 上肢控制**（非调试模式，锁定站立/运控 1/运控 2 下可用）:
- Topic: `rt/arm_sdk`，消息类型 `LowCmd_`
- 电机索引 12-28: 腰部与上肢
- 索引 29: 权重 [0.0, 1.0]
- 与身体运控独立——内置控制器管腿，MOSS 管手

**SportModeState**（Topic: `rt/sportmodestate`，固件 >= 1.5.1）:
| 字段 | 说明 |
|------|------|
| `fsm_id` | 当前模式 ID（见上表） |
| `fsm_mode` | 0=静态(可切模式), 1=动态(不可切，仅可切到阻尼) |
| `task_id` | 上肢互动动作 ID |
| `task_time` | 动作执行时间(秒)，握手保持期间不变 |

**对 MOSS 集成的关键影响**:
- 初始集成应该走 **RPC 路径 + 非调试模式**。内置运控兜底，比调试模式安全得多
- `SetFsmId()` + `GetFsmId()` + `SportModeState` 构成了 G1 的模式管理闭环
- 上肢控制（rt/arm_sdk）可以在运动模式下与内置运控共存——这是"模型控制上肢 + 遥控器控制身体"的混合路径
- 调试模式是终极目标但不是第一步

### 底层通讯与消息结构

| 项目 | 内容 |
|------|------|
| URL | https://support.unitree.com/home/zh/G1_developer/basic_services_interface |
| 记录时间 | 2026-06-07 |
| 来源层级 | 文档（待源码验证） |

**整体**: PC2 ↔ G1 通过 DDS 订阅 `rt/lowstate` 读状态，发布 `rt/lowcmd` 写控制。灵巧手独立 topic: `rt/dex3/left/right/cmd` 和 `state`。

**LowState_**（状态读取 — rt/lowstate）:
```cpp
struct LowState_ {
  unsigned long version[2];
  octet mode_pr;                       // 0:PR, 1:AB (脚踝/腰部并联机构)
  octet mode_machine;                  // 4:23Dof, 5:29Dof, 6:27Dof
  unsigned long tick;                  // 1ms 递增计时器
  IMUState_ imu_state;                 // IMU (四元数/陀螺仪/加速度/欧拉角/温度)
  MotorState_ motor_state[35];         // 全身 35 个电机状态
  octet wireless_remote[40];           // **遥控器原始数据 — MOSS 可读遥控器按键!**
  unsigned long reserve[4];
  unsigned long crc;                   // 校验和
};
```

**LowCmd_**（运动控制写入 — rt/lowcmd）:
```cpp
struct LowCmd_ {
  octet mode_pr;                       // 需与 LowState_ 匹配
  octet mode_machine;                  // 需与 LowState_ 匹配
  MotorCmd_ motor_cmd[35];             // 全身 35 个电机控制
  unsigned long reserve[4];
  unsigned long crc;                   // 校验和（必须正确）
};
```

**MotorCmd_**（单电机控制 — 完整阻抗控制接口）:
```cpp
struct MotorCmd_ {
  octet mode;                          // 0:Disable, 1:Enable
  float q;                             // 目标位置 (rad)
  float dq;                            // 目标速度 (rad/s)
  float tau;                           // 前馈力矩
  float kp;                            // 刚度系数
  float kd;                            // 阻尼系数
  unsigned long reserve[3];
};
```

**MotorState_**（单电机反馈）:
```cpp
struct MotorState_ {
  octet mode;                          // 当前模式
  float q;                             // 实际位置 (rad)
  float dq;                            // 实际速度 (rad/s)
  float ddq;                           // 实际加速度 (rad/s^2)
  float tau_est;                       // 估算力矩
  short temperature[2];                // 外表+绕组温度
  float vol;                           // 电机端电压
  unsigned long motorstate;            // 电机状态字
  // ...
};
```

**IMUState_**: `quaternion[4]`, `gyroscope[3]`, `accelerometer[3]`, `rpy[3]`, `temperature`

**HandCmd_** / **HandState_**: 灵巧手独立控制。`motor_cmd`/`motor_state` + `imu_state` + `press_sensor_state` + `power_v/a`

**PressSensorState_**: `pressure[12]` + `temperature[12]`

**关键发现 — `wireless_remote[40]`**:
- LowState_ 中包含遥控器原始数据。MOSS 可以读到遥控器的按键状态
- 这意味着可以实现软件层急停确认：读到 L2+B → MOSS 立即停止所有运动指令
- 这是硬件急停（不可绕过）之外的一层软件感知——MOSS 能知道自己被急停了

**初始集成决策（2026-06-07）**: 底层 DDS 通讯（rt/lowcmd + rt/lowstate）在初始集成阶段定位为**状态读取 + 调试自感知**，不作为控制通道。运动控制走 RPC 路径（LocoClient），有内置运控兜底。底层写入留到高阶阶段，届时遥控器保持急停权。

**其他关键点**:
- `MotorCmd_.mode` 是逐电机使能——可以只 enable 上肢电机，保持下肢 disable
- `kp`/`kd` 是阻抗控制参数——默认低刚度=柔性安全，需要时才提高
- `crc` 校验和必须正确，否则指令被拒绝
- `tick` 1ms 递增，提供高精度时序基准
- 所有调研结论待人类工程师在 PC2 上验证修订

### 设备状态服务接口

| 项目 | 内容 |
|------|------|
| URL | https://support.unitree.com/home/zh/G1_developer/robot_state_client_interface |
| 记录时间 | 2026-06-07 |
| 来源层级 | 文档（待源码验证） |

**RobotStateClient** — G1 内部服务治理的 RPC 客户端。复用自 B2 平台。

接口:
| 函数 | 说明 |
|------|------|
| `ServiceSwitch(name, swit, status)` | 服务开关。swit=1 开/0 关。status 返回操作后状态 |
| `SetReportFreq(interval, duration)` | 服务状态上报频率。interval/duration 单位秒 |

错误码:
| 错误号 | 说明 |
|--------|------|
| 5201 | 服务开关执行错误 |
| 5202 | 服务受保护，不允许开启或关闭 |

**G1 内部服务列表**:

| 服务名 | 描述 | MOSS 相关度 |
|--------|------|------------|
| `ai_sport` | 主运动控制服务 | **核心** — LocoClient RPC 依赖此服务运行 |
| `basic_service` | 底层服务 | 高 — 底层通讯基础 |
| `g1_arm_example` | 上肢动作服务 | 高 — 挥手/握手等预设动作 |
| `vui_service` | 音频灯光控制服务 | 中 — 音频交互 |
| `unitree_slam` | 导航服务 | 低 — 高阶阶段 |

**关键推断**（待源码验证）:
- `ServiceSwitch("ai_sport", 0)` 可以关闭主运控——相当于软件方式进入类似调试模式的状态
- `ServiceSwitch("ai_sport", 1)` 重新开启——MOSS 可能可以在 RPC 和 DDS 模式间切换而无需遥控器
- 但部分服务"受保护"(5202)——哪些服务可被开关、需要什么权限，需源码验证
- `SetReportFreq` 提供了服务状态轮询机制——可能可以用于 MOSS 的 alive 检测

**SDK 路径**: `unitree_sdk2py/b2/robot_state` (Python), `unitree_sdk2/include/unitree/robot/b2/robot_state` (C++)

### 里程计服务接口

| 项目 | 内容 |
|------|------|
| URL | [待补] |
| 记录时间 | 2026-06-07 |

- Topic: `rt/odommodestate` (500Hz 高频) / `rt/lf/odommodestate` (20Hz 低频)，类型 `SportModeState_`
- 数据: 位置(世界坐标系, m)、速度(机器人坐标系, m/s)、欧拉角(rad)、yaw角速度(rad/s)、四元数
- 坐标系: 世界=开机时机身地面投影为原点; 机器人=机身几何中心为原点
- 最低版本: State Estimator >= 1.0.0.1

**空间感知组合**: 里程计(位置/姿态) + 视觉通道(前方场景) → 模型获得完整空间上下文。呈现方式待定：可以文本叠加到视觉帧、独立 context message、或渲染空间示意。初始集成以文本形式从 G1 channel 的 context_messages 提供。

### 音频灯光服务接口

| 项目 | 内容 |
|------|------|
| URL | https://support.unitree.com/home/zh/G1_developer/VuiClient_Service |
| 记录时间 | 2026-06-07 |

**硬件**: 256 色 RGB 灯条 + 8Ω/3W 扬声器 + 四麦阵列(20mm 间距, 线性硅麦)

**AudioClient RPC**:
| 函数 | 说明 |
|------|------|
| `TtsMaker(text, speaker_id)` | 内置离线 TTS。0=中文女, 1=英文。不支持中英混合 |
| `GetVolume(volume)` / `SetVolume(volume)` | 0-100 |
| `LedControl(R, G, B)` | 调用间隔需 >200ms |
| `PlayStream(app_name, stream_id, pcm_data)` | PCM 16K/16bit/mono。相同 stream_id=续播, 不同=打断 |
| `PlayStop(app_name)` | 停止播放 |

**ASR**（麦克风打开状态下）:
- DDS topic: `rt/audio_msg` (String_)，JSON 格式
- 离线非流式。字段: text, angle(0-180), speaker_id, sense(情绪), confidence, is_final
- 自带 VAD（文档未描述控制接口）

**麦克风音频**: UDP 组播 239.168.123.161:5555，PCM 16K/16bit/mono，降噪后输出

**关键待验证**（源码+实测）:
- **架构**: 音频硬件大概率在 PC1，PC2 通过 RPC 间接控制。PC2 本身可能无直接音频设备
- **PlayStream 状态**: 是否有播放完成/被中断的回调？目前文档只有 play_state 的 DDS 通知（0=停止, 1=开始），不确定是否可靠
- **TTS 耗时**: 无返回值表示完成。PC2→PC1 延迟未知。需实测计时
- **Cancel 能力**: `PlayStop(app_name)` 存在，但 TTS 是否可 Cancel 不确定
- **流式推送**: `PlayStream` 接受 PCM buffer，相同 stream_id 可续播。能否边生成边推送需实测——底层是队列还是混音不明
- **VAD 控制**: ASR 自带 VAD，但无 API 开关/灵敏度调整

**替代方案**: PC2 是否可直连蓝牙耳机？PC2 性能应足够，MOSS 音频从 PC2 直接输出可绕过 PC1 音频链路的全部不确定性（PlayStream 状态/TTS 耗时/Cancel 可靠性）。需在阶段 C（硬件环境记录）中验证 PC2 蓝牙能力。

**已知能力缺口**: 目前文档未发现 G1 有自动避障或自动降速能力。运动安全完全依赖遥控器+内置运控，MOSS 侧需自行实现安全边界（速度上限、姿态限制、区域围栏）。

**MOSS 音频四语义 vs G1 现状**:

| 语义 | G1 现状 | 差距 |
|------|---------|------|
| 流式(streaming) | PlayStream 支持续播(同 stream_id)，可能可切片推送 | 需验证是否可边生成边推 |
| 取消(cancel) | PlayStop(app_name) 存在 | TTS 是否可取消不明 |
| 异常(exception) | 无文档说明 | 需源码查错误传播 |
| 完成(done) | play_state DDS 通知 | 是否可靠、延迟多大 |

### LiDAR 服务接口

| 项目 | 内容 |
|------|------|
| URL | https://support.unitree.com/home/zh/G1_developer/lidar_services_interface |
| 记录时间 | 2026-06-07 |

- 型号: Mid-360 (2026.4 后生产为 Mid360s)，位于头部中央，倒置安装
- LiDAR→机器人坐标系: (-0.0, 0.0, -0.47618)，pitch 倾角 -2.3°
- IMU→LiDAR: (0.011, 0.02329, -0.04412)
- 点云: `rt/utlidar/cloud_livox_mid360`，10Hz，ROS2 PointCloud2_ 格式
- IMU: `rt/utlidar/imu_livox_mid360`，200Hz，ROS2 Imu_ 格式
- LiDAR 独立 IP: 192.168.123.120。三条数据路径: DDS(简单)、Livox SDK2 直连(低延迟，绕过 PC1)、ROS2 driver
- Livox Viewer 2 不可在 Jetson Orin NX 上运行。噪点过滤: tag 低四位非零=噪点

**条件反射层**: LiDAR 可用于 channel 内部的安全条件反射——不经过模型思考，直接在 channel 层做：
- 近场急停（<X 米有障碍 → 强制停止运动指令）
- 速度自适应（点云密度 → 动态限制 vx/vy 上限）
- "虚拟保险杠"——独立于遥控器和模型判断的第三层安全围栏

这是之前避障缺口的填补方案。不同于模型驱动的 CTML 指令，条件反射是 channel 内部的自主安全层，持续运行，不可被模型绕过。

### 运控切换接口

| 项目 | 内容 |
|------|------|
| URL | https://support.unitree.com/home/zh/G1_developer/motion_witcher_service_interface |
| 记录时间 | 2026-06-07 |

**MotionSwitcherClient** — 运控模式切换 RPC。复用自 B2 平台。

| 函数 | 说明 |
|------|------|
| `CheckMode(form, name)` | 检测当前运控模式。name="ai"=主运控运行中 |
| `SelectMode(name)` | 选择运控模式。`SelectMode("ai")`=恢复主运控 |
| `ReleaseMode()` | **释放运控模式，进入用户调试模式** |

错误码: 7001(参数错误), 7002(服务繁忙), 7004(模式名不支持), 7005-7008(各类执行错误), 7009(自定义配置错误)

**关键意义**: `ReleaseMode()`/`SelectMode("ai")` 提供了 **软件路径** 在 RPC 模式和 DDS 调试模式之间切换——不需要遥控器 L2+R2。这意味着：

- MOSS 可以程序化从 Passenger 切到 Pilot：`ReleaseMode()` → 内置运控退出 → `rt/lowcmd` 可用
- MOSS 可以程序化回到 Passenger：`SelectMode("ai")` → 内置运控恢复 → LocoClient RPC 恢复
- 但切换前是否需要先进入阻尼/零力矩（如遥控器要求）？文档未说明，源码验证优先级高

**安全考量**: 这是强大的能力也是高风险点。MOSS 应在 channel 层对 `ReleaseMode()` 做人类确认围栏——模型请求释放运控时需要人类在遥控器或终端确认。不能让模型自主从 Passenger 切到 Pilot。

### 时间同步

| 项目 | 内容 |
|------|------|
| URL | https://support.unitree.com/home/zh/G1_developer/time_sync_interface |
| 记录时间 | 2026-06-07 |

- PC1 内置 NTP 服务器: 192.168.123.161
- PC2 用 Chrony 或 Systemd-Timesyncd 同步。WiFi 模式触发网络授时可能导致时间跳变，依赖系统时间的程序建议关 WiFi
- Chrony 已知 seccomp 问题: 修改 `/etc/default/chrony` 添加 `DAEMON_OPTS="-F 0"`
- Phase D 装机时处理

### 手臂控制例程

| 项目 | 内容 |
|------|------|
| URL | https://support.unitree.com/home/zh/G1_developer/arm_control_routine |
| 记录时间 | 2026-06-07 |

- Topic: `rt/arm_sdk`，类型 `LowCmd_`。**无需调试模式**——可在内置运控运行时使用
- 电机索引: 12-28(腰+上肢), 29=权重(`kNotUsedJoint`, 过渡平滑系数 [0.0, 1.0])
- Weight 机制: 0→1 电机从当前位置平滑过渡到目标位置。变化速度=过渡速度。退出时将 weight 从 1→0 实现慢复位
- 控制模式: 逐帧插值——每步计算 `current += clamp(target - current, -max_delta, max_delta)`，然后发布

**MOSS 动画模型**:

G1 手臂控制天然支持关键帧动画模式:
- **可编程录制**: MOSS 可以记录关节序列（从 `rt/lowstate` 订阅 `motor_state[].q`），存储为关键帧文件
- **生成式动画**: 模型生成关键帧序列 → 插值播放。weight 机制保证平滑过渡
- **慢复位**: 动画结束 weight → 0，电机逐渐回到自由状态，而非骤停
- **剪辑**: 关键帧文件可截取/拼接——纯数据，不依赖 G1 内部存储

**关键验证点**（待实测）:
- 下肢运控时上肢 DDS 控制是否可用（文档说"锁定站立/运控1/运控2 中使用"）
- 高阶动作（翻跟头等）是否有内置冲突防护——没有则 MOSS 侧需做模式互斥锁
- 轨迹录制 API: 目前文档未发现录制开关。替代路径是 MOSS 自己从 `rt/lowstate` 采样关节序列

### 手臂动作服务接口

| 项目 | 内容 |
|------|------|
| URL | https://support.unitree.com/home/zh/G1_developer/arm_action_interface |
| 记录时间 | 2026-06-07 |

**ArmAction RPC** — 内置手臂互动动作。依赖内置运控，调试模式下失效。

| 函数 | 说明 |
|------|------|
| `ExecuteAction(int action_id)` | 预设动作。**阻塞执行** |
| `ExecuteAction(string action_name)` | 示教动作。**非阻塞执行** |
| `StopCustomAction()` | 停止示教动作，手臂回初始位 |
| `GetActionList(data)` | 列出可用动作、FSM 要求、示教动作名和时长 |

预设动作: 99(复位), 11(双手飞吻), 15(平举), 17(鼓掌), 19(拥抱), 20(双手比心), 27(握手) 等 15+ 种

错误码: 7400(占用), 7401(手臂举起/用99复位), 7402(ID不存在), 7404(当前FSM不可触发)

**关键互斥**: 使用 `rt/arm_sdk` 做自定义手臂控制前，必须先关闭 `g1_arm_example` 服务：
```cpp
RobotStateClient rsc;
rsc.ServiceSwitch("g1_arm_example", 0, status);
```

**示教录制**: App 端有示教功能可录制自定义动作（存为命名动作），但 SDK 文档未暴露录制 API。MOSS 的录制路径仍然是自行采样 `rt/lowstate` 的 `motor_state[].q`，存为关键帧文件。

**MOSS 决策**: 初始阶段直接用 `ExecuteAction(int)` 调用预设动作（阻塞，简单可靠）。需要自定义动画时关闭 `g1_arm_example`，走 `rt/arm_sdk` 关键帧回放路径。

### 辅助路径（已记录，非初始集成必需）

| 路径 | URL | 用途 | 何时启用 |
|------|-----|------|---------|
| RL 仿真训练 | https://support.unitree.com/home/zh/G1_developer/rl_control_routine | Isaac Gym + unitree_rl_gym 步态训练 | Phase H+ 自定义步态 |
| ROS2 通讯 | https://support.unitree.com/home/zh/G1_developer/ros2_communication_routine | rosbag/rviz2/tf2 工具链 | 调试可视化时 |
| Livox SDK2 直连 | https://support.unitree.com/home/zh/G1_developer/lidar_Instructions | LiDAR 直连 192.168.123.120，绕过 PC1 | 条件反射层低延迟需求 |
| DDS 通信例程 | https://support.unitree.com/home/zh/G1_developer/dds_communication_routine | 最小 pub/sub 示例 | 参考（已消化） |
| 深度相机 D435i | https://support.unitree.com/home/zh/G1_developer/depth_camera_instruction | 桌面灵巧手 VLA/VLM 操作 | 高阶阶段（当前不启用） |

### 未覆盖的文档站条目

| 条目 | URL | 状态 |
|------|-----|------|
| G1 总览 | https://support.unitree.com/home/zh/G1_developer/about_G1 | ⬜ |
| 快速开始 | https://support.unitree.com/home/zh/G1_developer/G1_Quick_Start | ⬜ |

---

## 阶段 A 判断总结

### 人类工程师判断

1. **遥控器是第一公民。** L2+B 是硬件级急停底线，所有模式下不可绕过。MOSS 与遥控器的兼容性是首要设计约束。

2. **G1 没有自动避障/降速。** 安全完全依赖遥控器+内置运控。MOSS 侧需用 LiDAR 填补条件反射层。

3. **D435i 深度相机当前没用。** 斜下 75° 无脖子，是为桌面灵巧手 VLA/VLM 设计的。初始集成视觉走外部摄像头。

4. **PC2 应测试蓝牙耳机直连。** 绕过 PC1 音频链路的不确定性。AudioClient RPC 的状态反馈不完善。

5. **手臂动画用关键帧模型。** 生成式关键帧 + 平滑过渡 + 慢复位。不依赖示教录制 API。

6. **文档 < 源码 < 实测。** 所有结论待 PC2 实装验证。文档可能过期或与实际行为不一致。

7. **示教录制存在但 API 未暴露。** App 端可录制，SDK 可列表和回放但不可触发录制。MOSS 自己采样 `rt/lowstate`。

### 模型判断

1. **三层安全围栏**: 硬件层(L2+B, FSM 门控) → 条件反射层(LiDAR 近场检测, channel 内自主) → 模型层(CTML/RPC, 受前两层约束)。条件反射层不可被模型绕过。

2. **双路径控制模型**: RPC 路径(非调试, LocoClient, 内置运控兜底) 是初始集成首选。DDS 路径(调试, rt/lowcmd) 是高阶阶段目标。`MotionSwitcherClient` 提供了软件切换能力但需要人类确认围栏。

3. **最简 channel 原则**: App 进程 = 生命周期。Channel = 构造连硬件 + 方法暴露命令。G1 是此原则的示范。

4. **Channel 复杂度是历史的**: bootstrap/cleanup/factory/stateful 是进程内嵌入模式的补丁，app 模式天然解耦。

5. **空间感知组合**: 里程计文本(位置/姿态) + LiDAR 点云(障碍距离) + 外部视觉帧 = 模型空间上下文。初始以文本形式从 context_messages 提供。

6. **手臂控制分层**: 预设动作(ExecuteAction, 简单) → 自定义关键帧(rt/arm_sdk, 灵活) → 全身运动(rt/lowcmd, 高阶)。逐级解锁。

7. **LiDAR 只做距离检测**: 10Hz 点云做扇面距离判断和线性降速，不做 SLAM/识别。复杂空间分析通过 Matrix 总线广播给专门 cell。

8. **ROS2 当前不引入**: 初始集成 DDS 已覆盖所有通讯需求。ROS2 工具链留到需要 rosbag/rviz2 时。

---

## 关键概念索引

从文档中提取的核心概念，标注来源层级和来源页面。

| 概念 | 简要说明 | 来源层级 | 来源 |
|------|---------|---------|------|
| 调试模式 | SDK 唯一可控模式。停止运控程序，避免指令冲突。从阻尼/零力矩 + L2+R2 进入 | 文档 | remote_control |
| L2+B 急停 | 遥控器硬件急停，调试模式下仍有效。进入阻尼状态 | 文档 | remote_control |
| 运控状态 LED | 7 种颜色对应 7 种运控模式 + 异常红 | 文档 | remote_control |
| 控制权互斥 | 遥控器与 SDK 不可同时控制运动。调试模式下运控程序停止 | 文档（待源码验证） | remote_control |
| 双路径控制模型 | RPC路径(非调试,高层,安全) vs DDS路径(调试,底层,全权)。初始集成优先RPC | 架构决策(修正) | high_level_motion |
| LocoClient RPC | 高层运动服务客户端。Sit/StandUp/Move/SetFsmId。依赖内置运控 | 文档 | high_level_motion |
| FSM 模式 ID | 0-802 共 10+ 种模式。500+ 有平衡控制。SetFsmId/GetFsmId 管理 | 文档 | high_level_motion |
| SportModeState | rt/sportmodestate DDS topic。fsm_id + fsm_mode(静/动) + task_id | 文档 | high_level_motion |
| 上肢独立控制 | rt/arm_sdk DDS topic。电机 12-28。可在运控模式下与内置控制器共存 | 文档 | high_level_motion |
| 底层通讯定位 | 初始集成: DDS 仅用于状态读取+自感知。控制走 RPC。底层写入留高阶阶段 | 架构决策 | 本次会话 |
| wireless_remote[40] | LowState_ 中遥控器原始数据。MOSS 可软件感知急停按键 | 文档 | low_level_comm |
| MotorCmd_ 阻抗控制 | kp/kd 刚度阻尼系数。默认低刚度=柔性安全 | 文档 | low_level_comm |
| LowCmd_.crc | 校验和必须正确，否则指令被拒绝 | 文档 | low_level_comm |
| DDS 三层架构 | DDS → Channel(发布/订阅) → RPC(请求/响应)。统一内外部进程通讯 | 文档 | sdk_interface |
| ChannelFactory | DDS 单例工厂。外部开发 `enableSharedMemory=false`。需指定网卡 | 文档 | sdk_interface |
| rt/lowcmd | **MOSS 运控写入点**。LowCmd_ 类型，控制关节/运动 | 文档（待源码验结构） | sdk_interface |
| rt/lowstate | **MOSS 状态读取点**。LowState_ 类型，IMU+电机反馈 | 文档（待源码验结构） | sdk_interface |
| 灵巧手独立控制 | dex3 左右手有独立 read/write topic，与身体运控分离 | 文档 | sdk_interface |
| RPC 租约机制 | `WaitLeaseApplied()` 阻塞获取租约。用于独占资源访问 | 文档（待源码验证） | sdk_interface |
| 低频/高频双通道 | 大部分状态 topic 有 rt/ 和 rt/lf/ 两个版本 | 文档 | sdk_interface |

---

## 状态说明

- ⬜ 未读 — 尚未访问
- 🔄 进行中 — 正在阅读/消化
- ✅ 已完成 — 已提取关键信息，写入对应 `docs/*.md`
- ❌ 不可访问 — SPA 或 404
- ⏭️ 跳过 — 当前阶段不需要
