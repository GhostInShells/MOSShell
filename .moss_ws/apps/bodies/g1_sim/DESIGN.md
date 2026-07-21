# 技术设计文档：MuJoCo 人形（G1）纯软件演示 Channel

> 当前文件是 `g1_sim` app 的权威设计文档，替代了早期放在项目根目录的 `mojoco_humun.md`。
> 使用建议：实现与重构时优先看本文件；运行与使用方式看 `README.md` / `APP.md`。

> 版本: v1.0
> 状态: 设计定稿，待开发
> 适用范围: 纯软件仿真演示（NO hardware）。与 dev 分支中 `apps/bodies/g1` 的真机 SDK 通道（LocoClient/DDS/PC2）**无任何关系**。
> 目标读者: 负责实现的 AI Agent / 开发者
> 基线: MOSShell `dev` 分支

---

## 0. 阅读指引（给实现 Agent）

本文件是**可执行的施工蓝图**，不是概念讨论。请按以下顺序使用：

1. 先读 §1～§4，建立整体心智模型（分层、接缝、两阶段路线）。
2. 实现时以 §6（模块详细设计）为主，逐模块落地。
3. §10（观测/动作对齐）是**整个项目唯一真正的技术难点**，必须严格按检查清单执行，否则机器人会原地摔倒。
4. §12 是任务拆解（WBS），可直接转成 Task 列表。
5. 遇到"维度/索引/单位"等具体数值，**一律以 `unitree_rl_gym` 仓库的实际 config 为准**，不要相信本文里任何写死的数字（本文标注的数字仅为"预期量级"，用于帮助理解）。

---

## 1. 项目目标与非目标

### 1.1 目标（In Scope）

构建一个 MOSShell channel，使得：

- 用户用自然语言（语音/文字）说"往前走""向左转""停下"等指令；
- 大模型经 CTML 把指令路由到该 channel 的命令；
- channel 背后由一个**预训练小型 MLP 策略**驱动 **MuJoCo 仿真中的 G1 人形**完成对应动作；
- 全程**纯软件**，在一台普通机器（CPU 即可，GPU 可选）上跑出可视化 demo。

最终可演示形态：**"我说往前走 → 屏幕里的 G1 真的迈步走过去"**。

### 1.2 非目标（Out of Scope）

- ❌ 真实硬件、真机 SDK、DDS、sim2real、力矩安全限幅。
- ❌ 自己从头训练策略（直接用开源预训练权重）。
- ❌ 全身协调的精细上肢手势（whole-body control，属研究前沿，本期不做）。
- ❌ 蹲下/起立等**走路策略不支持**的动作（除非所选策略原生支持，否则本期不放出该命令）。
- ❌ 复杂动态运动规划（`abcd.py` 中 `Movement._plan_new_actions` 那套未完成的多轨规划，本期不碰）。

### 1.3 设计原则

- **能力边界决定指令集**：channel 暴露哪些命令，由所加载策略实际支持的能力反推，不预设。
- **接缝解耦**：channel 层与数据层不动，仿真后端是可替换零件。
- **分阶段可演示**：每一步都留一个能跑起来的版本。

---

## 2. 背景：dev 分支现状（可复用资产盘点）

| 资产 | 路径 | 本项目如何使用 |
|---|---|---|
| `RobotController` 抽象 | `src/ghoshell_moss_contrib/prototypes/ros2_robot/abcd.py` | **复用其"抽象接缝"哲学**；但其面向 Trajectory，需扩展（见 §5）。 |
| 后台线程控制循环范式 | `prototypes/ros2_robot/ros2_controller.py` | **复用线程模式**：守护线程 + 队列 + `ThreadSafeFuture`，`goal_interval=1/50`（50Hz）。我们的控制循环照此结构写。 |
| `MOSSRobotManager` / `MemoryRobotManager` | `abcd.py` / `manager.py` | 复用 `MemoryRobotManager` 持有 `RobotInfo`（机器人形体描述）。 |
| 数据模型 | `prototypes/ros2_robot/models.py` | `Joint`/`Pose`/`RobotInfo` 复用于"形体描述"；`Trajectory`/`Animation` 本期**不作为 locomotion 主路径**（见 §5）。 |
| `MockRobotController` | `prototypes/ros2_robot/mocks.py` | 作为新控制器的**结构模板**。 |
| channel 构建 API | `ghoshell_moss.core.blueprint.channel_builder.new_channel` / `PyChannel` | 用 `new_channel` + `@chan.build.command()` 定义命令。参考 `apps/bodies/g1/g1_channel.py` 的写法（loco channel 的 `move(vx,vy,vyaw)` 与我们语义天然一致）。 |
| Matrix 注册 | `ghoshell_moss.core.blueprint.matrix.Matrix` | `main.py` 中 `matrix.provide_channel(channel)`，参考 `apps/bodies/g1/main.py`。 |
| app 元信息 | `apps/bodies/g1/APP.md` | 新 app 照此格式写 `APP.md`（Circus 生命周期）。 |

> 关键事实：dev 分支的 `apps/bodies/g1/main.py` 当前是**阶段 A 空壳**——只有 `new_channel` 和一条 `instruction`，无任何命令。我们**不改它**，另起一个纯软新 app。

---

## 3. 核心架构决策（最重要的一节）

### 3.1 决策一：新建独立 app，不污染 g1 真机 app

- 路径：`.moss_ws/apps/bodies/g1_sim/`
- 理由：与硬件无关；g1 app 走真机 SDK，混在一起会引入 DDS/CycloneDDS 等不必要依赖。

### 3.2 决策二：locomotion 用"速度指令 + 高频闭环"模型，**不复用 Trajectory**

这是全文最关键的判断，实现 Agent 必须理解：

- 现有 `RobotController` 的 `run_trajectory` / `move_to_pose` / `add_trajectory_actions` 是**"一次性轨迹回放"**模型：给一条预先算好的关节序列，执行到底。这适合机械臂、关键帧动画。
- 但 MLP locomotion 是**"持续速度控制 + 每拍闭环"**：没有"终点轨迹"，只有一个**当前速度指令 (vx, vy, vyaw)**，控制循环每一拍读"当前观测 + 当前指令"→ MLP → 关节目标 → 推进一帧。机器人会一直走，直到你改指令或归零。
- 因此：**channel 命令的职责不是"算轨迹"，而是"修改一个目标速度指令变量"**。真正持续动关节的是后台一直在转的控制循环。

> 这与 `g1_channel.py` 中 `move(vx, vy, vyaw, continuous)` 的语义完全一致——设速度，不设轨迹。

### 3.3 决策三：定义新接口 `VelocityRobotController`，与 `RobotController` 平级

不强行继承面向 Trajectory 的 `RobotController`，而是新建一个贴合 locomotion 语义的抽象（保留同样的"接缝解耦"哲学）：

```
VelocityRobotController (ABC)
  ├─ start() / close() / closed()
  ├─ set_velocity_command(vx, vy, vyaw)   # channel 命令调它（非阻塞，只改状态）
  ├─ stop()                                # 指令归零
  ├─ get_observation() -> obs vector       # 调试用
  ├─ get_base_state() -> 位姿/速度          # 状态查询/instruction 用
  └─ (内部) _control_loop()                # 守护线程，50Hz，读 obs+cmd → MLP → ctrl → mj_step
```

> 若后续要兼容关键帧动画/位姿命令，可让具体实现**同时**满足 `RobotController`；但本期 locomotion demo 只需要 `VelocityRobotController`。

### 3.4 决策四：两阶段路线（先点亮管道，再换 G1）

| 阶段 | 仿真+策略 | 目的 | 验收 |
|---|---|---|---|
| **M0 管道点亮** | SB3 RL Zoo `Humanoid-v4`（火柴人） | 验证 语音→CTML→channel→推理→MuJoCo 有东西在动 | 说"走"，火柴人动起来 |
| **M1 正式演示** | `unitree_rl_gym` G1 预训练策略 + G1 MJCF | 真 G1 听话行走 | 说"往前走/左转/停"，G1 正确响应 |

---

## 4. 总体架构（分层）

```
┌─────────────────────────────────────────────────────────┐
│  用户 (语音/文字): "往前走两步"                              │
└───────────────────────┬─────────────────────────────────┘
                        │ ASR + 大模型理解
                        ▼
┌─────────────────────────────────────────────────────────┐
│  CTML 指令: <bodies_g1_sim:walk vx="0.5" duration="2.0" /> │
└───────────────────────┬─────────────────────────────────┘
                        │ Matrix 总线路由
                        ▼
┌─────────────────────────────────────────────────────────┐
│  Channel 层 (g1_sim_channel.py)                            │
│  命令: walk / move / turn / stop / state ...               │
│  职责: 把语义命令翻译成 set_velocity_command(...)           │
│        —— 只改"目标速度指令"，不算关节                       │
└───────────────────────┬─────────────────────────────────┘
                        │ contract 绑定
                        ▼
┌─────────────────────────────────────────────────────────┐
│  控制层 (MujocoVelocityController : VelocityRobotController)│
│   ┌─────────────────────────────────────────────────┐    │
│   │ 后台守护线程 _control_loop() @ 50Hz               │    │
│   │  while running:                                   │    │
│   │    obs   = build_observation(sim_state, cmd)      │    │
│   │    act   = policy(obs)        # MLP 推理           │    │
│   │    ctrl  = action_to_ctrl(act)                    │    │
│   │    sim.data.ctrl[:] = ctrl                        │    │
│   │    mj_step(model, data) × N                        │    │
│   └─────────────────────────────────────────────────┘    │
│   持有: MuJoCo model/data + 预训练 policy + 当前 cmd        │
└───────────────────────┬─────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────┐
│  MuJoCo 仿真 + viewer (G1 MJCF 模型，画面)                  │
└─────────────────────────────────────────────────────────┘
```

> 注意：`MOSSRobotManager`/`RobotInfo`（形体描述）在本架构中是**旁路**——用于生成给大模型看的 `robot_state()` 文字描述（有哪些关节、当前位姿），不参与高频控制回路。

---

## 5. 数据流与时序

### 5.1 指令下发（非阻塞）
```
walk(vx=0.5, duration=2.0)
  → controller.set_velocity_command(0.5, 0, 0)   # 立即返回
  → (可选) 起一个 asyncio 定时器，duration 后 controller.stop()
```

### 5.2 控制循环（后台线程，与指令异步）
```
每 20ms (50Hz):
  1. 从 MuJoCo 读 base 角速度、投影重力、各关节角/角速度
  2. 拼接观测向量 = [base_ang_vel, proj_gravity, current_cmd, dof_pos, dof_vel, last_action, (phase)]
  3. policy(obs) → action (各关节目标增量/位置)
  4. action → ctrl（按 config 的 action_scale + default_angles 还原）
  5. data.ctrl[:] = ctrl
  6. mj_step × decimation 次
  7. last_action = action
  8. viewer.sync()
```

### 5.3 "走两步"如何实现
- 模型**不理解"两步"**。`walk(vx=0.5, duration=2.0)` = 设前进速度 0.5 m/s，持续 2 秒后归零。
- "几步"由 `duration × 步频` 间接决定。第一版**只保证方向和大致时长正确**即可。

---

## 6. 模块详细设计

### 6.1 目录结构（最终交付）

```
.moss_ws/apps/bodies/g1_sim/
├── APP.md                  # Circus 元信息（参考 g1/APP.md）
├── DESIGN.md               # 本文件
├── README.md               # 安装/运行说明
├── main.py                 # 进程入口：build channel → matrix.provide_channel
├── g1_sim_channel.py       # channel 定义（命令 + instruction）
├── pyproject.toml          # 依赖：mujoco, numpy, torch(或onnxruntime)
├── control/
│   ├── __init__.py
│   ├── interface.py        # VelocityRobotController (ABC)
│   ├── mujoco_controller.py# MujocoVelocityController 实现 + 控制循环线程
│   ├── policy.py           # 策略加载与推理封装 (PolicyRunner)
│   └── obs.py              # 观测向量构造 + 动作还原（对齐逻辑集中于此）
├── assets/
│   ├── g1/                 # G1 的 MJCF + mesh（来自 menagerie / unitree_rl_gym）
│   └── policies/
│       ├── g1_motion.pt    # G1 预训练权重
│       └── humanoid_v4.zip # M0 用的 SB3 agent
└── config/
    ├── g1.yaml             # 观测/动作规格（从 unitree_rl_gym 抄并核对）
    └── humanoid_v4.yaml    # M0 配置
```

### 6.2 `control/interface.py` — VelocityRobotController

抽象接口，定义控制层与 channel 层的唯一握手点。方法见 §3.3。要点：
- `set_velocity_command` / `stop` **必须非阻塞**（只写一个被锁保护的状态变量）。
- 线程安全：当前指令用 `threading.Lock` 保护（参考 `ros2_controller.py` 的 `_joint_positions_lock`）。
- 生命周期：`start()` 启动控制线程 + viewer；`close()` 置停止事件、join 线程、关 viewer。

### 6.3 `control/policy.py` — PolicyRunner

封装"加载权重 + 推理"，对上层只暴露 `act(obs: np.ndarray) -> np.ndarray`。
- **格式抉择**：
  - `unitree_rl_gym` 导出的是 TorchScript `.pt`（`torch.jit.load`），输入输出是 tensor。推理用 CPU 即可（小 MLP，微秒级）。
  - M0 的 SB3 agent 是 `.zip`，用 `stable_baselines3` 的 `model.predict(obs)`。
  - 为隔离差异，`PolicyRunner` 做成接口，下面两个实现：`TorchScriptPolicy`、`SB3Policy`。
- 推理无梯度：`torch.no_grad()`。

### 6.4 `control/obs.py` — 观测/动作对齐（核心难点，见 §10）

集中放置：
- `build_observation(model, data, cmd, last_action, cfg) -> np.ndarray`
- `action_to_ctrl(action, cfg) -> np.ndarray`
- 关节顺序映射表（MuJoCo 关节顺序 ↔ 策略训练时的关节顺序）。

**所有"魔法数字"集中在 config，对齐逻辑集中在此文件**，便于调试和换模型。

### 6.5 `control/mujoco_controller.py` — MujocoVelocityController

```python
class MujocoVelocityController(VelocityRobotController):
    def __init__(self, model_path, policy: PolicyRunner, cfg, *, render=True):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data  = mujoco.MjData(self.model)
        self.policy = policy
        self.cfg = cfg
        self._cmd = (0.0, 0.0, 0.0)
        self._cmd_lock = threading.Lock()
        self._last_action = np.zeros(cfg.num_actions)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._control_loop, daemon=True)
        self._viewer = None
        self._render = render

    def set_velocity_command(self, vx, vy, vyaw):
        with self._cmd_lock:
            self._cmd = (vx, vy, vyaw)

    def stop(self):
        self.set_velocity_command(0.0, 0.0, 0.0)

    def _control_loop(self):
        # 初始化到默认站姿 default_angles
        self.data.qpos[7:] = self.cfg.default_angles
        mujoco.mj_forward(self.model, self.data)
        dt = self.cfg.sim_dt * self.cfg.decimation   # 控制周期
        while not self._stop.is_set():
            t0 = time.time()
            with self._cmd_lock:
                cmd = self._cmd
            obs = build_observation(self.model, self.data, cmd, self._last_action, self.cfg)
            action = self.policy.act(obs)
            self._last_action = action
            ctrl = action_to_ctrl(action, self.cfg)
            self.data.ctrl[:] = ctrl
            for _ in range(self.cfg.decimation):
                mujoco.mj_step(self.model, self.data)
            if self._viewer is not None:
                self._viewer.sync()
            # 控制节拍
            sleep = dt - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)
```

> 线程模型直接对标 `ros2_controller.py`：守护线程 + 停止事件 + 锁。区别是它不是消费 trajectory 队列，而是稳定地跑闭环。

### 6.6 `g1_sim_channel.py` — Channel 定义

```python
def build_g1_sim_channel(controller: VelocityRobotController) -> PyChannel:
    chan = PyChannel(name="bodies_g1_sim", blocking=False)
    chan.build.with_binding(VelocityRobotController, controller)

    @chan.build.command()
    async def walk(vx: float = 0.4, duration: float = 2.0) -> str:
        """让机器人向前/后行走。vx>0 前进, vx<0 后退。duration 秒后自动停。"""
        c = ChannelCtx.get_contract(VelocityRobotController)
        c.set_velocity_command(vx, 0.0, 0.0)
        if duration > 0:
            await asyncio.sleep(duration)
            c.stop()
        return "ok"

    @chan.build.command()
    async def turn(vyaw: float = 0.5, duration: float = 1.5) -> str:
        """原地转向。vyaw>0 左转, vyaw<0 右转。"""
        ...

    @chan.build.command()
    async def move(vx: float, vy: float, vyaw: float, duration: float = 1.0) -> str:
        """底层速度控制：同时设定前后/横移/转向速度。"""
        ...

    @chan.build.command()
    async def stop() -> str:
        """立即停止移动（速度归零）。"""
        ...

    @chan.build.command()
    async def state() -> str:
        """返回机器人当前状态（朝向、是否在移动）给大模型参考。"""
        ...

    chan.build.instruction(
        "G1 纯仿真身体控制。仅支持平地移动类动作：walk/turn/move/stop。"
        "不支持蹲下/起立/跳跃（当前策略不具备）。"
        "'往前走'用 walk(vx>0)，'后退'用 walk(vx<0)，'左/右转'用 turn。"
        "speed 单位 m/s，建议 vx 不超过 0.6，vyaw 不超过 0.8，过大会摔倒。"
    )
    return chan
```

> 命令集 = §1.3"能力边界决定指令集"：当前只放移动类。蹲/起立等留待策略支持后再加。

### 6.7 `main.py` — 进程入口

```python
async def main(matrix: Matrix):
    cfg = load_config("config/g1.yaml")
    policy = TorchScriptPolicy("assets/policies/g1_motion.pt")
    controller = MujocoVelocityController("assets/g1/scene.xml", policy, cfg, render=True)
    controller.start()
    channel = build_g1_sim_channel(controller)
    await matrix.provide_channel(channel)

if __name__ == "__main__":
    Matrix.discover().run(main)
```

---

## 7. 选型与素材来源（结论复述）

| 用途 | 选定方案 | 来源 |
|---|---|---|
| M1 正式演示策略 + 推理脚本 | `unitree_rl_gym`（自带 `deploy/pre_train/g1/motion.pt` + `deploy/deploy_mujoco/` 纯仿真推理脚本，照搬其观测/动作逻辑） | [[1]](https://github.com/unitreerobotics/unitree_rl_gym) |
| G1 机器人 MJCF 外形 | MuJoCo Menagerie 的 Unitree G1（Apache-2.0）；或直接用 unitree_rl_gym 自带的 MJCF | [[2]](https://github.com/google-deepmind/mujoco_menagerie) |
| M0 管道点亮策略 | SB3 RL Zoo 的 `Humanoid-v4` 预训练 agent（`pip install rl_zoo3`，`enjoy` 一条命令验证） | [[3]](https://stable-baselines3.readthedocs.io/en/master/guide/rl_zoo.html) |
| 仿真引擎 | MuJoCo 官方 Python 包（`pip install mujoco`，自带 `mujoco.viewer`） | [[4]](https://gymnasium.farama.org/environments/mujoco/humanoid) |

> 为何不选 MuJoCo Playground / PyBullet / Genesis / Isaac：在"纯软 + 最快出 demo + 画面最像 G1"目标下，unitree_rl_gym 把"权重+推理脚本+G1 外形+速度指令接口"四件套打包齐全，集成成本最低；其余方案要么是 JAX 训练框架（Playground 偏重），要么外形不像 G1（SB3 Humanoid 仅作管道验证）。

---

## 8. 依赖与环境

```toml
# pyproject.toml (节选)
dependencies = [
  "mujoco>=3.0",
  "numpy",
  "torch",            # TorchScript 策略推理；如权重是 onnx 则换 onnxruntime
  "pyyaml",
  # M0 阶段额外:
  "stable-baselines3", "rl_zoo3", "gymnasium[mujoco]",
]
```

- **CPU 即可**：MLP 推理 + 单体 MuJoCo 仿真，CPU 实时跑得动；GPU 非必需。
- **无网络监听**：viewer 是本地窗口，不起任何服务端口。
- 安装遵循仓库约定（`uv` / `pip install --user`）。

---

## 9. CTML 联动设计

大模型看到的"接口"= channel 命令签名 + instruction。映射示例：

| 自然语言 | CTML | 落到 |
|---|---|---|
| 往前走两步 | `<bodies_g1_sim:walk vx="0.5" duration="2.0" />` | `set_velocity_command(0.5,0,0)` 2s 后 stop |
| 后退一点 | `<bodies_g1_sim:walk vx="-0.3" duration="1.0" />` | 反向 |
| 向左转 | `<bodies_g1_sim:turn vyaw="0.5" duration="1.5" />` | 转向 |
| 停 | `<bodies_g1_sim:stop />` | 归零 |

instruction 必须**显式声明能力边界**（只会移动、不会蹲起跳），否则大模型会臆造不存在的动作。

---

## 10. 对齐检查清单（★ 唯一真正的技术难点 ★）

预训练策略假设了一套**固定的观测/动作约定**。喂错一位，机器人立刻摔。**实现时必须逐项核对 unitree_rl_gym 的实际 config，不要相信下表数字**（仅为预期量级）。

### 10.1 观测向量（obs）—— 逐项核对

| 段 | 预期含义 | 预期维度 | 核对点 |
|---|---|---|---|
| base 角速度 | IMU 角速度，机体系 | 3 | 缩放系数 `ang_vel_scale`？ |
| 投影重力 | 重力在机体系投影（表示躯干姿态） | 3 | 如何从 quat 算的？ |
| 速度指令 | (vx, vy, vyaw) | 3 | **指令缩放**？单位 m/s vs 归一化？ |
| 关节角 | (qpos - default_angles) × scale | ~12 | **关节顺序**！MuJoCo 顺序 ≠ 策略顺序？ |
| 关节角速度 | qvel × scale | ~12 | scale？ |
| 上一步动作 | last_action | ~12 | 第一拍用零向量 |
| 相位 | sin/cos(phase) 等 | 0~2 | 部分策略有步态相位 |
| **合计** | | **≈47** | **必须等于策略 `num_obs`** |

### 10.2 动作（action）→ 控制量（ctrl）
- 策略输出通常是**关节目标位置的偏移**：`target = default_angles + action × action_scale`。
- ctrl 可能是位置（PD 内置于 actuator）或力矩——**看 MJCF 的 actuator 类型**。
- 动作维度（如 ~12，只控腿，手臂冻结）必须等于 `num_actions`。

### 10.3 关键参数（从 config 抄）
`sim_dt`、`decimation`（每个控制步 step 几次物理）、`default_angles`、各 `*_scale`、`action_scale`、`num_obs`、`num_actions`、关节名顺序列表。

### 10.4 对齐自检流程（强制）
1. **静止自检**：cmd=0 启动，机器人应**稳稳站立 ≥10 秒**不倒。倒了 → 观测/default_angles/关节顺序错。
2. **直行自检**：cmd=(0.3,0,0)，应平稳前进。原地抽搐/打转 → 关节顺序或 scale 错。
3. **对照 deploy_mujoco**：先**原样跑通** unitree_rl_gym 自带的 `deploy_mujoco` 脚本，再把其观测/动作代码逐行搬进 `obs.py`，**逐字段 diff**。这是最快、最不易错的路径。

---

## 11. 测试与验收

### 11.1 单元/集成
- `PolicyRunner`：给定 mock obs，输出 shape == num_actions，无异常。
- `build_observation`：给定固定 sim 状态，输出 shape == num_obs。
- 控制器：`start()` 后线程在跑、`stop()` 后速度归零、`close()` 干净退出（无悬挂线程）。

### 11.2 阶段验收
- **M0 通过条件**：在 MOSS 里发出"走"指令 → SB3 Humanoid 在 MuJoCo 窗口里动起来。证明 CTML→channel→推理→仿真 链路打通。
- **M1 通过条件**：
  1. 静止自检：G1 站立 10s 不倒。
  2. 说"往前走" → G1 朝前行走；说"停" → 停下；说"左转" → 左转。
  3. 大模型能仅凭 instruction 正确选择 walk/turn/stop 并填合理参数。

### 11.3 演示脚本（建议录制）
"往前走 → 停 → 左转 → 往前走 → 停"，一镜到底，证明自然语言闭环。

---

## 12. 任务拆解（WBS，可直接转 Task 列表）

### Phase M0 — 点亮管道
1. 新建 `apps/bodies/g1_sim/` 骨架（APP.md / main.py / pyproject.toml）。
2. 写 `VelocityRobotController` 接口（`control/interface.py`）。
3. 写 `SB3Policy`（加载 RL Zoo Humanoid-v4 agent）。
4. 写一个**最简 MuJoCo 控制器**跑 Humanoid-v4（用 gymnasium env 或裸 MuJoCo 均可），接 `set_velocity_command`（Humanoid 无速度指令，可用"是否运动"开关近似）。
5. 写 `g1_sim_channel.py`（walk/stop 两个命令 + instruction）。
6. `main.py` 接 Matrix，`provide_channel`。
7. **验收 M0**：发指令 → 火柴人动。

### Phase M1 — G1 正式演示
8. clone `unitree_rl_gym`，本地**原样跑通** `deploy_mujoco`（基准）。
9. 拷贝 G1 MJCF + `motion.pt` 到 `assets/`。
10. 从其代码抄出 `config/g1.yaml`（§10.3 所有参数 + 关节顺序）。
11. 写 `control/obs.py`（`build_observation` / `action_to_ctrl`），**逐字段 diff 基准脚本**。
12. 写 `TorchScriptPolicy`。
13. 写 `MujocoVelocityController` 完整版（§6.5）。
14. **静止自检**（站 10s）→ **直行自检** → 调通。
15. channel 扩展为 walk/turn/move/stop/state，完善 instruction。
16. **验收 M1**：自然语言闭环 demo。

### Phase M2（可选，超出本期）
17. 若策略支持：加 squat/stand 命令。
18. 上肢手势：用 `models.py` 的 `Animation` 录关键帧（需确认有下肢平衡兜底）。
19. 评估升级到 MuJoCo Playground 多策略。

---

## 13. 风险登记

| 风险 | 等级 | 缓解 |
|---|---|---|
| 观测/动作对齐错 → 机器人摔 | 高 | §10 强制以基准脚本逐字段 diff；先过静止自检 |
| 写死维度数字与实际 config 不符 | 高 | 一切数值以仓库 config 为准；本文数字仅参考 |
| 控制循环节拍不稳 → 抖动 | 中 | 固定 50Hz；`time.sleep` 补偿；必要时实时性让步给确定性 |
| viewer 阻塞主线程 | 中 | viewer 在控制线程内 sync，或用 passive viewer |
| 策略只控腿、手臂下垂不好看 | 低 | 演示可接受；后续 M2 处理 |
| `Trajectory` 抽象被误用于 locomotion | 中 | §3.2 明确：locomotion 走速度指令，不走 Trajectory |

---

## 14. 附录

### 14.1 与 g1 真机 app 的关系
**无关**。本 app 纯软；g1 真机 app 走 SDK/DDS。两者只是恰好都建模 G1 这个身体，未来可共享 `RobotInfo` 形体描述，但控制后端完全独立。

### 14.2 术语
- **locomotion 策略**：把"速度指令 + 本体观测"映射成"关节目标"的小型神经网络（MLP），高频闭环运行。
- **decimation**：每个控制步执行多少次物理仿真步。
- **default_angles**：机器人默认站姿的各关节角，动作通常是相对它的偏移。
- **投影重力**：重力向量在机体坐标系下的投影，用来让策略感知躯干倾斜。

### 14.3 关键参考
- 策略与纯仿真推理脚本 [[1]](https://github.com/unitreerobotics/unitree_rl_gym)
- G1 MJCF 外形 [[2]](https://github.com/google-deepmind/mujoco_menagerie)
- M0 管道验证 [[3]](https://stable-baselines3.readthedocs.io/en/master/guide/rl_zoo.html)
- MuJoCo / Humanoid 环境 [[4]](https://gymnasium.farama.org/environments/mujoco/humanoid)
- 复用范式：`prototypes/ros2_robot/abcd.py`、`ros2_controller.py`、`mocks.py`、`manager.py`、`models.py`；channel 写法 `apps/bodies/g1/g1_channel.py`、`main.py`

---

文档同时也保存在仓库里 `.moss_ws/apps/bodies/g1_sim/DESIGN.md`,你本地能直接打开。需要我按 M0 的 WBS 开始落地骨架代码吗?

## References
1. [unitreerobotics/unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym)
2. [google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie)
3. [RL Baselines3 Zoo](https://stable-baselines3.readthedocs.io/en/master/guide/rl_zoo.html)
4. [Humanoid - Gymnasium](https://gymnasium.farama.org/environments/mujoco/humanoid)
