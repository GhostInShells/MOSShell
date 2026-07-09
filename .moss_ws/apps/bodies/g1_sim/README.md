# G1 Sim

`g1_sim` 是一个面向 MOSS 的 G1 纯软件仿真 app。

它的定位是:

- 只做 MuJoCo 仿真，不接真机 SDK
- 用高层 channel 语义驱动后台控制循环
- 支持从 CTML、自然语言和后续麦克风输入映射到机器人动作

## 文档分工

- `APP.md`: 面向使用者和模型的能力说明、启动方式、自然语言映射
- `README.md`: 面向开发者的当前实现说明、运行方式、资产准备
- `DESIGN.md`: 这个 app 的完整技术设计文档和实现蓝图
- `VOICE_MOTION_CTML.md`: 语音播报 + 动作编排模板
- `REVIEW_NOTES.md`: 当前代码审查记录与问题清单

## 当前架构

主链路如下:

1. `main.py` 选择配置、加载策略、启动控制器
2. `g1_sim_channel.py` 暴露高层命令
3. `MujocoVelocityController` 在后台线程中持续执行控制循环
4. 控制循环读取当前速度命令、拼接观测、调用策略、生成 PD 力矩、推进 MuJoCo

设计原则:

- channel 只负责改“目标速度指令”或恢复指令
- 高频关节控制全部放在后台控制循环
- M0 与 M1 共用同一套 channel / controller 接缝

## 当前实现

- 已落地独立 app: `.moss_ws/apps/bodies/g1_sim/`
- 已实现 `VelocityRobotController` 抽象: [interface.py](file:///Users/lipeng/TraeProject/MOSShell/.moss_ws/apps/bodies/g1_sim/control/interface.py)
- 已实现 `MujocoVelocityController`: [mujoco_controller.py](file:///Users/lipeng/TraeProject/MOSShell/.moss_ws/apps/bodies/g1_sim/control/mujoco_controller.py)
- 已实现两条后端路径:
- `gym_humanoid_v4`: 用于 M0 管道点亮和结构验证
- `mujoco_g1`: 用于 M1 的 G1 模型、47 维观测、12 维动作和 PD 控制
- 已实现 stand-first 高层语义:
- `prepare / recover / reset / stand`
- 已实现短时动作语义:
- `forward / backward / left / right / walk / turn / move / stop`
- 已实现持续动作语义:
- `go_forward / go_backward / keep_left / keep_right / end_showcase`
- 已实现 viewer 自动跟随相机与稳定地面视角
- 已实现语音编排模板和精准同步版 CTML
- 已接好麦克风控制的系统侧前提:
- `sensors/audio_capture + sensors/listener`
- `sensors/ptt_listener`

## 运行方式

### 直接运行

```bash
cd .moss_ws/apps/bodies/g1_sim
uv run main.py
```

当前默认 profile 选择逻辑:

- 如果 G1 资产齐全，默认走 `config/g1.yaml`
- 否则回退到 `config/humanoid_v4.yaml`

### 指定 profile

```bash
cd .moss_ws/apps/bodies/g1_sim
G1_SIM_PROFILE=g1 uv run main.py
```

或者:

```bash
cd .moss_ws/apps/bodies/g1_sim
G1_SIM_PROFILE=humanoid_v4 uv run main.py
```

### 通过 MOSS 启动

```ctml
<apps:start fullname="bodies/g1_sim" />
<apps.bodies_g1_sim:prepare />
<apps.bodies_g1_sim:forward speed="0.4" duration="2.0" />
<apps.bodies_g1_sim:left speed="0.5" duration="1.5" />
<apps.bodies_g1_sim:stop />
```

## 命令语义

### 恢复与站稳

- `prepare(timeout, poll)`: 先恢复，再等待 ready
- `recover() / stand()`: 进入站稳优先模式
- `reset()`: 强制 reset 到标准站姿

### 短时动作

- `forward(speed, duration)`
- `backward(speed, duration)`
- `left(speed, duration)`
- `right(speed, duration)`
- `walk(vx, duration)`
- `turn(vyaw, duration)`
- `move(vx, vy, vyaw, duration)`
- `stop()`

### 持续动作

- `go_forward(speed)`: 一直前进，直到收到停止或新的动作
- `go_backward(speed)`: 一直后退
- `keep_left(speed)`: 一直左转
- `keep_right(speed)`: 一直右转
- `end_showcase()`: 结束展示/巡逻/表演，立即停下并回到站稳

### 状态观察

- `health()`: 返回 ready / fallen / phase / reason / base_height
- `state()`: 返回命令、位姿、观测维度、错误信息

## 语音与自然语言

推荐映射:

- “往前走” -> `forward(...)`
- “一直往前走” -> `go_forward()`
- “左转” -> `left(...)`
- “一直右转” -> `keep_right()`
- “停下” -> `stop()`
- “结束展示” -> `end_showcase()`

如果需要“先说，再动”的展示脚本，请看 `VOICE_MOTION_CTML.md`。

如果需要麦克风联调，请优先看 `APP.md` 里的“麦克风指挥”部分。

## 资产准备

首次准备 M1 资产:

```bash
cd .moss_ws/apps/bodies/g1_sim
uv run python sync_unitree_g1_assets.py
```

要得到完整 G1 演示效果，需要:

- `assets/g1/scene.xml`
- `assets/g1/g1_12dof.xml`
- `assets/g1/meshes/*.STL`
- `assets/policies/g1_motion.pt`

这些资产可以通过 `sync_unitree_g1_assets.py` 自动下载。

## 策略与降级行为

- M0 默认可使用内置 `demo` 策略做管道点亮
- 如果配置中策略文件缺失，会回退到 `ZeroPolicy`
- 回退后 app 仍能注册 channel，但不会产生有效 locomotion

## 依赖

`pyproject.toml` 已声明核心依赖:

- `ghoshell-moss[host]`
- `numpy`
- `pyyaml`
- `mujoco`
- `gymnasium[mujoco]`
- `torch`

## 已知问题

当前审查中已确认的主要问题，见 `REVIEW_NOTES.md`:

- 定时动作还不能被同通道命令即时打断
- M1 的 `yaw` 状态读数目前并不准确
