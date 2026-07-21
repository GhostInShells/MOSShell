# G1 Sim Review Notes

日期: 2026-06-16

## Intent

`g1_sim` 的目标是提供一个独立的 G1 纯软件仿真 app, 用高层 channel 语义驱动后台控制循环, 并支持后续接入语音/自然语言控制。

核心设计意图:

- `main.py` 负责选择 profile、加载策略、启动控制器并向 Matrix 提供 channel
- `control/mujoco_controller.py` 负责持续控制循环、状态机、跌倒检测、PD 控制、viewer 跟随相机
- `g1_sim_channel.py` 负责把高层语义映射成速度指令和恢复指令
- `APP.md / README.md / VOICE_MOTION_CTML.md` 负责面向使用者的操作说明

## 架构概览

- M0 路径: `gym_humanoid_v4`
- M1 路径: `mujoco_g1`
- 控制模式: `channel -> VelocityRobotController -> control loop -> policy -> PD -> MuJoCo`
- 高层语义: `prepare / recover / reset / stand / forward / backward / left / right / go_forward / go_backward / keep_left / keep_right / stop / end_showcase`

## Findings

### 1. High: 定时动作不可被及时打断

位置:

- `g1_sim_channel.py`

说明:

- `_run_for_duration()` 在收到定时动作后会直接 `await asyncio.sleep(duration)`, 期间当前命令一直占用 `apps.bodies_g1_sim` 通道
- 这意味着 `forward(..., duration=3.0)`、`left(..., duration=2.0)` 这类命令在运行时, 后续同通道的 `stop()` / `end_showcase()` / 新动作命令无法立刻抢占
- 对“语音指挥”场景尤其明显: 如果自然语言被映射成定时动作, 用户中途再说“停下”, 也只能等当前 sleep 结束后再处理

影响:

- 交互性下降
- 语音指挥时用户感知为“机器人不听停”
- 也会让“结束展示”这类高优先级口令不能即时生效

建议:

- 不要在 channel 命令里 `sleep(duration)` 持有通道
- 改成由 controller 内部维护“动作截止时间”或“当前 motion session”, 让通道命令尽快返回
- `stop()` / `end_showcase()` 应该能覆盖当前 session

### 2. High: M1 的 yaw 状态读数错误

位置:

- `control/mujoco_controller.py`

说明:

- 在 M1 路径里, `BaseState.yaw` 直接取了 `data.qpos[6]`
- 但 free joint 的 `qpos[3:7]` 是四元数 `[w, x, y, z]`
- `qpos[6]` 只是四元数的 `z` 分量, 不是欧拉角 yaw

影响:

- `state()` / `summary()` 输出的 yaw 不可信
- 调试、展示、路线分析时会误判朝向
- 如果后续基于 `yaw` 做更高层语义或自动镜头/导航逻辑, 会埋下隐患

建议:

- 从 `qpos[3:7]` 四元数正确转换成 yaw
- 保证 `state.summary()`、`snapshot_to_dict()` 里的角度语义一致

## Documentation Gaps

### README.md 需要更新

当前问题:

- 仍写着“已实现 `walk / turn / move / stop / state` channel”, 与当前实际实现不一致
- 仍写着默认使用 M0 配置, 但当前 `main.py` 在资产齐全时默认会走 `g1`
- 没有覆盖 stand-first、持续动作语义、麦克风指挥接入

建议补充:

- 当前完整命令集
- 默认 profile 选择逻辑
- `listener / ptt_listener` 的接入说明
- “语音自然语言 -> g1_sim 高层命令”的推荐映射

### APP.md 仍可继续打磨

虽然已经比 README 新, 但还可以继续补:

- “定时动作”和“持续动作”的区别
- `stop()` 与 `end_showcase()` 的语义差别
- 对语音场景的推荐映射优先级
- 一个“最小可用麦克风联调流程”

## Recommended Priority

1. 先修复“定时动作不可及时打断”
2. 再修复“M1 yaw 读数错误”
3. 然后统一补 README / APP 文档

## Residual Risks

- `end_showcase()` 这类立即停机命令当前是异步线程消费, 紧接着查询 `health()` 可能看到短暂旧状态
- 语音控制链路虽然架构已接好, 但还缺一次真实麦克风联调来验证 Ghost 的自然语言映射质量
