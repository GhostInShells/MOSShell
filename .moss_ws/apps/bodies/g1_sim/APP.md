---
executable: uv
script: main.py
arguments: ''
description: 'G1 纯软件仿真身体控制通道，支持 stand-first 高层语义与基于策略的移动控制'
respawn: false
workers: 1
---

G1 pure software simulation app.

默认提供 `apps.bodies_g1_sim` 通道，使用高层语义 + 速度命令驱动后台控制循环，而不是一次性轨迹回放。

## 文档入口

- `DESIGN.md`: 完整技术设计与实现蓝图
- `README.md`: 当前实现、运行方式、资产准备、已知问题
- `VOICE_MOTION_CTML.md`: 语音播报 + 动作编排模板
- `REVIEW_NOTES.md`: 代码审查与问题清单

## 能力

- 纯软件 locomotion 演示
- `prepare / recover / reset / stand / forward / backward / left / right / go_forward / go_backward / keep_left / keep_right / walk / turn / move / stop / end_showcase / state / health` 命令
- M0: `Humanoid-v4` 管线验证
- M1: 预留 `unitree_rl_gym` G1 MuJoCo 模型与 TorchScript 策略接入口

## 启动

```ctml
<apps:start fullname="bodies/g1_sim" />
<bodies_g1_sim:prepare />
<bodies_g1_sim:forward speed="0.4" duration="2.0" />
<bodies_g1_sim:left speed="0.5" duration="1.5" />
<bodies_g1_sim:stop />
```

## 高层语义约定

- 自然表达“站稳、恢复、重新站好、别动、准备好再动”时，优先使用 `prepare()`、`recover()` 或 `stand()`
- 自然表达“往前走、后退、左转、右转”时，优先使用 `forward()/backward()/left()/right()`
- 自然表达“**一直**往前走、持续前进、继续往前走直到我说停”时，优先使用 `go_forward()`
- 自然表达“**一直**后退、持续后退”时，优先使用 `go_backward()`
- 自然表达“**一直**左转 / 右转、持续转向”时，优先使用 `keep_left()/keep_right()`
- 自然表达“停下、停止、结束展示、结束表演、结束巡逻”时，优先使用 `stop()` 或 `end_showcase()`
- 只有在明确需要同时控制前进、横移和转向时，才使用 `move(vx, vy, vyaw, duration)`
- 如果机器人已经跌倒或仍在 reset 阶段，先调用 `prepare()/recover()/reset()/stand()`，不要直接发移动命令
- 如果你希望“恢复完成后立刻开始动作”，优先使用 `prepare()`，它会等到机器人真正 `ready` 再返回

## 麦克风指挥

- MOSS 里已经有现成的麦克风链路，不需要为了 `g1_sim` 额外再造一个新的麦克风 app
- 连续监听链路：`sensors/audio_capture` + `aether/listener`
- 按键说话链路：`sensors/ptt_listener`
- `listener / ptt_listener` 会把用户语音识别成文本，发布成 `SpeechTopic` 并发出 `AudioSignal`
- 默认 Ghost 会通过 `audio_nucleus` 接收这些语音输入，再结合 `apps.bodies_g1_sim` 的动态接口与说明，把自然语言转成 CTML 调用

推荐启动顺序：

```ctml
<apps:start fullname="bodies/g1_sim" />
<apps:start fullname="sensors/audio_capture" />
<apps:start fullname="aether/listener" />
```

如果你更想避免环境噪声误触发，推荐先用 PTT：

```ctml
<apps:start fullname="bodies/g1_sim" />
<apps:start fullname="sensors/ptt_listener" />
```

自然语言示例：

- “往前走” -> `forward(...)` 或 `go_forward()`
- “一直往前走” -> `go_forward()`
- “左转” -> `left(...)`
- “一直右转” -> `keep_right()`
- “停下” -> `stop()`
- “结束展示” -> `end_showcase()`

## 语音 + 动作

- 语音先说、再驱动机器人动作的 CTML 模板，见 `VOICE_MOTION_CTML.md`
- 推荐结构：`prepare() -> say -> motion -> say -> motion -> stop()`

## 配置

- 默认读取 `config/humanoid_v4.yaml`
- 可通过环境变量 `G1_SIM_PROFILE=g1` 切换到 `config/g1.yaml`
