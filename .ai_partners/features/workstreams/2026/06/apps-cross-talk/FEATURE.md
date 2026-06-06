# FEATURE: apps-cross-talk

> status: partial
> updated: 2026-06-06
> owner: human
> codename: apps-cross-talk
> tags: [wip, parallel, mindflow]
> layer: "3"

## What

让 MOSS Workspace 下的多个 App 能够并行运行、相互对话，并共享数据与 buffer。同时支持 Ghost 通过 CTML 控制其他 App，使得 Ghost 具备 Agent 特质（执行长程任务、使用工具）。

## 起因

[voice.md](../../../../../../.moss_ws/apps/sensors/voice/FEATURE.md) — 碎碎念。

Voice 在输出语音时会打碎自身的输入（录音、VAD、ASR），所以需要 2 个进程。它需要跨进程通信和 buffer 共享机制。

随着更多传感器 app 加入（vision 人脸识别、gomoku 棋局），app 间通信的需求泛化了：传感器 app 产生的数据需要流式传递给显示/交互 app（ai_eye），且各 app 之间的通信不应阻塞 Ghost 的主对话循环。

## 评估

MOSS 在各个层都有 hook，可以逐层 hack 出跨 app 通信能力：

- Channel 实现可以跨进程，说明 Command 生命周期支持跨进程。
- Channel 本身是动态加载的。Channel 的构建过程就是 IoC 容器里一组 Provider 的组装过程。Provider 可以来自不同进程。
- Matrix 是个非阻塞无状态通信总线，通信协议由 Provider 的实现决定。MoMatrix 默认进程内，Matrix stream 默认 Zenoh，Matrix RPC 默认 Zenoh。
- Ghost 不需要知道这些，它只需要 CTML 表达意图，Ghost 的输出会流式解析为 CTML，CTML 会被 Interpreter 派发到 Channel，由 Channel 指挥 Provider 执行。
- 整个执行链路都是异步非阻塞的。只要 channel 能通过矩阵找到 provider，就能跨进程执行。
- 性能瓶颈在于 CTML 不支持同步等待。模型必须发出指令后继续输出，不能等待结果。

## 设计

分三层思考这个 feature 的实现。

### Layer 0: Zenoh Stream 共享 Buffer

最底层的技术方案：App 之间通过 Zenoh stream topic 传递数据，而不是共享内存或文件 buffer。为什么选 stream：

- voice、vision、gomoku 等 App 可能运行在不同进程甚至不同设备
- Zenoh 的 `pub_stream_delta` / `sub_stream` 提供增量发布/订阅，天然适合 buffer 数据
- 进程间没有 Python 对象的内存共享需求，全是字节流

vision 通过 `sub_stream("voice/shared_buffer", ...)` 订阅 voice 的共享 buffer 数据，用于本地 buffer 回放。gomoku 也有类似的 buffer 更新需要传递给 ai_eye。这是所有 app 间通信的基础传输层。

### Layer 1: App 跨进程通信

App 是独立进程，有自己的 Channel 和生命周期。它们之间的通信模式：

- **voice → ai_eye**: 发布语音状态。voice 通过 `pub_stream_delta("voice/state", payload)` 告知 ai_eye 当前是否在录音。payload: `"recording_started"` / `"recording_stopped"`。ai_eye 收到后进入/退出 voice_attention 模式（PTT 期间看前方，不追脸）。
- **vision → ai_eye**: 发布人脸坐标。vision 每 0.3s 本地检测到人脸后，通过 `pub_stream_delta("vision/face", payload)` 告知 ai_eye 人脸位置。payload 格式 `"cx,cy"`（归一化 0..1，量化到 2% 网格防抖）。
- **gomoku → ai_eye**: 发布棋局状态 (`pub_stream_delta("gomoku/state", ...)`，payload: `"human_moved"` / `"ai_moved"` / `"game_over"`)，ai_eye 订阅后触发表情闪烁。
- **voice shared_buffer**: voice 的 delta_since buffer 通过 `pub_stream_delta("voice/shared_buffer", ...)` 发布，vision 已订阅（回放逻辑待实现）。

### Layer 2: Ghost 执行 CTML

Ghost 的"输出到世界"本质上是 CTML 执行，逻辑上可以解耦。ai_eye 作为 app 接收 CTML 指令：

- `look_at`: 控制眼球看向坐标 (x, y)
- `look_away`: 分神/重置
- `react`: 做表情 (smile, surprise, blink, 棋局表情)
- `play`/`stop`: 控制音频播放

Ghost 只管发指令，不需要等结果。ai_eye 的眼睛移动有过渡动画，语音排队播放。这些都是 app 内部的 channel 行为。

### Layer 3: Voice 超时 fallback

Voice 有时 15 秒没有产出 attention 数据（比如说话人消失了）。这种情况下 ai_eye 会回到 auto_gaze 模式（idle 自动游移）。这是 fallback 行为，不是主路径。

**优先级**: voice attention > vision face tracking > auto_gaze。voice attention 活跃时覆盖 face tracking；voice 超时后切换回 face tracking 或 auto_gaze。

## 方案

```
Layer 0: Zenoh Stream 共享 Buffer
  - voice: @voice.utterance.timer 触发 _publish_shared_buffer()
    - delta_since 机制：只在 buffer 有变化时发布
    - topic: "voice/shared_buffer"
  - vision: sub_stream("voice/shared_buffer", ...) 接收并回放 buffer
  - gomoku: 类似机制发布棋局状态

Layer 1: App 间 Stream 通信
  - voice: pub_stream_delta("voice/state", payload) 发布录音状态
  - vision: pub_stream_delta("vision/face", payload) 发布人脸坐标
  - gomoku: pub_stream_delta("gomoku/state", payload) 发布棋局事件
  - ai_eye: sub_stream 接收上述三个 topic，直接更新内部状态
  - 通信全走 Zenoh stream，不经过 Ghost，不经过 CTML

Layer 2: Ghost → ai_eye CTML
  - ai_eye channel commands: look_at, look_away, react
  - Ghost 输出的 CTML 通过 channel interpreter 路由到 ai_eye
  - ai_eye 的行为是 channel 实现（动画、状态管理）

Layer 3: 超时 fallback
  - voice attention 15s 超时 → ai_eye 回退到 auto_gaze
  - voice attention 无人脸时 → 切换回 face tracking
```

实现不需要改内核，全是 App 层面的事。Matrix 通信、Channel 动态加载、CTML 解析都已支持。只需在 App 的 `main.py` 中组合这些能力。

关键决策：

- **Zenoh Stream 而非共享内存**: App 可能跨设备，stream 天然支持分布式
- **ai_eye 内部状态管理**: ai_eye 维护 gaze 模式状态机（voice_attention > face_tracking > auto_gaze），各 stream 只负责更新数据，不控制模式切换
- **CTML 只用于 Ghost→ai_eye**: app 间通信走 stream，Ghost 控制走 CTML，职责分离

## 评估

**智能体现在拥有的能力：**

1. **并行 App** — 每个 App 是独立进程，有自己的 Channel 和生命周期
2. **Stream 通信** — App 通过 Zenoh stream topic 发布/订阅数据，不需要 Python 内存共享
3. **数据复用** — Voice 的 buffer 数据通过 stream 传递给 Vision，不需要拷贝
4. **Ghost 控制** — Ghost 通过 CTML 指令控制 ai_eye（眼神、表情、音频），不需要等结果
5. **超时 fallback** — 传感器数据中断时系统优雅降级（auto_gaze）

**和前序 FEATURE 的关系：**

- voice-attention（已有）— voice 独立的注意力方向检测，现在通过 stream 发布给 ai_eye
- voice-vad-pipeline（已有）— voice 的 VAD/ASR 流水线
- vision-pipeline（进行中）— vision 的持续捕获 + 本地人脸检测，通过 stream 发布给 ai_eye

## 实现状态

### 已实现

1. **Layer 0**: voice 的 `_publish_shared_buffer()` 每 0.3s 通过 `pub_stream_delta("voice/shared_buffer")` 发布，带 delta_since 防抖。vision 已订阅接收。
2. **Layer 1**: voice→ai_eye stream (`voice/state`)，vision→ai_eye stream (`vision/face`)，gomoku→ai_eye stream (`gomoku/state`)，ai_eye 优先级状态机（voice > face > auto_gaze）。
3. **Layer 2**: ai_eye 已有 look_at/look_away/react channel commands，Ghost CTML 可控制。
4. **Layer 3**: voice attention 15s 超时 fallback 到 auto_gaze，vision face 持续更新。

### 未实现

1. vision 对 voice shared_buffer stream 的消费和 buffer 回放逻辑（stream 已订阅，但 vision 端的回放逻辑待实现）。

## 重要更新 (2026-06-06)

### Vision 持续运行，不依赖 voice 触发

Vision 之前设计为"voice 更新 delta_since 后触发 snapshot"。改为持续运行：每 0.3s 本地人脸检测，独立于 voice。理由：

- Vision 作为独立 sensor 应该持续感知，不依赖 voice 的状态变化
- Ghost 每轮对话时获取最新帧即可（通过 context_messages），不需要 vision 调用 VLM
- Voice→Vision 的触发机制已移除，voice shared_buffer stream 仅用于 buffer 数据传递

### app 间通信走 stream，不走 CTML/pub_logos

之前设计为 vision 通过 `pub_logos` 发布 CTML 指令让 ai_eye 看向人脸。改为 stream topic：

- `pub_logos` 是 Ghost articulate → CTML 的路径，app 发的 CTML 不会被 interpreter 消费
- ai_eye 的 look_at 是 channel command，需要 CTML interpreter 路由，但 app 间通信不经过 interpreter
- `pub_stream_delta("vision/face")` + `sub_stream` 是正确的 app 间通信方式

实现文件：

- `voice/main.py` — 持续抓帧 + 本地人脸检测 + stream 发布
- `ai_eye/main.py` — 订阅 voice/attention 和 vision/face stream
- `voice/pyproject.toml` — 移除了 `anthropic` 依赖（vision 不调 VLM）
