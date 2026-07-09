# Aether App

Aether 是 MOSS 的实时语音交互外壳。它把麦克风采集、ASR、Ghost 思考、TTS 播放和前端可视化连成一条闭环，用来验证 MOSS 能不能像一个在场的智能体一样听、想、说、被打断。

这个目录是 Aether 相关 app 的唯一维护入口。后续理解、启动、排错，优先看本 README 和本目录代码。

## 目录结构

```text
.moss_ws/apps/aether/
  core/          前端可视化和 WebSocket 状态聚合
  listener/      音频到 ASR，再到 SpeechTopic / AudioSignal
  vpio_capture/  macOS VPIO 音频采集和系统级回声消除
```

三个 app 的 canonical address 是:

```text
aether/core
aether/listener
aether/vpio_capture
```

旧地址 `ui/aether_core`、`sensors/listener`、`sensors/vpio_capture` 已不再作为 Aether 入口使用。

## 一键启动

从仓库根目录执行:

```bash
.venv/bin/moss-run-ghost echo --mode aether
```

启动后打开:

```text
http://127.0.0.1:8765/
```

`aether` mode 会自动拉起:

```text
aether/vpio_capture
aether/listener
aether/core
```

关闭全部 Aether/MOSS runtime 进程:

```bash
.venv/bin/moss --ai --mode aether runtime kill-all --yes
```

查看当前还在运行的 cell:

```bash
.venv/bin/moss --ai --mode aether runtime list-cells
```

## 单独调试

只启动前端可视化:

```bash
.venv/bin/moss --ai --mode aether apps test aether/core
```

只启动 ASR listener。排查时建议先用 manual 模式，避免连续 ASR 抢占调试过程:

```bash
LISTENER_ASR_MODE=manual .venv/bin/moss --ai --mode aether apps test aether/listener
```

只启动 macOS VPIO 采集:

```bash
.venv/bin/moss --ai --mode aether apps test aether/vpio_capture
```

## 组件职责

### aether/vpio_capture

`aether/vpio_capture` 是音频采集层。

它负责:

- 使用 macOS AVAudioEngine + VPIO 采集麦克风输入。
- 打开系统级 voice processing / echo cancellation，降低 TTS 外放被 ASR 收回去的概率。
- 把音频转成 listener 需要的 16kHz mono PCM。
- 发布 VPIO 运行诊断，例如 RMS、peak、channel、frame count、stall/restart。

它不负责 ASR、不负责判断用户意图、不负责调用 Ghost。

### aether/listener

`aether/listener` 是语音识别层。

它负责:

- 消费音频采集 topic。
- 连接 Volcengine streaming ASR。
- 发布 `SpeechTopic`，让 MOSS/Ghost 收到用户说完的一句话。
- 发布 `AudioSignal`，告诉 Mindflow 用户开始说话、最终文本完成。
- 监听 `asr_control`，在 continuous/manual 两种收音模式之间切换。
- 在用户开口或停止意图出现时发布 interrupt 相关信号。

它不负责前端绘制、不负责 TTS 播放、不负责 LLM 推理。

### aether/core

`aether/core` 是前端状态聚合层。

它负责:

- 提供 `http://127.0.0.1:8765/` 页面。
- 订阅 `SpeechTopic` 和 `AudioRuntimeTopic`。
- 把 listen、think、speak、interrupt、idle 等状态合成给 WebSocket 前端。
- 接收前端按钮或浏览器 VAD 产生的控制事件。
- 把 ASR 模式切换写回 `asr_control` topic。

它不负责直接启动其他 app，不直接调用 ASR/TTS/LLM，不绕过 MOSS runtime 伪造 Ghost 输入。

## 数据流

正常语音链路:

```text
麦克风
  -> aether/vpio_capture
  -> audio topic
  -> aether/listener
  -> SpeechTopic + AudioSignal
  -> MOSS Mindflow / Ghost
  -> TTS
  -> speaker diagnostics
  -> aether/core
  -> browser WebGL state
```

前端控制链路:

```text
browser button / VAD
  -> aether/core WebSocket
  -> AudioRuntimeTopic(asr_control / interrupt)
  -> aether/listener 或 MOSS host runtime
```

## 关键 Topic

| Topic | 发布者 | 消费者 | 作用 |
| --- | --- | --- | --- |
| `SpeechTopic` | `aether/listener` | Ghost / `aether/core` | 用户一句完整语音文本 |
| `AudioSignal(SPEECH_STARTED)` | `aether/listener` | Mindflow | 用户已经开始说话 |
| `AudioSignal(SPEECH_FINAL)` | `aether/listener` | Mindflow | 用户一句话完成 |
| `AudioRuntimeTopic(device_name="vpio")` | `aether/vpio_capture` | `aether/core` | 采集状态和音量诊断 |
| `AudioRuntimeTopic(device_name="asr")` | `aether/listener` | `aether/core` | ASR running/partial/final/error/idle |
| `AudioRuntimeTopic(device_name="asr_control")` | `aether/core` | `aether/listener` | 连续/手动 ASR 控制 |
| `AudioRuntimeTopic(device_name="speaker")` | TTS/player | `aether/core` | TTS 播放状态 |
| `AudioRuntimeTopic(device_name="interrupt")` | `aether/core` / `aether/listener` | `aether/core` / host runtime | 打断当前输出 |

## 常见排错

### 页面打不开

确认 `aether/core` 是否启动:

```bash
.venv/bin/moss --ai --mode aether runtime list-cells
```

确认 8765 端口是否有响应:

```bash
curl -s http://127.0.0.1:8765/
```

### 没有声音输入

先单独启动 VPIO:

```bash
.venv/bin/moss --ai --mode aether apps test aether/vpio_capture
```

看日志里是否持续出现 RMS/peak 变化。如果 RMS 长时间为 0，优先检查系统麦克风权限、默认输入设备、采样权限。

### 停顿后 ASR 不再识别

优先检查 `aether/listener` 日志:

- 是否还在读取 audio frames。
- 是否还有 ASR partial/final。
- 是否进入了 error 或 idle 后没有恢复。
- `asr_control` 当前是否被切到 manual。

如果只是调试 listener，先用:

```bash
LISTENER_ASR_MODE=manual .venv/bin/moss --ai --mode aether apps test aether/listener
```

这样可以把连续收音问题和 ASR 服务问题分开看。

### TTS 外放被 ASR 收进去

确认当前 mode 使用的是:

```text
aether/vpio_capture
```

不要用普通 `sensors/audio_capture` 来验证全双工效果。普通采集没有 macOS VPIO 的系统级 AEC，容易把扬声器声音重新送进 ASR。

### 停不下来

先杀当前 mode 下的 runtime:

```bash
.venv/bin/moss --ai --mode aether runtime kill-all --yes
```

再确认:

```bash
.venv/bin/moss --ai --mode aether runtime list-cells
```

输出 `No cells found in this scope` 才表示 MOSS runtime cell 已清空。

## 维护边界

Aether 代码以后尽量收敛在本目录内:

- 前端和 WebSocket 状态聚合放在 `aether/core/`。
- ASR listener 放在 `aether/listener/`。
- macOS VPIO 采集放在 `aether/vpio_capture/`。
- MOSS host、Mindflow、Speech provider 的公共能力仍放在 `src/ghoshell_moss/`，不要复制到 app 目录。

如果新增说明文档，优先更新这个 README。不要再在仓库根 `Docs/` 下新增 Aether 历史说明。
