---
title: Aether 双工 UI：并发 listen/think/speak 可视化
status: in-progress
priority: P1
created: 2026-06-29
updated: 2026-07-10
depends: []
milestone:
description: >-
  将 Aether Core 从互斥的五状态 UI，改造为面向 MOSS 全双工语音交互的并发活动层可视化。
---

# Aether 双工 UI

> 使用 `moss features set-status aether-duplex-ui <status> -m "note"` 更新状态。
> 目录布局见 [TOPOLOGY.md](TOPOLOGY.md)，完整约定见 [README.md](README.md)。

## 动机

Aether Core 本应证明 MOSS 的全双工交互能力，但第一版实现把 `listen`、`think`、`speak` 和 `interrupt` 压缩成了单一 UI 状态。这掩盖了核心现象：Ghost 可以一边说话一边继续听，也可以在语音输出已经开始后继续思考。

单状态路径还带来了一个糟糕的失败模式：前端 VAD 可能在后端真正停止 TTS 之前就显示 `interrupt`，于是画面声称“已经停止”，但音频还在继续播放。这个 workstream 要让 UI 保真：普通 VAD 输入只点亮 listen 层；真正的 interrupt 只保留给后端 barge-in 确认，或显式发送 MOSS interrupt signal 的 interrupt 命令。

## 设计索引

- 关键设计文档：`design/`
- 关键讨论记录：`discuss/`

## 关键决策

<!-- 记录每一个有意义的设计选择。这是下一个 AI 化身最先阅读的内容。 -->

1. Aether 状态现在是 `layers`，不是互斥枚举。
   WebSocket 载荷为兼容旧客户端仍保留 `state`，但权威契约是 `layers: {listen, think, speak, interrupt}`。`state` 按优先级派生：`interrupt > speak > think > listen > idle`。

2. `listen`、`think`、`speak` 可以同时为 true。
   视觉语言改为混合核心着色器加三条外层活动环：青色 listen、紫色 think、琥珀色 speak。中心标签可以显示 `LISTEN + THINK + SPEAK`，而不是假装只有一个状态获胜。

3. 前端 VAD 不再等同于 interrupt。
   VAD 只切换本地 `mic` 诊断层。主 `listen` 由后端 ASR 运行时活动驱动，而不是由浏览器能量检测驱动。真正的 interrupt 需要 listener 检测到唤醒词，或收到显式 `{type:"interrupt"}` 请求，并通过 session 发送 `new_interrupt_signal()`。这样可以避免旧的 UI/TTS 裂脑，以及“listen 但没有 think”的假信号。

4. `interrupt` 是抢占层，不是另一个普通状态。
   触发时，后端清空 `think` 和 `speak`，广播 `interrupt_burst`，并发送预期触发 `shell.clear()` 的 MOSS interrupt signal（TTS clear + interpretation cancellation）。

5. 演示优先优化语音回合延迟，而不是保守分段。
   listener 的 synthetic-final patience 是 1.0s，不是 3.0s。浏览器 VAD 默认阈值是 0.008，且 `listen` 会在 VAD 结束后短暂保持到 ASR final pending，所以 UI 不再在正常 ASR 延迟窗口里从 listen 直接掉回 idle。

6. Aether voice mode 将 Echo 约束为短自然语言回复。
   默认 Echo soul 足够宽，可以做 CTML/工具调用，这会让 DeepSeek 在语音演示中输出非法语音标签，例如 `<say emotion=...>`，甚至输出 app-start 命令。Aether 现在明确要求优先短自然语言、不要 Markdown、不要启动 app；只有确实需要 CTML 时，才允许合法的 `say` 属性。

## 实现记录

<!-- 记录坑点、非显然行为，以及拒绝更简单方案的理由。 -->

- `.moss_ws/apps/aether/core/main.py` 负责后端 layer snapshot，并且仍为旧客户端发出 `state`。
- `.moss_ws/apps/aether/core/webroot/web/state_mapper.js` 将 layers 映射到主状态和混合着色器目标。把映射集中在一个文件里，可以防止旧的五状态假设重新泄漏到 `main.js` 或 `scene.js`。
- `.moss_ws/apps/aether/core/webroot/web/main.js` 会为了低于 50ms 的 listen 反馈立即应用本地 VAD，但不会因为 VAD 调用 `sendInterrupt()`。
- 2026-06-29 尝试过浏览器视觉验证，但当前环境中的 Codex browser plugin 报告没有可用浏览器。因此改为运行 JS 语法检查、Python 编译，以及 layer-mapping 冒烟检查。
- 2026-06-29 现场测试反馈：初始 VAD 太不敏感，ASR final 体感偏晚，interrupt 没有明显触发，Echo 生成了非法 CTML。修复：降低 VAD 阈值；本地 VAD 结束后保持 listen；缩短 ASR patience；扩大唤醒词；即使 speaker runtime topic 滞后，也把唤醒词当作 interrupt；收紧 Aether voice prompt 纪律。
- 2026-06-29 追踪诊断：ASR 和 LLM 都在产生 turn，日志里 Volcengine TTS 也已成功返回音频。不稳定基线来自两个本地集成问题：Aether mode 没有为纯文本 speech output 注册 `__content__`；多 app 启动时可能撞上 Circus 的 transient arbiter lock。修复：在 Aether mode 中启用 `SpeechChannelModule(register_content=True)`；当 Circus 报告 arbiter 短暂忙碌时重试 watcher `start`。
- 2026-06-29 追踪诊断：部分可见的“think but no speak”其实是假阳性 think。浏览器 VAD 结束后，在 ASR 产生 final `SpeechTopic` 之前就要求后端设置 `pending_think`；如果 ASR 没有 finalize，Aether 会超时，且没有 Ghost turn 可以说话。Reset 也曾发布字面量 `/reset` human SpeechTopic，制造假 think。修复：VAD 现在只拥有 listen 层，并等待后端 ASR final；reset 只清空视觉上下文，不再创建 speech turn。
- 2026-06-29 现场双工 bug：listener 在 `speaker running=True` 时仍丢弃非 wake-word ASR 结果。在 VPIO AEC 路径中这是错误的：TTS 期间的用户语音已经经过 echo cancellation，必须成为正常 turn。症状很明确：浏览器显示 listen，ASR 产生 final text，但没有发布 `SpeechTopic`，所以 UI 从 listen 回到 idle，而不是进入 think/speak。修复：旧的保守行为只保留在 `LISTENER_GATE_DURING_TTS=1` 后面；Aether 默认在 TTS 期间也发布 speech。
- 2026-06-29 完整 review：“listen then idle”有多重原因。前端 VAD 结束后立即发送 `listen=false`，所以 UI 可能在后端 ASR final 抵达前回到 idle。listener 的 300ms ASR end window 加 1.0s synthetic-final patience 会把语音过度切成“你”/“我”这类碎片，制造低价值 Ghost turn 队列并延迟短回复。Aether Core 也没有在 speaker runtime 重新变成 true 时清理 `_tts_end_at`，所以一次瞬时 `speaker=false` 可能稍后清空 `speak`，即使音频已经恢复。修复：前端 listen clear 延迟到 3.5s；listener 发布 `asr` runtime activity；ASR end window 调整为 700ms，synthetic final patience 调整为 1.4s，并丢弃单字 timeout fragment；Aether Core 使用后端 ASR runtime 表示 listen，并在 speaker 恢复时取消陈旧 TTS end grace。
- 2026-06-29 语义修复：普通音频 impulse 不再设置 `interrupt=True`。只有唤醒词或显式 InterruptNucleus 应该调用 `shell.clear()`。这让 Aether 更接近全双工语义：用户可以在 speak 期间说话，而不会让每个 ASR final segment 都变成紧急停止。
- 2026-06-29 诊断细化：用户仍能看到很多“listen then no think/speak”，因为浏览器 VAD 被展示成权威 listen，即使后端 ASR 没有产生 partial/final。UI 现在区分 `MIC`（浏览器本地能量）和 `LISTEN`（后端 ASR 活动）。VPIO capture 也记录一秒粒度的 RMS/peak stats，所以下次失败可以定位到浏览器 mic、macOS VPIO capture、Volcengine ASR 或 Ghost/TTS，而不是折叠成一个视觉状态。
- 2026-06-30 ASR 完整性 bug：Volcengine ASR full-server response 可能不带 sequence 字段。旧 parser 总是跳过四个 sequence bytes，导致载荷从 `sult` 开始而不是 `{"result"`，进而 parse 失败、丢失 `LISTEN`、错过唤醒词 interrupt。修复：从 protocol flags 推导 response offset。listener segmentation 也太激进（`end_window_size=700`、synthetic-final patience `1.4s`），会截断“我有一”这类 utterance；Aether 现在使用 `end_window_size=1400`、patience `2.6s`，并加 60s first-result watchdog 来重启 stale ASR recognition。
- 2026-06-30 interrupt/queue 诊断：日志显示“立刻停下”已被识别，Aether Core 也收到了后端 interrupt signal，但 `BaseTTSSpeech.clear()` 只清理 text output buffer，没有清理 TTS backend 或 `StreamAudioPlayer`；因此有效 interrupt 后音频仍可能继续播放。修复：`BaseTTSSpeech.clear()` 同时清理 backend TTS 和 player。另新增 `queue` activity layer，用来表示 `speak` active 时 finalize 的人类语音。TTS grace shutdown 不再无条件清掉更新的 `think`，所以用户在第一个回答期间问的第二个问题会保持显示为 `QUEUE + THINK`，而不是被第一个回答的 TTS end 覆盖。
- 2026-06-30 现场 interrupt 延迟诊断：08:06:29 的 wake-word interrupt 立刻抵达 listener 和 Aether Core，但 host 到 08:06:58 才执行 `shell.clear()`，因为 interrupt 仍依赖 Mindflow attention/action abort checkpoint。到那时 Volcengine TTS 已经缓冲了 27.64s 的 player wait。修复：`GhostRuntimeImpl` 现在在 `AudioRuntimeTopic` 上运行一个带外 audio interrupt watcher；当看到 `device_name="interrupt"` 时立即调用 `shell.clear()`，而 Mindflow interrupt signal 仍保留为语义取消路径。
- 2026-06-30 ASR 截断调参：现场日志仍显示“我刚才让”、“发现有一个问题，就是他”、“我需要”等 partial utterance 在 2.6s silence timeout 后被合成为 final。这主要不是浏览器 VAD 问题：后端 ASR partial 本身就是不完整的。listener 现在默认改为更保守的 `LISTENER_ASR_END_WINDOW_MS=1800` 和 `LISTENER_SILENCE_PATIENCE=4.5`，二者都可通过环境变量调整。
- 2026-06-30 listen 语义修正：`LISTEN` 仍绑定在后端 ASR partial 上，所以 UI 只有在 Volcengine 已识别出文本后才显示 listen。现场日志显示 VPIO energy 在 ASR partial 前已经上升，例如 08:16:39/40 的 audio peaks，而 UI 到 08:16:40 的 ASR partial “我”才进入 listen。对 Aether 的视觉契约而言，listen 的含义是“音频层听到用户”，不是“语义 ASR 文本存在”。前端现在从本地浏览器 VAD（`mic`）组合出快速视觉 `listen` 层，同时仍只允许 ASR final/SpeechTopic 进入 `think`。
- 2026-06-30 语音 interrupt 诊断：手动 stop 证明 interrupt topic → host watcher → `shell.clear()` → TTS/player clear 路径有效。语音 stop 失败位于上游：ASR 经常没有发出包含“停下”的文本。VPIO capture 现在默认 `VPIO_CHANNEL_MODE=best`，按每帧 RMS 选择最强 channel，而不是总选 channel 0，并发布 `best_ch/ch_rms` 诊断。listener 在 ASR runtime topic 中发布 ASR partial/final text，Aether Core 将最近三条 ASR 结果和 VPIO diagnostics 转发给 UI，让现场测试能区分硬件/channel capture、ASR/matcher 等问题。
- 2026-06-30 ASR 控制变量排查：现场日志显示两个不同问题。Volcengine 增量 partial 在 UI 中看起来像多个 utterance，虽然只有 final result 会发布给 Ghost；panel 现在把当前 partial 和 final history 分开展示。Volcengine `server_error: 106` 过去被吞成空 final，listener 会立刻重连并进入 tight loop，UI 没有任何信号。recognizer 现在发出内部 error marker，listener 发布结构化 ASR error diagnostics 并退避重连，Aether UI 显示 code/backoff。local ASR 和 VPIO disabling 被有意推迟，以保持 headset-mic 测试隔离。
- 2026-06-30 local ASR 转向：Volcengine ASR 停止产生可用结果。native sherpa-onnx runtime 已安装，但 Python/native model download 被模型托管网络卡住。已经下载完成的官方 wasm zh-en Paraformer bundle 现在通过 `/asr` 提供，并由 Aether UI 加载。浏览器本地 sherpa-onnx 在端侧完成 partial/final ASR，并通过 WebSocket 发送 `{type:"speech"}` 到 `aether_core`；后端从 final text 发布 `SpeechTopic`/`SPEECH_FINAL`。local partial 只要包含停止唤醒词，也会触发已有 interrupt path。listener 默认 `LISTENER_ASR_BACKEND=browser`，因此可以在不打开 Volcengine 的情况下保持运行。
- 2026-06-30 native ASR 修正：浏览器 wasm bundle 下载了 226MB `.data` 文件，但在 `onRuntimeInitialized` 前卡住；即使加了 COOP/COEP headers，也很可能是 pthread worker/bootstrap 脆弱性。已从 wasm data package 中提取 `encoder.onnx`、`decoder.onnx` 和 `tokens.txt` 到 `.moss_ws/models/asr/sherpa-onnx-paraformer-zh-en-native/`。native `sherpa_onnx.OnlineRecognizer.from_paraformer(...)` 在这台 Mac 上约 0.68s 初始化，所以 listener 默认改成 `LISTENER_ASR_BACKEND=sherpa`，主 Aether 页面不再加载沉重的浏览器 wasm ASR scripts。
- 2026-06-30 ASR fallback 决策：local sherpa ASR 对交互式 Aether 测试仍太不稳定（首轮可能识别，后续经常失败或准确率很差）。local ASR 代码、提取出的模型和 wasm debug page 保留给未来工作，但 listener 默认切回 `LISTENER_ASR_BACKEND=volcengine`。
- 2026-06-30 Volcengine 文档排查：当前 ASR-only endpoint `wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_async` 保持为默认值，但文档澄清了两个运行事实。第一，公开 ASR 错误表没有定义观察到的 raw `106`；error frame 会在 code 后携带 UTF-8 message，所以 parser 现在按文档中的 `Header + code + message_size + message` 布局解析，并把 message/backoff 暴露到 UI。第二，continuous listener mode 确实会打开 ASR recognition session 并发送 audio，所以 Aether 现在有前端 ASR control gate：`continuous` 保留旧行为，`manual + disabled` 阻止 listener 调用 `asr.recognize()`，从而避免打开 Volcengine WebSocket。完整当前技术设计记录在 `Docs/aether-voice-runtime-technical-design.md`。
  `wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_async` 是官方优化版双向 streaming 路径，不是废弃路径。更大的延迟回退来自本地调参漂移：Aether 曾漂到 `end_window_size=1800` 和 `silence_patience=4.5`，而 Volcengine 推荐低延迟 finalization 使用大约 800ms 或 1000ms。listener 默认值现在是 `LISTENER_ASR_END_WINDOW_MS=1000` 和 `LISTENER_SILENCE_PATIENCE=1.8`。protocol request 也开始发送 `enable_ddc` 和可配置的 `force_to_speech_time`，并通过 `VOLCENGINE_BM_ASR_API_KEY` 支持新版控制台 `X-Api-Key` header。官方 S2S 全双工 `wss://openspeech.bytedance.com/api/v3/duplex/realtime/dialogue` 是独立的未来架构轨道，因为它同时组合 ASR/LLM/TTS；如果不刻意包装，会绕过 Ghost/CTML。
- 2026-06-30 DeepSeek/TTS streaming 文档排查：DeepSeek Chat Completions 可以 stream response delta，但不接受一个仍在增长的 ASR transcript 作为单个输入流。因此 Aether 保持把 ASR partial 作为 UI diagnostics，只把 final ASR text 提交给 Ghost；如果把每个 partial 都喂给 LLM，会产生“你现在”/“你现在给我讲”这类重复 turn。`Atom.articulate()` 已经使用 `agent.run_stream()`，并把 text delta yield 给 CTML/TTS。DeepSeek V4 现在通过 `thinking: {type: "disabled"}` 明确关闭 thinking，以降低语音延迟。Volcengine bidirectional TTS 已经能边消费 text chunk 边产生 audio chunk；其 headers/env overrides 现在与官方 `api/v3/tts/bidirection` 协议对齐，包括 `X-Api-Key`、resource ID selection、可选 usage-token 返回，以及 proxy-free WebSocket dialing。
- 2026-07-01 清理：local ASR 实验不再属于当前 Aether baseline。browser wasm debug page/bridge、提取出的 local model 目录、listener backend selection branch 都从主路径移除。Aether 现在把 Volcengine ASR → `SpeechTopic` → Mindflow → DeepSeek V4 Flash → Volcengine TTS 视作唯一支持的演示回路。未来 offline ASR 工作应作为独立 feature 重新引入，并复用同一 Topic contract，而不是临时浏览器 `{type:"speech"}` shortcut。
- 2026-07-01 交接完成：定向验证暴露了 Volcengine ASR error-frame parsing 中最后一个未清理 bug。parser 现在同时支持观察到的 sequence-wrapped error frame 和文档定义的无 sequence 布局 `Header + code + message_size + message`。新增无 sequence variant 的 protocol regression coverage。已验证 Python 编译、ASR protocol tests、speech/mindflow interrupt tests 和 WebGL JS syntax checks。完整 `pytest tests -q` 在这个 sandbox 中不是有用 gate，因为 zenoh POSIX shm 和 ZMQ tcp bind 会被环境拒绝。
- 2026-07-01 现场音频/控制诊断：VPIO capture 看起来不是主要识别问题。当前 VPIO 路径报告 48k native capture、9 channels、input/output VPIO enabled、重复的 post-VPIO channel RMS、约 0.0007-0.0015 的低 silence RMS，以及常见 0.04-0.35、偶尔更高的 speech peaks。观察到的坏 turn 更像 ASR segmentation/finalization 问题，而不是错误 capture channel。listener 的 `LISTENER_SILENCE_PATIENCE` 默认值现在不那么激进（`3.2s`），同时保持 Volcengine `end_window_size=1000ms`。另确认一个缺陷：前端 “manual ASR disabled” 只是 transient topic，可能被 VPIO/ASR diagnostics 驱逐，之后 listener 会回退到 continuous mode；旧前端代码也会在 connect 时重新发送默认 continuous control。listener 现在保持 ASR control state sticky，使用更大的 runtime topic window，并在 manual mode disabled 时立即 gate live audio generator。WebGL bridge 只有在用户显式修改后才重发 ASR control。通过 Aether WebSocket 验证看到 `mode=manual enabled=False` 后跟随 `ASR manual gate closed`；以这种方式关闭 live Volcengine stream 仍会暴露 server EOF warning，这是清理风险，但不再表示 manual gate 失败。
- 2026-07-01 Volcengine ASR timeout 诊断：再次检查官方 bigmodel streaming ASR 文档。`bigmodel_async` 仍是推荐的优化双向路径，audio packet duration 应保持约 100-200ms，观察到的 `45000081` 被文档定义为等待下一个 packet 超时。现场日志匹配这一点：VPIO stats 在 23:30:12 后停止，但进程仍存活；之后 listener 打开 Volcengine session，却没有发送 PCM，并每 8s 命中 `45000081`。第二个本地缺陷也被发现：listener 把 VPIO stream 当成 generic `AudioCaptureConfig` 默认 44.1kHz 输入，虽然 Aether VPIO 实际发布 16k PCM，于是把已经是 16k 的语音当成 44.1k 再采样，破坏 ASR timing。listener 现在为 Aether 默认 input sample rate 16k；如果在 `LISTENER_AUDIO_FRAME_TIMEOUT` 内没有 audio frame，就中止 ASR stream；Volcengine ASR 将外发 PCM 聚合成可配置的 200ms packet；VPIO 在 process 仍存活但 frames 停止时发布 stalled runtime topic。这修复了直接的 timeout/quality bug，但尚未自动重启 stalled VPIO engine，也还没移除下一次 ASR session 前剩余的 post-utterance cooldown gap。
- 2026-07-04 Aether 项目布局决策：Aether 不再被当成松散的 `examples/` WebGL 演示。它是一个可运行的 MOSS mode 加 `aether/core` app，前端耦合到该 app 的 WebSocket/Topic contract。WebGL static files 已从 `examples/web_gl` 移到 `.moss_ws/apps/aether/core/webroot`，后端现在从 app-local root 提供服务。这让 app 在 `.moss_ws` 下自包含，同时保留已有 HTML/JS/着色器结构和 runtime protocol。
- 2026-07-09 长暂停 ASR 诊断：官方 Volcengine SAUC WebSocket 文档确认优化版 streaming endpoint、100-200ms audio packet 建议、16k PCM 要求、final negative audio package、`end_window_size`，以及 `45000081` 表示 wait-for-next-packet timeout。现场失败位于 ASR 上游：listener 在没有 audio frame 抵达后正确拒绝打开新的 ASR session，而 VPIO 持续报告 `no frames`。VPIO watchdog 现在会在持续 stall 后重启 AVAudioEngine/tap，清空 stale queued frames，并发布 `restarting/restarted/restart_failed` runtime diagnostics。ASR protocol sender 也现在返回并保留 `create_audio_only_request` 发出的 packet sequence，所以 final packet 不再依赖 caller 侧重复 sequence arithmetic。
- 2026-07-10 公共表面兼容性审查：Aether 特有行为已经收回到显式 mode/env 配置之后。公共默认值现在保留 `AudioNucleus` complete-impulse interruption、Atom 默认 Anthropic provider、AppStore 并行 bringup、listener TTS gating、listener capture sample-rate defaults、Volcengine WebSockets 使用系统 proxy，以及 silent ASR error finals。Aether mode 通过 `MOSS_ENABLE_AUDIO_INTERRUPT_TOPIC`、`MOSS_APPSTORE_BRINGUP_SERIAL`、`MOSS_ATOM_TEXT_PROMPT_COMPAT`、`MOSS_ATOM_DISABLE_HISTORY`、`LISTENER_INPUT_SAMPLE_RATE=16000`、`LISTENER_GATE_DURING_TTS=0`、`VOLCENGINE_BM_ASR_URL=.../bigmodel_async` 和 `VOLCENGINE_BM_ASR_PROPAGATE_ERRORS=1` 显式启用全双工变体。已添加聚焦的 AudioNucleus regression coverage，用来锁住公共默认行为和 Aether opt-in `interrupt_on_complete=False` 行为。
