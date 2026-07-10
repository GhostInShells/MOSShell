---
apps:
  - '*/*'
bringup_apps:
  - 'aether/vpio_capture'
  - 'sensors/listener'
  - 'aether/core'
ctml_version: ''
description: 'Aether Core · 能量核心 UI 模式 — 听→想→说 完整回路 + 急刹打断 (macOS VPIO AEC)'
name: aether
---

Aether Core 模式：拉起 `aether/vpio_capture` + `sensors/listener` + `aether/core` 三个 app，ghost 通过语音对话（ASR→LLM→TTS），前端能量核心实时反映 idle/listen/think/speak/interrupt 状态。

语音演示优先级：低延迟短回应优先于复杂 CTML。Ghost 收到语音时应优先直接输出一句自然语言纯文本，让 SpeechChannel 立刻播放；不要主动启动 app，不要输出 Markdown 代码块，不要使用 `<say emotion=...>` 等 speech channel 不支持的属性。

**macOS AEC 升级**：本模式默认使用 `aether/vpio_capture`（基于 macOS VPIO 的系统级回声消除），让 TTS 外放时 ASR 仍能干净地收人声，真正实现全双工可打断。

**Fallback**：若在非 macOS 或 PyObjC 不可用的环境，把 `bringup_apps` 里的 `aether/vpio_capture` 改回 `sensors/audio_capture`（miniaudio 采集，依赖 listener 三重门控防回声）。
