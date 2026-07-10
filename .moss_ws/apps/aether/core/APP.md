---
arguments: ''
description: 'Aether Core UI — canonical aether/core app，聚合 SpeechTopic/AudioRuntimeTopic 并通过 WebSocket 驱动能量核心'
executable: uv
respawn: false
script: main.py
workers: 1
---

Aether Core UI app — MOSS ghost 的能量核心可视化通道。Canonical app
address: `aether/core`.

订阅 listener 的 SpeechTopic（用户说完一句 → think）和 TTS player 的 AudioRuntimeTopic（speaker running → speak/idle），通过 WebSocket 把状态推给前端 WebGL 能量核心。

前端 VAD 快线检测到开口时，通过 WebSocket 发 interrupt 消息，后端推 interrupt 状态（急刹冻结视觉），实现 v2 技术设计的 <50ms 爆点。
