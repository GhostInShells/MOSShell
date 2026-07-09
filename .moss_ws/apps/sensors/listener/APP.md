---
arguments: ''
description: 'ASR consumer — audio → Volcengine ASR → SpeechTopic + AudioSignal'
executable: uv
respawn: false
script: main.py
workers: 1
---

ASR consumer — audio → Volcengine ASR → SpeechTopic + AudioSignal.
Canonical app address: `sensors/listener`.

The listener is a shared sensor app. Aether reuses it through the original
`sensors/listener` address instead of moving it into `.moss_ws/apps/aether/`.
Frontend ASR controls can switch between continuous listening and manual
capture, but both modes use the same backend recognizer.
