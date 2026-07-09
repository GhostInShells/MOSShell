---
arguments: ''
description: 'Aether listener — audio → Volcengine ASR → SpeechTopic + AudioSignal'
executable: uv
respawn: false
script: main.py
workers: 1
---

Aether listener — audio → Volcengine ASR → SpeechTopic + AudioSignal.
Canonical app address: `aether/listener`.

The Aether baseline uses the Volcengine streaming ASR path only. Frontend ASR
controls can switch between continuous listening and manual capture, but both
modes use the same backend recognizer.
