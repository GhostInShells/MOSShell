---
arguments: ''
description: 'System audio capture — opens miniaudio CaptureDevice, publishes raw PCM to Zenoh stream (audio/pcm) with per-frame FFT metadata.'
executable: uv
respawn: false
script: main.py
workers: 1
---
