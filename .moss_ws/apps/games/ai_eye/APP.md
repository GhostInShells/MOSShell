---
executable: uv
script: main.py
arguments: ''
description: 'AI 眼球 — pygame 图形化身，AI 通过 Channel 控制注视方向、瞳孔缩放、眨眼和表情'
respawn: false
workers: 1
---

AI eyeball app — animated pygame eyes that react to the real world. Auto-tracks faces via `vision/face` stream, focuses attention when voice recording is active via `voice/state` stream, and reacts to gomoku game events via `gomoku/state` stream. Ghost controls gaze direction, pupil size, blinking, and expressions (idle/thinking/speaking) through the Channel interface.