---
executable: uv
script: main.py
arguments: ''
description: '飞书 IM 集成 — WebSocket 长连接，消息感知与回复'
respawn: true
workers: 1
---
飞书 Channel App。通过 lark-channel-sdk 接入飞书。
接收消息递轻量 Signal 给 Ghost，Ghost 通过 Channel pull 消息详情并回复。
