---
arguments: ''
description: 'Moss CLI self-control channel. Execute moss commands via CTML for ghost self-bootstrapping.'
executable: uv
respawn: false
script: main.py
workers: 1
---

Moss Self Channel — 将 moss CLI 命令树反射为 Channel 命令，使 Ghost 能通过 CTML 调用 moss 工具开发 moss 自身。