---
executable: uv
script: main.py
arguments: ''
description: '五子棋 — 15x15 pygame 棋盘，人类点击落子，AI 通过 Channel 命令参与对弈'
respawn: false
workers: 1
---

Gomoku game app — 15x15 pygame board. Human clicks to place black stones, AI responds with white stones via Channel commands (`ai_move`). Also supports `human_move` for voice-controlled placement. Publishes game events (`human_moved`, `ai_moved`, `game_over`) to `gomoku/state` stream for cross-app reactions.