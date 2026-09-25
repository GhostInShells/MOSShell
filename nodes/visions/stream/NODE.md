---
name: 'stream'
description: 'Stream vision — open vision on a protocol address (RTMP/RTSP/SRT/MJPEG), capture a frame, toggle per-round watch.'
category: visions
# 非单例: 一个 node = 一路流, argument(--address) 绑定身份.
singleton: false
# 共享 visions venv 在 nodes/visions/ — 相对 node cwd 解析
exec:
  command: ../.venv/bin/python
  args: main.py
# 启动前探针: 依赖 / 策略, nonzero+stderr 即拒绝拉起并给出原因
check:
  command: ../.venv/bin/python
  args: check.py
---

Stream vision node. The stream is bound at launch via `--address`; one node =
one stream, so `capture` / `watch` never take an address — they read the latest
frame of the node's own stream. `watch` only gates whether a fresh frame rides
each round of context; it does not touch the connection.

Launch (address is the instance identity):

    moss nodes run nodes/visions/stream -- --address rtmp://127.0.0.1/live/desk --label desk
