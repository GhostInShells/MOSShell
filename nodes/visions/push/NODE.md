---
name: 'push'
description: 'unified local visual push (screen / camera) — a request becomes a shared, revocable stream; the human approves and watches, either face stops'
category: visions
# 非单例: 一个 node = 一台机器的视觉推流 supervisor; 每个 producer 子进程 = 一个设备.
# 设备独占由 producer 子进程生灭兜底, 不用治理域 singleton 锁.
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

Push is the unified local visual push node. A push is not a command — it is a
**session**: the model requests a source (`screen` / `camera`), the human
approves or denies on the node's own web page, and a live session is an ffmpeg
child the node owns and serves as MJPEG. Either face can stop it; the model may
stop only sessions it asked for, never one the human opened.

```
<push:request source="screen" label="desk"/>
<push:sessions/>
<push:stop session_id="1"/>
```

One process, two faces over one store: the channel (the ghost's request/stop
surface) and the web surface (the human's approval and watch surface). The
surface binds an **ephemeral port by default**; read its live URL from this
channel's `url` notice — never assume a fixed port. To pin one, start with
`--port N` (or set `MOSS_PUSH_PORT`).

## Approve, watch, stop

A pending request appears on the human's page as an approval card. Accepting
spawns the producer; the page shows a live preview of every streaming session,
and any of them can be closed with one click. There is an **accept all** toggle
for "stop asking, just run" — a front-end flag, not a persistent switch.

The model is signalled each verdict and each live stream's URL; it then opens a
*separate* `stream` node on that address to consume the vision (this node only
pushes).

## Sources

`screen` and `camera` are the enumerated sources. The model names a source plus
tuning knobs (`fps` / `max_width` / `quality`); it never names a device, an
ffmpeg flag, or a shell line. The capture device index is node-side config.

## Configuration

Cell-level env (copy `.env.example`): `PUSH_SCREEN_INDEX` / `PUSH_CAMERA_INDEX`
(the avfoundation capture indices), and the policy gate `PUSH_ALLOW` (set to
`0` / `false` / `no` / `off` to refuse launch).

## macOS permission

The `screen` source needs Screen Recording permission (System Settings →
Privacy & Security) granted to whatever process runs ffmpeg. Without it, capture
blocks rather than errors — the probe deliberately never opens the screen.

## Debug

    ../.venv/bin/python main.py
