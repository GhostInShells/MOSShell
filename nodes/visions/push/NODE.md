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
approves or denies on the node's own page, and a live session is an ffmpeg child
the node owns and serves as MJPEG. Either face can stop it; the model may stop
only sessions it asked for, never one the human opened.

```
<push:request source="screen" label="desk"/>
<push:sessions/>
<push:stop session_id="1"/>
```

One process, two faces over one store: the channel (the ghost's request/stop
surface) and the page (the human's approval and watch surface). The page
announces itself on the mesh as a **web view**, so a screen takes it in by itself —
nobody opens a URL by hand. It lands as a floating item on that screen's desktop,
and only a window arranged into a group is clickable: **the ghost arranges it**
(an unarranged item is a proxy, not yet the page). It binds an ephemeral port by
default so siblings never collide; pin one with `--port N` (or `MOSS_PUSH_PORT`).

## Approve, watch, stop

A pending request appears on the page as an approval card. Accepting spawns the
producer; the page shows a live preview of every streaming session, and any of
them can be closed with one click. There is an **accept all** toggle for "stop
asking, just run" — a front-end flag, not a persistent switch.

The model is signalled each verdict and each live stream's URL; it then opens a
*separate* `stream` node on that address to consume the vision (this node only
pushes).

## Sources

`screen` and `camera` are the enumerated sources. The model names a source plus
tuning knobs (`fps` / `max_width` / `quality`) — never an ffmpeg flag or a shell
line. Omit `fps` and each source takes its own default rate: screen 10, camera
30. A capture device only opens at a rate it advertises, so a rate it does not
offer fails the open rather than being clamped.

Which device to read is the optional `index` argument — omit it and this
machine's configured default applies (screen 1, camera 0 on a stock Mac). Which
device a request reads is a per-machine fact, so it lives here rather than in the
request; pin it with `PUSH_SCREEN_INDEX` / `PUSH_CAMERA_INDEX` when this display
does not sit at the stock index.

## Configuration

Cell-level env (copy `.env.example`): the policy gate `PUSH_ALLOW` (set to
`0` / `false` / `no` / `off` to refuse launch), plus the two index escape hatches
above.

## macOS permission

The `screen` source needs Screen Recording permission granted to whatever
process runs ffmpeg (System Settings → Privacy & Security). Without it capture
blocks rather than errors — the probe deliberately never opens the screen.

## Debug

    ../.venv/bin/python main.py
