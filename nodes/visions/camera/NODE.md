---
name: 'camera'
description: 'Camera vision — capture a frame, toggle continuous perception (watch), and stream the live field of view for a human viewer.'
category: visions
singleton: true
# 共享 visions venv 在 nodes/visions/ — 相对 node cwd 解析
exec:
  command: ../.venv/bin/python
  args: main.py
# 启动前探针: 依赖 / 策略 / 设备可用性, nonzero+stderr 即拒绝拉起并给出原因
check:
  command: ../.venv/bin/python
  args: check.py
---

Camera vision node. The device is owned by the node lifecycle (opened on start,
closed on stop). `watch` only gates whether a fresh frame rides each round of
context — it does not touch the device.

## Configuration

Cell-level env (copy `.env.example`): `CAMERA_INDEX`, `CAMERA_WIDTH` /
`CAMERA_HEIGHT`, `CAMERA_FPS`, `VIEWER_HOST` / `VIEWER_PORT`, and the policy gate
`CAMERA_ALLOW` (set to `0` / `false` / `no` / `off` to refuse launch).

Two launch arguments override env defaults:

    moss nodes run nodes/visions/camera -- --camera 1 --port 9000

Runtime re-config within safe bounds via `set_config(fps=0.5..30,
resolution="640x480"|"1280x720"|"1920x1080")`.

## View

Open `http://127.0.0.1:8765/stream` in a browser for the live field of view (MJPEG).

## Debug

    ../.venv/bin/python main.py
