---
name: 'camera'
description: 'Camera stream producer — own the device, run a capture loop, serve a live MJPEG stream for a stream node to consume.'
category: visions
# 非单例: 一个 node = 一个设备 (--camera N), 多摄像头 = 多实例.
# 设备独占由 cv2 打开失败兜底, 不用治理域 singleton 锁.
singleton: false
# 共享 visions venv 在 nodes/visions/ — 相对 node cwd 解析
exec:
  command: ../.venv/bin/python
  args: main.py
# 启动前探针: 依赖 / 策略 / 设备可用性, nonzero+stderr 即拒绝拉起并给出原因
check:
  command: ../.venv/bin/python
  args: check.py
---

Camera stream producer node. The device is owned by the node lifecycle (opened
on start, closed on stop). It exposes **no channel** — it only produces: a
continuous capture loop, a device-facing FaceTopic, and an MJPEG stream served
at `/stream`. Perception happens in a `stream` node consuming that address.

## Stream

The stream address is announced on start. Consume it:

    moss nodes run nodes/visions/stream -- --address http://127.0.0.1:8765/stream --label camera

## Configuration

Cell-level env (copy `.env.example`): `CAMERA_INDEX`, `CAMERA_WIDTH` /
`CAMERA_HEIGHT`, `CAMERA_FPS`, `VIEWER_HOST` / `VIEWER_PORT`, and the policy gate
`CAMERA_ALLOW` (set to `0` / `false` / `no` / `off` to refuse launch).

Two launch arguments override env defaults:

    moss nodes run nodes/visions/camera -- --camera 1 --port 9000

## View

Open `http://127.0.0.1:8765/stream` in a browser for the live field of view (MJPEG).

## Debug

    ../.venv/bin/python main.py
