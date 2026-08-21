---
name: 'camera'
description: 'Camera vision — persistent capture, face detection, local MJPEG viewer of the ghost''s field of view.'
category: visions
singleton: true
# 共享 visions venv 在 nodes/visions/ — 相对 node cwd 解析
exec:
  command: ../.venv/bin/python
  args: main.py
---

Camera vision node. Persistent OpenCV capture (green light stays on), rolling
frame cache, face detection → typed `vision/face` FaceTopic when watch on, and
a local MJPEG stream so a human can see what the ghost's camera sees.

The camera perception bit is privacy-sensitive — this is a vision family
awareness ("知情"), see `nodes/visions/README.md` for the authorization seed.

## Configuration (safe bounds)

Cell-level defaults via `.env` (copy `.env.example`): `CAMERA_INDEX`,
`CAMERA_WIDTH`/`CAMERA_HEIGHT`, `CAMERA_FPS`, `WATCH_ON_START`,
`VIEWER_HOST`/`VIEWER_PORT`. Loaded at startup by dotenv; runtime re-config
via commands:

- `set_config(fps=0.5..30, resolution="640x480"|"1280x720"|"1920x1080")`
- `get_config()` — read current safe-bound config
- `list_cameras()` / `set_camera(index)`

## CTML invocation

    <camera:watch on="true" />
    <camera:capture />
    <camera:detect_faces />
    <camera:status />
    <camera:set_config fps="5.0" resolution="1280x720" />

## View

Open `http://127.0.0.1:8765/stream` in a browser to see the field of view (MJPEG).

## Debug

    ../.venv/bin/python main.py
