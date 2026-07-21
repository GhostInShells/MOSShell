# MOSS App: sensors/vision

Camera vision input channel. Persistent OpenCV camera handle (green light always on),
real-time face detection, and face→eye tracking via stream topic.

No LLM calls — Ghost gets the latest frame via `context_messages` and sees for itself.

## Setup

```bash
cd .moss_ws/apps/sensors/vision
uv sync
```

## How it works

- Persistent `cv2.VideoCapture` — camera stays open, green light on, ~0.3s per frame
- Face detection via OpenCV Haar cascade every ~0.3s (pure local, no LLM)
- Face positions published to `vision/face` stream topic — ai_eye subscribes directly
- `context_messages` provides latest frame + face coords to Ghost on each think cycle

## Architecture

```
Camera (cv2.VideoCapture — persistent, green light always on)
    │
    ▼
_capture_loop (standalone asyncio task, every 0.3s)
    ├── _grab_frame() → _latest_frame         [< 10ms]
    ├── _detect_faces() → _latest_faces        [~20ms]
    └── pub_stream_delta("vision/face")        → ai_eye (stream topic)

Ghost think cycle:
    └── context_messages → latest frame + face coords
```

## Commands

- `capture(camera_index)` — grab a single frame, return summary
- `list_cameras()` — list available AVFoundation video devices
- `set_camera(camera_index)` — switch the active camera
- `detect_faces()` — detect faces in latest frame
- `pause_tracking()` — pause automatic face→eye tracking
- `resume_tracking()` — resume automatic face→eye tracking

## Stream topic: vision/face

Vision publishes face coordinates to `vision/face` stream. Payload format:
```
"cx,cy"   (e.g. "0.44,0.74" — normalized 0..1, comma-separated)
```

ai_eye subscribes to this topic and updates gaze directly. No CTML, no Ghost in the loop.

## Dependencies

- `opencv-python-headless` (declared in app's own `pyproject.toml`)
- `Pillow`, `numpy` — used directly, provided transitively by `ghoshell-moss[host]`
- `ghoshell-moss[host]` via editable install from workspace root
