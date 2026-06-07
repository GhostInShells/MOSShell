"""Camera Vision Channel for MOSS — persistent capture + local face detection.

Captures frames from a persistent OpenCV camera handle (green light stays on),
runs face detection every ~0.3s for real-time eye tracking. No LLM — Ghost
gets the latest frame via context_messages and sees for itself.

Architecture:
  main() → capture loop as standalone asyncio task (lifetime)
    every 0.3s: grab frame + detect faces → pub_stream_delta("vision/face")
  face→eye: stream topic (vision → ai_eye), no Ghost in the loop
  commands: capture, detect_faces, list_cameras, set_camera,
            pause_tracking, resume_tracking
  context_messages: latest frame + face coords for Ghost to see
"""

import asyncio
import logging
import subprocess
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from dotenv import load_dotenv
from PIL import Image

from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.core.blueprint.channel_builder import (
    new_channel,
    Message,
)

# Explicitly load workspace .env, overriding any shell-cached values
_ws_root = Path(__file__).resolve().parents[3]  # apps/sensors/vision/main.py → .moss_ws
load_dotenv(_ws_root / ".env", override=True)

# ── Face detection (Haar cascade, bundled with opencv) ──

_face_cascade: cv2.CascadeClassifier | None = None


def _get_face_cascade() -> cv2.CascadeClassifier:
    global _face_cascade
    if _face_cascade is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(cascade_path)
    return _face_cascade


def _detect_faces(frame: Image.Image) -> list[dict]:
    """Detect faces in a PIL frame. Returns list of {x, y, w, h, cx, cy} normalized 0..1."""
    cascade = _get_face_cascade()
    if cascade.empty():
        return []

    # Convert PIL to BGR numpy for OpenCV
    cv_frame = cv2.cvtColor(np.array(frame.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2GRAY)

    # CLAHE histogram equalization for better contrast in varied lighting
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    faces = cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=2, minSize=(30, 30))
    h, w = gray.shape

    results = []
    for (fx, fy, fw, fh) in faces:
        results.append({
            "x": round(fx / w, 3),
            "y": round(fy / h, 3),
            "w": round(fw / w, 3),
            "h": round(fh / h, 3),
            "cx": round((fx + fw / 2) / w, 3),
            "cy": round((fy + fh / 2) / h, 3),
        })
    return results

# ── Camera discovery ──

def _scan_cameras() -> list[dict]:
    try:
        result = subprocess.run(
            ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            capture_output=True, text=True, timeout=5,
        )
        stderr = result.stderr
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []

    cameras = []
    in_video = False
    for line in stderr.splitlines():
        if "AVFoundation video devices:" in line:
            in_video = True
            continue
        if "AVFoundation audio devices:" in line:
            break
        if in_video and "]" in line:
            try:
                idx_str = line[line.rindex("[") + 1 : line.rindex("]")]
                idx = int(idx_str)
                name = line[line.rindex("]") + 1 :].strip()
                cameras.append({"index": idx, "name": name})
            except (ValueError, IndexError):
                continue
    return cameras


# ── Persistent camera (OpenCV VideoCapture) ──

_cap: cv2.VideoCapture | None = None
_cap_lock = threading.Lock()  # guards camera init/release across threads


def _init_camera(camera_index: int) -> bool:
    """Open a persistent camera handle. Green light stays on."""
    global _cap
    with _cap_lock:
        if _cap is not None and _cap.isOpened():
            _cap.release()
        _cap = cv2.VideoCapture(camera_index)
        if not _cap.isOpened():
            _cap = None
            return False
        _cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        _cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return True


def _grab_frame() -> Optional[Image.Image]:
    """Grab one frame from the persistent camera handle. Fast (<10ms)."""
    with _cap_lock:
        if _cap is None or not _cap.isOpened():
            return None
        ret, frame = _cap.read()
        if not ret:
            return None
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def _release_camera():
    global _cap
    with _cap_lock:
        if _cap is not None:
            _cap.release()
            _cap = None


# ── Channel definition ──

channel = new_channel(
    name="sensors_vision",
    description=(
        "Camera vision channel — persistent capture + local face detection. "
        "Auto-tracks faces → eyes every 0.3s (pure OpenCV, no LLM). "
        "Ghost sees the latest frame via context_messages. "
        "Ghost can pause_tracking() / resume_tracking() to take over gaze control."
    ),
)

# Mutable state
_cameras: list[dict] = []
_current_camera_index: int = 0
_latest_frame: Optional[Image.Image] = None
_latest_faces: list[dict] = []
_logger: Optional[logging.Logger] = None
_matrix: Optional[Matrix] = None
_last_look_at: str = ""  # avoid sending duplicate stream messages for same position
_tracking_enabled: bool = True  # Ghost can pause face→eye tracking


@channel.build.command()
async def pause_tracking() -> str:
    """Pause automatic face→eye tracking. Ghost calls this when it wants to control gaze."""
    global _tracking_enabled
    _tracking_enabled = False
    return "Face tracking paused. Ghost has gaze control."


@channel.build.command()
async def resume_tracking() -> str:
    """Resume automatic face→eye tracking."""
    global _tracking_enabled
    _tracking_enabled = True
    return "Face tracking resumed."


@channel.build.command()
async def list_cameras() -> list[dict]:
    """List all available camera devices with their indices."""
    global _cameras
    _cameras = _scan_cameras()
    return _cameras


@channel.build.command()
async def set_camera(camera_index: int) -> str:
    """Switch to a different camera by index. Use list_cameras to see available devices."""
    global _current_camera_index, _cameras
    _cameras = _scan_cameras()
    valid = {c["index"] for c in _cameras}
    if camera_index not in valid:
        return f"Camera {camera_index} not available. Choices: {sorted(valid)}"
    _current_camera_index = camera_index
    _init_camera(camera_index)
    name = next(c["name"] for c in _cameras if c["index"] == camera_index)
    return f"Switched to [{camera_index}] {name}"


@channel.build.command()
async def capture(camera_index: int = -1) -> str:
    """Capture a single frame from the current camera. Returns dimensions and format."""
    global _latest_frame, _current_camera_index

    idx = _current_camera_index if camera_index < 0 else camera_index
    if idx != _current_camera_index:
        if not _init_camera(idx):
            return f"Failed to open camera {idx}"
        _current_camera_index = idx

    frame = _grab_frame()
    if frame is None:
        return f"Failed to capture from camera {idx}"

    _latest_frame = frame
    return f"Frame captured: {frame.size[0]}x{frame.size[1]}, mode={frame.mode}"


@channel.build.command()
async def detect_faces() -> list[dict]:
    """Detect faces in the latest frame. Returns list of face bboxes (normalized 0..1).
    Each face: {x, y, w, h, cx, cy} where cx,cy is the face center."""
    global _latest_faces, _latest_frame
    if _latest_frame is None:
        frame = _grab_frame()
        if frame is None:
            return []
        _latest_frame = frame
    _latest_faces = _detect_faces(_latest_frame)
    return _latest_faces


@channel.build.context_messages
async def context() -> list:
    """Provide camera frame + face position as dynamic context for Ghost."""
    parts: list = []

    if _latest_frame is not None:
        w, h = _latest_frame.size
        parts.append(
            Message.new().with_content(f"[sensors/vision] Camera {_current_camera_index} — {w}x{h}")
        )
        parts.append(Message.new().with_content(_latest_frame.copy()))

    if _latest_faces:
        face = _latest_faces[0]  # primary face
        parts.append(
            Message.new().with_content(
                f"[sensors/vision] Face at (cx={face['cx']:.2f}, cy={face['cy']:.2f}). "
                f"[{'auto-tracking' if _tracking_enabled else 'paused'}]. "
                f"<apps.sensors_vision:pause_tracking /> / <apps.sensors_vision:resume_tracking /> to toggle."
            )
        )
        if len(_latest_faces) > 1:
            parts.append(
                Message.new().with_content(
                    f"[sensors/vision] {len(_latest_faces)} faces detected in frame."
                )
            )

    if not parts:
        parts.append(
            Message.new().with_content(
                "[sensors/vision] No frame yet. Use apps.sensors_vision:capture."
            )
        )

    return parts


# ── Capture loop (standalone background task) ──

async def _capture_loop():
    """Persistent capture loop: grab frame + detect faces every 0.3s.

    Runs as a standalone asyncio task in main(), independent of channel lifecycle.
    Pure local — no LLM calls. Ghost sees frames via context_messages.
    """
    global _latest_frame, _latest_faces, _last_look_at

    _init_camera(_current_camera_index)
    if _logger:
        _logger.info(f"Camera opened: index={_current_camera_index}")

    try:
        while True:
            frame = _grab_frame()
            if frame is not None:
                _latest_frame = frame
                _latest_faces = _detect_faces(frame)

                # Face → eye tracking: publish face coords via stream topic
                if _tracking_enabled and _latest_faces and _matrix is not None:
                    face = _latest_faces[0]
                    # Quantize to ~2% to avoid jitter and duplicate messages
                    cx = round(face["cx"] / 0.02) * 0.02
                    cy = round(face["cy"] / 0.02) * 0.02
                    payload = f"{cx:.2f},{cy:.2f}"
                    if payload != _last_look_at:
                        _last_look_at = payload
                        _matrix.session.pub_stream_delta("vision/face", payload.encode())

            await asyncio.sleep(0.3)
    except asyncio.CancelledError:
        pass
    finally:
        _release_camera()
        if _logger:
            _logger.info("Camera released.")


# ── App entry point ──

async def main(matrix: Matrix):
    global _logger, _cameras, _matrix

    _matrix = matrix
    _logger = logging.getLogger("Vision")
    logging.basicConfig(level=logging.WARNING)

    _cameras = _scan_cameras()
    names = ", ".join(f"[{c['index']}] {c['name']}" for c in _cameras)
    _logger.info(f"sensors/vision started. Cameras: {names or 'none'}")

    # Start channel (non-blocking)
    matrix.provide_channel(channel)

    # Start capture loop as standalone background task
    capture_task = asyncio.create_task(_capture_loop())

    # Keep process alive
    quit_event = asyncio.Event()
    try:
        await quit_event.wait()
    except asyncio.CancelledError:
        pass
    finally:
        capture_task.cancel()
        try:
            await capture_task
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    Matrix.discover().run(main)