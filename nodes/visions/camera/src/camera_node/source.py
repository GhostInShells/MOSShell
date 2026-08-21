"""OpenCV-backed camera source for the camera vision node.

The CameraController is cv2-agnostic; it receives a live ``source`` object plus
``list_cameras`` / ``detect_faces``. This module provides the real OpenCV
implementations, loaded only by the camera node venv (which has opencv).
"""
from __future__ import annotations

import subprocess
from typing import Callable, Optional

import cv2
import numpy as np
from PIL import Image


class OpenCVSource:
    """Persistent ``cv2.VideoCapture`` as a frame source.

    ``grab()`` is blocking; the controller wraps it in ``asyncio.to_thread`` so
    it never stalls the matrix event loop.
    """

    def __init__(self, index: int = 0, width: int = 640, height: int = 480):
        self._index = index
        self._width = width
        self._height = height
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self, index: int | None = None, width: int | None = None, height: int | None = None) -> bool:
        """Open the given camera (default: current index/resolution)."""
        if index is not None:
            self._index = index
        if width is not None:
            self._width = width
        if height is not None:
            self._height = height
        self.close()
        cap = cv2.VideoCapture(self._index)
        if not cap.isOpened():
            cap.release()
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap = cap
        return True

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def grab(self) -> Optional[Image.Image]:
        """Grab one frame, returned as RGB PIL Image (or None)."""
        if not self.is_opened():
            return None
        ret, frame = self._cap.read()
        if not ret:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def list_cameras() -> list[dict]:
    """Scan AVFoundation video devices via ffmpeg (macOS)."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            capture_output=True,
            text=True,
            timeout=5,
        )
        stderr = result.stderr
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []
    cameras: list[dict] = []
    in_video = False
    for line in stderr.splitlines():
        if "AVFoundation video devices:" in line:
            in_video = True
            continue
        if "AVFoundation audio devices:" in line:
            break
        if in_video and "]" in line:
            try:
                idx = int(line[line.rindex("[") + 1 : line.rindex("]")])
                name = line[line.rindex("]") + 1 :].strip()
                cameras.append({"index": idx, "name": name})
            except (ValueError, IndexError):
                continue
    return cameras


def make_face_detector() -> Callable[[Image.Image], list[dict]]:
    """Haar cascade face detector; returns normalized bboxes {x,y,w,h,cx,cy}.

    Constructed defensively: if the cascade XML is missing (e.g. opencv 5.x
    removed Haar), returns a no-op detector — the node still streams frames
    (watch/capture/context) even though face tracking degrades to empty.
    """
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
    except Exception:
        cascade = cv2.CascadeClassifier()

    def detect(frame: Image.Image) -> list[dict]:
        if cascade.empty():
            return []
        cv_frame = cv2.cvtColor(np.array(frame.convert("RGB")), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=2, minSize=(30, 30))
        h, w = gray.shape
        out: list[dict] = []
        for (fx, fy, fw, fh) in faces:
            out.append({
                "x": round(fx / w, 3),
                "y": round(fy / h, 3),
                "w": round(fw / w, 3),
                "h": round(fh / h, 3),
                "cx": round((fx + fw / 2) / w, 3),
                "cy": round((fy + fh / 2) / h, 3),
            })
        return out

    return detect
