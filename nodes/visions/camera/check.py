"""Pre-launch probe for the camera node (NODE.md `check:`).

Runs as an independent process, zero cooperation with main.py. Exit code only:
0 = the environment can run; nonzero + stderr = the reason (surfaces to the
launcher and, via the nodes channel, to the model). Write "why", not a stack
trace.
"""
from __future__ import annotations

import os
import pathlib
import shutil
import sys

_NODE_DIR = pathlib.Path(__file__).resolve().parent

try:
    from dotenv import load_dotenv

    load_dotenv(_NODE_DIR / ".env")
except ImportError:
    pass

_FALSE = ("0", "false", "no", "off")


def _fail(reason: str) -> int:
    print(reason, file=sys.stderr)
    return 1


def main() -> int:
    allow = os.getenv("CAMERA_ALLOW", "1").strip().lower()
    if allow in _FALSE:
        return _fail(f"camera disabled by policy: CAMERA_ALLOW={allow!r}")

    if shutil.which("ffmpeg") is None:
        return _fail("ffmpeg not found (required for device enumeration)")

    try:
        import cv2  # noqa: F401
    except ImportError:
        return _fail("opencv (cv2) not importable in the visions venv")

    index = int(os.getenv("CAMERA_INDEX", "0"))
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        return _fail(f"camera {index} not available (cannot open device)")
    cap.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
