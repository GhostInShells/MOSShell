"""Camera vision node entry point.

Start:  moss nodes run nodes/visions/camera    # fore, CLI is owner
Debug:  ../.venv/bin/python main.py             # ad-hoc (from_proc identity)
See it: open http://127.0.0.1:8765/stream       # MJPEG viewer

Config is cell-level via `.env` (copy `.env.example`); loaded first via
dotenv. Defaults: camera 0, 640x480, fps 2.0, watch on at start.

Follows the qt_screen node pattern: heavy deps (cv2/aiohttp) live in this
node's venv; main() is a thin shell that wires an OpenCV source into a
cv2-agnostic CameraController, serves a local MJPEG viewer, and provides the
channel. See nodes/visions/README.md for the family contract.
"""
from __future__ import annotations

import asyncio
import logging
import os
import pathlib
import sys

_NODE_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_NODE_DIR / "src"))

from dotenv import load_dotenv

# Cell-level config — .env (gitignored) overrides .env.example defaults.
load_dotenv(_NODE_DIR / ".env")

from ghoshell_moss.core.blueprint.matrix import Matrix

from camera_node.camera import CameraController
from camera_node.source import OpenCVSource, list_cameras, make_face_detector
from camera_node.viewer import MjpegViewer


def _bool(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


def _read_config() -> dict:
    return {
        "index": int(os.getenv("CAMERA_INDEX", "0")),
        "width": int(os.getenv("CAMERA_WIDTH", "640")),
        "height": int(os.getenv("CAMERA_HEIGHT", "480")),
        "fps": float(os.getenv("CAMERA_FPS", "2.0")),
        "watch_on_start": _bool(os.getenv("WATCH_ON_START", "true")),
        "viewer_host": os.getenv("VIEWER_HOST", "127.0.0.1"),
        "viewer_port": int(os.getenv("VIEWER_PORT", "8765")),
    }


async def main(matrix: Matrix) -> None:
    logger = matrix.logger or logging.getLogger("moss.visions.camera")
    cfg = _read_config()
    logger.info("camera node starting (config=%s)", cfg)

    source = OpenCVSource(cfg["index"], cfg["width"], cfg["height"])
    cameras = list_cameras()

    controller = CameraController(
        matrix,
        source=source,
        list_cameras=list_cameras,
        detect_faces=make_face_detector(),
        logger=logger,
        camera_index=cfg["index"],
        fps=cfg["fps"],
        resolution=(cfg["width"], cfg["height"]),
    )

    # Minimal GUI: local MJPEG viewer so a human sees the ghost's view.
    viewer = MjpegViewer(
        controller.latest_jpeg,
        host=cfg["viewer_host"],
        port=cfg["viewer_port"],
    )
    await viewer.start()

    # Perception loop: capture/analyze/publish only when watch is on.
    watch_task = asyncio.create_task(controller.run_loop())

    # Watch on at start (cell-level default) — open camera + begin capture.
    if cfg["watch_on_start"]:
        await controller.watch(True)

    # Presence announcement (authorization seed — see CameraController.authorize).
    try:
        await matrix.publish_event(
            f"camera node alive ({len(cameras)} camera(s)); "
            f"viewer http://{cfg['viewer_host']}:{cfg['viewer_port']}/stream"
        )
    except Exception as e:
        logger.debug("publish_event failed: %s", e)

    try:
        await matrix.provide_channel(controller.as_channel())
    finally:
        watch_task.cancel()
        await viewer.stop()
        source.close()


if __name__ == "__main__":
    Matrix.discover().run(main)
