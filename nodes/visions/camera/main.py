"""Camera vision node entry point.

Start:  moss nodes run nodes/visions/camera                    # fore, CLI is owner
        moss nodes run nodes/visions/camera -- --camera 1 --port 9000
Debug:  ../.venv/bin/python main.py                            # ad-hoc (from_proc identity)
See it: open http://127.0.0.1:8765/stream                       # MJPEG viewer

Config is cell-level env (dotenv loads `.env`; copy `.env.example`). Two launch
arguments override env defaults: `--camera N` (which device) and `--port N`
(where the local viewer binds). The device is owned by the node lifecycle —
opened here on start, closed on stop; `watch` does not touch the device.
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

load_dotenv(_NODE_DIR / ".env")

from ghoshell_moss.core.blueprint.matrix import Matrix

from camera_node.camera import CameraController
from camera_node.source import OpenCVSource, list_cameras, make_face_detector
from camera_node.viewer import MjpegViewer


def _parse_args(argv: list[str]) -> dict[str, int | None]:
    """Minimal argv parse for the two launch overrides: --camera N, --port N."""
    out: dict[str, int | None] = {"camera": None, "port": None}
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ("--camera", "--port") and i + 1 < len(argv):
            out[arg.lstrip("--")] = int(argv[i + 1])
            i += 2
            continue
        i += 1
    return out


def _read_config(argv: list[str]) -> dict:
    overrides = _parse_args(argv)
    index = overrides["camera"] if overrides["camera"] is not None else int(os.getenv("CAMERA_INDEX", "0"))
    port = overrides["port"] if overrides["port"] is not None else int(os.getenv("VIEWER_PORT", "8765"))
    return {
        "index": index,
        "width": int(os.getenv("CAMERA_WIDTH", "640")),
        "height": int(os.getenv("CAMERA_HEIGHT", "480")),
        "fps": float(os.getenv("CAMERA_FPS", "10.0")),
        "viewer_host": os.getenv("VIEWER_HOST", "127.0.0.1"),
        "viewer_port": port,
    }


async def main(matrix: Matrix) -> None:
    logger = matrix.logger or logging.getLogger("moss.visions.camera")
    cfg = _read_config(sys.argv[1:])
    logger.info("camera node starting (config=%s)", cfg)

    source = OpenCVSource(cfg["index"], cfg["width"], cfg["height"])
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
    controller.open()  # device owned by node lifecycle; the probe already gated launch

    viewer = MjpegViewer(
        controller.latest_jpeg,
        host=cfg["viewer_host"],
        port=cfg["viewer_port"],
    )
    await viewer.start()

    # Presence announcement (authorization seed — the ghost learns a camera came online).
    try:
        await matrix.publish_event(
            f"camera node alive; viewer http://{cfg['viewer_host']}:{cfg['viewer_port']}/stream"
        )
    except Exception as e:
        logger.debug("publish_event failed: %s", e)

    loop_task = asyncio.create_task(controller.run_loop())

    try:
        await matrix.provide_channel(controller.as_channel())
    finally:
        loop_task.cancel()
        await viewer.stop()
        controller.close()


if __name__ == "__main__":
    Matrix.discover().run(main)
