"""Camera stream producer — a device turned into a continuous MJPEG stream.

The producer owns the camera device and runs a capture loop that feeds two
consumers: the local MJPEG viewer (the stream a stream node consumes) and the
device-facing FaceTopic. It exposes no model-facing channel — the camera is a
producer; perception happens in the stream node.
"""
from __future__ import annotations

import asyncio
import io
import time
from typing import Optional

from PIL import Image

from ghoshell_moss.types.topics.vision import FaceTopic

_DEFAULT_FPS = 10.0


class CameraProducer:
    """Owns the camera device and turns it into a continuous MJPEG stream."""

    def __init__(
        self,
        matrix,
        *,
        source,
        detect_faces,
        logger,
        camera_index: int = 0,
        fps: float = _DEFAULT_FPS,
        resolution: tuple[int, int] = (640, 480),
    ):
        self._matrix = matrix
        self._src = source
        self._detect = detect_faces
        self._logger = logger
        self._camera_index = camera_index
        self._fps = fps
        self._resolution = resolution

        self._latest_jpeg: Optional[bytes] = None
        self._failed: Optional[str] = None
        self._grab_lock = asyncio.Lock()

    # ---- device lifecycle (node owns the device) ---- #

    def open(self) -> bool:
        """Open the device once for the node's lifetime. False on failure."""
        if not self._src.open(self._camera_index, *self._resolution):
            self._failed = f"camera {self._camera_index} not available"
            return False
        return True

    def close(self) -> None:
        self._src.close()

    # ---- capture loop (feeds the human stream + FaceTopic) ---- #

    async def run_loop(self) -> None:
        while True:
            frame = await self._grab()
            if frame is not None:
                ts = await self._ingest(frame)
                faces = await asyncio.to_thread(self._detect, frame)
                if faces:
                    self._publish_face(faces[0], ts)
            await asyncio.sleep(max(0.05, 1.0 / self._fps))

    async def _grab(self) -> Optional[Image.Image]:
        async with self._grab_lock:
            try:
                frame = await asyncio.to_thread(self._src.grab)
            except Exception as e:
                self._failed = str(e)
                return None
            self._failed = None if frame is not None else f"camera {self._camera_index} grab failed"
            return frame

    async def _ingest(self, frame: Image.Image) -> float:
        ts = time.time()
        self._latest_jpeg = await asyncio.to_thread(self._encode_jpeg, frame)
        return ts

    @staticmethod
    def _encode_jpeg(frame: Image.Image) -> bytes:
        buf = io.BytesIO()
        frame.convert("RGB").save(buf, format="JPEG", quality=80)
        return buf.getvalue()

    def _publish_face(self, face: dict, ts: float) -> None:
        if self._matrix is None:
            return
        try:
            self._matrix.session.topics.pub(FaceTopic(
                camera=str(self._camera_index),
                x=face["x"], y=face["y"], w=face["w"], h=face["h"],
                cx=face["cx"], cy=face["cy"], ts=ts,
            ))
        except Exception as e:
            self._logger.debug("face topic pub failed: %s", e)

    # ---- stream ---- #

    def latest_jpeg(self) -> Optional[bytes]:
        return self._latest_jpeg
