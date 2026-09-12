"""Camera vision node core — a cv2-agnostic controller.

The controller owns the perception state and the channel surface. It does not
import cv2; the physical frame acquisition (``source``) plus ``list_cameras``
and ``detect_faces`` are injected, so it is testable without camera hardware.

Device ownership follows the vision family contract: the source is opened when
the node starts and closed when it stops. ``watch`` only gates whether a fresh
frame rides the per-round context message — it does not touch the device.
"""
from __future__ import annotations

import asyncio
import io
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from PIL import Image

from ghoshell_moss.core.blueprint.channel_builder import new_channel, CommandUtil
from ghoshell_moss.message import Message, Base64Image
from ghoshell_moss.topics.vision import FaceTopic

_DEFAULT_FPS = 10.0
_FPS_MIN, _FPS_MAX = 0.5, 30.0
_RESOLUTIONS = [(640, 480), (1280, 720), (1920, 1080)]
# A frame older than this is no longer "live" for the per-round context image.
_FRESH_WINDOW_SECONDS = 3.0


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


class CameraController:
    """Owns camera perception state and exposes it as a channel."""

    def __init__(
        self,
        matrix,
        *,
        source,
        list_cameras,
        detect_faces,
        logger,
        camera_index: int = 0,
        fps: float = _DEFAULT_FPS,
        resolution: tuple[int, int] = (640, 480),
    ):
        self._matrix = matrix
        self._src = source
        self._list_cameras = list_cameras
        self._detect = detect_faces
        self._logger = logger
        self._home = Path(matrix.home) if matrix is not None else Path.cwd()

        self._camera_index = camera_index
        self._fps = fps
        self._resolution = resolution
        self._watch_on = False

        self._latest: Optional[tuple[float, Image.Image]] = None
        self._latest_jpeg: Optional[bytes] = None
        self._last_injected_ts: Optional[float] = None
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

    # ---- capture loop (continuous: feeds the human stream + FaceTopic) ---- #

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
        self._latest = (ts, frame)
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

    def _check_health(self) -> None:
        """Raise on refresh when the device is dead — the channel short-circuits."""
        if self._failed:
            raise RuntimeError(f"camera {self._camera_index} unavailable: {self._failed}")

    # ---- commands ---- #

    async def watch(self, on: bool) -> str:
        """Gate the per-round context image. Does not touch the device."""
        self._watch_on = _to_bool(on)
        return self._help()

    async def capture(self, path: str | None = None):
        """Grab one frame and hand it back as an image observation."""
        frame = await self._grab()
        if frame is None:
            return CommandUtil.observe(f"capture failed: {self._failed}")
        ts = await self._ingest(frame)
        text = (
            f"frame {frame.size[0]}x{frame.size[1]} "
            f"@ {datetime.fromtimestamp(ts):%H:%M:%S}.{int(ts % 1 * 1000):03d}"
        )
        if path:
            try:
                dest = self._resolve_export_path(path)
                await asyncio.to_thread(self._write_jpeg, frame, dest)
                text += f" -> {dest}"
            except ValueError as e:
                text += f" (not saved: {e})"
        return CommandUtil.observe_image(text, frame)

    async def list_cameras(self) -> list[dict]:
        return await asyncio.to_thread(self._list_cameras)

    async def set_config(self, fps: float | None = None, resolution: str | None = None) -> str:
        if fps is not None:
            if not (_FPS_MIN <= fps <= _FPS_MAX):
                return f"fps {fps} out of bounds [{_FPS_MIN}, {_FPS_MAX}]"
            self._fps = fps
        if resolution is not None:
            try:
                w, h = [int(x) for x in str(resolution).lower().split("x")]
            except (ValueError, AttributeError):
                return f"resolution '{resolution}' not parseable as WxH"
            if (w, h) not in _RESOLUTIONS:
                return f"resolution {(w, h)} not allowed; choices {_RESOLUTIONS}"
            self._resolution = (w, h)
            self._src.set_resolution(w, h)
        return self._help()

    async def get_config(self) -> dict:
        return {
            "camera": self._camera_index,
            "fps": self._fps,
            "resolution": list(self._resolution),
            "watch_on": self._watch_on,
        }

    async def status(self) -> dict:
        latest_ts = self._latest[0] if self._latest else None
        return {
            "camera": self._camera_index,
            "watch_on": self._watch_on,
            "fps": self._fps,
            "resolution": list(self._resolution),
            "latest_frame": datetime.fromtimestamp(latest_ts).strftime("%H:%M:%S") if latest_ts else None,
            "error": self._failed,
        }

    async def authorize(self) -> str:
        self._logger.info("authorize called (consent seed, non-blocking)")
        try:
            await self._matrix.publish_event("camera awaits authorization (seed)")
        except Exception as e:
            self._logger.debug("auth event failed: %s", e)
        return "Perception is not gated; authorization is a consent seed, not a blocker."

    # ---- context ---- #

    def _context(self) -> list[Message]:
        if self._latest is None:
            return [Message.new(name="__camera__").with_content("no frame yet")]
        ts, img = self._latest
        watch = "on" if self._watch_on else "off"
        label = f"watch:{watch} frame {img.size[0]}x{img.size[1]} {time.time() - ts:.1f}s ago"
        if self._watch_on and self._is_fresh(ts) and self._is_newer(ts):
            self._last_injected_ts = ts
            return [Message.new(name="__camera__").with_content(
                label, Base64Image.from_pil_image(img, format="JPEG"))]
        return [Message.new(name="__camera__").with_content(label)]

    def _is_fresh(self, ts: float) -> bool:
        return (time.time() - ts) < max(1.0, _FRESH_WINDOW_SECONDS / self._fps)

    def _is_newer(self, ts: float) -> bool:
        return self._last_injected_ts is None or ts > self._last_injected_ts

    def _help(self) -> str:
        state = "watch:on" if self._watch_on else "watch:off"
        last = "no frame"
        if self._latest:
            last = f"frame {datetime.fromtimestamp(self._latest[0]):%H:%M:%S}"
        return (
            f"{state} fps={self._fps} res={self._resolution[0]}x{self._resolution[1]} "
            f"device={self._camera_index} {last}"
        )

    # ---- export path (files land under node home or tempdir only) ---- #

    def _resolve_export_path(self, path: str) -> Path:
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = self._home / p
        resolved = p.resolve()
        allowed = [self._home.resolve(), Path(tempfile.gettempdir()).resolve()]
        if not any(resolved == root or root in resolved.parents for root in allowed):
            raise ValueError(f"path {path} outside allowed roots (node home or tempdir)")
        return resolved

    @staticmethod
    def _write_jpeg(frame: Image.Image, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        frame.convert("RGB").save(dest, format="JPEG", quality=80)

    # ---- channel surface ---- #

    def as_channel(self):
        chan = new_channel(
            "camera",
            description="Camera vision — capture a frame, toggle continuous perception, and expose the live field of view over a local MJPEG stream.",
        )
        chan.build.instruction(
            "Continuous vision is expensive: with watch on, every round of context "
            "carries one current frame image. Turn watch off when you are not actively "
            "using the camera."
        )
        chan.build.refresh_meta(self._check_health)
        chan.build.context_messages(self._context)
        chan.build.command(name="watch")(self.watch)
        chan.build.command(name="capture")(self.capture)
        chan.build.command(name="list_cameras")(self.list_cameras)
        chan.build.command(name="set_config")(self.set_config)
        chan.build.command(name="get_config")(self.get_config)
        chan.build.command(name="status")(self.status)
        chan.build.command(name="authorize")(self.authorize)
        return chan

    # ---- viewer ---- #

    def latest_jpeg(self) -> Optional[bytes]:
        return self._latest_jpeg
