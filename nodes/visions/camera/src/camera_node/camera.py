"""Camera vision node core — a cv2-agnostic controller.

The controller owns the perception state and the channel tree. It does NOT
import cv2; the physical frame acquisition (``source``) plus ``list_cameras``
and ``detect_faces`` are injected, so the controller can be tested without
camera hardware (inject a fake source in tests).

Node contract (nodes/visions/README.md):
  - config surface   -> set_config / get_config (safe bounds)
  - watch toggle     -> watch(on), background capture + FaceTopic publish
  - status via help  -> _help() (live state; shell-trajectory reads it)
  - minimal GUI      -> viewer, served by camera_node/viewer.py (MJPEG)
  - timed capture    -> rolling (ts, frame) cache; capture() returns snapshot
"""
from __future__ import annotations

import asyncio
import collections
import time
from datetime import datetime
from typing import Optional

from PIL import Image

from ghoshell_moss.core.blueprint.channel_builder import new_channel
from ghoshell_moss.message import Message, Base64Image
from ghoshell_moss.topics.vision import FaceTopic

_CACHE_SIZE = 32
_DEFAULT_FPS = 2.0
_FPS_MIN, _FPS_MAX = 0.5, 30.0
_RESOLUTIONS = [(640, 480), (1280, 720), (1920, 1080)]


def _to_bool(v) -> bool:
    """Normalize a CTML bool (often the string 'true'/'false') to a real bool."""
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "on")


class CameraController:
    """Owns camera perception state and exposes it as a ``camera`` channel."""

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
        self._src = source                  # .open(i,w,h)->bool .grab()->Image|None .is_opened() .close()
        self._list_cameras = list_cameras   # () -> [{"index", "name"}]
        self._detect = detect_faces         # (Image) -> [bbox dict]
        self._logger = logger

        # Rolling frame cache (timesteped snapshots) — touched only in the loop.
        self._cache: collections.deque[tuple[float, Image.Image]] = collections.deque(maxlen=_CACHE_SIZE)
        self._watch_on = False
        self._camera_index = camera_index
        self._fps = fps
        self._resolution = resolution
        self._last_faces: list[dict] = []
        self._last_error: Optional[str] = None

    # ---- perception loop ---- #

    async def run_loop(self) -> None:
        """Continuously run; captures + analyzes + publishes only when watch on."""
        while True:
            if self._watch_on and self._src.is_opened():
                frame = await self._grab()
                if frame is not None:
                    ts = time.time()
                    self._cache.append((ts, frame))
                    faces = self._detect(frame)
                    if faces:
                        self._last_faces = faces
                        # watch on -> camera pushes typed FaceTopic (not a raw
                        # stream string). sender = this cell's address.
                        f = faces[0]
                        try:
                            self._matrix.session.topics.pub(FaceTopic(
                                camera=str(self._camera_index),
                                x=f["x"], y=f["y"], w=f["w"], h=f["h"],
                                cx=f["cx"], cy=f["cy"], ts=ts,
                            ))
                        except Exception as e:  # topic closed etc.
                            self._logger.debug("face topic pub failed: %s", e)
                await asyncio.sleep(max(0.05, 1.0 / self._fps))
            else:
                await asyncio.sleep(0.2)

    async def _grab(self) -> Optional[Image.Image]:
        try:
            return await asyncio.to_thread(self._src.grab)
        except Exception as e:
            self._last_error = str(e)
            self._logger.debug("grab failed: %s", e)
            return None

    def _ensure_open(self) -> bool:
        if not self._src.is_opened():
            return self._src.open(self._camera_index, *self._resolution)
        return True

    # ---- channel commands ---- #

    async def watch(self, on: bool) -> str:
        """Toggle continuous perception. on: open camera + background capture + face topic. off: idle."""
        on = _to_bool(on)
        if on:
            if not self._ensure_open():
                self._last_error = f"camera {self._camera_index} not available"
                return f"failed to enable watch: {self._last_error}"
            self._watch_on = True
        else:
            self._watch_on = False
        return self._help()

    async def capture(self) -> str:
        """Capture one frame now (snapshot at call time) into the rolling cache."""
        if not self._ensure_open():
            self._last_error = f"camera {self._camera_index} not available"
            return f"capture failed: {self._last_error}"
        frame = await self._grab()
        if frame is None:
            return "capture failed: no frame"
        ts = time.time()
        self._cache.append((ts, frame))
        faces = self._detect(frame)
        if faces:
            self._last_faces = faces
        return f"captured {frame.size[0]}x{frame.size[1]} @ {datetime.fromtimestamp(ts):%H:%M:%S}.{int(ts % 1 * 1000):03d}"

    async def list_cameras(self) -> list[dict]:
        """List available camera devices (index + name)."""
        return self._list_cameras()

    async def set_camera(self, index: int) -> str:
        """Switch the active camera by index; use list_cameras to see options."""
        valid = {c["index"] for c in self._list_cameras()} or {index}
        if index not in valid:
            return f"camera {index} not available; choices {sorted(valid)}"
        if not self._src.open(index, *self._resolution):
            return f"failed to open camera {index}"
        self._camera_index = index
        return f"switched to camera [{index}]"

    async def set_config(self, fps: float | None = None, resolution: str | None = None) -> str:
        """Configure within safe bounds. fps: 0.5..30. resolution: 'WxH' from {640x480, 1280x720, 1920x1080}."""
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
            if self._src.is_opened():
                self._src.open(self._camera_index, w, h)
        return self._help()

    async def get_config(self) -> dict:
        """Current safe-bound config resolution."""
        return {"camera": self._camera_index, "fps": self._fps, "resolution": list(self._resolution),
                "watch_on": self._watch_on}

    async def detect_faces(self) -> list[dict]:
        """Detect faces in the latest frame (grabs one if watch is off)."""
        if not self._cache and not self._ensure_open():
            return []
        if not self._cache:
            frame = await self._grab()
            if frame is not None:
                self._cache.append((time.time(), frame))
        if not self._cache:
            return []
        faces = self._detect(self._cache[-1][1])
        if faces:
            self._last_faces = faces
        return faces

    async def status(self) -> dict:
        """Full current state."""
        latest_ts = self._cache[-1][0] if self._cache else None
        return {
            "camera": self._camera_index,
            "watch_on": self._watch_on,
            "fps": self._fps,
            "resolution": list(self._resolution),
            "cache_size": len(self._cache),
            "latest_frame": datetime.fromtimestamp(latest_ts).strftime("%H:%M:%S") if latest_ts else None,
            "faces": len(self._last_faces),
            "error": self._last_error,
        }

    async def authorize(self) -> str:
        """Authorization (informed consent) seed — privacy-sensitive perception.

        Full per-node consent / warrant (P2) is a known extension; this is a
        light seed that announces presence rather than gating perception.
        """
        self._logger.info("authorize called (consent seed, non-blocking)")
        try:
            await self._matrix.publish_event("camera awaits authorization (seed); fuller warrant to be layered")
        except Exception as e:
            self._logger.debug("auth event failed: %s", e)
        return "Authorization is a seed; perception is not gated. See nodes/visions/README.md."

    # ---- channel surface ---- #

    def _help(self) -> str:
        state = "watch:on" if self._watch_on else "watch:off"
        last = "no frame"
        if self._cache:
            last = f"frame {datetime.fromtimestamp(self._cache[-1][0]):%H:%M:%S}"
        return (
            f"cam[{self._camera_index}] {state} fps={self._fps} "
            f"res={self._resolution[0]}x{self._resolution[1]} "
            f"cache={len(self._cache)} {last} faces={len(self._last_faces)}"
        )

    async def _context(self) -> list[Message]:
        msgs: list[Message] = []
        if self._cache:
            ts, img = self._cache[-1]
            # Real-time vision frame → JPEG, not PNG. from_pil_image defaults to
            # PNG when image.format is None (frames come from Image.fromarray),
            # which balloons a 640x480 frame to ~500KB. Explicit JPEG keeps the
            # context image compact (~40KB) for the model / MCP transport.
            msgs.append(Message.new(name="__camera_frame__").with_content(
                f"[camera] frame {img.size[0]}x{img.size[1]}",
                Base64Image.from_pil_image(img, format="JPEG"),
            ))
        else:
            msgs.append(Message.new(name="__camera_state__").with_content(
                '[camera] no frame yet. Use <camera:watch on="true" /> or <camera:capture />.'
            ))
        if self._last_faces:
            f = self._last_faces[0]
            msgs.append(Message.new(name="__camera_face__").with_content(
                f"[camera] face at (cx={f['cx']:.2f}, cy={f['cy']:.2f}). "
                f"<camera:detect_faces /> to list all."
            ))
        msgs.append(Message.new(name="__camera_status__").with_content(self._help()))
        return msgs

    def as_channel(self):
        chan = new_channel("camera", description="Camera vision — persistent capture, face detection, and a local MJPEG viewer of the ghost's field of view. Configure via set_config; toggle perception via watch; the latest frame is always in context.")
        chan.build.help(self._help)
        chan.build.context_messages(self._context)
        chan.build.command(name="watch")(self.watch)
        chan.build.command(name="capture")(self.capture)
        chan.build.command(name="list_cameras")(self.list_cameras)
        chan.build.command(name="set_camera")(self.set_camera)
        chan.build.command(name="set_config")(self.set_config)
        chan.build.command(name="get_config")(self.get_config)
        chan.build.command(name="detect_faces")(self.detect_faces)
        chan.build.command(name="status")(self.status)
        chan.build.command(name="authorize")(self.authorize)
        return chan

    # ---- viewer accessor ---- #

    def latest_jpeg(self) -> Optional[bytes]:
        """Latest cached frame as JPEG bytes (for the MJPEG viewer)."""
        if not self._cache:
            return None
        img = self._cache[-1][1].convert("RGB")
        import io
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return buf.getvalue()
