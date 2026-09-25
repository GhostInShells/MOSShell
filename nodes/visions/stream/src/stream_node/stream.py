"""Stream vision controller — URL-agnostic; a stream is the latest JPEG frame.

The controller owns perception state and the channel surface. The physical
ingest (``source``) is injected, so it is testable without ffmpeg or a live
stream. The source keeps the latest frame; capture/watch read only that tail
frame (no buffering, no replay). Every model-bound image passes a threshold
gate: under max_edge/max_bytes it is sent as-is, over it is resampled once.
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
from ghoshell_moss.core.concepts.command import Observe
from ghoshell_moss.message import Message, Base64Image

_DEFAULT_FPS = 2.0
_DEFAULT_MAX_EDGE = 1568
_DEFAULT_MAX_BYTES = 512 * 1024
_DEFAULT_JPEG_QUALITY = 80
_DEFAULT_STALE_SECONDS = 5.0


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _jpeg_dims(jpeg: bytes) -> tuple[int, int]:
    try:
        with Image.open(io.BytesIO(jpeg)) as img:
            return img.size
    except Exception:
        return (0, 0)


class StreamController:
    """Owns stream perception state and exposes it as a channel."""

    def __init__(
        self,
        *,
        source,
        address: str,
        label: str = "",
        fps: float = _DEFAULT_FPS,
        max_edge: int = _DEFAULT_MAX_EDGE,
        max_bytes: int = _DEFAULT_MAX_BYTES,
        jpeg_quality: int = _DEFAULT_JPEG_QUALITY,
        stale_seconds: float = _DEFAULT_STALE_SECONDS,
        home: Path | None = None,
        logger=None,
    ):
        self._src = source
        self._address = address
        self._label = label or address
        self._fps = fps
        self._max_edge = max_edge
        self._max_bytes = max_bytes
        self._jpeg_quality = jpeg_quality
        self._stale_seconds = stale_seconds
        self._home = home or Path.cwd()
        self._logger = logger
        self._watch_on = False
        self._last_injected_ts: Optional[float] = None

    # ---- commands ---- #

    async def capture(self):
        """Capture the latest frame as an image observation."""
        latest = self._src.latest()
        if latest is None:
            return CommandUtil.observe(
                f"capture failed: no frame ({self._src.failed or 'connecting'})"
            )
        ts, jpeg = latest
        jpeg, dims, resampled = self._emission(jpeg)
        age = time.time() - ts
        text = (
            f"frame {dims[0]}x{dims[1]} "
            f"@ {datetime.fromtimestamp(ts):%H:%M:%S}.{int(ts % 1 * 1000):03d} "
            f"age={age:.1f}s{' (resampled)' if resampled else ''}"
        )
        return Observe(messages=[
            Message.new().with_content(text, Base64Image.from_binary("image/jpeg", jpeg))
        ])

    async def watch(self, on: bool) -> str:
        """Gate the per-round context image. Does not touch the connection."""
        self._watch_on = _to_bool(on)
        return self._help()

    async def status(self) -> dict:
        latest = self._src.latest()
        ts = latest[0] if latest else None
        return {
            "address": self._address,
            "label": self._label,
            "watch_on": self._watch_on,
            "latest_frame": datetime.fromtimestamp(ts).strftime("%H:%M:%S") if ts else None,
            "age": round(time.time() - ts, 1) if ts else None,
            "stale": self._is_stale(),
            "error": self._src.failed,
        }

    async def export(self, path: str) -> str:
        """Save the latest raw frame to a file under project home."""
        latest = self._src.latest()
        if latest is None:
            return f"export failed: no frame ({self._src.failed or 'connecting'})"
        _, jpeg = latest
        try:
            dest = self._resolve_export_path(path)
        except ValueError as e:
            return f"export failed: {e}"
        await asyncio.to_thread(self._write_jpeg, jpeg, dest)
        return f"exported {dest} ({len(jpeg)} bytes)"

    # ---- context ---- #

    def _context(self) -> list[Message]:
        watch = "on" if self._watch_on else "off"
        latest = self._src.latest()
        if latest is None:
            return [Message.new(name="__stream__").with_content(
                f"watch:{watch} {self._label}: no frame")]
        ts, jpeg = latest
        if self._watch_on and self._is_fresh(ts) and self._is_newer(ts):
            self._last_injected_ts = ts
            jpeg, dims, _ = self._emission(jpeg)
            label = (
                f"watch:{watch} {self._label} frame {dims[0]}x{dims[1]} "
                f"{time.time() - ts:.1f}s ago"
            )
            return [Message.new(name="__stream__").with_content(
                label, Base64Image.from_binary("image/jpeg", jpeg))]
        return [Message.new(name="__stream__").with_content(
            f"watch:{watch} {self._label}")]

    # ---- health / helpers ---- #

    def _check_health(self) -> None:
        if self._src.failed:
            raise RuntimeError(f"stream unavailable: {self._src.failed}")
        if self._is_stale():
            raise RuntimeError(f"stream stale: no frame for {self._stale_seconds}s")

    def _is_stale(self) -> bool:
        latest = self._src.latest()
        if latest is None:
            return False
        return (time.time() - latest[0]) > self._stale_seconds

    def _is_fresh(self, ts: float) -> bool:
        return (time.time() - ts) < max(1.0, self._stale_seconds)

    def _is_newer(self, ts: float) -> bool:
        return self._last_injected_ts is None or ts > self._last_injected_ts

    def _emission(self, jpeg: bytes) -> tuple[bytes, tuple[int, int], bool]:
        dims = _jpeg_dims(jpeg)
        if dims != (0, 0) and max(dims) <= self._max_edge and len(jpeg) <= self._max_bytes:
            return jpeg, dims, False
        img = Image.open(io.BytesIO(jpeg)).convert("RGB")
        w, h = img.size
        scale = min(1.0, self._max_edge / max(w, h)) if max(w, h) else 1.0
        if scale < 1.0:
            img = img.resize(
                (max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS,
            )
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=self._jpeg_quality)
        return buf.getvalue(), img.size, True

    def _resolve_export_path(self, path: str) -> Path:
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = self._home / p
        resolved = p.resolve()
        allowed = [self._home.resolve(), Path(tempfile.gettempdir()).resolve()]
        if not any(resolved == root or root in resolved.parents for root in allowed):
            raise ValueError(f"path {path} outside allowed roots (project home or tempdir)")
        return resolved

    @staticmethod
    def _write_jpeg(jpeg: bytes, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(jpeg)

    def _help(self) -> str:
        state = "watch:on" if self._watch_on else "watch:off"
        latest = self._src.latest()
        last = "no frame"
        if latest:
            last = f"frame {datetime.fromtimestamp(latest[0]):%H:%M:%S}"
        return f"{state} {self._label} fps={self._fps} {last}"

    # ---- channel ---- #

    def as_channel(self):
        chan = new_channel(
            "stream",
            description=(
                f"Stream vision on {self._label} — capture a frame or toggle "
                "per-round watch."
            ),
        )
        chan.build.instruction(
            f"Watching stream: {self._address}"
            + (f" ({self._label})" if self._label != self._address else "")
            + ". Continuous vision is expensive: with watch on, every round carries "
              "one current frame image. Turn watch off when not actively using it."
        )
        chan.build.refresh_meta(self._check_health)
        chan.build.context_messages(self._context)
        chan.build.command(name="capture")(self.capture)
        chan.build.command(name="watch")(self.watch)
        chan.build.command(name="status")(self.status)
        chan.build.command(name="export")(self.export)
        return chan
