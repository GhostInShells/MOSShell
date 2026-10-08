"""ffmpeg argument construction for the push node — the limited interface.

The model never names a device, a format, or an ffmpeg flag. It names a
**source** plus a few tuning knobs, and this module translates that into argv.
That is the whole authorization surface: a finite, enumerable set of things the
model can ask to stream.

argv is a list of strings, never a shell line, so nothing here can grow into a
`bash -c` by accident.

Which capture device to read comes in as a parameter (``index``); when the caller
does not name one, it falls back to node-side config (``PUSH_SCREEN_INDEX`` /
``PUSH_CAMERA_INDEX``) whose defaults match a stock Mac. On macOS both screen and
camera come through avfoundation; the screen is a distinct capture index from the
cameras, and which index it lands on depends on what is plugged in — so the
default stays generic and pinning one display is a per-machine escape hatch.

macOS gate: the ``screen`` source needs Screen Recording permission (TCC) granted
to whatever process runs ffmpeg. Without it the capture does not error — it
blocks. The probe must therefore never open the screen (it would hang); the
permission requirement is documented and surfaced, not auto-tested.
"""
from __future__ import annotations

import os
import sys
from typing import Optional

__all__ = ["SOURCES", "build_argv", "source_summary"]

_SOURCES: dict[str, str] = {
    "screen": "the local display — whatever is on screen right now",
    "camera": "the local camera device",
}

_SOURCE_FPS: dict[str, float] = {
    "screen": 10.0,
    "camera": 30.0,
}
"""Per-source capture rate — avfoundation is a *discrete-rate* device family: a rate the
device does not advertise fails the open outright instead of being clamped, so a default
has to be a rate the source actually offers. Camera hardware advertises 30 (often also
15); screen capture is synthetic and takes any rate, and 10 keeps a JPEG near 100KB. A
human preview is the most demanding consumer — nothing here needs more than 30."""

DEFAULT_MAX_WIDTH = 1280
"""Downscale ceiling. The screen is the only source that can exceed it (Retina
displays are 2880+ wide); 1280 keeps a JPEG near 100KB without visible loss at
preview size."""

DEFAULT_QUALITY = 5
"""ffmpeg mjpeg `-q:v`, 2 (best) .. 31 (worst). Same default as the stream node."""

_FPS_BOUNDS = (1.0, 30.0)
_QUALITY_BOUNDS = (2, 31)
_WIDTH_BOUNDS = (320, 3840)

SOURCES = _SOURCES
"""The enumerable source menu. Keys are what the model may pass as ``source``."""


def source_summary() -> str:
    """The source menu as one line, for the channel instruction."""
    return ", ".join(f"{name} ({desc})" for name, desc in sorted(_SOURCES.items()))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def capture_index_for(source: str) -> int:
    """The avfoundation capture index this node streams for ``source`` by default.

    Node-side identity, so it comes from env rather than the request. The
    defaults match a stock Mac: index 0 is the built-in camera, index 1 is the
    main display.
    """
    if source == "camera":
        return _env_int("PUSH_CAMERA_INDEX", 0)
    return _env_int("PUSH_SCREEN_INDEX", 1)


def default_fps_for(source: str) -> float:
    """The capture rate this node streams ``source`` at when the model omits ``fps``."""
    if source not in _SOURCES:
        raise ValueError(
            f"unknown source {source!r} (one of: {', '.join(sorted(_SOURCES))})"
        )
    return _SOURCE_FPS[source]


def build_argv(
    source: str,
    *,
    index: int | None = None,
    fps: float | None = None,
    max_width: int = DEFAULT_MAX_WIDTH,
    quality: int = DEFAULT_QUALITY,
) -> list[str]:
    """Build the ffmpeg argv for one push session.

    Raises ``ValueError`` for an unknown source so the caller can reject the
    request before a human ever sees it. Numeric knobs are clamped, not
    rejected — a model asking for 60fps gets 30, not an error. ``fps`` None means
    "this source's default" (see ``default_fps_for``); ``index`` None means
    "whatever this node is configured for" (see ``capture_index_for``).
    """
    if source not in _SOURCES:
        raise ValueError(
            f"unknown source {source!r} (one of: {', '.join(sorted(_SOURCES))})"
        )
    if sys.platform != "darwin":
        raise ValueError(
            f"push capture is implemented for macOS only (running on {sys.platform})"
        )
    fps = _clamp(float(default_fps_for(source) if fps is None else fps), *_FPS_BOUNDS)
    quality = int(_clamp(int(quality), *_QUALITY_BOUNDS))
    max_width = int(_clamp(int(max_width), *_WIDTH_BOUNDS))
    if index is None:
        index = capture_index_for(source)
    else:
        index = int(index)
    argv = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "avfoundation",
        "-framerate", f"{fps:g}",
    ]
    # The screen device rejects avfoundation's default yuv420p negotiation; it
    # must be told to deliver uyvy422. The camera accepts the default, so only
    # the screen carries this flag.
    if source == "screen":
        argv += ["-pixel_format", "uyvy422"]
    argv += [
        "-i", f"{index}:none",
        "-an",
        "-vf", f"fps={fps:g},scale='min({max_width},iw)':-2",
        "-f", "image2pipe",
        "-c:v", "mjpeg",
        "-q:v", str(quality),
        "pipe:1",
    ]
    return argv
