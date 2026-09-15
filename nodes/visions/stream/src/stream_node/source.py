"""ffmpeg-backed stream source for the stream vision node.

One ffmpeg subprocess per node decodes a live stream (RTMP/RTSP/SRT/MJPEG/...)
and emits a JPEG frame stream on stdout. The node keeps only the latest frame;
older frames are dropped. The subprocess is the isolation point — a broken
stream kills only the child, never the node event loop.
"""
from __future__ import annotations

import asyncio
import contextlib
from typing import Optional

import time

_SOI = b"\xff\xd8"
_EOI = b"\xff\xd9"
_READ_CHUNK = 65536
_MAX_BUFFER = 8 * 1024 * 1024  # drop stale partial data if no EOI after this


def split_latest_jpeg(buffer: bytes) -> tuple[Optional[bytes], bytes]:
    """Split the last complete JPEG (SOI..EOI) out of a byte stream.

    Returns ``(jpeg, remaining)`` where ``jpeg`` is the last complete frame (or
    None if none is complete yet) and ``remaining`` is the trailing partial bytes
    to keep buffering. ``image2pipe`` concatenates JPEGs with no delimiter, so
    framing is recovered from the SOI/EOI markers.
    """
    eoi = buffer.rfind(_EOI)
    if eoi == -1:
        return None, buffer
    soi = buffer.rfind(_SOI, 0, eoi)
    if soi == -1:
        return None, buffer[eoi + 2:]
    return buffer[soi:eoi + 2], buffer[eoi + 2:]


class FfmpegSource:
    """A live stream as a latest-frame JPEG source.

    ``start()`` spawns ffmpeg (``-i <address> ... -f image2pipe -c:v mjpeg``) and
    a reader task that keeps only the most recent JPEG. ``latest()`` returns
    ``(ts, jpeg_bytes)`` or None. ``failed`` carries the reason once the child
    exits or cannot start.
    """

    def __init__(self, address: str, *, fps: float = 2.0, quality: int = 4, logger=None):
        self._address = address
        self._fps = fps
        self._quality = quality
        self._logger = logger
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._task: Optional[asyncio.Task] = None
        self._latest: Optional[tuple[float, bytes]] = None
        self._failed: Optional[str] = None

    async def start(self) -> bool:
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", self._address,
            "-an", "-sn", "-dn",
            "-vf", f"fps={self._fps}",
            "-f", "image2pipe", "-c:v", "mjpeg", "-q:v", str(self._quality),
            "pipe:1",
        ]
        try:
            self._proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except FileNotFoundError:
            self._failed = "ffmpeg not found"
            return False
        self._task = asyncio.create_task(self._read_loop())
        return True

    async def _read_loop(self) -> None:
        proc = self._proc
        buffer = b""
        stdout = proc.stdout
        while True:
            try:
                chunk = await stdout.read(_READ_CHUNK)
            except Exception as exc:
                self._failed = f"read error: {exc}"
                break
            if not chunk:
                break
            buffer += chunk
            if len(buffer) > _MAX_BUFFER:
                # No complete frame in sight — drop stale partial data rather
                # than grow unbounded on a corrupt/garbled stream.
                buffer = buffer[-_MAX_BUFFER:]
            jpeg, buffer = split_latest_jpeg(buffer)
            if jpeg:
                self._latest = (time.time(), jpeg)
        code = proc.returncode
        if code is None:
            try:
                code = await proc.wait()
            except Exception:
                code = "?"
        self._failed = f"stream ended (ffmpeg exit {code})"

    async def stop(self) -> None:
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        proc = self._proc
        self._proc = None
        if proc is not None and proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                proc.kill()

    def latest(self) -> Optional[tuple[float, bytes]]:
        return self._latest

    @property
    def failed(self) -> Optional[str]:
        return self._failed
