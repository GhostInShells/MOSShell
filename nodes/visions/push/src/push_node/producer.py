"""ProducerManager — spawn and govern one ffmpeg child per approved session.

Each live session is an ffmpeg subprocess emitting a JPEG frame stream on
stdout. The manager reads that pipe itself (binary, not the text-capture path),
keeps only the latest frame per session, and hands ``.stop()`` back to the
subprocess owner — so the child never outlives the node.

Binary reading is the reason this does not use ``CaptureSpec``: capture drains
stdout as text lines, which would corrupt a JPEG stream. ``execute`` with a
manual ``stdout=PIPE`` still runs the owner's reclaim + graceful-stop machinery,
so the process is as governed as any terminal child, just read as bytes.

A natural exit is a *failure* (the device vanished or the stream broke); a stop
is not. The ``stopping`` flag on ``_Live`` keeps the two apart so cancelling a
reader never reports a spurious failure.
"""
from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Awaitable, Callable
from typing import Optional

from ghoshell_moss.contracts.subprocesses import ManagedProcess, Subprocesses

from .ffmpeg import build_argv
from .session import PushSession, SessionState
from .store import PushStore

__all__ = ["ProducerManager"]

_SOI = b"\xff\xd8"
_EOI = b"\xff\xd9"
_READ_CHUNK = 65536
_MAX_BUFFER = 8 * 1024 * 1024


def _split_latest_jpeg(buffer: bytes) -> tuple[Optional[bytes], bytes]:
    """Split the last complete JPEG out of a concatenated image2pipe byte stream."""
    eoi = buffer.rfind(_EOI)
    if eoi == -1:
        return None, buffer
    soi = buffer.rfind(_SOI, 0, eoi)
    if soi == -1:
        return None, buffer[eoi + 2:]
    return buffer[soi:eoi + 2], buffer[eoi + 2:]


class _Live:
    """One running producer: its process, its latest frame, and its stop flag."""

    def __init__(self, session: PushSession, managed: ManagedProcess) -> None:
        self.session = session
        self.managed = managed
        self.latest: Optional[bytes] = None
        self.latest_at: Optional[float] = None
        self.task: Optional[asyncio.Task] = None
        self.stopping = False


class ProducerManager:
    def __init__(
        self,
        processes: Subprocesses,
        store: PushStore,
        *,
        on_frame: Callable[[int], Awaitable[None]] | None = None,
        on_failed: Callable[[int, str], Awaitable[None]] | None = None,
    ) -> None:
        self._processes = processes
        self._store = store
        self._on_frame = on_frame
        self._on_failed = on_failed
        self._live: dict[int, _Live] = {}

    async def start(self, session: PushSession) -> None:
        """Spawn the producer for an approved session and begin reading frames."""
        argv = build_argv(
            session.source,
            fps=session.fps,
            max_width=session.max_width,
            quality=session.quality,
        )
        try:
            managed = await self._processes.execute(
                *argv,
                name=f"push:{session.id}",
                description=f"{session.source} push",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except Exception as exc:
            self._store.fail(session.id, f"failed to spawn: {exc}")
            if self._on_failed is not None:
                await self._on_failed(session.id, f"failed to spawn: {exc}")
            return
        self._store.attach_process(session.id, managed.meta.index)
        live = _Live(session, managed)
        self._live[session.id] = live
        live.task = asyncio.create_task(self._read_loop(live))

    async def _read_loop(self, live: _Live) -> None:
        proc = live.managed.process
        buffer = b""
        try:
            while True:
                chunk = await proc.stdout.read(_READ_CHUNK)
                if not chunk:
                    break
                buffer += chunk
                if len(buffer) > _MAX_BUFFER:
                    buffer = buffer[-_MAX_BUFFER:]
                jpeg, buffer = _split_latest_jpeg(buffer)
                if jpeg:
                    live.latest = jpeg
                    live.latest_at = time.time()
                    self._store.mark_frame(live.session.id)
                    if self._on_frame is not None:
                        await self._on_frame(live.session.id)
        except asyncio.CancelledError:
            raise
        except Exception:
            pass
        finally:
            if not live.stopping and self._store.get(live.session.id).state == SessionState.LIVE:
                code = proc.returncode
                self._store.fail(live.session.id, f"producer exited (code {code})")
                if self._on_failed is not None:
                    await self._on_failed(live.session.id, f"producer exited (code {code})")
            self._live.pop(live.session.id, None)

    async def stop(self, session_id: int) -> None:
        """Stop a running producer: flag it, cancel the reader, stop the child."""
        live = self._live.get(session_id)
        if live is None:
            return
        live.stopping = True
        task = live.task
        live.task = None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await live.managed.stop()
        self._live.pop(session_id, None)

    async def stop_all(self) -> None:
        for session_id in list(self._live):
            await self.stop(session_id)

    def frame(self, session_id: int) -> Optional[bytes]:
        live = self._live.get(session_id)
        return live.latest if live is not None else None
