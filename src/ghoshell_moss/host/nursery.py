"""ProcessNursery — lightweight subprocess spawn with pipe fencing and graceful shutdown.

Not a process manager.  No restart, no health check, no bringup orchestration.
Answers one question only: parent alive → child alive, parent dead → child dead.

Two exit paths:
- **Graceful**: parent ``__aexit__`` → SIGTERM to each child pgid → wait timeout → SIGKILL
- **SIGKILL-proof**: parent killed by kernel → write end of nursery pipe closed → child
  read-fd returns EOF → child exits via its own pipe-watchdog
"""

from __future__ import annotations

import asyncio
import os
import signal
from logging import Logger
from typing import Callable

__all__ = ["ProcessNursery", "watch_nursery_pipe"]


async def watch_nursery_pipe(close: Callable[[], None]) -> None:
    """Watch ``MOSS_NURSERY_FD`` for parent death (pipe EOF).

    When the parent dies — including SIGKILL — the kernel closes all its fds.
    The write end of the nursery pipe disappears, and this read end returns
    EOF.  On EOF, *close* is called to trigger graceful shutdown.

    No-op if ``MOSS_NURSERY_FD`` is not set.
    """
    fd_str = os.environ.get("MOSS_NURSERY_FD")
    if not fd_str:
        return

    fd = int(fd_str)
    loop = asyncio.get_running_loop()
    try:
        while True:
            data = await loop.run_in_executor(None, os.read, fd, 1)
            if not data:  # EOF — parent dead
                close()
                return
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


class ProcessNursery:
    """Spawn subprocesses with pipe-fencing support and graceful shutdown.

    Async context manager — enter via ``enter_async_context(nursery)`` so
    ``shutdown()`` runs in LIFO order via the exit stack, guaranteeing it
    executes even if later cleanup steps fail.

    Does NOT store environment state — the caller provides the full env dict
    on each ``spawn()`` call.  This ensures env values reflect the current
    runtime state (session_id, mode, etc.) rather than a frozen snapshot.
    """

    def __init__(self, logger: Logger | None = None):
        self._pgids: set[int] = set()
        self._logger = logger

    # -- async context manager --------------------------------------------- #

    async def __aenter__(self) -> "ProcessNursery":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.shutdown()

    # -- spawn ------------------------------------------------------------ #

    async def spawn(
        self,
        *args: str,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        nursery_fd: int | None = None,
    ) -> asyncio.subprocess.Process:
        """Spawn a subprocess.

        *env* is the complete environment dict — the caller is responsible for
        assembling it from the current runtime state.

        Pipe fencing: pass a pipe read-fd as *nursery_fd*.  The parent holds
        the write end; when the parent dies (SIGKILL included), the kernel
        closes all fds, the child's read returns EOF.  Caller creates the pipe
        with ``os.pipe()`` and closes the read end after spawn.
        """
        env = dict(env) if env is not None else dict(os.environ)

        pass_fds: tuple[int, ...] = ()
        if nursery_fd is not None:
            env["MOSS_NURSERY_FD"] = str(nursery_fd)
            pass_fds = (nursery_fd,)

        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=cwd,
            env=env,
            start_new_session=True,
            pass_fds=pass_fds,
        )
        self._pgids.add(proc.pid)  # start_new_session → pgid == pid
        return proc

    # -- shutdown --------------------------------------------------------- #

    async def shutdown(self, timeout: float = 5.0) -> None:
        """Gracefully shut down all spawned children.

        SIGTERM → wait *timeout* seconds → SIGKILL stragglers.
        """
        if not self._pgids:
            return

        # phase 1: gentle
        for pgid in list(self._pgids):
            try:
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                self._pgids.discard(pgid)

        # phase 2: wait
        await asyncio.sleep(timeout)

        # phase 3: force
        for pgid in list(self._pgids):
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self._pgids.clear()

    @property
    def child_count(self) -> int:
        return len(self._pgids)
