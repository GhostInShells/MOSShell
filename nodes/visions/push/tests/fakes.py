"""Stand-ins for the contracts the channel/producer consume: Subprocesses and
the surface. Both are injected, so tests need nothing from a running node."""
from __future__ import annotations

import asyncio
from typing import Any, Optional

from ghoshell_moss.contracts.subprocesses import ProcessMeta


class FakeStdout:
    """A binary stdout pipe. ``hold`` blocks forever (a live stream); otherwise
    the next read is EOF."""

    def __init__(self, hold: bool = False) -> None:
        self.hold = hold

    async def read(self, n: int) -> bytes:
        if self.hold:
            await asyncio.sleep(3600)
        return b""


class FakeStderr:
    """A binary stderr pipe: hands over ``payload`` once, then EOF."""

    def __init__(self, payload: bytes = b"") -> None:
        self._payload = payload

    async def read(self, n: int) -> bytes:
        chunk, self._payload = self._payload[:n], self._payload[n:]
        return chunk


class FakeProcess:
    def __init__(self, hold: bool = False, stderr: bytes = b"") -> None:
        self.returncode: Optional[int] = None
        self.stdout = FakeStdout(hold=hold)
        self.stderr = FakeStderr(stderr)


class FakeManaged:
    def __init__(
        self, *, index: int, command: str, name: str, hold: bool = False,
        stderr: bytes = b"",
    ) -> None:
        self.process = FakeProcess(hold=hold, stderr=stderr)
        self.meta = ProcessMeta(
            index=index, pid=1000 + index, command=command, name=name, cwd="",
        )
        self.stopped = False

    async def stop(self, timeout: float = 5.0) -> None:
        self.stopped = True
        self.process.returncode = -15


class FakeSubprocesses:
    """Records spawns; returns a held-open (live) or EOF (crashed) child."""

    def __init__(self, *, hold: bool = True, stderr: bytes = b"") -> None:
        self.hold = hold
        self.stderr = stderr
        self.spawned: list[FakeManaged] = []
        self._counter = 0

    async def execute(
        self, *args: str, name: str | None = None, description: str | None = None,
        cwd=None, stdout=None, stderr=None, **_: Any,
    ) -> FakeManaged:
        self._counter += 1
        managed = FakeManaged(
            index=self._counter, command=" ".join(args), name=name or "cmd",
            hold=self.hold, stderr=self.stderr,
        )
        self.spawned.append(managed)
        return managed

    async def shell(self, cmd: str, **_: Any) -> FakeManaged:
        return await self.execute(cmd)


class Recorder:
    """Collects broadcast frames and the signals the channel sends. Stands in for
    both the surface (broadcast) and the signaler (callable)."""

    def __init__(self) -> None:
        self.frames: list[dict] = []
        self.signals: list[object] = []

    async def broadcast(self, frame: dict) -> None:
        self.frames.append(frame)

    def bind_frame(self, frame) -> None:
        self.frame = frame

    def __call__(self, signal: object) -> None:
        self.signals.append(signal)

    def types(self) -> list[str]:
        return [f["type"] for f in self.frames]

    def of(self, kind: str) -> list[dict]:
        return [f for f in self.frames if f["type"] == kind]
