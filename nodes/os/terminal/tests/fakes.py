"""Stand-ins for the two contracts the channel consumes: Subprocesses and the
surface. Both are injected, so tests need nothing from a running node."""

from __future__ import annotations

from ghoshell_moss.contracts.subprocesses import ProcessMeta


class FakeOutput:
    def __init__(self, lines: list[str] | None = None) -> None:
        self._lines = list(lines or [])

    def push(self, text: str) -> None:
        self._lines.extend(text.splitlines(keepends=True))

    def stdout(self, *, offset: int = 0, limit: int = 0) -> str:
        window = self._lines[offset:]
        if limit > 0:
            window = window[:limit]
        return "".join(window)

    def stderr(self, *, offset: int = 0, limit: int = 0) -> str:
        return ""

    async def wait_drained(self) -> None:
        return None


class FakeProcess:
    def __init__(self) -> None:
        self.returncode: int | None = None


class FakeManaged:
    def __init__(
        self,
        *,
        index: int,
        command: str,
        name: str,
        cwd: str,
        description: str = "",
        lines: list[str] | None = None,
        exit_code: int = 0,
    ) -> None:
        self.output = FakeOutput(lines)
        self.process = FakeProcess()
        self.meta = ProcessMeta(
            index=index, pid=1000 + index, command=command, name=name, cwd=cwd,
            description=description,
        )
        self._exit_code = exit_code
        self.stopped = False

    def finish(self, exit_code: int | None = None) -> None:
        self.process.returncode = self._exit_code if exit_code is None else exit_code

    async def stop(self, timeout: float = 5.0) -> None:
        self.stopped = True
        self.process.returncode = -15


class FakeSubprocesses:
    """Spawns immediately-finished (or held-open) fake processes."""

    def __init__(
        self,
        *,
        lines: list[str] | None = None,
        exit_code: int = 0,
        hold: bool = False,
    ) -> None:
        self.lines = list(lines or [])
        self.exit_code = exit_code
        self.hold = hold
        self.spawned: list[FakeManaged] = []
        self._counter = 0

    async def shell(
        self, cmd: str, *, name: str | None = None, description: str | None = None,
        cwd: str | None = None, capture=None, **_: object,
    ) -> FakeManaged:
        self._counter += 1
        managed = FakeManaged(
            index=self._counter, command=cmd, name=name or "cmd",
            cwd=cwd or "", description=description or "",
            lines=self.lines, exit_code=self.exit_code,
        )
        # Mirror the subprocess layer: if the channel asked for a stdout file,
        # the full output lands there (for the keep/delete threshold decision).
        if capture is not None and capture.stdout_file is not None:
            capture.stdout_file.parent.mkdir(parents=True, exist_ok=True)
            capture.stdout_file.write_text("".join(self.lines))
        if not self.hold:
            managed.finish()
        self.spawned.append(managed)
        return managed


class Recorder:
    """Collects the frames the channel broadcasts and the signals it sends."""

    def __init__(self) -> None:
        self.frames: list[dict] = []
        self.signals: list[object] = []

    async def broadcast(self, frame: dict) -> None:
        self.frames.append(frame)

    def __call__(self, signal: object) -> None:
        self.signals.append(signal)

    def types(self) -> list[str]:
        return [f["type"] for f in self.frames]

    def of(self, kind: str) -> list[dict]:
        return [f for f in self.frames if f["type"] == kind]


def chunks(parts: list[str]):
    async def gen():
        for part in parts:
            yield part

    return gen()
