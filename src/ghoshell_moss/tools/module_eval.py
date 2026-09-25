"""ModuleEval — wrap a .py file as a live eval-server subprocess.

The domain module becomes a persistent, side-effecting runtime the model drives
by writing Python. Requests are fire-and-forget (one-way) with responses matched
back by ``id``, so multiple requests can be in flight and each carries a real
timeout. The subprocess itself is a single serial eval loop (step by step, no
threads) — this layer only stops the parent from being blocked round-trip by
round-trip.

Two spawn paths:
  subprocesses=Subprocesses → subprocesses.execute() with MOSS lifecycle
  subprocesses=None         → asyncio.create_subprocess_exec()

Usage::

    eval = ModuleEval("./my_domain.py", subprocesses=subprocesses)
    await eval.start()
    result = await eval.exec("page.goto('https://example.com')", timeout=30)
    await eval.aexec("page.reload()")          # fire, result lands in history
    await eval.shutdown()
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from ghoshell_moss.contracts.subprocesses import ManagedProcess, Subprocesses

__all__ = ["JsonLineProcess", "ModuleEval"]


class JsonLineProcess:
    """One-way fire + reader task with id-matched responses.

    The lock only guards the write (atomic single-line JSON), never the
    request/response round trip — that is what allowed the old implementation
    to block everything behind one in-flight request. A dedicated reader task
    consumes stdout and resolves pending futures by ``id``.
    """

    def __init__(
            self,
            proc: asyncio.subprocess.Process,
            *,
            on_result: Callable[[dict], None] | None = None,
    ):
        self._proc = proc
        self._on_result = on_result
        self._write_lock = asyncio.Lock()
        self._pending: dict[str, asyncio.Future] = {}
        self._next_id = 0
        self._reader_task: asyncio.Task | None = None
        self._closed = False

    def start(self) -> None:
        self._reader_task = asyncio.create_task(self._reader())

    async def close(self) -> None:
        self._closed = True
        if self._reader_task is not None and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
        self._fail_pending(RuntimeError("JsonLineProcess closed"))

    async def send(self, msg: dict) -> None:
        """Write a raw JSON line (no id injected). For protocol control messages."""
        data = json.dumps(msg) + "\n"
        async with self._write_lock:
            self._proc.stdin.write(data.encode())
            await self._proc.stdin.drain()

    async def _write(self, msg: dict) -> None:
        data = json.dumps(msg) + "\n"
        async with self._write_lock:
            self._proc.stdin.write(data.encode())
            await self._proc.stdin.drain()

    async def _reader(self) -> None:
        while not self._closed:
            try:
                line = await self._proc.stdout.readline()
            except asyncio.CancelledError:
                raise
            if not line:
                self._fail_pending(RuntimeError("eval server closed (child exited)"))
                break
            try:
                result = json.loads(line.decode())
            except json.JSONDecodeError:
                continue
            rid = result.get("id")
            fut = self._pending.pop(rid, None)
            if fut is not None and not fut.done():
                fut.set_result(result)
            elif self._on_result is not None:
                self._on_result(result)

    def _fail_pending(self, exc: Exception) -> None:
        for fut in list(self._pending.values()):
            if not fut.done():
                fut.set_exception(exc)
        self._pending.clear()

    async def request(self, msg: dict, timeout: float = 30.0) -> dict:
        """Send a request and await its response, matched by id.  Raises on timeout."""
        rid = str(self._next_id)
        self._next_id += 1
        fut = asyncio.get_running_loop().create_future()
        self._pending[rid] = fut
        await self._write({**msg, "id": rid})
        try:
            return await asyncio.wait_for(fut, timeout)
        except asyncio.TimeoutError:
            self._pending.pop(rid, None)
            raise

    async def fire(self, msg: dict) -> str:
        """Send a request without waiting.  Returns the id; the response goes to
        ``on_result`` when it arrives."""
        rid = str(self._next_id)
        self._next_id += 1
        await self._write({**msg, "id": rid})
        return rid


class ModuleEval:
    """Wrap a .py file as a persistent, side-effecting eval runtime.

    The module's source is the domain (a live browser, a DB connection, a ROS
    node…) — its objects are materialized at child startup and stay alive across
    exec calls.  Builtins are unrestricted: the domain module's own imports are
    the declared boundary, surfaced to the model as instruction.

    Parameters
    ----------
    module_path:
        Path to a .py file.  Read at __init__; compiled and imported in the child
        at start().
    subprocesses:
        If given, ``subprocesses.execute()`` is used (MOSS Subprocesses owns the
        child lifecycle).  If None, bare ``asyncio.create_subprocess_exec``.
    history_size:
        How many recent commands to retain for ``history()``.
    """

    def __init__(
            self,
            module_path: str,
            *,
            subprocesses: Subprocesses | None = None,
            history_size: int = 20,
    ):
        self._module_path = Path(module_path).resolve()
        self._subprocesses = subprocesses
        self._source = self._module_path.read_text()
        self._module_name = self._module_path.stem
        self._proc: asyncio.subprocess.Process | None = None
        self._managed: ManagedProcess | None = None
        self._jsonline: JsonLineProcess | None = None
        self._history: deque[tuple[str, str]] = deque(maxlen=history_size)
        self._in_flight: dict[str, str] = {}  # id -> code, for aexec history

    # -- read-only ----------------------------------------------------------

    @property
    def source(self) -> str:
        """Domain module source text."""
        return self._source

    @property
    def module_name(self) -> str:
        return self._module_name

    # -- lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        """Spawn the eval server subprocess and wait for the ready signal."""
        server_script = str(Path(__file__).parent / "_eval_server.py")
        args = [sys.executable, "-u", server_script]
        extra_env = {
            "MODULE_FILE": str(self._module_path),
            "MODULE_NAME": self._module_name,
        }

        if self._subprocesses:
            self._managed = await self._subprocesses.execute(
                *args,
                name=f"module_eval/{self._module_name}",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                extra_env=extra_env,
            )
            self._proc = self._managed.process
        else:
            self._proc = await asyncio.create_subprocess_exec(
                *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                env={**os.environ, **extra_env},
            )

        line = await self._proc.stdout.readline()
        ready_data = line.decode().strip()
        if ready_data != "ready":
            try:
                error = json.loads(ready_data)
                raise RuntimeError(
                    f"eval server init failed: {error.get('error', ready_data)}"
                )
            except json.JSONDecodeError:
                raise RuntimeError(
                    f"eval server unexpected output: {ready_data!r}"
                )

        self._jsonline = JsonLineProcess(self._proc, on_result=self._on_result)
        self._jsonline.start()

    async def shutdown(self) -> None:
        """Send __SHUTDOWN__ and wait for the child to exit."""
        if self._proc is None:
            return
        try:
            if self._jsonline is not None:
                await self._jsonline.send({"code": "__SHUTDOWN__"})
                await self._jsonline.close()
            if self._managed is not None:
                await self._managed.stop(timeout=5.0)
            else:
                await asyncio.wait_for(self._proc.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            if self._proc.returncode is None:
                self._proc.kill()
        except Exception:
            if self._proc.returncode is None:
                self._proc.kill()

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *args):
        await self.shutdown()

    # -- commands -----------------------------------------------------------

    async def exec(self, code: str, *, timeout: float = 30.0) -> str:
        """Execute *code* in the live runtime.  Blocks; raises on timeout."""
        self._ensure_started()
        try:
            result = await self._jsonline.request({"code": code}, timeout=timeout)
        except asyncio.TimeoutError:
            self._record(code, f"(timeout after {timeout}s)")
            raise
        text = self._format_result(result)
        self._record(code, text)
        return text

    async def aexec(self, code: str) -> str:
        """Fire *code* without waiting.  Result is recorded into history when it
        arrives.  Returns the request id."""
        self._ensure_started()
        rid = await self._jsonline.fire({"code": code})
        self._in_flight[rid] = code
        return rid

    def history(self, n: int = 10) -> str:
        """Recent executed commands + result summaries (in-memory, oldest→newest)."""
        if not self._history:
            return "(no executed commands)"
        lines: list[str] = []
        for code, text in list(self._history)[-n:]:
            lines.append(f">>> {self._summarize(code, limit=80)}")
            if text:
                lines.append(f"    {self._summarize(text, limit=200)}")
        return "\n".join(lines)

    # -- internal -----------------------------------------------------------

    def _ensure_started(self) -> None:
        if self._jsonline is None:
            raise RuntimeError("ModuleEval not started")

    def _on_result(self, result: dict) -> None:
        rid = result.get("id")
        code = self._in_flight.pop(rid, None)
        if code is None:
            return
        self._record(code, self._format_result(result))

    def _record(self, code: str, text: str) -> None:
        self._history.append((code, text))

    @staticmethod
    def _format_result(result: dict) -> str:
        parts: list[str] = []
        if result.get("std_output"):
            parts.append(result["std_output"].rstrip())
        if result.get("exception"):
            parts.append(f"Error: {result['exception']}")
            tb = result.get("traceback")
            if tb:
                parts.append(tb.rstrip())
        ret = result.get("returns")
        if ret is not None:
            parts.append(f"__result__: {ret}")
        return "\n".join(parts) if parts else "(executed, no output)"

    @staticmethod
    def _summarize(s: str, *, limit: int) -> str:
        s = " ".join(s.split())
        return s if len(s) <= limit else s[: limit - 1] + "…"
