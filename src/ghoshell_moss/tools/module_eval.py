"""ModuleEval — wrap a .py file as sandboxed eval server subprocess.

Module source becomes channel instruction (Code as Prompt).
Child compiles with full builtins, execs with SANDBOX_BUILTINS.

Two spawn paths:
  matrix=Matrix → matrix.processes.execute() with MOSS lifecycle
  matrix=None   → asyncio.create_subprocess_exec()

Usage::

    eval = ModuleEval("./my_domain.py", matrix=matrix)
    await eval.start()
    result = await eval.exec("page.goto('https://example.com')")
    await eval.shutdown()
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ghoshell_moss.core.blueprint.matrix import Matrix
    from ghoshell_moss.contracts.subprocesses import ManagedProcess

__all__ = ["JsonLineProcess", "ModuleEval"]


class JsonLineProcess:
    """Async JSON-line protocol adapter for subprocess PIPE streams.

    Wraps an asyncio.subprocess.Process with stdin=PIPE, stdout=PIPE.
    Thread-safe via asyncio.Lock for request-response pairing.
    """

    def __init__(self, proc: asyncio.subprocess.Process):
        self._proc = proc
        self._lock = asyncio.Lock()

    async def send(self, msg: dict) -> None:
        data = json.dumps(msg) + "\n"
        self._proc.stdin.write(data.encode())
        await self._proc.stdin.drain()

    async def recv(self) -> dict:
        line = await self._proc.stdout.readline()
        return json.loads(line.decode())

    async def request(self, msg: dict, timeout: float = 30.0) -> dict:
        """Send a request and wait for the response.  Atomic per-process."""
        async with self._lock:
            await self.send(msg)
            return await self.recv()


class ModuleEval:
    """Wrap a .py file as sandboxed eval server subprocess.

    The module's source is the channel instruction — the AI sees exactly
    what objects are available, their types, and methods.  This is Code as
    Prompt in its purest form.

    Child process architecture::

        Compiler (builtins unrestricted) → compile module, execute imports
        init_sandbox (builtins=None)     → hold compiled namespace objects
        sandbox (parent=init, SANDBOX_BUILTINS) → AI exec namespace

    Parameters
    ----------
    module_path:
        Path to a .py file.  The file is read at __init__ time; compilation
        and import happen in the child process at start() time.
    matrix:
        If given, ``matrix.processes.execute()`` is used (the child is managed
        by MOSS Subprocesses lifecycle).  If None, bare ``asyncio.create_subprocess_exec``.
    """

    def __init__(self, module_path: str, *, matrix: Matrix | None = None):
        self._module_path = Path(module_path).resolve()
        self._matrix = matrix
        self._source = self._module_path.read_text()
        self._module_name = self._module_path.stem
        self._proc: asyncio.subprocess.Process | None = None
        self._managed: ManagedProcess | None = None
        self._jsonline: JsonLineProcess | None = None

    # -- read-only ----------------------------------------------------------

    @property
    def source(self) -> str:
        """Module source text."""
        return self._source

    @property
    def instruction(self) -> str:
        """Channel instruction — module source as prompt."""
        return self._source

    @property
    def module_name(self) -> str:
        return self._module_name

    # -- lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        """Spawn the eval server subprocess and wait for ready signal."""
        server_script = str(Path(__file__).parent / "_eval_server.py")
        args = [sys.executable, "-u", server_script]
        extra_env = {
            "MODULE_FILE": str(self._module_path),
            "MODULE_NAME": self._module_name,
        }

        if self._matrix:
            self._managed = await self._matrix.processes.execute(
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

        # Wait for "ready" signal
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

        self._jsonline = JsonLineProcess(self._proc)

    async def shutdown(self) -> None:
        """Send __SHUTDOWN__ and wait for the child to exit."""
        if self._proc is None:
            return
        try:
            if self._jsonline is not None:
                await self._jsonline.send({"code": "__SHUTDOWN__"})
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

    async def exec(self, code: str) -> str:
        """Execute *code* in the sandbox.  Returns formatted result string."""
        if self._jsonline is None:
            raise RuntimeError("ModuleEval not started")
        result = await self._jsonline.request({"code": code})
        return self._format_result(result)

    async def vars(self) -> str:
        """List namespace contents — name → type.

        Uses __vars__ protocol command (not exec), so works even when
        SANDBOX_BUILTINS blocks __import__.
        """
        if self._jsonline is None:
            raise RuntimeError("ModuleEval not started")
        result = await self._jsonline.request({"code": "__vars__"})
        ns = result.get("vars", {})
        if not ns:
            return "(empty namespace)"
        return json.dumps(ns, indent=2)

    async def api(self, name: str | None = None) -> str:
        """Reflect on the sandbox namespace.

        Without *name*: full get_interface() — source + attr blocks.
        With *name*: single-object detail.
        """
        if self._jsonline is None:
            raise RuntimeError("ModuleEval not started")
        msg: dict = {"code": "__api__"}
        if name:
            msg["name"] = name
        result = await self._jsonline.request(msg)
        return result.get("api", "(no api info)")

    # -- internal -----------------------------------------------------------

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
