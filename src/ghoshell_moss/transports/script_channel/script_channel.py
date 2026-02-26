import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ghoshell_container import Container, IoCContainer, get_container

from ghoshell_moss.core.duplex.connection import ChannelEvent, Connection, ConnectionClosedError
from ghoshell_moss.core.duplex.provider import DuplexChannelProvider
from ghoshell_moss.core.duplex.proxy import DuplexChannelProxy

logger = logging.getLogger(__name__)


class _JSONLineIO:
    @staticmethod
    def dumps(event: ChannelEvent) -> bytes:
        return (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")

    @staticmethod
    def loads(line: bytes) -> ChannelEvent:
        return json.loads(line.decode("utf-8"))


class StdioProviderConnection(Connection):
    """Provider side of a stdio (stdin/stdout) duplex connection.

    This connection is meant to run inside the child python process.
    It reads proxy events from stdin and writes provider events to stdout.
    """

    def __init__(
        self,
        *,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        logger: logging.Logger | None = None,
    ):
        self._reader = reader
        self._writer = writer
        self._closed_event = asyncio.Event()
        self._recv_lock = asyncio.Lock()
        self._send_lock = asyncio.Lock()
        self._logger = logger or logging.getLogger(__name__)

    async def recv(self, timeout: Optional[float] = None) -> ChannelEvent:
        if self._closed_event.is_set():
            raise ConnectionClosedError("Connection closed")

        async with self._recv_lock:
            try:
                line = await asyncio.wait_for(self._reader.readline(), timeout=timeout)
            except asyncio.TimeoutError:
                raise

            if not line:
                self._closed_event.set()
                raise ConnectionClosedError("Connection closed")

            try:
                event = _JSONLineIO.loads(line)
            except Exception as exc:
                self._logger.warning("Invalid event line from stdin: %s", exc)
                raise
            return event

    async def send(self, event: ChannelEvent) -> None:
        if self._closed_event.is_set():
            raise ConnectionClosedError("Connection closed")

        async with self._send_lock:
            try:
                self._writer.write(_JSONLineIO.dumps(event))
                await self._writer.drain()
            except (BrokenPipeError, ConnectionResetError) as exc:
                self._closed_event.set()
                raise ConnectionClosedError("Connection closed") from exc

    def is_closed(self) -> bool:
        return self._closed_event.is_set()

    def is_connected(self) -> bool:
        return not self.is_closed()

    async def close(self) -> None:
        if self._closed_event.is_set():
            return
        self._closed_event.set()
        try:
            self._writer.close()
            await self._writer.wait_closed()
        except Exception:
            return

    async def start(self) -> None:
        return


class SubprocessStdioConnection(Connection):
    """Proxy side of a stdio duplex connection backed by a subprocess."""

    def __init__(
        self,
        *,
        script_path: str,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        python_executable: str | None = None,
        args: list[str] | None = None,
        logger: logging.Logger | None = None,
    ):
        self._script_path = str(Path(script_path).expanduser())
        self._env = dict(env or {})
        self._cwd = cwd
        self._python_executable = python_executable or sys.executable
        self._args = list(args or [])
        self._logger = logger or logging.getLogger(__name__)

        self._process: asyncio.subprocess.Process | None = None
        self._start_time: float | None = None
        self._closed_event = asyncio.Event()
        self._recv_lock = asyncio.Lock()
        self._send_lock = asyncio.Lock()
        self._stdout_buffer = bytearray()
        self._monitor_task: asyncio.Task | None = None

    def is_closed(self) -> bool:
        return self._closed_event.is_set()

    def is_connected(self) -> bool:
        return not self.is_closed() and self._process is not None and self._process.returncode is None

    async def start(self) -> None:
        if self._process is not None:
            return

        script = Path(self._script_path)
        if not script.is_absolute():
            script = (Path.cwd() / script).resolve()
        if not script.exists() or not script.is_file():
            raise FileNotFoundError(f"script file not found: {script}")

        env = os.environ.copy()
        env.update(self._env)
        env.setdefault("PYTHONUNBUFFERED", "1")

        self._process = await asyncio.create_subprocess_exec(
            self._python_executable,
            str(script),
            *self._args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=self._cwd,
        )
        self._start_time = time.time()
        self._monitor_task = asyncio.create_task(self._monitor_stderr())

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process is not None else None

    @property
    def start_time(self) -> float | None:
        return self._start_time

    async def _monitor_stderr(self) -> None:
        if self._process is None or self._process.stderr is None:
            return
        try:
            while True:
                line = await self._process.stderr.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="ignore").rstrip()
                self._logger.error("[script_channel stderr] %s", text)
        except asyncio.CancelledError:
            pass
        except Exception:
            self._logger.exception("script_channel stderr monitor error")

    async def _readline_unlimited(self, timeout: Optional[float] = None) -> bytes:
        if self._process is None or self._process.stdout is None:
            raise RuntimeError("Connection not started")

        while True:
            nl = self._stdout_buffer.find(b"\n")
            if nl != -1:
                line = bytes(self._stdout_buffer[: nl + 1])
                del self._stdout_buffer[: nl + 1]
                return line

            read_coro = self._process.stdout.read(4096)
            if timeout is None:
                chunk = await read_coro
            else:
                chunk = await asyncio.wait_for(read_coro, timeout=timeout)

            if not chunk:
                self._closed_event.set()
                raise ConnectionClosedError("Connection closed")
            self._stdout_buffer.extend(chunk)

    async def recv(self, timeout: Optional[float] = None) -> ChannelEvent:
        if self._closed_event.is_set():
            raise ConnectionClosedError("Connection closed")
        if self._process is None or self._process.stdout is None:
            raise RuntimeError("Connection not started")

        async with self._recv_lock:
            if self._process.returncode is not None:
                self._closed_event.set()
                raise ConnectionClosedError("Connection closed")

            line = await self._readline_unlimited(timeout)
            try:
                return _JSONLineIO.loads(line)
            except Exception as exc:
                self._logger.warning("Invalid event line from subprocess stdout: %s", exc)
                raise

    async def send(self, event: ChannelEvent) -> None:
        if self._closed_event.is_set():
            raise ConnectionClosedError("Connection closed")
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("Connection not started")

        async with self._send_lock:
            if self._process.returncode is not None:
                self._closed_event.set()
                raise ConnectionClosedError("Connection closed")
            try:
                self._process.stdin.write(_JSONLineIO.dumps(event))
                await self._process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError) as exc:
                self._closed_event.set()
                raise ConnectionClosedError("Connection closed") from exc

    async def close(self) -> None:
        if self._closed_event.is_set():
            return
        self._closed_event.set()

        if self._monitor_task is not None:
            self._monitor_task.cancel()

        if self._process is None:
            return

        try:
            if self._process.stdin is not None:
                try:
                    self._process.stdin.close()
                except Exception as exc:
                    self._logger.debug("Failed to close subprocess stdin: %s", exc)
            if self._process.returncode is None:
                self._process.terminate()
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=3.0)
                except asyncio.TimeoutError:
                    self._process.kill()
                    await self._process.wait()
        finally:
            self._process = None
            self._start_time = None


@dataclass
class ScriptProviderConfig:
    """Configuration passed to the child script provider.

    Carry `ModuleChannel` constructor kwargs directly to avoid duplication.
    """

    module_channel_kwargs: dict[str, Any]


class ScriptChannelProvider(DuplexChannelProvider):
    """Provider that runs inside a script process (stdin/stdout transport)."""

    def __init__(
        self,
        *,
        connection: Connection,
        container: IoCContainer | None = None,
    ):
        container = Container(parent=container or get_container(), name="ScriptChannelProvider")
        super().__init__(provider_connection=connection, container=container)


class ScriptChannelProxy(DuplexChannelProxy):
    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        provider_launcher: str,
        channel_autostart: bool = True,
        provider_target: str | None = None,
        provider_channel_file: str | None = None,
        provider_expect_channel_val: str = "__channel__",
        provider_include: list[str] | None = None,
        provider_reload_on_bootstrap: bool = False,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        python_executable: str | None = None,
        args: list[str] | None = None,
        logger: logging.Logger | None = None,
    ):
        provider_script = str(Path(provider_launcher).expanduser())
        provider_env = dict(env or {})
        provider_args = list(args or [])

        # If `provider_target` is set, we treat `script_path` as the provider launcher.
        # The launcher should accept `--config-json` and run the provider main.
        if provider_target is not None:
            module_channel_kwargs: dict[str, Any] = {
                "name": name,
                "description": description,
                "module_name": provider_target,
                "include": provider_include,
                "expect_channel_val": provider_expect_channel_val,
                "channel_file": provider_channel_file,
                "reload_on_bootstrap": provider_reload_on_bootstrap,
            }
            config = ScriptProviderConfig(module_channel_kwargs=module_channel_kwargs)
            provider_args = [*provider_args, "--config-json", json.dumps(config.__dict__, ensure_ascii=False)]
            if logger is not None:
                logger.debug("provider_args=%s", provider_args)

        connection = SubprocessStdioConnection(
            script_path=provider_script,
            env=provider_env,
            cwd=cwd,
            python_executable=python_executable,
            args=provider_args,
            logger=logger,
        )
        super().__init__(
            name=name,
            description=description,
            to_provider_connection=connection,
            close_connection_on_close=True,
        )
        self._autostart = channel_autostart

    @property
    def connection(self) -> SubprocessStdioConnection:
        return self._provider_connection  # type: ignore[return-value]

    # NOTE: `DuplexChannelContext.start()` will call `connection.start()`.
    # We keep `channel_autostart` as a future extension point for hubs.
