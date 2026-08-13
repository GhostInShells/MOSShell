"""Subprocesses 默认实现 — asyncio 子进程 + 输出捕获 + 生命周期治理.

契约见 ghoshell_moss.contracts.subprocesses.
"""

# 实现要点 (reviewer 上下文):
# - "子进程不比 owner 活得久" 三件套: start_new_session + aexit killpg /
#   capture 管道随 owner 关闭 (pipe fencing) / 每进程一个 reclaim 协程.
# - stop 信号策略只有一份: ManagedProcess.stop 经 _stop_impl 注入回到
#   本类的 _stop_managed, 与 __aexit__ 关停路径同源 (SIGINT → grace → SIGKILL pg).
# - spawn 在未启动/已停止时严格抛错 (旧版仅 log): 停机后 spawn 出的进程
#   没有 owner 清场路径, 是孤儿源头, 必须堵死.

from __future__ import annotations

import asyncio
import logging
import os
import signal
import time
from pathlib import Path
from typing import Callable

from ghoshell_moss.contracts.subprocesses import (
    Subprocesses,
    ManagedProcess,
    ProcessMeta,
    ProcessOutput,
    CaptureSpec,
    ErrorInfo,
)
from ghoshell_moss.contracts.logger import get_moss_logger
from ghoshell_moss.core.subprocesses._utils import killpg as _kill_process_group_util

__all__ = ["SubprocessesImpl", "Subprocesses"]

_MAX_EXECUTED_HISTORY = 200
_SHUTDOWN_GRACE = 3.0


# -- 输出捕获实现 --


class _ProcessOutputImpl(ProcessOutput):
    """内存 tail 窗口 + 完整落盘文件. drain 协程持续填充."""

    def __init__(
            self,
            stdout_file: Path,
            stderr_file: Path,
            buffer_lines: int,
    ):
        self._stdout_file = stdout_file
        self._stderr_file = stderr_file
        self.buffer_lines = buffer_lines
        self.stdout_buf: list[str] = []
        self.stderr_buf: list[str] = []
        self.drain_tasks: set[asyncio.Task] = set()

    @property
    def stdout_file(self) -> Path | None:
        return self._stdout_file

    @property
    def stderr_file(self) -> Path | None:
        return self._stderr_file

    def stdout(self, *, offset: int = 0, limit: int = 0) -> str:
        return self._read_window(self.stdout_buf, offset, limit)

    def stderr(self, *, offset: int = 0, limit: int = 0) -> str:
        return self._read_window(self.stderr_buf, offset, limit)

    def _read_window(self, buf: list[str], offset: int, limit: int) -> str:
        if self.buffer_lines <= 0 or not buf:
            return ""
        window = buf[offset:]
        if limit > 0:
            window = window[:limit]
        return "".join(window)

    async def wait_drained(self) -> None:
        if self.drain_tasks:
            await asyncio.gather(*self.drain_tasks, return_exceptions=True)


# -- Subprocesses 实现 --


class SubprocessesImpl(Subprocesses):
    """契约的 asyncio 实现.

    cwd: spawn 未显式传 cwd 时的默认工作目录. None = 进程当前目录.
    """

    def __init__(
            self,
            cwd: str | Path | None = None,
            logger: logging.Logger | None = None,
    ):
        self._default_cwd = Path(cwd).resolve() if cwd else Path.cwd()
        self.logger = logger or get_moss_logger()

        self._started = False
        self._stopped = False

        self._counter: int = 0
        self._executing: dict[int, ManagedProcess] = {}
        self._executed: list[ProcessMeta] = []
        self._tasks: set[asyncio.Task] = set()

    # -- spawn --

    async def execute(
            self,
            *args: str,
            name: str | None = None,
            description: str | None = None,
            cwd: str | Path | None = None,
            extra_env: dict[str, str] | None = None,
            capture: CaptureSpec | None = None,
            stdin: int | None = None,
            stdout: int | None = None,
            stderr: int | None = None,
            start_new_session: bool = True,
            with_os_env: bool = True,
            on_exit: Callable[[ProcessMeta], None] | None = None,
            **kwargs,
    ) -> ManagedProcess:
        if not args:
            raise ValueError("execute() requires at least one argument")
        return await self._spawn(
            exec_args=args, shell_cmd=None,
            name=name or args[0],
            description=description or "",
            cwd=cwd, extra_env=extra_env, capture=capture,
            stdin=stdin, stdout=stdout, stderr=stderr,
            start_new_session=start_new_session, with_os_env=with_os_env,
            on_exit=on_exit,
        )

    async def shell(
            self,
            cmd: str,
            *,
            name: str | None = None,
            description: str | None = None,
            cwd: str | Path | None = None,
            extra_env: dict[str, str] | None = None,
            capture: CaptureSpec | None = None,
            stdin: int | None = None,
            stdout: int | None = None,
            stderr: int | None = None,
            start_new_session: bool = True,
            with_os_env: bool = True,
            on_exit: Callable[[ProcessMeta], None] | None = None,
            **kwargs,
    ) -> ManagedProcess:
        return await self._spawn(
            exec_args=(), shell_cmd=cmd,
            name=name or cmd[:60],
            description=description or "",
            cwd=cwd, extra_env=extra_env, capture=capture,
            stdin=stdin, stdout=stdout, stderr=stderr,
            start_new_session=start_new_session, with_os_env=with_os_env,
            on_exit=on_exit,
        )

    # -- 查询 --

    def executing(self) -> dict[int, ProcessMeta]:
        return {idx: managed.meta for idx, managed in self._executing.items()}

    def get(self, index: int) -> ManagedProcess | None:
        return self._executing.get(index)

    def executed(self) -> list[ProcessMeta]:
        return list(self._executed)

    # -- 信号 --

    def kill(self, pid: int) -> ErrorInfo | None:
        try:
            os.kill(pid, signal.SIGKILL)
            return None
        except ProcessLookupError:
            return None
        except PermissionError as e:
            self.logger.warning("kill(%d) denied: %s", pid, e)
            return f"kill({pid}) denied: {e}"
        except OSError as e:
            self.logger.error("kill(%d) failed: %s", pid, e)
            return f"kill({pid}) failed: {e}"

    def killpg(self, process_group: int, sig: int) -> ErrorInfo | None:
        return _kill_process_group_util(process_group, sig)

    # -- 生命周期 --

    def is_running(self) -> bool:
        return self._started and not self._stopped

    async def __aenter__(self) -> "SubprocessesImpl":
        self._started = True
        self.logger.info("Subprocesses started, default_cwd=%s", self._default_cwd)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self._stopped = True
        self.logger.info("Subprocesses stopping — %d executing", len(self._executing))

        # SIGINT 全体 → grace → SIGKILL 全体 (进程组覆盖未分离子孙)
        for managed in list(self._executing.values()):
            if managed.process.returncode is None:
                self._signal_managed(managed, signal.SIGINT)
        try:
            await asyncio.wait_for(
                asyncio.gather(
                    *(managed.process.wait()
                      for managed in self._executing.values()
                      if managed.process.returncode is None),
                    return_exceptions=True,
                ),
                timeout=_SHUTDOWN_GRACE,
            )
        except asyncio.TimeoutError:
            self.logger.warning(
                "shutdown: %d alive after SIGINT, SIGKILL",
                sum(1 for m in self._executing.values() if m.process.returncode is None),
            )
        for managed in list(self._executing.values()):
            if managed.process.returncode is None:
                self._signal_managed(managed, signal.SIGKILL)
        for managed in list(self._executing.values()):
            try:
                await managed.process.wait()
            except (ProcessLookupError, OSError):
                pass

        if self._tasks:
            self.logger.debug("shutdown: cancel %d internal tasks", len(self._tasks))
            for task in self._tasks:
                task.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)
            self._tasks.clear()
        self._executing.clear()
        self.logger.info("Subprocesses stopped — %d executed", len(self._executed))

    # ================================================================
    # 内部方法
    # ================================================================

    def _check_can_spawn(self) -> None:
        # 惰性启动: IoC 全局单例可能无人显式 aenter, 首次 spawn 自动进入启动态.
        if not self._started:
            self._started = True
        # 严格守卫: 停机后 spawn 的进程无人清场, 必然孤儿, 直接抛错.
        if self._stopped:
            raise RuntimeError("Subprocesses already stopped")

    def _next_index(self) -> int:
        self._counter += 1
        return self._counter

    def _resolve_cwd(self, cwd: str | Path | None) -> str:
        if cwd is None:
            return str(self._default_cwd)
        path = Path(cwd)
        if not path.is_absolute():
            path = self._default_cwd / path
        return str(path.resolve())


    @staticmethod
    def _build_env(with_os_env: bool, extra_env: dict[str, str] | None) -> dict:
        env = os.environ.copy() if with_os_env else {}
        if extra_env:
            env.update(extra_env)
        return env

    def _signal_managed(self, managed: ManagedProcess, sig: int) -> None:
        """优先信号进程组 (覆盖未分离子孙), 无进程组则信号单进程."""
        if managed.meta.pgid is not None:
            err = self.killpg(managed.meta.pgid, sig)
            if err is None:
                return
        try:
            managed.process.send_signal(sig)
        except ProcessLookupError:
            pass

    async def _stop_managed(self, managed: ManagedProcess, timeout: float) -> None:
        """ManagedProcess.stop 的唯一实现: SIGINT → grace → SIGKILL (进程组)."""
        proc = managed.process
        if proc.returncode is not None:
            return
        self._signal_managed(managed, signal.SIGINT)
        try:
            await asyncio.wait_for(asyncio.shield(proc.wait()), timeout=timeout)
            return
        except asyncio.TimeoutError:
            pass
        if proc.returncode is None:
            self._signal_managed(managed, signal.SIGKILL)
            try:
                await proc.wait()
            except (ProcessLookupError, OSError):
                pass

    async def _spawn(
            self,
            exec_args: tuple[str, ...],
            shell_cmd: str | None,
            name: str,
            description: str,
            cwd: str | Path | None,
            extra_env: dict[str, str] | None,
            capture: CaptureSpec | None,
            stdin: int | None,
            stdout: int | None,
            stderr: int | None,
            start_new_session: bool,
            with_os_env: bool,
            on_exit: Callable[[ProcessMeta], None] | None,
    ) -> ManagedProcess:
        self._check_can_spawn()
        if capture is not None and (stdout is not None or stderr is not None):
            raise ValueError("capture and manual stdout/stderr are mutually exclusive")

        work_dir = self._resolve_cwd(cwd)
        env = self._build_env(with_os_env, extra_env)

        if capture is not None:
            stdout = asyncio.subprocess.PIPE
            stderr = asyncio.subprocess.PIPE

        if shell_cmd is not None:
            command = shell_cmd
            self.logger.debug("spawn shell: %s", command)
            proc = await asyncio.create_subprocess_shell(
                shell_cmd, cwd=work_dir, env=env,
                start_new_session=start_new_session,
                stdin=stdin, stdout=stdout, stderr=stderr,
            )
        else:
            command = " ".join(exec_args)
            self.logger.debug("spawn exec: %s", command)
            proc = await asyncio.create_subprocess_exec(
                *exec_args, cwd=work_dir, env=env,
                start_new_session=start_new_session,
                stdin=stdin, stdout=stdout, stderr=stderr,
            )
        self.logger.info("spawned [%s] pid=%d", name, proc.pid)

        index = self._next_index()
        meta = ProcessMeta(
            index=index, pid=proc.pid,
            pgid=self._get_pgid(proc.pid, start_new_session),
            command=command, name=name, description=description,
            cwd=work_dir, with_os_env=with_os_env, extra_env=extra_env or {},
            start_new_session=start_new_session,
        )

        output: _ProcessOutputImpl | None = None
        if capture is not None:
            output = self._setup_capture(proc, index, capture)

        managed = ManagedProcess(meta=meta, process=proc, output=output)
        managed._stop_impl = lambda timeout: self._stop_managed(managed, timeout)
        self._executing[index] = managed
        self._start_reclaim(managed)

        if on_exit is not None:
            managed.add_done_callback(on_exit)
        return managed

    def _get_pgid(self, pid: int, start_new_session: bool) -> int | None:
        if not start_new_session:
            return None
        try:
            return os.getpgid(pid)
        except (OSError, AttributeError):
            # 平台不支持 (Windows) 或进程已退出
            return None

    def _setup_capture(
            self,
            proc: asyncio.subprocess.Process,
            index: int,
            capture: CaptureSpec,
    ) -> _ProcessOutputImpl:
        stdout_file = capture.stdout_file
        stderr_file = capture.stderr_file

        output = _ProcessOutputImpl(
            stdout_file=stdout_file,
            stderr_file=stderr_file,
            buffer_lines=capture.buffer_lines,
        )
        if proc.stdout:
            self._start_drain(
                proc.stdout, stdout_file, output.stdout_buf,
                capture.buffer_lines, f"stdout:{index}", output,
            )
        if proc.stderr:
            self._start_drain(
                proc.stderr, stderr_file, output.stderr_buf,
                capture.buffer_lines, f"stderr:{index}", output,
            )
        return output

    def _start_drain(
            self,
            stream: asyncio.StreamReader,
            out_file: Path,
            buffer: list[str],
            buffer_lines: int,
            label: str,
            output: _ProcessOutputImpl,
    ) -> None:
        task = asyncio.create_task(
            self._drain(stream, out_file, buffer, buffer_lines, label)
        )
        output.drain_tasks.add(task)
        self._tasks.add(task)
        task.add_done_callback(lambda t: self._tasks.discard(t))

    async def _drain(
            self,
            stream: asyncio.StreamReader,
            out_file: Path | None,
            buffer: list[str],
            buffer_lines: int,
            label: str,
    ) -> None:
        """从 stream 持续读取, 维护内存 tail 窗口; out_file 非空时同步落盘."""
        maintain_buffer = buffer_lines > 0
        try:
            if out_file is not None:
                out_file.parent.mkdir(parents=True, exist_ok=True)
                f = open(out_file, "a")
            else:
                f = None
            try:
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    decoded = line.decode() if isinstance(line, bytes) else line
                    if f is not None:
                        f.write(decoded)
                    if maintain_buffer:
                        buffer.append(decoded)
                        if len(buffer) > buffer_lines:
                            buffer.pop(0)
            finally:
                if f is not None:
                    f.close()
        except asyncio.CancelledError:
            pass
        except Exception:
            self.logger.warning("_drain %s error", label, exc_info=True)

    def _start_reclaim(self, managed: ManagedProcess) -> None:

        async def _reclaim():
            try:
                await managed.process.wait()
            except asyncio.CancelledError:
                # owner 关停路径: aexit 已统一发信号, 这里只兜底等待
                if managed.process.returncode is None:
                    try:
                        managed.process.send_signal(signal.SIGINT)
                        try:
                            await asyncio.wait_for(managed.process.wait(), timeout=2.0)
                            return
                        except asyncio.TimeoutError:
                            pass
                        if managed.process.returncode is None:
                            managed.process.kill()
                            await managed.process.wait()
                    except ProcessLookupError:
                        pass
            finally:
                managed.meta.exit_code = managed.process.returncode
                managed.meta.updated = time.time()
                self.logger.debug(
                    "reclaimed [%s] pid=%d exit=%s",
                    managed.meta.name, managed.process.pid, managed.process.returncode,
                )
                self._executing.pop(managed.meta.index, None)
                self._executed.append(managed.meta)
                while len(self._executed) > _MAX_EXECUTED_HISTORY:
                    self._executed.pop(0)
                # 先 snapshot 再清空, 防 callback 内 re-register 引发递归.
                # _exit_fired=True 后, 后续 add_done_callback 走立即 fire 路径.
                exit_callbacks = list(managed._on_exit_callbacks)
                managed._on_exit_callbacks.clear()
                managed._exit_fired = True
                for cb in exit_callbacks:
                    try:
                        cb(managed.meta)
                    except Exception:
                        self.logger.exception("on_exit callback failed")

        task = asyncio.create_task(_reclaim())
        self._tasks.add(task)
        task.add_done_callback(lambda t: self._tasks.discard(t))
