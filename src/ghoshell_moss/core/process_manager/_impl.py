import asyncio
import os
import signal
import time
from pathlib import Path
from typing import Callable

from ghoshell_moss.contracts.process_manager import (
    ProcessManager,
    ProcessMeta,
    ProcessTask,
    BackgroundTask,
    ManagedProcess,
    BackgroundRunType,
    LoopTimes,
    ErrorInfo,
)
from ghoshell_moss.contracts.logger import get_moss_logger
from ghoshell_moss.message import unique_id
from ghoshell_moss.core.process_manager._utils import killpg as _kill_process_group_util

import logging

__all__ = ["ProcessManagerImpl"]

_MAX_EXECUTED_HISTORY = 200


# -- 内部实现类 --


class _ProcessTaskImpl(ProcessTask):
    """ProcessTask 内部实现 — 持有 ManagedProcess + tail buffer + 文件.

    两种模式, 由 output_file 决定:
    - output_file=path → drain 持文件句柄持续写入
    - output_file=None → drain 只写内存 buffer

    buffer_lines 控制 tail 窗口大小, 两模式均生效. 0 = 不维护 buffer.
    """

    def __init__(
            self,
            id: str,
            managed: ManagedProcess,
            stdout_file: Path | None,
            stderr_file: Path | None,
            stdout_buf: list[str] | None,
            stderr_buf: list[str] | None,
            buffer_lines: int,
    ):
        self._id = id
        self.managed = managed
        self._stdout_file = stdout_file
        self._stderr_file = stderr_file
        self.stdout_buf = stdout_buf if stdout_buf is not None else []
        self.stderr_buf = stderr_buf if stderr_buf is not None else []
        self.buffer_lines = buffer_lines
        self.drain_tasks: set[asyncio.Task] = set()

    @property
    def id(self) -> str:
        return self._id

    @id.setter
    def id(self, value: str) -> None:
        self._id = value

    @property
    def stdout_file(self) -> Path | None:
        return self._stdout_file

    @property
    def stderr_file(self) -> Path | None:
        return self._stderr_file

    def add_drain_task(self, task: asyncio.Task) -> None:
        self.drain_tasks.add(task)

    @property
    def meta(self) -> ProcessMeta:
        return self.managed.meta

    def stdout_buffer(self, *, offset: int = 0, limit: int = 0) -> str:
        if self.buffer_lines <= 0 or not self.stdout_buf:
            return ""
        buf = self.stdout_buf[offset:]
        if limit > 0:
            buf = buf[:limit]
        return "".join(buf)

    def stderr_buffer(self, *, offset: int = 0, limit: int = 0) -> str:
        if self.buffer_lines <= 0 or not self.stderr_buf:
            return ""
        buf = self.stderr_buf[offset:]
        if limit > 0:
            buf = buf[:limit]
        return "".join(buf)

    @property
    def is_alive(self) -> bool:
        return self.managed.process.returncode is None

    @property
    def return_code(self) -> int | None:
        return self.managed.process.returncode

    def send_signal(self, sig: int) -> None:
        self.managed.process.send_signal(sig)

    def terminate(self) -> None:
        self.managed.process.terminate()

    def kill(self) -> None:
        self.managed.process.kill()

    async def wait(self) -> None:
        await self.managed.process.wait()
        if self.drain_tasks:
            await asyncio.gather(*self.drain_tasks, return_exceptions=True)


class _BackgroundTaskImpl(BackgroundTask):
    """BackgroundTask 内部实现 — 持有调度状态 + 复用文件 + 复用 buffer.

    output_file 在首轮创建后复用, stdout_buf / stderr_buf 每轮 drain 直接写入.
    """

    def __init__(
            self,
            id: str,
            name: str,
            type: BackgroundRunType,
            description: str,
            loop: int,
            sleep: float,
            exec_args: tuple[str, ...],
            shell_cmd: str | None,
            cwd: str | None,
            extra_env: dict | None,
            output_file: Path | None,
            buffer_lines: int,
    ):
        self._id = id
        self._name = name
        self._type = type
        self._description = description
        self._loop = loop
        self._sleep = sleep
        self.exec_args = exec_args
        self.shell_cmd = shell_cmd
        self.cwd = cwd
        self.extra_env = extra_env
        self.output_file = output_file
        self.buffer_lines = buffer_lines

        # 复用 buffer — 每轮 drain 更新, 不反复分配
        self.stdout_buf: list[str] = []
        self.stderr_buf: list[str] = []

        self._last: ProcessTask | None = None
        self._executed = 0
        self._is_running = False
        self._last_finish: float = 0.0
        self.error: str = ""

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def type(self) -> BackgroundRunType:
        return self._type

    @property
    def description(self) -> str:
        return self._description

    @property
    def loop(self) -> int:
        return self._loop

    @property
    def executed(self) -> int:
        return self._executed

    @property
    def sleep(self) -> float:
        return self._sleep

    @property
    def is_running(self) -> bool:
        return self._is_running

    def stop(self) -> None:
        self._is_running = False
        self._last_finish = time.time()

    @property
    def last(self) -> ProcessTask | None:
        return self._last

    def last_stdout(self, *, offset: int = 0, limit: int = 0) -> str:
        if self.buffer_lines <= 0 or not self.stdout_buf:
            return ""
        buf = self.stdout_buf[offset:]
        if limit > 0:
            buf = buf[:limit]
        return "".join(buf)

    def last_stderr(self, *, offset: int = 0, limit: int = 0) -> str:
        if self.buffer_lines <= 0 or not self.stderr_buf:
            return ""
        buf = self.stderr_buf[offset:]
        if limit > 0:
            buf = buf[:limit]
        return "".join(buf)


# -- ProcessManager 实现 --


class ProcessManagerImpl(ProcessManager):

    def __init__(
            self,
            root: Path,
            output_path: Path,
            logger: logging.Logger | None = None,
    ):
        self._root = root.resolve()
        self._pwd = self._root
        self._output_path = output_path.resolve()
        self._output_path.mkdir(parents=True, exist_ok=True)
        self.logger = logger or get_moss_logger()

        self._started = False
        self._stopped = False

        self._counter: int = 0
        self._executing: dict[int, ManagedProcess] = {}
        self._executed: list[ProcessMeta] = []

        self._bg_tasks: dict[str, _BackgroundTaskImpl] = {}
        self._stopped_bg: list[BackgroundTask] = []

        self._tasks: set[asyncio.Task] = set()

    # -- 生命周期守卫 --

    def _check_running(self) -> None:
        if not self._started:
            raise RuntimeError("ProcessManager not started — use 'async with'")
        if self._stopped:
            raise RuntimeError("ProcessManager already stopped")

    def _assert_running(self) -> None:
        if not self._started:
            self.logger.error("spawn called before ProcessManager started")
        if self._stopped:
            self.logger.error("spawn called after ProcessManager stopped")

    # -- 目录作用域 --

    @property
    def root(self) -> Path:
        return self._root

    @property
    def pwd(self) -> Path:
        return self._pwd

    def cd(self, path: str, *, from_pwd: bool = True) -> Path:
        self._check_running()
        if from_pwd:
            new_path = (self._pwd / path).resolve()
        else:
            new_path = Path(path).resolve()
        try:
            new_path.relative_to(self._root)
        except ValueError:
            raise ValueError(f"cd: {new_path} is outside root {self._root}")
        self._pwd = new_path
        return self._pwd

    # -- 进程查询 --

    def executing(self) -> dict[int, ProcessMeta]:
        return {idx: managed.meta for idx, managed in self._executing.items()}

    def get_executing(self, index: int) -> ManagedProcess | None:
        return self._executing.get(index)

    def executed(self) -> list[ProcessMeta]:
        return list(self._executed)

    # -- Layer 1: execute / shell --

    async def execute(
            self,
            *args: str,
            name: str | None = None,
            description: str | None = None,
            cwd: str | Path | None = None,
            extra_env: dict | None = None,
            stdin: int | None = None,
            stdout: int | None = None,
            stderr: int | None = None,
            start_new_session: bool = True,
            with_os_env: bool = True,
            on_exit: Callable[[ProcessMeta], None] | None = None,
            **kwargs,
    ) -> ManagedProcess:
        self._assert_running()
        work_dir = self._resolve_cwd(cwd)
        env = self._build_env(with_os_env, extra_env)
        command = " ".join(args)
        process_name = name or (args[0] if args else "unknown")
        self.logger.debug("spawn exec: %s", command)
        proc = await asyncio.create_subprocess_exec(
            *args, cwd=work_dir, env=env,
            start_new_session=start_new_session,
            stdin=stdin, stdout=stdout, stderr=stderr,
        )
        self.logger.info("spawned [%s] pid=%d", process_name, proc.pid)
        managed = self._wrap_and_track(
            proc=proc, command=command, name=process_name,
            description=description or "", cwd=str(work_dir),
            with_os_env=with_os_env, extra_env=extra_env or {},
            start_new_session=start_new_session,
        )
        if on_exit is not None:
            managed.add_done_callback(on_exit)
        return managed

    async def shell(
            self,
            cmd: str,
            *,
            name: str | None = None,
            description: str | None = None,
            cwd: str | Path | None = None,
            extra_env: dict | None = None,
            stdin: int | None = None,
            stdout: int | None = None,
            stderr: int | None = None,
            start_new_session: bool = True,
            with_os_env: bool = True,
            on_exit: Callable[[ProcessMeta], None] | None = None,
            **kwargs,
    ) -> ManagedProcess:
        self._assert_running()
        work_dir = self._resolve_cwd(cwd)
        env = self._build_env(with_os_env, extra_env)
        process_name = name or cmd[:60]
        self.logger.debug("spawn shell: %s", cmd)
        proc = await asyncio.create_subprocess_shell(
            cmd, cwd=work_dir, env=env,
            start_new_session=start_new_session,
            stdin=stdin, stdout=stdout, stderr=stderr,
        )
        self.logger.info("spawned shell [%s] pid=%d", process_name, proc.pid)
        managed = self._wrap_and_track(
            proc=proc, command=cmd, name=process_name,
            description=description or "", cwd=str(work_dir),
            with_os_env=with_os_env, extra_env=extra_env or {},
            start_new_session=start_new_session,
        )
        if on_exit is not None:
            managed.add_done_callback(on_exit)
        return managed

    # -- Layer 2: execute_task / shell_task --

    async def execute_task(
            self,
            *args: str,
            name: str | None = None,
            description: str | None = None,
            cwd: str | Path | None = None,
            extra_env: dict | None = None,
            output_file: Path | None = None,
            with_os_env: bool = True,
            start_new_session: bool = True,
            background_run: tuple[BackgroundRunType, LoopTimes] | None = None,
            sleep: float = 0.0,
            callback: Callable[[ProcessTask], None] | None = None,
            buffer_lines: int = 100,
            **kwargs,
    ) -> ProcessTask:
        self._assert_running()
        process_name = name or (args[0] if args else "unknown")

        if background_run is not None:
            run_type, loop = self._parse_background_run(background_run)
            bt = _BackgroundTaskImpl(
                id="",
                name=process_name,
                type=run_type,
                description=description or "",
                loop=loop, sleep=sleep,
                exec_args=args, shell_cmd=None,
                cwd=str(cwd) if cwd else None,
                extra_env=extra_env,
                output_file=output_file,
                buffer_lines=buffer_lines,
            )
            task = await self._spawn_task(
                exec_args=args, shell_cmd=None,
                name=process_name, description=description or "",
                cwd=cwd, extra_env=extra_env, output_file=output_file,
                with_os_env=with_os_env, start_new_session=start_new_session,
                buffer_lines=buffer_lines,
                stdout_buffer=bt.stdout_buf,
                stderr_buffer=bt.stderr_buf,
            )
            bt._id = task.id
            bt._last = task
            bt._is_running = True
            bt._executed = 1
            task.meta.background_task_id = bt._id
            self._bg_tasks[bt._id] = bt
            self.logger.info("registered bg task [%s] type=%s", process_name, run_type)
            self._create_task_done_watcher(task, lambda t: self._on_bg_iteration_done(bt, t))
            return task

        return await self._spawn_task(
            exec_args=args, shell_cmd=None,
            name=process_name, description=description or "",
            cwd=cwd, extra_env=extra_env, output_file=output_file,
            with_os_env=with_os_env, start_new_session=start_new_session,
            buffer_lines=buffer_lines,
            done_callback=callback,
        )

    async def shell_task(
            self,
            cmd: str,
            *,
            name: str | None = None,
            description: str | None = None,
            cwd: str | Path | None = None,
            extra_env: dict | None = None,
            output_file: Path | None = None,
            with_os_env: bool = True,
            start_new_session: bool = True,
            background_run: tuple[BackgroundRunType, LoopTimes] | None = None,
            sleep: float = 0.0,
            callback: Callable[[ProcessTask], None] | None = None,
            buffer_lines: int = 100,
            **kwargs,
    ) -> ProcessTask:
        self._assert_running()
        process_name = name or cmd[:60]

        if background_run is not None:
            run_type, loop = self._parse_background_run(background_run)
            bt = _BackgroundTaskImpl(
                id="",
                name=process_name,
                type=run_type,
                description=description or "",
                loop=loop, sleep=sleep,
                exec_args=(), shell_cmd=cmd,
                cwd=str(cwd) if cwd else None,
                extra_env=extra_env,
                output_file=output_file,
                buffer_lines=buffer_lines,
            )
            task = await self._spawn_task(
                exec_args=(), shell_cmd=cmd,
                name=process_name, description=description or "",
                cwd=cwd, extra_env=extra_env, output_file=output_file,
                with_os_env=with_os_env, start_new_session=start_new_session,
                buffer_lines=buffer_lines,
                stdout_buffer=bt.stdout_buf,
                stderr_buffer=bt.stderr_buf,
            )
            bt._id = task.id
            bt._last = task
            bt._is_running = True
            bt._executed = 1
            task.meta.background_task_id = bt._id
            self._bg_tasks[bt._id] = bt
            self.logger.info("registered bg task [%s] type=%s", process_name, run_type)
            self._create_task_done_watcher(task, lambda t: self._on_bg_iteration_done(bt, t))
            return task

        return await self._spawn_task(
            exec_args=(), shell_cmd=cmd,
            name=process_name, description=description or "",
            cwd=cwd, extra_env=extra_env, output_file=output_file,
            with_os_env=with_os_env, start_new_session=start_new_session,
            buffer_lines=buffer_lines,
            done_callback=callback,
        )

    # -- Layer 3: 后台任务管理 --

    def background_tasks(self) -> list[BackgroundTask]:
        return list(self._bg_tasks.values())

    async def refresh_background(self) -> list[BackgroundTask]:
        self._check_running()
        updated: list[BackgroundTask] = []
        now = time.time()

        for bt in list(self._bg_tasks.values()):
            if bt.type == "once":
                if bt.executed > 0:
                    continue
                await self._start_bg_iteration(bt)
                updated.append(bt)

            elif bt.type == "loop":
                if bt.is_running:
                    continue
                if bt.error:
                    self.logger.debug(
                        "bg [%s] error state, skip auto-loop: %s", bt.name, bt.error,
                    )
                    bt.error = ""
                    continue
                if bt.loop > 0 and bt.executed >= bt.loop:
                    continue
                if bt.sleep > 0 and bt._last_finish > 0:
                    if now - bt._last_finish < bt.sleep:
                        continue
                await self._start_bg_iteration(bt)
                updated.append(bt)

            elif bt.type == "on_prompt":
                if bt.is_running and bt.last is not None:
                    bt.last.kill()
                    try:
                        await asyncio.wait_for(bt.last.wait(), timeout=5.0)
                    except asyncio.TimeoutError:
                        self.logger.warning("bg %s on_prompt kill timeout", bt.id)
                await self._start_bg_iteration(bt)
                updated.append(bt)

        return updated

    async def stop_background_task(self, task_id: str) -> bool:
        bt = self._bg_tasks.pop(task_id, None)
        if bt is None:
            self.logger.warning("stop_background_task: unknown id %s", task_id)
            return False
        self.logger.info("stop bg [%s] %s", bt.name, task_id)
        if bt.is_running and bt.last is not None:
            bt.last.kill()
            try:
                await asyncio.wait_for(bt.last.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self.logger.warning("bg %s kill timeout, force stop", task_id)
        bt.stop()
        self._stopped_bg.append(bt)
        return True

    def stopped_background_tasks(self) -> list[BackgroundTask]:
        return list(self._stopped_bg)

    # -- 进程组治理 --

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

    async def __aenter__(self) -> "ProcessManagerImpl":
        self._started = True
        self.logger.info("ProcessManager started, root=%s", self._root)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self._stopped = True
        self.logger.info(
            "ProcessManager stopping — %d executing, %d bg",
            len(self._executing), len(self._bg_tasks),
        )
        for bt in list(self._bg_tasks.values()):
            await self.stop_background_task(bt.id)
        self._bg_tasks.clear()

        grace_period = 3.0
        for managed in list(self._executing.values()):
            if managed.process.returncode is not None:
                continue
            self.killpg(managed.process.pid, signal.SIGINT)
        try:
            await asyncio.wait_for(
                asyncio.gather(
                    *(managed.process.wait()
                      for managed in self._executing.values()
                      if managed.process.returncode is None),
                    return_exceptions=True,
                ),
                timeout=grace_period,
            )
        except asyncio.TimeoutError:
            self.logger.warning(
                "shutdown: %d alive after SIGINT, SIGKILL",
                sum(1 for m in self._executing.values() if m.process.returncode is None),
            )
        for managed in list(self._executing.values()):
            if managed.process.returncode is not None:
                continue
            self.killpg(managed.process.pid, signal.SIGKILL)
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
        self.logger.info("ProcessManager stopped — %d executed", len(self._executed))

    # ================================================================
    # 内部方法
    # ================================================================

    def _next_index(self) -> int:
        self._counter += 1
        return self._counter

    def _resolve_cwd(self, cwd: str | Path | None) -> str:
        if cwd is None:
            return str(self._pwd)
        path = Path(cwd)
        if not path.is_absolute():
            path = self._pwd / path
        return str(path.resolve())

    @staticmethod
    def _build_env(with_os_env: bool, extra_env: dict | None) -> dict:
        env = os.environ.copy() if with_os_env else {}
        if extra_env:
            env.update(extra_env)
        return env

    def _wrap_and_track(
            self,
            proc: asyncio.subprocess.Process,
            command: str,
            name: str,
            description: str,
            cwd: str,
            with_os_env: bool,
            extra_env: dict,
            start_new_session: bool,
            done_callback: Callable[[], None] | None = None,
    ) -> ManagedProcess:
        index = self._next_index()
        meta = ProcessMeta(
            index=index, pid=proc.pid, command=command,
            name=name, description=description, cwd=cwd,
            with_os_env=with_os_env, extra_env=extra_env,
            start_new_session=start_new_session,
        )
        managed = ManagedProcess(meta=meta, process=proc)
        self._executing[index] = managed
        self._start_reclaim(managed, done_callback=done_callback)
        return managed

    def _start_reclaim(
            self,
            managed: ManagedProcess,
            done_callback: Callable[[], None] | None = None,
    ) -> None:

        async def _reclaim():
            try:
                await managed.process.wait()
            except asyncio.CancelledError:
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
                if done_callback:
                    try:
                        done_callback()
                    except Exception:
                        self.logger.exception("done_callback failed")
                # SS-10: fire ManagedProcess.add_done_callback registered callbacks.
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

    # -- 内部: 一次性任务 spawn --

    async def _spawn_task(
            self,
            exec_args: tuple[str, ...],
            shell_cmd: str | None,
            name: str,
            description: str,
            cwd: str | Path | None,
            extra_env: dict | None,
            output_file: Path | None,
            with_os_env: bool,
            start_new_session: bool,
            buffer_lines: int,
            stdout_buffer: list[str] | None = None,
            stderr_buffer: list[str] | None = None,
            done_callback: Callable[[ProcessTask], None] | None = None,
    ) -> _ProcessTaskImpl:
        work_dir = self._resolve_cwd(cwd)
        env = self._build_env(with_os_env, extra_env)
        task_id = unique_id()

        # 确定 stdout/stderr 文件路径
        task_dir = self._output_path / f"task_{task_id}"
        if output_file is not None:
            stdout_file: Path | None = output_file
        else:
            stdout_file = None
        stderr_file: Path | None = task_dir / "stderr.txt" if buffer_lines > 0 else None

        # spawn
        if shell_cmd is not None:
            self.logger.debug("spawn_task shell: %s", shell_cmd)
            proc = await asyncio.create_subprocess_shell(
                shell_cmd, cwd=work_dir, env=env,
                start_new_session=start_new_session,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            command = shell_cmd
        else:
            command = " ".join(exec_args)
            self.logger.debug("spawn_task exec: %s", command)
            proc = await asyncio.create_subprocess_exec(
                *exec_args, cwd=work_dir, env=env,
                start_new_session=start_new_session,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

        self.logger.info("spawned task [%s] pid=%d buf=%d", name, proc.pid, buffer_lines)

        managed = self._wrap_and_track(
            proc=proc, command=command, name=name, description=description,
            cwd=str(work_dir), with_os_env=with_os_env, extra_env=extra_env or {},
            start_new_session=start_new_session,
        )

        process_task = _ProcessTaskImpl(
            id=task_id,
            managed=managed,
            stdout_file=stdout_file,
            stderr_file=stderr_file,
            stdout_buf=stdout_buffer,
            stderr_buf=stderr_buffer,
            buffer_lines=buffer_lines,
        )
        managed.meta.task_id = task_id

        # 启动 drain 协程
        if proc.stdout:
            drain_out = asyncio.create_task(
                self._drain(
                    stream=proc.stdout,
                    out_file=stdout_file,
                    buffer=process_task.stdout_buf,
                    buffer_lines=buffer_lines,
                    label=f"stdout:{task_id}",
                )
            )
            process_task.add_drain_task(drain_out)
            self._tasks.add(drain_out)
            drain_out.add_done_callback(lambda t: self._tasks.discard(t))

        if proc.stderr:
            drain_err = asyncio.create_task(
                self._drain(
                    stream=proc.stderr,
                    out_file=stderr_file,
                    buffer=process_task.stderr_buf,
                    buffer_lines=buffer_lines,
                    label=f"stderr:{task_id}",
                )
            )
            process_task.add_drain_task(drain_err)
            self._tasks.add(drain_err)
            drain_err.add_done_callback(lambda t: self._tasks.discard(t))

        if done_callback:
            self._create_task_done_watcher(process_task, done_callback)

        return process_task

    def _create_task_done_watcher(
            self,
            process_task: _ProcessTaskImpl,
            done_callback: Callable[[ProcessTask], None],
    ) -> None:

        async def _watcher():
            await process_task.wait()
            self.logger.debug(
                "task done [%s] exit=%d",
                process_task.meta.name, process_task.return_code,
            )
            try:
                done_callback(process_task)
            except Exception:
                self.logger.exception(
                    "done_callback failed for [%s]", process_task.meta.name,
                )

        watcher = asyncio.create_task(_watcher())
        self._tasks.add(watcher)
        watcher.add_done_callback(lambda t: self._tasks.discard(t))

    async def _drain(
            self,
            stream: asyncio.StreamReader,
            out_file: Path | None,
            buffer: list[str],
            buffer_lines: int,
            label: str,
    ) -> None:
        """从 stream 持续读取, 写入文件 (持有句柄) + 维护 tail buffer.

        out_file=None  → 只写 buffer.
        out_file=path → 持有文件句柄持续写入, 同时维护 buffer.
        buffer_lines=0 → 不维护 buffer.
        """
        write_file = out_file is not None
        maintain_buffer = buffer_lines > 0

        try:
            if write_file:
                out_file.parent.mkdir(parents=True, exist_ok=True)
                with open(out_file, "a") as f:
                    while True:
                        line = await stream.readline()
                        if not line:
                            break
                        decoded = line.decode() if isinstance(line, bytes) else line
                        f.write(decoded)
                        if maintain_buffer:
                            buffer.append(decoded)
                            if len(buffer) > buffer_lines:
                                buffer.pop(0)
            else:
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    decoded = line.decode() if isinstance(line, bytes) else line
                    if maintain_buffer:
                        buffer.append(decoded)
                        if len(buffer) > buffer_lines:
                            buffer.pop(0)
        except asyncio.CancelledError:
            pass
        except Exception:
            self.logger.warning("_drain %s error", label, exc_info=True)

    # -- 内部: 后台任务调度 --

    def _on_bg_iteration_done(
            self, bt: _BackgroundTaskImpl, task: ProcessTask
    ) -> None:
        bt.stop()
        if bt.type == "loop" and task.return_code != 0:
            bt.error = f"exit={task.return_code}"
            self.logger.warning(
                "bg [%s] loop exit=%d, paused until next refresh",
                bt.name, task.return_code,
            )

    @staticmethod
    def _parse_background_run(
            bg_run: tuple[BackgroundRunType, LoopTimes] | tuple[BackgroundRunType, ...],
    ) -> tuple[BackgroundRunType, int]:
        run_type = bg_run[0]
        if len(bg_run) >= 2:
            return run_type, bg_run[1]  # type: ignore[return-value]
        if run_type == "once":
            return run_type, 1
        elif run_type == "loop":
            return run_type, 0
        else:
            return run_type, 1

    async def _start_bg_iteration(self, bt: _BackgroundTaskImpl) -> None:
        """为后台任务启动新一轮执行. 复用 bt 的 output_file + buffer."""
        # 清空 buffer, 新 drain 重新填充
        bt.stdout_buf.clear()
        bt.stderr_buf.clear()

        task = await self._spawn_task(
            exec_args=bt.exec_args,
            shell_cmd=bt.shell_cmd,
            name=bt.name,
            description=bt.description,
            cwd=bt.cwd,
            extra_env=bt.extra_env,
            output_file=bt.output_file,
            with_os_env=True,
            start_new_session=True,
            buffer_lines=bt.buffer_lines,
            stdout_buffer=bt.stdout_buf,
            stderr_buffer=bt.stderr_buf,
            done_callback=lambda t: self._on_bg_iteration_done(bt, t),
        )
        bt._last = task
        bt._is_running = True
        bt._executed += 1
        task.meta.background_task_id = bt.id
