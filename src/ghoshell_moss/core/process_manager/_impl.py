import asyncio
import os
import signal
import time
import uuid
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
from ghoshell_moss.core.process_manager._utils import killpg as _killpg_util

import logging

__all__ = ["ProcessManagerImpl"]

_MAX_EXECUTED_HISTORY = 200


# -- 内部实现类 --


class _ProcessTaskImpl(ProcessTask):
    """ProcessTask 内部实现 — 持有 ManagedProcess + buffer + file output."""

    def __init__(
        self,
        id: str,
        mp: ManagedProcess,
        stdout_file: Path,
        stderr_file: Path,
        buffer_lines: int,
        logger: logging.Logger,
    ):
        self._id = id
        self._mp = mp
        self._stdout_file = stdout_file
        self._stderr_file = stderr_file
        self._stdout_buf: list[str] = []
        self._stderr_buf: list[str] = []
        self._buffer_lines = buffer_lines
        self._logger = logger
        self._drain_tasks: set[asyncio.Task] = set()

    @property
    def id(self) -> str:
        return self._id

    @property
    def meta(self) -> ProcessMeta:
        return self._mp.meta

    def stdout_buffer(self, *, offset: int = 0, limit: int = 0) -> str:
        buf = self._stdout_buf[offset:]
        if limit > 0:
            buf = buf[:limit]
        return "".join(buf)

    def stderr_buffer(self, *, offset: int = 0, limit: int = 0) -> str:
        buf = self._stderr_buf[offset:]
        if limit > 0:
            buf = buf[:limit]
        return "".join(buf)

    @property
    def stdout_file(self) -> Path:
        return self._stdout_file

    @property
    def stderr_file(self) -> Path:
        return self._stderr_file

    @property
    def is_alive(self) -> bool:
        return self._mp.process.returncode is None

    @property
    def return_code(self) -> int | None:
        return self._mp.process.returncode

    def send_signal(self, sig: int) -> None:
        self._mp.process.send_signal(sig)

    def terminate(self) -> None:
        self._mp.process.terminate()

    def kill(self) -> None:
        self._mp.process.kill()

    async def wait(self) -> None:
        """等待进程退出 + 所有 drain 任务完成."""
        await self._mp.process.wait()
        if self._drain_tasks:
            await asyncio.gather(*self._drain_tasks, return_exceptions=True)


class _BackgroundTaskImpl(BackgroundTask):
    """BackgroundTask 内部实现 — 持有调度状态和 spawn 参数."""

    def __init__(
        self,
        id: str,
        name: str,
        type: BackgroundRunType,
        description: str,
        loop: int,
        sleep: float,
        # 保存参数以在 refresh 时重新 spawn
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
        self._exec_args = exec_args
        self._shell_cmd = shell_cmd
        self._cwd = cwd
        self._extra_env = extra_env
        self._output_file = output_file
        self._buffer_lines = buffer_lines
        self._last: ProcessTask | None = None
        self._executed = 0
        self._is_running = False
        self._last_finish: float = 0.0

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

    @property
    def last(self) -> ProcessTask | None:
        return self._last

    def last_stdout(self, *, offset: int = 0, limit: int = 0) -> str:
        if self._last is None:
            return ""
        return self._last.stdout_buffer(offset=offset, limit=limit)

    def last_stderr(self, *, offset: int = 0, limit: int = 0) -> str:
        if self._last is None:
            return ""
        return self._last.stderr_buffer(offset=offset, limit=limit)


# -- ProcessManager 实现 --


class ProcessManagerImpl(ProcessManager):
    """ProcessManager 实现.

    使用 asyncio.create_subprocess_exec / create_subprocess_shell
    作为底层, 提供 exec / shell / task / background 四种模式.
    """

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
        self._logger = logger or get_moss_logger()

        # 进程簿记
        self._counter: int = 0
        self._executing: dict[int, ManagedProcess] = {}
        self._executed: list[ProcessMeta] = []

        # 后台任务
        self._bg_tasks: dict[str, _BackgroundTaskImpl] = {}
        self._stopped_bg: list[BackgroundTask] = []

        # 内部 asyncio task 追踪 (reclaim / drain)
        self._tasks: set[asyncio.Task] = set()

    # -- 目录作用域 --

    @property
    def root(self) -> Path:
        return self._root

    @property
    def pwd(self) -> Path:
        return self._pwd

    def cd(self, path: str, *, from_pwd: bool = True) -> Path:
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
        return {idx: mp.meta for idx, mp in self._executing.items()}

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
        **kwargs,
    ) -> ManagedProcess:
        work_dir = self._resolve_cwd(cwd)
        env = self._build_env(with_os_env, extra_env)

        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=work_dir,
            env=env,
            start_new_session=start_new_session,
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
        )

        return self._wrap_and_track(
            proc=proc,
            command=" ".join(args),
            name=name or (args[0] if args else "unknown"),
            description=description or "",
            cwd=str(work_dir),
            with_os_env=with_os_env,
            extra_env=extra_env or {},
            start_new_session=start_new_session,
        )

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
        **kwargs,
    ) -> ManagedProcess:
        work_dir = self._resolve_cwd(cwd)
        env = self._build_env(with_os_env, extra_env)

        proc = await asyncio.create_subprocess_shell(
            cmd,
            cwd=work_dir,
            env=env,
            start_new_session=start_new_session,
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
        )

        return self._wrap_and_track(
            proc=proc,
            command=cmd,
            name=name or cmd[:60],
            description=description or "",
            cwd=str(work_dir),
            with_os_env=with_os_env,
            extra_env=extra_env or {},
            start_new_session=start_new_session,
        )

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
        task = await self._spawn_task(
            exec_args=args,
            shell_cmd=None,
            name=name or (args[0] if args else "unknown"),
            description=description or "",
            cwd=cwd,
            extra_env=extra_env,
            output_file=output_file,
            with_os_env=with_os_env,
            start_new_session=start_new_session,
            buffer_lines=buffer_lines,
        )

        if callback:
            callback(task)

        if background_run is not None:
            run_type, loop = self._parse_background_run(background_run)
            self._register_background(
                task=task,
                run_type=run_type,
                loop=loop,
                sleep=sleep,
                exec_args=args,
                shell_cmd=None,
                cwd=cwd,
                extra_env=extra_env,
                output_file=output_file,
                buffer_lines=buffer_lines,
                name=name or (args[0] if args else "unknown"),
                description=description or "",
            )

        return task

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
        task = await self._spawn_task(
            exec_args=(),
            shell_cmd=cmd,
            name=name or cmd[:60],
            description=description or "",
            cwd=cwd,
            extra_env=extra_env,
            output_file=output_file,
            with_os_env=with_os_env,
            start_new_session=start_new_session,
            buffer_lines=buffer_lines,
        )

        if callback:
            callback(task)

        if background_run is not None:
            run_type, loop = self._parse_background_run(background_run)
            self._register_background(
                task=task,
                run_type=run_type,
                loop=loop,
                sleep=sleep,
                exec_args=(),
                shell_cmd=cmd,
                cwd=cwd,
                extra_env=extra_env,
                output_file=output_file,
                buffer_lines=buffer_lines,
                name=name or cmd[:60],
                description=description or "",
            )

        return task

    # -- Layer 3: 后台任务管理 --

    def background_tasks(self) -> list[BackgroundTask]:
        return list(self._bg_tasks.values())

    async def refresh_background(self) -> list[BackgroundTask]:
        updated: list[BackgroundTask] = []
        now = time.time()

        for bt in list(self._bg_tasks.values()):
            if bt._type == "once":
                if bt._executed > 0:
                    continue
                await self._start_bg_iteration(bt)
                updated.append(bt)

            elif bt._type == "loop":
                if bt._is_running:
                    continue
                if bt._loop > 0 and bt._executed >= bt._loop:
                    continue
                # 检查 sleep 间隔
                if bt._sleep > 0 and bt._last_finish > 0:
                    if now - bt._last_finish < bt._sleep:
                        continue
                await self._start_bg_iteration(bt)
                updated.append(bt)

            elif bt._type == "on_prompt":
                # 杀掉当前运行的, 起新的
                if bt._is_running and bt._last is not None:
                    bt._last.kill()
                    try:
                        await asyncio.wait_for(bt._last.wait(), timeout=5.0)
                    except asyncio.TimeoutError:
                        pass
                await self._start_bg_iteration(bt)
                updated.append(bt)

        return updated

    async def stop_background_task(self, task_id: str) -> bool:
        bt = self._bg_tasks.pop(task_id, None)
        if bt is None:
            return False
        if bt._is_running and bt._last is not None:
            bt._last.kill()
            try:
                await asyncio.wait_for(bt._last.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                pass
        bt._is_running = False
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
            return f"kill({pid}) denied: {e}"
        except OSError as e:
            return f"kill({pid}) failed: {e}"

    def killpg(self, process_group: int, sig: int) -> ErrorInfo | None:
        return _killpg_util(process_group, sig)

    # -- 生命周期 --

    async def __aenter__(self) -> "ProcessManagerImpl":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        # 1. 停所有后台任务
        for bt in list(self._bg_tasks.values()):
            await self.stop_background_task(bt._id)
        self._bg_tasks.clear()

        # 2. 先礼后兵 — 对所有执行中的进程发 SIGINT, 等 3 秒, 然后 SIGKILL
        for mp in list(self._executing.values()):
            if mp.process.returncode is not None:
                continue
            try:
                mp.process.send_signal(signal.SIGINT)
            except ProcessLookupError:
                continue

        # 等 3 秒
        try:
            await asyncio.wait_for(
                asyncio.gather(
                    *(mp.process.wait() for mp in self._executing.values()
                      if mp.process.returncode is None),
                    return_exceptions=True,
                ),
                timeout=3.0,
            )
        except asyncio.TimeoutError:
            pass

        # 还不死的强制 kill
        for mp in list(self._executing.values()):
            if mp.process.returncode is not None:
                continue
            try:
                mp.process.kill()
            except ProcessLookupError:
                pass

        # 等 kill 生效
        for mp in list(self._executing.values()):
            try:
                await mp.process.wait()
            except (ProcessLookupError, OSError):
                pass

        # 3. 等待 reclaim / drain 任务完成
        if self._tasks:
            for t in self._tasks:
                t.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)
            self._tasks.clear()

        self._executing.clear()

    # ================================================================
    # 内部方法
    # ================================================================

    def _next_index(self) -> int:
        self._counter += 1
        return self._counter

    def _resolve_cwd(self, cwd: str | Path | None) -> str:
        if cwd is None:
            return str(self._pwd)
        p = Path(cwd)
        if not p.is_absolute():
            p = self._pwd / p
        return str(p.resolve())

    def _build_env(self, with_os_env: bool, extra_env: dict | None) -> dict:
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
    ) -> ManagedProcess:
        index = self._next_index()
        meta = ProcessMeta(
            index=index,
            pid=proc.pid,
            command=command,
            name=name,
            description=description,
            cwd=cwd,
            with_os_env=with_os_env,
            extra_env=extra_env,
            start_new_session=start_new_session,
        )
        mp = ManagedProcess(meta=meta, process=proc)
        self._executing[index] = mp
        self._start_reclaim(mp)
        return mp

    def _start_reclaim(self, mp: ManagedProcess) -> None:
        """启动收尸协程 — 等进程退出, 迁入 executed 历史."""

        async def _reclaim():
            try:
                await mp.process.wait()
            except asyncio.CancelledError:
                if mp.process.returncode is None:
                    try:
                        mp.process.send_signal(signal.SIGINT)
                        try:
                            await asyncio.wait_for(mp.process.wait(), timeout=2.0)
                            return
                        except asyncio.TimeoutError:
                            pass
                        if mp.process.returncode is None:
                            mp.process.kill()
                            await mp.process.wait()
                    except ProcessLookupError:
                        pass
            finally:
                mp.meta.exit_code = mp.process.returncode
                mp.meta.updated = time.time()
                self._executing.pop(mp.meta.index, None)
                self._executed.append(mp.meta)
                # 保持 executed 列表不无限膨胀
                while len(self._executed) > _MAX_EXECUTED_HISTORY:
                    self._executed.pop(0)

        t = asyncio.create_task(_reclaim())
        self._tasks.add(t)
        t.add_done_callback(lambda _t: self._tasks.discard(_t))

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
    ) -> _ProcessTaskImpl:
        work_dir = self._resolve_cwd(cwd)
        env = self._build_env(with_os_env, extra_env)
        task_id = uuid.uuid4().hex[:12]

        # 确定输出文件路径
        task_dir = self._output_path / f"task_{task_id}"
        task_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_file or (task_dir / "stdout.txt")
        err_path = task_dir / "stderr.txt"

        # spawn 进程, stdout/stderr 走 PIPE
        if shell_cmd is not None:
            proc = await asyncio.create_subprocess_shell(
                shell_cmd,
                cwd=work_dir,
                env=env,
                start_new_session=start_new_session,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            command = shell_cmd
        else:
            proc = await asyncio.create_subprocess_exec(
                *exec_args,
                cwd=work_dir,
                env=env,
                start_new_session=start_new_session,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            command = " ".join(exec_args)

        mp = self._wrap_and_track(
            proc=proc,
            command=command,
            name=name,
            description=description,
            cwd=str(work_dir),
            with_os_env=with_os_env,
            extra_env=extra_env or {},
            start_new_session=start_new_session,
        )

        pt = _ProcessTaskImpl(
            id=task_id,
            mp=mp,
            stdout_file=out_path,
            stderr_file=err_path,
            buffer_lines=buffer_lines,
            logger=self._logger,
        )

        # 启动 drain 协程
        if proc.stdout:
            t_out = asyncio.create_task(
                self._drain(proc.stdout, pt._stdout_buf, out_path, buffer_lines)
            )
            pt._drain_tasks.add(t_out)
            self._tasks.add(t_out)
            t_out.add_done_callback(lambda _t: self._tasks.discard(_t))

        if proc.stderr:
            t_err = asyncio.create_task(
                self._drain(proc.stderr, pt._stderr_buf, err_path, buffer_lines)
            )
            pt._drain_tasks.add(t_err)
            self._tasks.add(t_err)
            t_err.add_done_callback(lambda _t: self._tasks.discard(_t))

        # 关联 task_id 到 meta
        mp.meta.task_id = task_id

        return pt

    @staticmethod
    async def _drain(
        stream: asyncio.StreamReader,
        buffer: list[str],
        out_file: Path,
        max_lines: int,
    ) -> None:
        """从 stream 持续读取, 同时写入 buffer (窗口) 和文件 (完整)."""
        try:
            # 确保父目录存在
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, "a") as f:
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    decoded = line.decode() if isinstance(line, bytes) else line
                    buffer.append(decoded)
                    if len(buffer) > max_lines:
                        buffer.pop(0)
                    f.write(decoded)
                    f.flush()
        except Exception:
            pass  # stream 关闭或其它 I/O 错误, 静默结束

    # -- 内部: 后台任务 --

    @staticmethod
    def _parse_background_run(
        bg_run: tuple[BackgroundRunType, LoopTimes] | tuple[BackgroundRunType, ...],
    ) -> tuple[BackgroundRunType, int]:
        run_type = bg_run[0]
        if len(bg_run) >= 2:
            return run_type, bg_run[1]  # type: ignore[return-value]  # tuple[..., ...] 推断为 int
        # 默认 loop 值
        if run_type == "once":
            return run_type, 1
        elif run_type == "loop":
            return run_type, 0  # 无限
        else:  # on_prompt
            return run_type, 1

    def _register_background(
        self,
        task: ProcessTask,
        run_type: BackgroundRunType,
        loop: int,
        sleep: float,
        exec_args: tuple[str, ...],
        shell_cmd: str | None,
        cwd: str | Path | None,
        extra_env: dict | None,
        output_file: Path | None,
        buffer_lines: int,
        name: str,
        description: str,
    ) -> None:
        bt = _BackgroundTaskImpl(
            id=task.id,
            name=name,
            type=run_type,
            description=description,
            loop=loop,
            sleep=sleep,
            exec_args=exec_args,
            shell_cmd=shell_cmd,
            cwd=str(cwd) if cwd else None,
            extra_env=extra_env,
            output_file=output_file,
            buffer_lines=buffer_lines,
        )
        bt._last = task
        bt._is_running = True
        bt._executed = 1

        # 监听本轮执行完成
        asyncio.create_task(self._watch_bg_iteration(bt))

        task.meta.background_task_id = bt._id
        self._bg_tasks[bt._id] = bt

    async def _watch_bg_iteration(self, bt: _BackgroundTaskImpl) -> None:
        """等待本轮 ProcessTask 结束, 更新状态."""
        if bt._last is not None:
            try:
                await bt._last.wait()
            except Exception:
                pass
        bt._is_running = False
        bt._last_finish = time.time()

    async def _start_bg_iteration(self, bt: _BackgroundTaskImpl) -> None:
        """为后台任务启动新一轮执行."""
        task = await self._spawn_task(
            exec_args=bt._exec_args,
            shell_cmd=bt._shell_cmd,
            name=bt._name,
            description=bt._description,
            cwd=bt._cwd,
            extra_env=bt._extra_env,
            output_file=bt._output_file,
            with_os_env=True,
            start_new_session=True,
            buffer_lines=bt._buffer_lines,
        )
        bt._last = task
        bt._is_running = True
        bt._executed += 1
        task.meta.background_task_id = bt._id
        asyncio.create_task(self._watch_bg_iteration(bt))