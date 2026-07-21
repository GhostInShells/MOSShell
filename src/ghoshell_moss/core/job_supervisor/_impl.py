"""JobSupervisor 默认实现 — asyncio 调度器, 组合 Subprocesses 执行短命进程.

契约见 ghoshell_moss.contracts.job_supervisor.
"""

# 实现要点 (reviewer 上下文):
# - 每 Job 一个 runner 协程, 由 asyncio.sleep 实现 interval, 无全局 tick.
# - Job 的执行体是 Subprocesses.execute/shell(capture=CaptureSpec), 单轮
#   的进程治理 (setsid/killpg/reclaim) 全部继承下层, 本层零重叠.
# - 每轮 buffer 从零重开: bt.stdout_buf.clear() 的等价物在这里是
#   "新 spawn 拿到新 ProcessOutput 实例", 快照就是最近一轮的 output.
# - stop 语义唯一路径: cancel runner → 若有正在跑的 ManagedProcess 走 stop(timeout).
# - PAUSED/on_failure=pause: runner 在 wait_event 上阻塞, resume 唤醒即可.
#   与 SLEEPING 状态区分 — SLEEPING 是自愿间隔, PAUSED 是失败守门.

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from ghoshell_moss.contracts.job_supervisor import (
    Job,
    JobSnapshot,
    JobSpec,
    JobStatus,
    JobSupervisor,
)
from ghoshell_moss.contracts.logger import get_moss_logger
from ghoshell_moss.contracts.subprocesses import (
    CaptureSpec,
    ManagedProcess,
    Subprocesses,
)
from ghoshell_moss.message import unique_id

__all__ = ["JobSupervisorImpl", "JobImpl"]

_MAX_JOB_HISTORY = 200


class JobImpl(Job):
    """一个 Job 的运行时状态 + 调度 runner 协程."""

    def __init__(
            self,
            spec: JobSpec,
            subprocesses: Subprocesses,
            logger: logging.Logger,
    ):
        self._id = unique_id()
        self._spec = spec
        self._sp = subprocesses
        self.logger = logger

        self._status: JobStatus = JobStatus.PENDING
        self._executed: int = 0
        self._last_exit_code: int | None = None
        self._last_started: float | None = None
        self._last_finished: float | None = None
        self._last_managed: ManagedProcess | None = None

        self._resume_event = asyncio.Event()
        self._stop_flag = False
        self._done_event = asyncio.Event()
        self._runner: asyncio.Task[Any] | None = None

    @property
    def id(self) -> str:
        return self._id

    @property
    def spec(self) -> JobSpec:
        return self._spec

    def snapshot(self) -> JobSnapshot:
        stdout_tail = ""
        stderr_tail = ""
        if self._last_managed is not None and self._last_managed.output is not None:
            stdout_tail = self._last_managed.output.stdout()
            stderr_tail = self._last_managed.output.stderr()
        return JobSnapshot(
            id=self._id,
            name=self._spec.name,
            description=self._spec.description,
            status=self._status,
            executed=self._executed,
            times=self._spec.times,
            last_exit_code=self._last_exit_code,
            last_started=self._last_started,
            last_finished=self._last_finished,
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
        )

    async def stop(self, timeout: float = 5.0) -> None:
        # 幂等: 已在终态直接返回
        if self._status in (JobStatus.FINISHED, JobStatus.STOPPED):
            return
        self._stop_flag = True
        # 唤醒可能阻塞在 PAUSED 的 runner, 让它看见 stop_flag
        self._resume_event.set()
        # 杀正在跑的一轮
        managed = self._last_managed
        if managed is not None and managed.process.returncode is None:
            try:
                await managed.stop(timeout=timeout)
            except Exception:
                self.logger.exception("job [%s] stop managed failed", self._spec.name)
        # 等 runner 收尾
        if self._runner is not None:
            try:
                await asyncio.wait_for(self._runner, timeout=timeout + 1.0)
            except asyncio.TimeoutError:
                self._runner.cancel()
                try:
                    await self._runner
                except (asyncio.CancelledError, Exception):
                    pass
        self._status = JobStatus.STOPPED
        self._done_event.set()

    def resume(self) -> None:
        if self._status != JobStatus.PAUSED:
            return
        # runner 会在 _resume_event 上苏醒, 重置 status 也交给它
        self._resume_event.set()

    async def wait(self) -> JobSnapshot:
        await self._done_event.wait()
        return self.snapshot()

    # -- runner 内部 --

    def _launch(self) -> None:
        # 由 JobSupervisor 在 submit 时调用一次.
        self._runner = asyncio.create_task(self._run())

    async def _run(self) -> None:
        try:
            # 首轮延迟
            if self._spec.at is not None:
                delay = self._spec.at - time.time()
                if delay > 0:
                    await self._sleep_interruptible(delay)
                    if self._stop_flag:
                        return

            while True:
                if self._stop_flag:
                    return
                if self._spec.times > 0 and self._executed >= self._spec.times:
                    self._status = JobStatus.FINISHED
                    return

                # 执行一轮
                self._status = JobStatus.RUNNING
                self._last_started = time.time()
                try:
                    managed = await self._spawn_one()
                except Exception:
                    self.logger.exception("job [%s] spawn failed", self._spec.name)
                    self._status = JobStatus.STOPPED
                    return
                self._last_managed = managed
                # 等这一轮 + drain 完成 (输出窗口读到的是完整数据)
                await managed.process.wait()
                if managed.output is not None:
                    await managed.output.wait_drained()
                self._executed += 1
                self._last_finished = time.time()
                self._last_exit_code = managed.process.returncode

                if self._stop_flag:
                    return

                # 失败守门
                if self._last_exit_code != 0 and self._spec.on_failure == "pause":
                    self.logger.warning(
                        "job [%s] paused after exit=%s",
                        self._spec.name, self._last_exit_code,
                    )
                    self._status = JobStatus.PAUSED
                    self._resume_event.clear()
                    await self._resume_event.wait()
                    if self._stop_flag:
                        return

                # 判终
                if self._spec.times > 0 and self._executed >= self._spec.times:
                    self._status = JobStatus.FINISHED
                    return

                # 间隔
                if self._spec.interval > 0:
                    self._status = JobStatus.SLEEPING
                    await self._sleep_interruptible(self._spec.interval)
        finally:
            # runner 落幕 — 终态未定则视为 STOPPED
            if self._status not in (JobStatus.FINISHED, JobStatus.STOPPED):
                self._status = JobStatus.STOPPED
            self._done_event.set()

    async def _spawn_one(self) -> ManagedProcess:
        capture = CaptureSpec(buffer_lines=self._spec.buffer_lines)
        if self._spec.shell_cmd is not None:
            return await self._sp.shell(
                self._spec.shell_cmd,
                name=self._spec.name,
                description=self._spec.description,
                cwd=self._spec.cwd,
                extra_env=self._spec.extra_env or None,
                capture=capture,
            )
        return await self._sp.execute(
            *self._spec.args,
            name=self._spec.name,
            description=self._spec.description,
            cwd=self._spec.cwd,
            extra_env=self._spec.extra_env or None,
            capture=capture,
        )

    async def _sleep_interruptible(self, seconds: float) -> None:
        # stop 通过 _resume_event.set() 提前中断 sleep — 语义: 任何"苏醒信号"都跳出等待
        self._resume_event.clear()
        try:
            await asyncio.wait_for(self._resume_event.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass


class JobSupervisorImpl(JobSupervisor):
    """契约的默认实现. 组合外部注入的 Subprocesses."""

    def __init__(
            self,
            subprocesses: Subprocesses,
            logger: logging.Logger | None = None,
    ):
        self._sp = subprocesses
        self.logger = logger or get_moss_logger()

        self._started = False
        self._stopped = False
        self._jobs: dict[str, JobImpl] = {}
        self._history: list[JobImpl] = []

    def submit(self, spec: JobSpec) -> Job:
        if not self._started:
            raise RuntimeError("JobSupervisor not started — use 'async with'")
        if self._stopped:
            raise RuntimeError("JobSupervisor already stopped")
        if not spec.args and not spec.shell_cmd:
            raise ValueError("JobSpec must specify either args or shell_cmd")
        if spec.args and spec.shell_cmd:
            raise ValueError("JobSpec.args and shell_cmd are mutually exclusive")

        job = JobImpl(spec=spec, subprocesses=self._sp, logger=self.logger)
        self._jobs[job.id] = job
        self.logger.info(
            "submit job [%s] id=%s interval=%s times=%s",
            spec.name, job.id, spec.interval, spec.times,
        )
        job._launch()
        return job

    def jobs(self) -> list[Job]:
        # 活跃优先, 历史其后 (FIFO). 与 Subprocesses.executing()/executed() 同风格.
        return list(self._jobs.values()) + list(self._history)

    def get(self, job_id: str) -> Job | None:
        job = self._jobs.get(job_id)
        if job is not None:
            return job
        for j in self._history:
            if j.id == job_id:
                return j
        return None

    def new(self) -> "JobSupervisorImpl":
        # 复制内部依赖引用, 状态归零 (jobs/history 独立). peer 未启动, owner 自负 async with.
        return JobSupervisorImpl(subprocesses=self._sp, logger=self.logger)

    async def __aenter__(self) -> "JobSupervisorImpl":
        self._started = True
        self.logger.info("JobSupervisor started")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self._stopped = True
        self.logger.info("JobSupervisor stopping — %d jobs", len(self._jobs))
        for job in list(self._jobs.values()):
            try:
                await job.stop(timeout=3.0)
            except Exception:
                self.logger.exception("job [%s] stop failed on shutdown", job.spec.name)
            self._retire(job)
        self.logger.info("JobSupervisor stopped")

    def _retire(self, job: JobImpl) -> None:
        self._jobs.pop(job.id, None)
        self._history.append(job)
        while len(self._history) > _MAX_JOB_HISTORY:
            self._history.pop(0)
