"""JobSupervisorImpl 单测 — 覆盖提交、调度、状态、stop/resume、快照."""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from ghoshell_moss.contracts.job_supervisor import JobSpec, JobStatus
from ghoshell_moss.core.job_supervisor._impl import JobSupervisorImpl
from ghoshell_moss.core.subprocesses._impl import SubprocessesImpl


@pytest.fixture
def tmp_cwd(tmp_path: Path) -> Path:
    path = tmp_path / "cwd"
    path.mkdir()
    return path


@pytest.fixture
def tmp_output(tmp_path: Path) -> Path:
    path = tmp_path / "output"
    path.mkdir()
    return path


@asynccontextmanager
async def running_jobs(cwd: Path):
    sp = SubprocessesImpl(cwd=cwd)
    async with sp:
        jobs = JobSupervisorImpl(subprocesses=sp)
        async with jobs:
            yield jobs


# ============================================================
# 生命周期 & 守卫
# ============================================================


class TestLifecycle:

    @pytest.mark.asyncio
    async def test_submit_before_enter(self, tmp_cwd, tmp_output):
        sp = SubprocessesImpl(cwd=tmp_cwd)
        async with sp:
            jobs = JobSupervisorImpl(subprocesses=sp)
            with pytest.raises(RuntimeError, match="not started"):
                jobs.submit(JobSpec(name="test", args=("true",)))

    @pytest.mark.asyncio
    async def test_submit_after_exit(self, tmp_cwd, tmp_output):
        async with running_jobs(tmp_cwd) as jobs:
            pass
        with pytest.raises(RuntimeError, match="already stopped"):
            jobs.submit(JobSpec(name="test", args=("true",)))

    @pytest.mark.asyncio
    async def test_shutdown_stops_all_jobs(self, tmp_cwd, tmp_output):
        async with running_jobs(tmp_cwd) as jobs:
            j1 = jobs.submit(JobSpec(name="j1", args=("sleep", "30")))
            j2 = jobs.submit(JobSpec(name="j2", args=("sleep", "30")))
            await asyncio.sleep(0.2)
            assert len(jobs.jobs()) == 2
        snap1 = j1.snapshot()
        snap2 = j2.snapshot()
        assert snap1.status == JobStatus.STOPPED
        assert snap2.status == JobStatus.STOPPED


# ============================================================
# 提交与验证
# ============================================================


class TestSubmit:

    @pytest.mark.asyncio
    async def test_submit_exec_mode(self, tmp_cwd, tmp_output):
        async with running_jobs(tmp_cwd) as jobs:
            job = jobs.submit(JobSpec(name="echo_test", args=("echo", "hello")))
            assert job.id is not None
            assert job.spec.name == "echo_test"

    @pytest.mark.asyncio
    async def test_submit_shell_mode(self, tmp_cwd, tmp_output):
        async with running_jobs(tmp_cwd) as jobs:
            job = jobs.submit(JobSpec(name="shell_test", shell_cmd="echo hello"))
            assert job.id is not None

    @pytest.mark.asyncio
    async def test_submit_rejects_no_command(self, tmp_cwd, tmp_output):
        async with running_jobs(tmp_cwd) as jobs:
            with pytest.raises(ValueError):
                jobs.submit(JobSpec(name="bad"))

    @pytest.mark.asyncio
    async def test_submit_rejects_both_args_and_shell(self, tmp_cwd, tmp_output):
        async with running_jobs(tmp_cwd) as jobs:
            with pytest.raises(ValueError):
                jobs.submit(JobSpec(name="bad", args=("true",), shell_cmd="true"))


# ============================================================
# 执行与状态转换
# ============================================================


class TestExecution:

    @pytest.mark.asyncio
    async def test_single_execution_completes(self, tmp_cwd, tmp_output):
        async with running_jobs(tmp_cwd) as jobs:
            job = jobs.submit(JobSpec(name="once", args=("echo", "done"), times=1))
            snap = await job.wait()
            assert snap.status == JobStatus.FINISHED
            assert snap.executed == 1
            assert snap.last_exit_code == 0

    @pytest.mark.asyncio
    async def test_multiple_times(self, tmp_cwd, tmp_output):
        async with running_jobs(tmp_cwd) as jobs:
            job = jobs.submit(JobSpec(
                name="multi",
                args=("echo", "tick"),
                times=3,
                interval=0.1,
            ))
            snap = await job.wait()
            assert snap.status == JobStatus.FINISHED
            assert snap.executed == 3

    @pytest.mark.asyncio
    async def test_infinite_loop_stopped_by_supervisor(self, tmp_cwd, tmp_output):
        async with running_jobs(tmp_cwd) as jobs:
            job = jobs.submit(JobSpec(
                name="infinite",
                args=("echo", "loop"),
                times=0,
                interval=0.1,
            ))
            await asyncio.sleep(0.5)
            snap = job.snapshot()
            assert snap.executed >= 2
            assert snap.status in (JobStatus.RUNNING, JobStatus.SLEEPING)
        # shutdown 停了它
        snap = job.snapshot()
        assert snap.status == JobStatus.STOPPED

    @pytest.mark.asyncio
    async def test_interval_timing(self, tmp_cwd, tmp_output):
        """interval 控制轮间延迟."""
        async with running_jobs(tmp_cwd) as jobs:
            job = jobs.submit(JobSpec(
                name="timed",
                args=("true",),
                times=2,
                interval=0.3,
            ))
            await asyncio.sleep(0.1)
            snap = job.snapshot()
            assert snap.executed == 1
            await asyncio.sleep(0.3)
            snap = job.snapshot()
            assert snap.executed == 2


# ============================================================
# 快照
# ============================================================


class TestSnapshot:

    @pytest.mark.asyncio
    async def test_snapshot_captures_stdout(self, tmp_cwd, tmp_output):
        async with running_jobs(tmp_cwd) as jobs:
            job = jobs.submit(JobSpec(
                name="output_test",
                args=("echo", "captured"),
                times=1,
            ))
            await job.wait()
            snap = job.snapshot()
            assert "captured" in snap.stdout_tail

    @pytest.mark.asyncio
    async def test_snapshot_captures_stderr(self, tmp_cwd, tmp_output):
        async with running_jobs(tmp_cwd) as jobs:
            job = jobs.submit(JobSpec(
                name="err_test",
                shell_cmd="echo err_msg >&2",
                times=1,
            ))
            await job.wait()
            snap = job.snapshot()
            assert "err_msg" in snap.stderr_tail

    @pytest.mark.asyncio
    async def test_snapshot_buffer_lines(self, tmp_cwd, tmp_output):
        async with running_jobs(tmp_cwd) as jobs:
            job = jobs.submit(JobSpec(
                name="buffer_test",
                shell_cmd="for i in $(seq 1 200); do echo line$i; done",
                times=1,
                buffer_lines=50,
            ))
            await job.wait()
            snap = job.snapshot()
            lines = snap.stdout_tail.strip().split("\n")
            assert len(lines) <= 50
            assert "line200" in snap.stdout_tail


# ============================================================
# stop / resume
# ============================================================


class TestStopResume:

    @pytest.mark.asyncio
    async def test_stop_terminates_job(self, tmp_cwd, tmp_output):
        async with running_jobs(tmp_cwd) as jobs:
            job = jobs.submit(JobSpec(
                name="stoppable",
                args=("sleep", "30"),
                times=0,
            ))
            await asyncio.sleep(0.2)
            await job.stop(timeout=1.0)
            snap = job.snapshot()
            assert snap.status == JobStatus.STOPPED

    @pytest.mark.asyncio
    async def test_stop_kills_running_iteration(self, tmp_cwd, tmp_output):
        async with running_jobs(tmp_cwd) as jobs:
            job = jobs.submit(JobSpec(
                name="long_running",
                args=("sleep", "30"),
            ))
            await asyncio.sleep(0.2)
            snap = job.snapshot()
            assert snap.status == JobStatus.RUNNING
            await job.stop(timeout=1.0)
            snap = job.snapshot()
            assert snap.status == JobStatus.STOPPED

    @pytest.mark.asyncio
    async def test_stop_idempotent(self, tmp_cwd, tmp_output):
        async with running_jobs(tmp_cwd) as jobs:
            job = jobs.submit(JobSpec(name="once", args=("true",), times=1))
            await job.wait()
            # 已 FINISHED, 再 stop 不抛
            await job.stop()
            snap = job.snapshot()
            assert snap.status == JobStatus.FINISHED

    @pytest.mark.asyncio
    async def test_on_failure_pause(self, tmp_cwd, tmp_output):
        async with running_jobs(tmp_cwd) as jobs:
            job = jobs.submit(JobSpec(
                name="fail_once",
                shell_cmd="exit 1",
                times=3,
                interval=0.1,
                on_failure="pause",
            ))
            await asyncio.sleep(0.3)
            snap = job.snapshot()
            assert snap.status == JobStatus.PAUSED
            assert snap.executed == 1
            assert snap.last_exit_code == 1

    @pytest.mark.asyncio
    async def test_resume_from_pause(self, tmp_cwd, tmp_output):
        async with running_jobs(tmp_cwd) as jobs:
            job = jobs.submit(JobSpec(
                name="resume_test",
                shell_cmd="exit 1",
                times=2,
                on_failure="pause",
            ))
            await asyncio.sleep(0.2)
            snap = job.snapshot()
            assert snap.status == JobStatus.PAUSED
            assert snap.executed == 1
            job.resume()
            await asyncio.sleep(0.2)
            snap = job.snapshot()
            # resume 后又跑一轮 exit 1, 又暂停
            assert snap.status == JobStatus.PAUSED
            assert snap.executed == 2

    @pytest.mark.asyncio
    async def test_on_failure_continue(self, tmp_cwd, tmp_output):
        async with running_jobs(tmp_cwd) as jobs:
            job = jobs.submit(JobSpec(
                name="continue_on_fail",
                shell_cmd="exit 42",
                times=3,
                interval=0.05,
                on_failure="continue",
            ))
            snap = await job.wait()
            assert snap.status == JobStatus.FINISHED
            assert snap.executed == 3
            assert snap.last_exit_code == 42


# ============================================================
# 查询
# ============================================================


class TestQuery:

    @pytest.mark.asyncio
    async def test_jobs_list_active_and_history(self, tmp_cwd, tmp_output):
        async with running_jobs(tmp_cwd) as jobs:
            j1 = jobs.submit(JobSpec(name="active", args=("sleep", "30")))
            j2 = jobs.submit(JobSpec(name="done", args=("true",), times=1))
            await j2.wait()
            await asyncio.sleep(0.1)
            all_jobs = jobs.jobs()
            # active 在前, history 在后
            assert len(all_jobs) >= 2
            assert j1 in all_jobs
            assert j2 in all_jobs

    @pytest.mark.asyncio
    async def test_get_by_id(self, tmp_cwd, tmp_output):
        async with running_jobs(tmp_cwd) as jobs:
            job = jobs.submit(JobSpec(name="findme", args=("true",)))
            found = jobs.get(job.id)
            assert found is job

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, tmp_cwd, tmp_output):
        async with running_jobs(tmp_cwd) as jobs:
            found = jobs.get("nonexistent-id")
            assert found is None


# ============================================================
# 并发
# ============================================================


class TestConcurrency:

    @pytest.mark.asyncio
    async def test_multiple_jobs_run_concurrently(self, tmp_cwd, tmp_output):
        async with running_jobs(tmp_cwd) as jobs:
            j1 = jobs.submit(JobSpec(name="job1", args=("sleep", "0.3"), times=1))
            j2 = jobs.submit(JobSpec(name="job2", args=("sleep", "0.3"), times=1))
            j3 = jobs.submit(JobSpec(name="job3", args=("sleep", "0.3"), times=1))
            snaps = await asyncio.gather(j1.wait(), j2.wait(), j3.wait())
            for snap in snaps:
                assert snap.status == JobStatus.FINISHED


# ============================================================
# .new() — IoC 复制姿态
# ============================================================


class TestNew:

    @pytest.mark.asyncio
    async def test_new_shares_subprocesses(self, tmp_cwd, tmp_output):
        async with running_jobs(tmp_cwd) as root:
            peer = root.new()
            assert peer._sp is root._sp

    @pytest.mark.asyncio
    async def test_new_state_isolated(self, tmp_cwd, tmp_output):
        async with running_jobs(tmp_cwd) as root:
            peer = root.new()
            async with peer:
                j_root = root.submit(JobSpec(name="on-root", args=("true",), times=1))
                j_peer = peer.submit(JobSpec(name="on-peer", args=("true",), times=1))
                await asyncio.gather(j_root.wait(), j_peer.wait())
                assert j_root in root.jobs()
                assert j_peer not in root.jobs()
                assert j_peer in peer.jobs()
                assert j_root not in peer.jobs()

    @pytest.mark.asyncio
    async def test_new_before_root_entered(self, tmp_cwd, tmp_output):
        # 根实例即使未 async with, .new() 也能派生可启用 peer.
        from ghoshell_moss.core.subprocesses._impl import SubprocessesImpl
        sp = SubprocessesImpl(cwd=tmp_cwd)
        async with sp:
            root = JobSupervisorImpl(subprocesses=sp)
            peer = root.new()
            async with peer:
                job = peer.submit(JobSpec(name="peer_only", args=("true",), times=1))
                snap = await job.wait()
                assert snap.status == JobStatus.FINISHED

    @pytest.mark.asyncio
    async def test_peer_shutdown_leaves_root_alive(self, tmp_cwd, tmp_output):
        async with running_jobs(tmp_cwd) as root:
            peer = root.new()
            async with peer:
                j_peer = peer.submit(JobSpec(name="peer", args=("true",), times=1))
                await j_peer.wait()
            # peer 已关, root 仍可 submit
            j_root = root.submit(JobSpec(name="root", args=("true",), times=1))
            snap = await j_root.wait()
            assert snap.status == JobStatus.FINISHED

    @pytest.mark.asyncio
    async def test_new_produces_new_instance(self, tmp_cwd, tmp_output):
        async with running_jobs(tmp_cwd) as root:
            peer1 = root.new()
            peer2 = root.new()
            assert peer1 is not peer2
            assert peer1 is not root
            assert peer2 is not root
