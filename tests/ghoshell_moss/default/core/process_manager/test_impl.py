"""ProcessManagerImpl 单测 — 覆盖生命周期、exec/shell、task、background、进程治理."""

import asyncio
import signal
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from ghoshell_moss.core.process_manager._impl import ProcessManagerImpl


@pytest.fixture
def pm_root(tmp_path: Path) -> Path:
    path = tmp_path / "project"
    path.mkdir()
    return path


@pytest.fixture
def pm_output(tmp_path: Path) -> Path:
    path = tmp_path / "output"
    path.mkdir()
    return path


@asynccontextmanager
async def running_pm(root: Path, output: Path):
    pm = ProcessManagerImpl(root=root, output_path=output)
    async with pm:
        yield pm


# ============================================================
# 生命周期 & 守卫
# ============================================================


class TestLifecycle:

    def test_guard_before_enter(self, pm_root, pm_output):
        pm = ProcessManagerImpl(root=pm_root, output_path=pm_output)
        with pytest.raises(RuntimeError, match="not started"):
            pm.cd("subdir")

    @pytest.mark.asyncio
    async def test_guard_after_exit(self, pm_root, pm_output):
        async with running_pm(pm_root, pm_output) as pm:
            pass
        with pytest.raises(RuntimeError, match="already stopped"):
            pm.cd("subdir")

    @pytest.mark.asyncio
    async def test_enter_exit(self, pm_root, pm_output):
        pm = ProcessManagerImpl(root=pm_root, output_path=pm_output)
        assert pm._started is False
        async with pm:
            assert pm._started is True
            assert pm._stopped is False
        assert pm._stopped is True


# ============================================================
# 目录作用域
# ============================================================


class TestDirectory:

    @pytest.mark.asyncio
    async def test_pwd_defaults_to_root(self, pm_root, pm_output):
        async with running_pm(pm_root, pm_output) as pm:
            assert pm.pwd == pm_root.resolve()

    @pytest.mark.asyncio
    async def test_cd_relative(self, pm_root, pm_output):
        async with running_pm(pm_root, pm_output) as pm:
            sub = pm_root / "sub"
            sub.mkdir()
            result = pm.cd("sub")
            assert result == sub.resolve()
            assert pm.pwd == sub.resolve()

    @pytest.mark.asyncio
    async def test_cd_outside_root(self, pm_root, pm_output):
        async with running_pm(pm_root, pm_output) as pm:
            outside = pm_root.parent / "outside"
            outside.mkdir(exist_ok=True)
            with pytest.raises(ValueError, match="outside root"):
                pm.cd(str(outside))

    @pytest.mark.asyncio
    async def test_cd_absolute_inside_root(self, pm_root, pm_output):
        async with running_pm(pm_root, pm_output) as pm:
            sub = pm_root / "sub"
            sub.mkdir()
            result = pm.cd(str(sub))
            assert result == sub.resolve()


# ============================================================
# Layer 1: execute / shell
# ============================================================


class TestExecute:

    @pytest.mark.asyncio
    async def test_execute_echo(self, pm_root, pm_output):
        async with running_pm(pm_root, pm_output) as pm:
            managed = await pm.execute("echo", "hello")
            assert managed.process.pid > 0
            assert managed.meta.name == "echo"
            await managed.process.wait()
            assert managed.process.returncode == 0

    @pytest.mark.asyncio
    async def test_shell_echo(self, pm_root, pm_output):
        async with running_pm(pm_root, pm_output) as pm:
            managed = await pm.shell("echo hello")
            assert managed.process.pid > 0
            await managed.process.wait()
            assert managed.process.returncode == 0

    @pytest.mark.asyncio
    async def test_execute_reclaim(self, pm_root, pm_output):
        async with running_pm(pm_root, pm_output) as pm:
            managed = await pm.execute("echo", "reclaim_test")
            assert len(pm.executing()) >= 1
            assert pm.get_executing(managed.meta.index) is not None
            await managed.process.wait()
            await asyncio.sleep(0.1)
            assert managed.meta.exit_code == 0
            assert pm.get_executing(managed.meta.index) is None
            assert len(pm.executed()) >= 1

    @pytest.mark.asyncio
    async def test_execute_cwd(self, pm_root, pm_output):
        async with running_pm(pm_root, pm_output) as pm:
            sub = pm_root / "workdir"
            sub.mkdir()
            managed = await pm.execute("pwd", cwd=str(sub))
            await managed.process.wait()
            assert managed.process.returncode == 0

    @pytest.mark.asyncio
    async def test_execute_env(self, pm_root, pm_output):
        async with running_pm(pm_root, pm_output) as pm:
            managed = await pm.execute(
                "sh", "-c", "echo $TEST_VAR",
                extra_env={"TEST_VAR": "moss_value"},
            )

    @pytest.mark.asyncio
    async def test_execute_failure(self, pm_root, pm_output):
        async with running_pm(pm_root, pm_output) as pm:
            managed = await pm.execute("sh", "-c", "exit 3")
            await managed.process.wait()
            assert managed.process.returncode == 3

    @pytest.mark.asyncio
    async def test_execute_order_in_executed(self, pm_root, pm_output):
        async with running_pm(pm_root, pm_output) as pm:
            m1 = await pm.execute("echo", "1")
            m2 = await pm.execute("echo", "2")
            await m1.process.wait()
            await m2.process.wait()
            await asyncio.sleep(0.1)
            names = [m.name for m in pm.executed()]
            assert "echo" in names


# ============================================================
# Layer 2: execute_task / shell_task
# ============================================================


class TestExecuteTask:

    @pytest.mark.asyncio
    async def test_stdout_captured(self, pm_root, pm_output):
        async with running_pm(pm_root, pm_output) as pm:
            task = await pm.execute_task("echo", "hello world")
            await task.wait()
            assert "hello world" in task.stdout_buffer()

    @pytest.mark.asyncio
    async def test_file_output(self, pm_root, pm_output):
        async with running_pm(pm_root, pm_output) as pm:
            out = pm_output / "test_out.txt"
            task = await pm.execute_task("echo", "file content", output_file=out)
            await task.wait()
            assert out.exists()
            assert "file content" in out.read_text()

    @pytest.mark.asyncio
    async def test_no_file_when_not_specified(self, pm_root, pm_output):
        async with running_pm(pm_root, pm_output) as pm:
            task = await pm.execute_task("echo", "memory only")
            await task.wait()
            assert task.stdout_file is None
            assert "memory only" in task.stdout_buffer()

    @pytest.mark.asyncio
    async def test_buffer_tail_window(self, pm_root, pm_output):
        async with running_pm(pm_root, pm_output) as pm:
            cmd = "for i in $(seq 1 150); do echo line$i; done"
            task = await pm.execute_task("sh", "-c", cmd, buffer_lines=50)
            await task.wait()
            buf = task.stdout_buffer()
            lines = buf.strip().split("\n")
            assert len(lines) <= 50
            assert "line150" in buf

    @pytest.mark.asyncio
    async def test_buffer_zero(self, pm_root, pm_output):
        async with running_pm(pm_root, pm_output) as pm:
            task = await pm.execute_task("echo", "no buffer", buffer_lines=0)
            await task.wait()
            assert task.stdout_buffer() == ""

    @pytest.mark.asyncio
    async def test_shell_task(self, pm_root, pm_output):
        async with running_pm(pm_root, pm_output) as pm:
            task = await pm.shell_task("echo shell_task_ok")
            await task.wait()
            assert "shell_task_ok" in task.stdout_buffer()

    @pytest.mark.asyncio
    async def test_done_callback(self, pm_root, pm_output):
        async with running_pm(pm_root, pm_output) as pm:
            results: list[str] = []

            def on_done(t):
                results.append(t.stdout_buffer().strip())

            task = await pm.execute_task("echo", "callback_test", callback=on_done)
            await task.wait()
            await asyncio.sleep(0.1)
            assert len(results) == 1
            assert "callback_test" in results[0]

    @pytest.mark.asyncio
    async def test_stderr_captured(self, pm_root, pm_output):
        async with running_pm(pm_root, pm_output) as pm:
            task = await pm.execute_task("sh", "-c", "echo ok; echo err >&2")
            await task.wait()
            assert "ok" in task.stdout_buffer()
            assert "err" in task.stderr_buffer()

    @pytest.mark.asyncio
    async def test_large_output(self, pm_root, pm_output):
        async with running_pm(pm_root, pm_output) as pm:
            cmd = "for i in $(seq 1 1000); do echo line$i; done"
            task = await pm.execute_task("sh", "-c", cmd, buffer_lines=200)
            await task.wait()
            assert task.return_code == 0
            buf = task.stdout_buffer()
            assert len(buf.strip().split("\n")) <= 200


# ============================================================
# Layer 3: BackgroundTask
# ============================================================


class TestBackgroundTask:

    @pytest.mark.asyncio
    async def test_once(self, pm_root, pm_output):
        async with running_pm(pm_root, pm_output) as pm:
            task = await pm.execute_task(
                "echo", "once_test",
                background_run=("once",),
            )
            assert len(pm.background_tasks()) == 1
            bg = pm.background_tasks()[0]
            assert bg.type == "once"
            await task.wait()
            await asyncio.sleep(0.1)
            assert bg.executed == 1
            assert bg.is_running is False
            await pm.refresh_background()
            assert bg.executed == 1

    @pytest.mark.asyncio
    async def test_loop_fixed(self, pm_root, pm_output):
        async with running_pm(pm_root, pm_output) as pm:
            task = await pm.execute_task(
                "echo", "loop_test",
                background_run=("loop", 3),
            )
            bg = pm.background_tasks()[0]
            assert bg.type == "loop"
            assert bg.loop == 3
            await task.wait()
            await asyncio.sleep(0.1)
            assert bg.executed == 1
            await pm.refresh_background()
            await asyncio.sleep(0.1)
            assert bg.executed >= 1

    @pytest.mark.asyncio
    async def test_on_prompt(self, pm_root, pm_output):
        async with running_pm(pm_root, pm_output) as pm:
            task = await pm.execute_task(
                "sleep", "10",
                background_run=("on_prompt",),
            )
            bg = pm.background_tasks()[0]
            assert bg.is_running
            await pm.refresh_background()
            await asyncio.sleep(0.1)
            assert bg.executed >= 1

    @pytest.mark.asyncio
    async def test_exit_code_pauses_loop(self, pm_root, pm_output):
        async with running_pm(pm_root, pm_output) as pm:
            task = await pm.execute_task(
                "sh", "-c", "exit 42",
                background_run=("loop", 5),
            )
            bg = pm.background_tasks()[0]
            await task.wait()
            await asyncio.sleep(0.1)
            assert bg.error != ""
            assert "42" in bg.error
            await pm.refresh_background()
            assert bg.error == ""

    @pytest.mark.asyncio
    async def test_stop(self, pm_root, pm_output):
        async with running_pm(pm_root, pm_output) as pm:
            task = await pm.execute_task(
                "sleep", "30",
                background_run=("loop",),
            )
            bg = pm.background_tasks()[0]
            assert bg.is_running
            ok = await pm.stop_background_task(bg.id)
            assert ok is True
            assert bg.is_running is False
            assert len(pm.background_tasks()) == 0
            assert len(pm.stopped_background_tasks()) == 1

    @pytest.mark.asyncio
    async def test_stop_unknown_id(self, pm_root, pm_output):
        async with running_pm(pm_root, pm_output) as pm:
            ok = await pm.stop_background_task("nonexistent")
            assert ok is False

    @pytest.mark.asyncio
    async def test_buffer_reuse(self, pm_root, pm_output):
        async with running_pm(pm_root, pm_output) as pm:
            task = await pm.execute_task(
                "echo", "iteration_1",
                background_run=("loop", 2),
            )
            bg = pm.background_tasks()[0]
            await task.wait()
            await asyncio.sleep(0.1)
            first = bg.last_stdout()
            assert "iteration_1" in first
            await pm.refresh_background()
            if bg.last:
                await bg.last.wait()
            await asyncio.sleep(0.1)
            assert bg.executed >= 2


# ============================================================
# 进程治理
# ============================================================


class TestProcessGovernance:

    @pytest.mark.asyncio
    async def test_kill(self, pm_root, pm_output):
        async with running_pm(pm_root, pm_output) as pm:
            managed = await pm.execute("sleep", "30")
            assert managed.process.returncode is None
            err = pm.kill(managed.process.pid)
            assert err is None
            await managed.process.wait()
            assert managed.process.returncode == -signal.SIGKILL

    @pytest.mark.asyncio
    async def test_kill_nonexistent(self, pm_root, pm_output):
        async with running_pm(pm_root, pm_output) as pm:
            err = pm.kill(99999)
            assert err is None

    @pytest.mark.asyncio
    async def test_killpg(self, pm_root, pm_output):
        async with running_pm(pm_root, pm_output) as pm:
            managed = await pm.execute("sleep", "30")
            err = pm.killpg(managed.process.pid, signal.SIGKILL)
            assert err is None
            await managed.process.wait()
            assert managed.process.returncode == -signal.SIGKILL

    @pytest.mark.asyncio
    async def test_shutdown_kills_all(self, pm_root, pm_output):
        pm = ProcessManagerImpl(root=pm_root, output_path=pm_output)
        async with pm:
            m1 = await pm.execute("sleep", "30")
            m2 = await pm.execute("sleep", "30")
            assert len(pm.executing()) >= 2
        assert pm._stopped is True
        assert m1.process.returncode is not None
        assert m2.process.returncode is not None

    @pytest.mark.asyncio
    async def test_shutdown_clears_background_tasks(self, pm_root, pm_output):
        pm = ProcessManagerImpl(root=pm_root, output_path=pm_output)
        async with pm:
            await pm.execute_task(
                "sleep", "30",
                background_run=("loop",),
            )
            assert len(pm.background_tasks()) == 1
        assert len(pm.background_tasks()) == 0


# ============================================================
# 边界情况
# ============================================================


class TestEdgeCases:

    @pytest.mark.asyncio
    async def test_concurrent_execute(self, pm_root, pm_output):
        async with running_pm(pm_root, pm_output) as pm:
            results = await asyncio.gather(
                pm.execute("echo", "a"),
                pm.execute("echo", "b"),
                pm.execute("echo", "c"),
            )
            assert len(results) == 3
            for managed in results:
                await managed.process.wait()
                assert managed.process.returncode == 0

    @pytest.mark.asyncio
    async def test_concurrent_tasks(self, pm_root, pm_output):
        async with running_pm(pm_root, pm_output) as pm:
            t1 = await pm.execute_task("echo", "task_one")
            t2 = await pm.execute_task("echo", "task_two")
            await asyncio.gather(t1.wait(), t2.wait())
            assert "task_one" in t1.stdout_buffer()
            assert "task_two" in t2.stdout_buffer()
            assert "task_one" not in t2.stdout_buffer()

    @pytest.mark.asyncio
    async def test_cd_then_execute(self, pm_root, pm_output):
        async with running_pm(pm_root, pm_output) as pm:
            sub = pm_root / "subdir"
            sub.mkdir()
            pm.cd("subdir")
            managed = await pm.execute("pwd")
            await managed.process.wait()
            assert managed.process.returncode == 0
