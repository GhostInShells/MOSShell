"""SubprocessesImpl 单测 — 覆盖生命周期、exec/shell、capture、stop、on_exit、并发."""

import asyncio
import signal
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from ghoshell_moss.contracts.subprocesses import CaptureSpec
from ghoshell_moss.core.subprocesses._impl import SubprocessesImpl


@pytest.fixture
def sp_cwd(tmp_path: Path) -> Path:
    path = tmp_path / "cwd"
    path.mkdir()
    return path


@pytest.fixture
def sp_output(tmp_path: Path) -> Path:
    path = tmp_path / "output"
    path.mkdir()
    return path


@asynccontextmanager
async def running_sp(cwd: Path):
    sp = SubprocessesImpl(cwd=cwd)
    async with sp:
        yield sp


# ============================================================
# 生命周期 & 守卫
# ============================================================


class TestLifecycle:

    @pytest.mark.asyncio
    async def test_spawn_without_enter_lazy_start(self, sp_cwd, sp_output):
        # 惰性启动: 无需 async with, 首次 spawn 自动进入启动态.
        sp = SubprocessesImpl(cwd=sp_cwd)
        proc = await sp.execute("true")
        assert await proc.process.wait() == 0

    @pytest.mark.asyncio
    async def test_spawn_after_exit(self, sp_cwd, sp_output):
        async with running_sp(sp_cwd) as sp:
            pass
        with pytest.raises(RuntimeError, match="already stopped"):
            await sp.execute("true")

    @pytest.mark.asyncio
    async def test_enter_exit(self, sp_cwd, sp_output):
        sp = SubprocessesImpl(cwd=sp_cwd)
        assert sp.is_running() is False
        async with sp:
            assert sp.is_running() is True
        assert sp.is_running() is False


# ============================================================
# Layer 1: execute / shell (裸)
# ============================================================


class TestExecute:

    @pytest.mark.asyncio
    async def test_execute_echo(self, sp_cwd, sp_output):
        async with running_sp(sp_cwd) as sp:
            managed = await sp.execute("echo", "hello")
            assert managed.process.pid > 0
            assert managed.meta.name == "echo"
            assert managed.meta.pgid == managed.process.pid  # setsid → pgid=pid
            await managed.process.wait()
            assert managed.process.returncode == 0

    @pytest.mark.asyncio
    async def test_shell_echo(self, sp_cwd, sp_output):
        async with running_sp(sp_cwd) as sp:
            managed = await sp.shell("echo hello")
            assert managed.process.pid > 0
            await managed.process.wait()
            assert managed.process.returncode == 0

    @pytest.mark.asyncio
    async def test_execute_reclaim(self, sp_cwd, sp_output):
        async with running_sp(sp_cwd) as sp:
            managed = await sp.execute("echo", "reclaim_test")
            assert len(sp.executing()) >= 1
            assert sp.get(managed.meta.index) is not None
            await managed.process.wait()
            await asyncio.sleep(0.1)
            assert managed.meta.exit_code == 0
            assert sp.get(managed.meta.index) is None
            assert len(sp.executed()) >= 1

    @pytest.mark.asyncio
    async def test_execute_cwd_explicit(self, sp_cwd, sp_output):
        async with running_sp(sp_cwd) as sp:
            sub = sp_cwd / "workdir"
            sub.mkdir()
            managed = await sp.execute("pwd", cwd=str(sub), capture=CaptureSpec())
            await managed.process.wait()
            await managed.output.wait_drained()
            assert str(sub.resolve()) in managed.output.stdout()

    @pytest.mark.asyncio
    async def test_execute_env(self, sp_cwd, sp_output):
        async with running_sp(sp_cwd) as sp:
            managed = await sp.execute(
                "sh", "-c", "echo $TEST_VAR",
                extra_env={"TEST_VAR": "moss_value"},
                capture=CaptureSpec(),
            )
            await managed.process.wait()
            await managed.output.wait_drained()
            assert "moss_value" in managed.output.stdout()

    @pytest.mark.asyncio
    async def test_execute_failure(self, sp_cwd, sp_output):
        async with running_sp(sp_cwd) as sp:
            managed = await sp.execute("sh", "-c", "exit 3")
            await managed.process.wait()
            assert managed.process.returncode == 3

    @pytest.mark.asyncio
    async def test_execute_no_args_rejected(self, sp_cwd, sp_output):
        async with running_sp(sp_cwd) as sp:
            with pytest.raises(ValueError):
                await sp.execute()


# ============================================================
# CaptureSpec — 输出捕获
# ============================================================


class TestCapture:

    @pytest.mark.asyncio
    async def test_stdout_captured(self, sp_cwd, sp_output):
        async with running_sp(sp_cwd) as sp:
            managed = await sp.execute("echo", "hello world", capture=CaptureSpec())
            await managed.process.wait()
            await managed.output.wait_drained()
            assert "hello world" in managed.output.stdout()

    @pytest.mark.asyncio
    async def test_stderr_captured(self, sp_cwd, sp_output):
        async with running_sp(sp_cwd) as sp:
            managed = await sp.execute(
                "sh", "-c", "echo ok; echo err >&2",
                capture=CaptureSpec(),
            )
            await managed.process.wait()
            await managed.output.wait_drained()
            assert "ok" in managed.output.stdout()
            assert "err" in managed.output.stderr()

    @pytest.mark.asyncio
    async def test_file_output_explicit(self, sp_cwd, sp_output):
        async with running_sp(sp_cwd) as sp:
            out = sp_output / "custom" / "test_out.txt"
            managed = await sp.execute(
                "echo", "file content",
                capture=CaptureSpec(stdout_file=out),
            )
            await managed.process.wait()
            await managed.output.wait_drained()
            assert out.exists()
            assert "file content" in out.read_text()

    @pytest.mark.asyncio
    async def test_file_output_memory_only(self, sp_cwd):
        """CaptureSpec without explicit file paths is memory-only, no disk spill."""
        sp = SubprocessesImpl(cwd=sp_cwd)
        async with sp:
            managed = await sp.execute("echo", "auto", capture=CaptureSpec())
            await managed.process.wait()
            await managed.output.wait_drained()
            assert managed.output.stdout_file is None
            assert "auto" in managed.output.stdout()

    @pytest.mark.asyncio
    async def test_buffer_tail_window(self, sp_cwd, sp_output):
        async with running_sp(sp_cwd) as sp:
            managed = await sp.execute(
                "sh", "-c", "for i in $(seq 1 150); do echo line$i; done",
                capture=CaptureSpec(buffer_lines=50),
            )
            await managed.process.wait()
            await managed.output.wait_drained()
            buf = managed.output.stdout()
            lines = buf.strip().split("\n")
            assert len(lines) <= 50
            assert "line150" in buf

    @pytest.mark.asyncio
    async def test_buffer_zero(self, sp_cwd):
        sp = SubprocessesImpl(cwd=sp_cwd)
        async with sp:
            managed = await sp.execute(
                "echo", "no_buffer",
                capture=CaptureSpec(buffer_lines=0),
            )
            await managed.process.wait()
            await managed.output.wait_drained()
            assert managed.output.stdout() == ""
            assert managed.output.stdout_file is None

    @pytest.mark.asyncio
    async def test_capture_conflicts_with_manual_stdout(self, sp_cwd, sp_output):
        async with running_sp(sp_cwd) as sp:
            with pytest.raises(ValueError, match="mutually exclusive"):
                await sp.execute(
                    "true",
                    capture=CaptureSpec(),
                    stdout=asyncio.subprocess.PIPE,
                )


# ============================================================
# ManagedProcess.stop
# ============================================================


class TestStop:

    @pytest.mark.asyncio
    async def test_stop_running_process(self, sp_cwd, sp_output):
        async with running_sp(sp_cwd) as sp:
            managed = await sp.execute("sleep", "30")
            assert managed.process.returncode is None
            await managed.stop(timeout=2.0)
            assert managed.process.returncode is not None

    @pytest.mark.asyncio
    async def test_stop_already_exited(self, sp_cwd, sp_output):
        async with running_sp(sp_cwd) as sp:
            managed = await sp.execute("true")
            await managed.process.wait()
            # 幂等: 已退出再 stop 不抛
            await managed.stop(timeout=1.0)

    @pytest.mark.asyncio
    async def test_stop_forces_kill_after_timeout(self, sp_cwd, sp_output):
        """SIGINT 无效的进程 → grace 超时 → SIGKILL."""
        async with running_sp(sp_cwd) as sp:
            # Python 子进程显式 ignore INT/TERM, 只有 KILL 能杀.
            # 打印 ready 后由测试观察 stdout 确认 handler 已装, 避免竞态.
            managed = await sp.execute(
                "python3", "-u", "-c",
                "import signal, time; "
                "signal.signal(signal.SIGINT, signal.SIG_IGN); "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "print('ready', flush=True); "
                "time.sleep(30)",
                capture=CaptureSpec(),
            )
            # 等到 handler 已装 (最多 3s)
            for _ in range(60):
                await asyncio.sleep(0.05)
                if "ready" in managed.output.stdout():
                    break
            else:
                pytest.fail("child did not signal ready in time")
            await managed.stop(timeout=0.5)
            assert managed.process.returncode is not None
            assert managed.process.returncode == -signal.SIGKILL


# ============================================================
# 信号
# ============================================================


class TestSignals:

    @pytest.mark.asyncio
    async def test_kill(self, sp_cwd, sp_output):
        async with running_sp(sp_cwd) as sp:
            managed = await sp.execute("sleep", "30")
            err = sp.kill(managed.process.pid)
            assert err is None
            await managed.process.wait()
            assert managed.process.returncode == -signal.SIGKILL

    @pytest.mark.asyncio
    async def test_kill_nonexistent(self, sp_cwd, sp_output):
        async with running_sp(sp_cwd) as sp:
            err = sp.kill(99999)
            assert err is None

    @pytest.mark.asyncio
    async def test_killpg(self, sp_cwd, sp_output):
        async with running_sp(sp_cwd) as sp:
            managed = await sp.execute("sleep", "30")
            err = sp.killpg(managed.process.pid, signal.SIGKILL)
            assert err is None
            await managed.process.wait()
            assert managed.process.returncode == -signal.SIGKILL


# ============================================================
# owner 关停 — "子进程不比 owner 活得久"
# ============================================================


class TestOwnerShutdown:

    @pytest.mark.asyncio
    async def test_shutdown_kills_all(self, sp_cwd, sp_output):
        sp = SubprocessesImpl(cwd=sp_cwd)
        async with sp:
            m1 = await sp.execute("sleep", "30")
            m2 = await sp.execute("sleep", "30")
            assert len(sp.executing()) >= 2
        assert sp.is_running() is False
        assert m1.process.returncode is not None
        assert m2.process.returncode is not None


# ============================================================
# on_exit / add_done_callback
# ============================================================


class TestOnExit:

    @pytest.mark.asyncio
    async def test_on_exit_param_fires(self, sp_cwd, sp_output):
        async with running_sp(sp_cwd) as sp:
            received: list = []
            managed = await sp.execute("true", on_exit=lambda meta: received.append(meta))
            await managed.process.wait()
            await asyncio.sleep(0.1)
            assert len(received) == 1
            assert received[0] is managed.meta
            assert received[0].exit_code == 0

    @pytest.mark.asyncio
    async def test_add_done_callback_before_exit(self, sp_cwd, sp_output):
        async with running_sp(sp_cwd) as sp:
            received: list = []
            managed = await sp.execute("true")
            managed.add_done_callback(lambda meta: received.append(meta.exit_code))
            await managed.process.wait()
            await asyncio.sleep(0.1)
            assert received == [0]

    @pytest.mark.asyncio
    async def test_add_done_callback_after_exit_fires_immediately(self, sp_cwd, sp_output):
        async with running_sp(sp_cwd) as sp:
            managed = await sp.execute("true")
            await managed.process.wait()
            await asyncio.sleep(0.1)

            received: list = []
            managed.add_done_callback(lambda meta: received.append(meta.exit_code))
            assert received == [0]

    @pytest.mark.asyncio
    async def test_multiple_callbacks_all_fire(self, sp_cwd, sp_output):
        async with running_sp(sp_cwd) as sp:
            received: list = []
            managed = await sp.execute("true")
            managed.add_done_callback(lambda meta: received.append("a"))
            managed.add_done_callback(lambda meta: received.append("b"))
            managed.add_done_callback(lambda meta: received.append("c"))
            await managed.process.wait()
            await asyncio.sleep(0.1)
            assert received == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_callback_exception_isolated(self, sp_cwd, sp_output):
        async with running_sp(sp_cwd) as sp:
            received: list = []

            def bad(meta):
                raise ValueError("boom")

            managed = await sp.execute("true")
            managed.add_done_callback(bad)
            managed.add_done_callback(lambda meta: received.append("ok"))
            await managed.process.wait()
            await asyncio.sleep(0.1)
            assert received == ["ok"]
            assert managed.meta in sp.executed()


# ============================================================
# 并发
# ============================================================


class TestConcurrency:

    @pytest.mark.asyncio
    async def test_spawn_10_concurrent_sleep(self, sp_cwd, sp_output):
        sp = SubprocessesImpl(cwd=sp_cwd)
        async with sp:
            procs = await asyncio.gather(*(
                sp.execute("sleep", str(30 + i)) for i in range(10)
            ))
            assert len(sp.executing()) == 10
            for p in procs:
                assert p.process.pid > 0
                assert p.process.returncode is None
        assert sp.is_running() is False
        for p in procs:
            assert p.process.returncode is not None

    @pytest.mark.asyncio
    async def test_spawn_10_concurrent_captures(self, sp_cwd, sp_output):
        async with running_sp(sp_cwd) as sp:
            procs = await asyncio.gather(*(
                sp.execute("echo", f"task_{i}", capture=CaptureSpec())
                for i in range(10)
            ))
            for p in procs:
                await p.process.wait()
                await p.output.wait_drained()
            for i, p in enumerate(procs):
                assert f"task_{i}" in p.output.stdout()
                for j in range(10):
                    if j != i:
                        assert f"task_{j}" not in p.output.stdout()
