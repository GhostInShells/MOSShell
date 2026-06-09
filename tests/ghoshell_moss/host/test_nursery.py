"""ProcessNursery 单元测试 — 接口契约 + env 组装 + shutdown 行为."""

import asyncio
import os
import signal
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from ghoshell_moss.host.nursery import ProcessNursery, watch_nursery_pipe

pytestmark = pytest.mark.asyncio


# ------------------------------------------------------------------
# fixtures
# ------------------------------------------------------------------

@pytest.fixture
def nursery():
    return ProcessNursery()


async def _spawn_and_get_call_args(nursery, **kwargs):
    """spawn 一次并返回 create_subprocess_exec 的调用参数。"""
    with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_spawn:
        mock_spawn.return_value = MagicMock(pid=12345)
        await nursery.spawn("python", "main.py", **kwargs)
        return mock_spawn.call_args


# ------------------------------------------------------------------
# spawn — env 组装与参数传递
# ------------------------------------------------------------------

class TestSpawnArgs:

    async def test_args_passed_through(self, nursery):
        ca = await _spawn_and_get_call_args(nursery)
        assert ca.args == ("python", "main.py")

    async def test_env_passed_verbatim_when_no_nursery_fd(self, nursery):
        ca = await _spawn_and_get_call_args(nursery, env={"MOSS_WORKSPACE": "/ws"})
        env = ca.kwargs["env"]
        assert env["MOSS_WORKSPACE"] == "/ws"
        assert "MOSS_NURSERY_FD" not in env

    async def test_nursery_fd_injected_into_env(self, nursery):
        ca = await _spawn_and_get_call_args(
            nursery, env={"MOSS_WORKSPACE": "/ws"}, nursery_fd=42,
        )
        env = ca.kwargs["env"]
        assert env["MOSS_NURSERY_FD"] == "42"

    async def test_pipe_read_fd_in_pass_fds(self, nursery):
        ca = await _spawn_and_get_call_args(nursery, env={}, nursery_fd=42)
        assert 42 in ca.kwargs["pass_fds"]

    async def test_no_pass_fds_when_no_nursery_fd(self, nursery):
        ca = await _spawn_and_get_call_args(nursery, env={})
        assert ca.kwargs["pass_fds"] == ()

    async def test_start_new_session_true(self, nursery):
        ca = await _spawn_and_get_call_args(nursery, env={})
        assert ca.kwargs["start_new_session"] is True

    async def test_cwd_passed(self, nursery):
        ca = await _spawn_and_get_call_args(nursery, env={}, cwd="/tmp")
        assert ca.kwargs["cwd"] == "/tmp"

    async def test_caller_env_not_mutated(self, nursery):
        original = {"MOSS_WORKSPACE": "/ws"}
        await _spawn_and_get_call_args(nursery, env=original)
        assert "MOSS_NURSERY_FD" not in original

    async def test_pgid_tracked(self, nursery):
        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_spawn:
            mock_spawn.return_value = MagicMock(pid=12345)
            await nursery.spawn("python", "main.py", env={})
            assert 12345 in nursery._pgids
            assert nursery.child_count == 1


# ------------------------------------------------------------------
# shutdown — 两阶段退出
# ------------------------------------------------------------------

class TestShutdown:

    async def test_shutdown_empty_noop(self, nursery):
        await nursery.shutdown()
        assert nursery.child_count == 0

    async def test_shutdown_sends_sigterm(self, nursery):
        nursery._pgids = {99999}
        with patch("os.killpg") as mock_killpg:
            await nursery.shutdown(timeout=0.01)
            mock_killpg.assert_any_call(99999, signal.SIGTERM)

    async def test_shutdown_sends_sigkill_after_timeout(self, nursery):
        nursery._pgids = {99999}
        with patch("os.killpg") as mock_killpg:
            await nursery.shutdown(timeout=0.01)
            mock_killpg.assert_any_call(99999, signal.SIGKILL)

    async def test_shutdown_clears_pgids(self, nursery):
        nursery._pgids = {99999}
        with patch("os.killpg"):
            await nursery.shutdown(timeout=0.01)
            assert nursery.child_count == 0

    async def test_shutdown_discards_on_process_lookup_error(self, nursery):
        nursery._pgids = {99999}
        with patch("os.killpg", side_effect=ProcessLookupError):
            await nursery.shutdown(timeout=0.01)
        assert nursery.child_count == 0

    async def test_shutdown_sigterm_before_sigkill(self, nursery):
        nursery._pgids = {99999}
        calls = []
        with patch("os.killpg", side_effect=lambda pgid, sig: calls.append(sig)):
            await nursery.shutdown(timeout=0.01)
        assert calls[0] == signal.SIGTERM
        assert calls[1] == signal.SIGKILL


# ------------------------------------------------------------------
# async context manager
# ------------------------------------------------------------------

class TestAsyncContextManager:

    async def test_aenter_returns_self(self, nursery):
        result = await nursery.__aenter__()
        assert result is nursery

    async def test_aexit_calls_shutdown(self, nursery):
        nursery.shutdown = AsyncMock()
        await nursery.__aexit__(None, None, None)
        nursery.shutdown.assert_awaited_once()


# ------------------------------------------------------------------
# watch_nursery_pipe — pipe fencing 子进程侧
# ------------------------------------------------------------------

class TestWatchNurseryPipe:

    async def test_noop_when_env_not_set(self):
        with patch.dict(os.environ, {}, clear=True):
            close = MagicMock()
            await watch_nursery_pipe(close)
            close.assert_not_called()

    async def test_eof_triggers_close(self):
        """pipe read 读到 EOF 时调用 close。"""
        read_fd, write_fd = os.pipe()
        try:
            with patch.dict(os.environ, {"MOSS_NURSERY_FD": str(read_fd)}):
                close = MagicMock()

                async def _close_after_delay():
                    await asyncio.sleep(0.05)
                    os.close(write_fd)

                await asyncio.gather(watch_nursery_pipe(close), _close_after_delay())
                close.assert_called_once()
        finally:
            pass  # fd handled by watchdog

    async def test_closes_fd_on_exit(self):
        """watch_nursery_pipe 退出后 fd 应已关闭。"""
        read_fd, write_fd = os.pipe()
        os.close(write_fd)  # 立即关闭写端 → 下一个 read 就 EOF
        try:
            with patch.dict(os.environ, {"MOSS_NURSERY_FD": str(read_fd)}):
                close = MagicMock()
                await watch_nursery_pipe(close)
            with pytest.raises(OSError):
                os.fstat(read_fd)
        except OSError:
            pass
