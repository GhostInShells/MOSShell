"""ProcessNursery 集成测试 — 真实子进程 spawn / pipe fencing / SIGTERM / SIGKILL.

Usage:
    pytest scripts/nursery/test_integration.py -v
    python scripts/nursery/test_integration.py
"""

import asyncio
import os
import signal
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ghoshell_moss.host.nursery import ProcessNursery

pytestmark = pytest.mark.asyncio

CHILD = str(Path(__file__).resolve().parent / "test_child.py")


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------

async def _wait_alive(pid: int, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            await asyncio.sleep(0.05)
    return False


async def _wait_dead(pid: int, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
            await asyncio.sleep(0.05)
        except OSError:
            return True
    return False


def _spawn(nursery, *args, **kw):
    return nursery.spawn(sys.executable, CHILD, *args, **kw)


# ------------------------------------------------------------------
# 进程组隔离
# ------------------------------------------------------------------

async def test_child_has_different_pgid(tmp_path):
    pgid_file = tmp_path / "pgid.txt"
    nursery = ProcessNursery()
    proc = await _spawn(nursery, "pgid", str(pgid_file)
)
    try:
        await asyncio.wait_for(proc.wait(), timeout=3.0)
        child_pgid = int(pgid_file.read_text().strip())
        assert child_pgid != os.getpgrp()
    finally:
        await nursery.shutdown(timeout=1.0)


# ------------------------------------------------------------------
# pipe fencing
# ------------------------------------------------------------------

async def test_pipe_fencing_eof_detection():
    read_fd, write_fd = os.pipe()
    nursery = ProcessNursery()
    proc = await _spawn(nursery, "watch_pipe"
, nursery_fd=read_fd)
    os.close(read_fd)

    try:
        assert await _wait_alive(proc.pid)
        os.close(write_fd)
        assert await _wait_dead(proc.pid, timeout=3.0)
    finally:
        await nursery.shutdown(timeout=1.0)


# ------------------------------------------------------------------
# 优雅退出
# ------------------------------------------------------------------

async def test_graceful_shutdown_sends_sigterm(tmp_path):
    # 父进程先写文件确认权限 OK
    sig_file = tmp_path / "signal.txt"
    sig_file.write_text("PARENT_OK")

    nursery = ProcessNursery()
    proc = await _spawn(nursery, "log_signal", str(sig_file))
    try:
        assert await _wait_alive(proc.pid)
        # 子进程启动后会覆写为 STARTED
        for _ in range(20):
            if sig_file.read_text().startswith("STARTED"):
                break
            await asyncio.sleep(0.05)
        content = sig_file.read_text()
        assert "STARTED" in content, f"expected STARTED but got: {content!r}"

        await nursery.shutdown(timeout=3.0)
        assert await _wait_dead(proc.pid, timeout=2.0)

        content = sig_file.read_text()
        assert ("SIGTERM" in content or "Terminated" in content), f"got: {content!r}"
        assert "STARTED" in content
    finally:
        await nursery.shutdown(timeout=1.0)


# ------------------------------------------------------------------
# SIGKILL 回退
# ------------------------------------------------------------------

async def test_sigkill_fallback_when_child_ignores_sigterm(tmp_path):
    sig_file = tmp_path / "signal.txt"
    nursery = ProcessNursery()
    proc = await _spawn(nursery, "ignore_term", str(sig_file)
)
    try:
        assert await _wait_alive(proc.pid)
        await nursery.shutdown(timeout=1.0)  # 短 timeout 快速进 SIGKILL
        assert await _wait_dead(proc.pid, timeout=2.0)
        if sig_file.exists():
            assert "SIGTERM" in sig_file.read_text()
    finally:
        await nursery.shutdown(timeout=1.0)


# ------------------------------------------------------------------
# 已退出子进程不报错
# ------------------------------------------------------------------

async def test_shutdown_handles_already_exited_child():
    nursery = ProcessNursery()
    proc = await _spawn(nursery, "exit_ok"
)
    try:
        await asyncio.wait_for(proc.wait(), timeout=3.0)
        assert proc.returncode == 0
        await nursery.shutdown(timeout=1.0)  # no error
    finally:
        await nursery.shutdown(timeout=1.0)


# ------------------------------------------------------------------
# 多子进程全杀
# ------------------------------------------------------------------

async def test_shutdown_kills_multiple_children():
    nursery = ProcessNursery()
    pids = []
    try:
        for _ in range(3):
            proc = await _spawn(nursery, "sleep"
)
            pids.append(proc.pid)

        for pid in pids:
            assert await _wait_alive(pid)

        await nursery.shutdown(timeout=3.0)

        for pid in pids:
            assert await _wait_dead(pid, timeout=2.0)
    finally:
        await nursery.shutdown(timeout=1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
