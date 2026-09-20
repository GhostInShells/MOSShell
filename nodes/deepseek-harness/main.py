"""dsh node — connect to a running dsh web (host/port/token), expose a code-driven channel.

The model starts this node with a dsh web token it knows, then drives dsh by writing
Python against the connection/session surfaces (ghoshell_moss.deepseek_harness.surfaces).

per-host:port flock: a second node to the same host:port fails fast (advisory, OS-released
on exit) — prevents two connections from both claiming the same dsh instance.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import sys
import tempfile
from pathlib import Path

from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.deepseek_harness.channel import new_dsh_runtime_channel
from ghoshell_moss.deepseek_harness.launcher import DshConnection, DshConnectionConfig
from ghoshell_moss.deepseek_harness.runtime import DshRuntime


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="dsh", description="dsh node")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3080)
    parser.add_argument("--token", default=None, help="dsh web token (or DSH_WEB_TOKEN env)")
    return parser.parse_args(argv)


def _lock_fd(host: str, port: int) -> int | None:
    """per-host:port advisory flock; None = already held by another node."""
    path = Path(tempfile.gettempdir()) / f"moss-dsh-{host}-{port}.lock"
    fd = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None
    return fd


async def main(matrix: Matrix) -> None:
    args = _parse_args(sys.argv[1:])

    fd = _lock_fd(args.host, args.port)
    if fd is None:
        raise RuntimeError(f"another dsh node already connected to {args.host}:{args.port}")

    connection = DshConnection(
        DshConnectionConfig(host=args.host, port=args.port, token=args.token)
    )
    runtime = DshRuntime(connection, send_signal=matrix.send_signal_to_ghost)
    channel = new_dsh_runtime_channel(runtime)
    # fd 保持打开 → 进程生命周期内持有 flock, 进程退出时 OS 释放.
    await matrix.provide_channel(channel)


if __name__ == "__main__":
    Matrix.discover().run(main)
