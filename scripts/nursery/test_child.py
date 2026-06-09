"""ProcessNursery 集成测试用子进程脚本。行为由命令行参数决定。

模式:
  pgid                    打印自己的进程组 ID 并退出
  watch_pipe              阻塞读 MOSS_NURSERY_FD, EOF 时退出
  log_signal <file>       注册 SIGTERM handler, 写信号名到文件后退出, 否则 sleep 60
  ignore_term <file>      注册 SIGTERM handler 仅记录, 不退出 (SIG_IGN 语义)
  exit_ok                 立即 exit(0)
  sleep                   sleep 60 (被 kill 用)
"""

import os
import signal
import sys
import time
from pathlib import Path


def mode_pgid(pgid_file: str):
    Path(pgid_file).write_text(str(os.getpgrp()))
    sys.exit(0)


def mode_watch_pipe():
    fd_str = os.environ.get("MOSS_NURSERY_FD")
    if not fd_str:
        print("NO_FD")
        sys.exit(1)
    fd = int(fd_str)
    # 阻塞读 EOF
    data = os.read(fd, 1)
    if not data:
        sys.exit(42)  # EOF exit code
    sys.exit(0)


def mode_log_signal(sig_file: str):
    path = Path(sig_file)
    path.write_text("STARTED")

    def handler(sig, frame):
        with open(path, "a") as f:
            f.write(f"|{signal.strsignal(sig)}")
            f.flush()
            os.fsync(f.fileno())
        sys.exit(0)

    signal.signal(signal.SIGTERM, handler)
    time.sleep(60)


def mode_ignore_term(sig_file: str):
    path = Path(sig_file)

    def handler(sig, frame):
        path.write_text(signal.strsignal(sig))

    signal.signal(signal.SIGTERM, handler)
    time.sleep(60)


def mode_exit_ok():
    sys.exit(0)


def mode_sleep():
    time.sleep(60)


MODES = {
    "pgid": mode_pgid,
    "watch_pipe": mode_watch_pipe,
    "log_signal": mode_log_signal,
    "ignore_term": mode_ignore_term,
    "exit_ok": mode_exit_ok,
    "sleep": mode_sleep,
}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: test_child.py <mode> [args...]", file=sys.stderr)
        sys.exit(1)

    mode = sys.argv[1]
    fn = MODES.get(mode)
    if fn is None:
        print(f"unknown mode: {mode}", file=sys.stderr)
        sys.exit(1)

    fn(*sys.argv[2:])
