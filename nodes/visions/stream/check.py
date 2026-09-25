"""Pre-launch probe for the stream vision node (NODE.md `check:`).

Runs as an independent process, zero cooperation with main.py. Exit code only:
0 = the environment can run; nonzero + stderr = the reason (surfaces to the
launcher and, via the nodes channel, to the model). Write "why", not a stack
trace.

The stream address itself is a runtime concern (ffmpeg exit surfaces through
the channel health short-circuit), so the probe only gates the environment:
policy + ffmpeg presence.
"""
from __future__ import annotations

import os
import shutil
import sys

_FALSE = ("0", "false", "no", "off")


def _fail(reason: str) -> int:
    print(reason, file=sys.stderr)
    return 1


def main() -> int:
    allow = os.getenv("STREAM_ALLOW", "1").strip().lower()
    if allow in _FALSE:
        return _fail(f"stream vision disabled by policy: STREAM_ALLOW={allow!r}")

    if shutil.which("ffmpeg") is None:
        return _fail("ffmpeg not found (required for stream ingest)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
