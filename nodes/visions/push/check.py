"""Pre-launch probe for the push node (NODE.md `check:`).

Runs as an independent process, zero cooperation with main.py. Exit code only:
0 = the environment can run; nonzero + stderr = the reason (surfaces to the
launcher and, via the nodes channel, to the model). Write "why", not a stack
trace.

The probe gates only the environment — policy and ffmpeg presence. It never
opens a device: opening the screen without macOS Screen Recording permission
blocks instead of erroring, and a probe that hangs is worse than one that
degrades.
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
    allow = os.getenv("PUSH_ALLOW", "1").strip().lower()
    if allow in _FALSE:
        return _fail(f"push disabled by policy: PUSH_ALLOW={allow!r}")

    if shutil.which("ffmpeg") is None:
        return _fail("ffmpeg not found (required for capture)")

    if sys.platform != "darwin":
        return _fail(f"push capture is macOS-only (running on {sys.platform})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
