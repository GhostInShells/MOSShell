#!/usr/bin/env python3
"""Launch a dsh of our own and prove it reached readiness — the sample for `launching-a-dsh`.

    python skills/runtime/launching-a-dsh/sample.py
    python skills/runtime/launching-a-dsh/sample.py --profile web --port 3099 --keep-home

Exits 0 only when the launcher's readiness wait returned (token discovered + mux connected).
A timeout exits 1 with the reason, so this doubles as the launch-path check.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import shutil
import socket
import sys
import tempfile
from pathlib import Path

from ghoshell_dsh.v0_2_0 import DSH_VERSION, DshLauncher, DshLauncherConfig


def free_port() -> int:
    """A port nothing is listening on right now. Racy in principle, fine for a sample."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def redact(url: str | None) -> str:
    """Print the surface without the credential in it."""
    if not url:
        return "<none>"
    return url.split("token=")[0] + "token=***"


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="web", help="dsh process profile (default: web)")
    parser.add_argument("--port", type=int, default=None, help="port to bind (default: a free one)")
    parser.add_argument("--home", default=None, help="DSH_HOME to use (default: a throwaway dir)")
    parser.add_argument("--keep-home", action="store_true", help="do not delete the home on exit")
    parser.add_argument("--timeout", type=float, default=30.0, help="readiness timeout in seconds")
    args = parser.parse_args()

    # One stream, so the landmarks and the package's own diagnostics keep their order.
    # WARNING is the right floor: below it, startup chatter is the reconnect loop working.
    logging.basicConfig(
        level=logging.WARNING, stream=sys.stdout, format="%(levelname)s: %(message)s",
    )

    home = Path(args.home) if args.home else Path(tempfile.mkdtemp(prefix="dsh-skill-"))
    port = args.port or free_port()

    # dsh opens a browser by default; a sample must never do that on somebody's desktop.
    os.environ["DSH_WEB_NO_OPEN"] = "1"

    config = DshLauncherConfig(
        home=home,
        profile=args.profile,
        port=port,
        readiness_timeout=args.timeout,
    )
    print(f"dsh_version_declared: {DSH_VERSION}")
    print(f"home: {home}")
    print(f"port: {port}")

    launcher = DshLauncher(config)
    try:
        async with launcher:  # spawn → token → authorize → mux connected → ready
            token = launcher.token()
            print(f"token: {'discovered' if token else 'MISSING'}")
            print(f"web_url: {redact(launcher.web_url())}")
            print(f"running: {launcher.is_running()}")
            print("READY")
    except TimeoutError as exc:
        print(f"NOT READY: {exc}", file=sys.stderr)
        return 1
    finally:
        if not args.keep_home:
            shutil.rmtree(home, ignore_errors=True)

    print("stopped cleanly")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
