#!/usr/bin/env python
"""Pre-launch probe for the dsh node — environment and installed home, no side effects.

Run by the node manager before spawn (NODE.md `check:`). Exit 0 = ready; nonzero + stderr
is the reason the node must not start. It checks, in order: the `[dsh]` extra, the dsh
executable, and the artifacts the install step leaves in the node home.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    try:
        from ghoshell_dsh.v0_2_0 import DSH_VERSION
        from ghoshell_dsh.v0_2_0.install import installed_patch
        from ghoshell_dsh.v0_2_0.launcher import resolve_dsh_binary
    except Exception as exc:
        print(f"dsh node: the [dsh] extra is not installed ({exc})", file=sys.stderr)
        return 1

    try:
        binary = resolve_dsh_binary()
    except Exception as exc:
        print(f"dsh node: {exc}", file=sys.stderr)
        return 1

    root = Path(__file__).resolve().parent
    dsh_home = root / ".dsh"
    missing = [
        path
        for path in (dsh_home, root / ".memento", installed_patch(dsh_home))
        if not path.exists()
    ]
    if missing:
        print(
            "dsh node: home is not initialized — follow INSTALL.md "
            "(python -m ghoshell_moss.deepseek_harness.v0_2_0.node --install).",
            file=sys.stderr,
        )
        for path in missing:
            print(f"  missing: {path}", file=sys.stderr)
        return 1

    print(f"dsh node ready: dsh {DSH_VERSION} at {binary}; home {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
