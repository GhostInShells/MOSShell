"""Safely clear the current workspace's Aurelius test Memento."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TARGET = _PROJECT_ROOT / ".moss" / "ghosts" / "aurelius" / "memento"
_EXPECTED_ENTRIES = frozenset({".gitignore", "branches", "moments"})


def _running_aurelius() -> list[str]:
    try:
        result = subprocess.run(
            ["pgrep", "-fl", "aurelius"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise RuntimeError(f"cannot verify running Aurelius processes: {error}") from error
    if result.returncode not in {0, 1}:
        raise RuntimeError(f"pgrep failed with exit code {result.returncode}")
    project = str(_PROJECT_ROOT)
    return [
        line
        for line in result.stdout.splitlines()
        if project in line and "moss-run-ghost aurelius" in line
    ]


def main() -> int:
    if not (_PROJECT_ROOT / "pyproject.toml").is_file():
        print(f"REFUSED: project root marker is missing: {_PROJECT_ROOT}", file=sys.stderr)
        return 2
    if _TARGET.is_symlink():
        print(f"REFUSED: target is a symlink: {_TARGET}", file=sys.stderr)
        return 2
    try:
        running = _running_aurelius()
    except RuntimeError as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2
    if running:
        print("REFUSED: stop Aurelius before clearing its Memento:", file=sys.stderr)
        for process in running:
            print(f"  {process}", file=sys.stderr)
        return 2
    if not _TARGET.exists():
        print(f"SKIP: no Aurelius Memento exists at {_TARGET}")
        return 0
    if not _TARGET.is_dir() or _TARGET.parent.resolve() != (
        _PROJECT_ROOT / ".moss" / "ghosts" / "aurelius"
    ).resolve():
        print(f"REFUSED: target is not the expected Memento directory: {_TARGET}", file=sys.stderr)
        return 2
    unexpected = {path.name for path in _TARGET.iterdir()} - _EXPECTED_ENTRIES
    if unexpected:
        print(f"REFUSED: unexpected entries in Memento: {sorted(unexpected)}", file=sys.stderr)
        return 2
    shutil.rmtree(_TARGET)
    print(f"CLEARED: {_TARGET}")
    print("The next Aurelius start will create an empty Memento.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
