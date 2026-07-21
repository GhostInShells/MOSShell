from __future__ import annotations

import importlib.util
import os
import platform
import sys
from pathlib import Path


def _resolve_mjpython_bin() -> str:
    spec = importlib.util.find_spec("mujoco")
    if spec is None or spec.origin is None:
        raise RuntimeError("mujoco package not found in current environment")
    module_dir = Path(spec.origin).resolve().parent
    mjpython_bin = module_dir / "MuJoCo_(mjpython).app" / "Contents" / "MacOS" / "mjpython"
    if not mjpython_bin.exists():
        raise RuntimeError(f"mjpython binary not found: {mjpython_bin}")
    return str(mjpython_bin)


def main(argv: list[str]) -> None:
    if platform.system() != "Darwin":
        raise RuntimeError("mjpython launcher is only required on macOS")
    if len(argv) < 2:
        raise SystemExit("Usage: mjpython_launcher.py <target-script> [args...]")

    os.environ["MJPYTHON_BIN"] = _resolve_mjpython_bin()
    # uv virtualenv uses symlinked python shims; mjpython needs the real binary.
    real_python = Path(os.path.realpath(sys.executable))
    os.environ["MJPYTHON_LIBPYTHON"] = str(real_python)
    lib_dir = real_python.parent.parent / "lib"
    if lib_dir.exists():
        fallback_paths = [str(lib_dir)]
        existing = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
        if existing:
            fallback_paths.extend([p for p in existing.split(":") if p])
        else:
            fallback_paths.extend(["/usr/local/lib", "/usr/lib"])
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(dict.fromkeys(fallback_paths))
    os.environ["MOSS_G1_SIM_UNDER_MJPYTHON"] = "1"

    target = Path(argv[1]).resolve()
    forwarded_argv = [sys.executable, str(target), *argv[2:]]
    os.execve(os.environ["MJPYTHON_BIN"], forwarded_argv, os.environ)


if __name__ == "__main__":
    main(sys.argv)
