"""Installing this generation's plugin into a dsh home — copy, then prove it loaded.

The plugin is not published and not built. It is three plain files that ship inside this wheel,
and installing them is: copy the directory into the home's shared `node_modules`, then let dsh
apply the bundle's patch layer at startup. No registry, no pnpm, no TypeScript toolchain — a
Python project should not need a JavaScript build to deliver its own capability.

Two conditions make a plugin loadable, and both are placement problems, not packaging problems:

1. **Resolvable by name.** The patch entries say `name: '@local/ghoshell-dsh'`, and node resolves
   that by walking up from the importing file. Copying the bundle into
   `<home>/profiles/node_modules/@local/ghoshell-dsh/` puts it beside dsh's own packages (which
   dsh hoists to `profiles/node_modules`), so one copy serves every profile under that home.
2. **Present in the patch layer.** dsh applies `--patch <file>` after the profile layer. The
   bundle ships its own patch file, so starting with `--patch <copy>/cordis.patch.yml` is the
   whole registration — the user's profile config is never edited.

Copy, not symlink: a symlink goes stale the moment the package is reinstalled somewhere else,
and its failure mode is "the plugin silently stops loading".
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

__all__ = [
    "BUNDLE_NAME",
    "bundle_dir",
    "install_plugin",
    "installed_patch",
    "installed_root",
]

#: The bundle's package name — also its path under `node_modules`.
BUNDLE_NAME = "@local/ghoshell-dsh"

#: The marker that must appear in `dsh --dump-config` once the patch layer is applied.
BUNDLE_ENTRY_ID = "ghoshell"


def bundle_dir() -> Path:
    """Where this generation's plugin lives inside the installed package."""
    return Path(__file__).resolve().parent / "plugin"


def installed_root(home: str | Path) -> Path:
    """Where the bundle goes inside a dsh home — shared by every profile under it."""
    return Path(home) / "profiles" / "node_modules" / BUNDLE_NAME


def installed_patch(home: str | Path) -> Path:
    """The patch file to hand dsh as `--patch` once the bundle is installed."""
    return installed_root(home) / "cordis.patch.yml"


def install_plugin(
    *,
    home: str | Path,
    profile: str = "web",
    binary: str | Path | None = None,
    init: bool = False,
) -> Path:
    """Copy the bundle into `home`, and (when a binary is given) prove dsh can see it.

    Two scopes, because they are different promises:

    - `init=False` (default) — **plugin only**. The home is somebody else's; we drop our bundle
      into it and touch nothing else. Re-running is safe, and it is the right mode for a home
      that is already in service.
    - `init=True` — **prepare the home too**. `dsh --profile <p> --dump-config` is run first,
      which is how dsh provisions a home and a profile without booting anything; only then is
      the bundle copied in. Use it when the directory may be empty, e.g. from a ghost's own
      initialisation.

    Booting is never used for either: `--dump-config` composes the same tree a boot would, costs
    a second, and cannot leave a process behind.

    Idempotent: the target is replaced wholesale, so re-installing after an upgrade leaves no
    half-old bundle behind. Verification is a `--dump-config` composition, not a start: it costs
    a second, touches no port, and answers the only question worth asking at this point —
    "would the patch layer load this?".

    Returns the installed patch path, which is what a launcher passes as `--patch`.
    """
    source = bundle_dir()
    if not source.is_dir():
        raise RuntimeError(f"plugin bundle missing from the package: {source}")

    if init:
        if binary is None:
            raise RuntimeError("init=True needs a dsh binary: provisioning a home is dsh's job")
        _provision(binary=Path(binary), home=Path(home), profile=profile)

    target = installed_root(home)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)

    patch = installed_patch(home)
    if binary is not None:
        _verify(binary=Path(binary), home=Path(home), profile=profile, patch=patch)
    return patch


def _provision(*, binary: Path, home: Path, profile: str) -> None:
    """Let dsh create the home and the profile — `--dump-config` provisions without booting."""
    home.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [str(binary), "--profile", profile, "--dump-config"],
        cwd=str(home),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if result.returncode != 0:
        tail = f"{result.stdout}\n{result.stderr}"[-2000:]
        raise RuntimeError(f"dsh could not provision profile {profile!r}:\n{tail}")


def _verify(*, binary: Path, home: Path, profile: str, patch: Path) -> None:
    """Compose the profile with our patch layer and check the entry is in it."""
    home.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [str(binary), "--profile", profile, "--patch", str(patch), "--dump-config"],
        cwd=str(home),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    output = f"{result.stdout}\n{result.stderr}"
    if result.returncode != 0:
        raise RuntimeError(
            f"dsh refused the patch layer (exit {result.returncode}):\n{output[-2000:]}"
        )
    if BUNDLE_ENTRY_ID not in output:
        raise RuntimeError(
            f"dsh composed a config without our entry '{BUNDLE_ENTRY_ID}' — the patch layer was "
            f"ignored. Tail:\n{output[-2000:]}"
        )
