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

## The bundle carries its own fingerprint

An installed bundle is only as fresh as the last copy, and nothing about the old one looks
wrong — a missing verb surfaces as a silent failure much later. So every copy writes
``ghoshell-bundle.json`` beside the plugin: the content hash of the files that were copied
plus the package version. ``refresh_plugin`` compares that stamp with the source and copies
only when the bytes differ; the check names no verb, so a new plugin capability can never
require a matching new check.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

__all__ = [
    "BUNDLE_NAME",
    "BUNDLE_STAMP",
    "bundle_dir",
    "bundle_fingerprint",
    "install_plugin",
    "installed_fingerprint",
    "installed_patch",
    "installed_root",
    "plugin_changed",
    "refresh_plugin",
]

#: The bundle's package name — also its path under `node_modules`.
BUNDLE_NAME = "@local/ghoshell-dsh"

#: The marker that must appear in `dsh --dump-config` once the patch layer is applied.
BUNDLE_ENTRY_ID = "ghoshell"

#: The stamp written into every installed copy — its content hash and version.
BUNDLE_STAMP = "ghoshell-bundle.json"


def bundle_dir() -> Path:
    """Where this generation's plugin lives inside the installed package."""
    return Path(__file__).resolve().parent / "plugin"


def installed_root(home: str | Path) -> Path:
    """Where the bundle goes inside a dsh home — shared by every profile under it."""
    return Path(home) / "profiles" / "node_modules" / BUNDLE_NAME


def installed_patch(home: str | Path) -> Path:
    """The patch file to hand dsh as `--patch` once the bundle is installed."""
    return installed_root(home) / "cordis.patch.yml"


def _digest(root: Path) -> str:
    """A stable content fingerprint: relative names plus bytes, in sorted order.

    The stamp itself is excluded, so writing it into an installed copy does not change what
    the copy is a copy *of*.
    """
    sha = hashlib.sha256()
    for item in sorted(path for path in root.rglob("*") if path.is_file()):
        relative = item.relative_to(root).as_posix()
        if relative == BUNDLE_STAMP:
            continue
        sha.update(relative.encode("utf-8"))
        sha.update(b"\0")
        sha.update(item.read_bytes())
        sha.update(b"\0")
    return sha.hexdigest()


def bundle_fingerprint() -> str:
    """The fingerprint of the plugin as it ships in this package."""
    return _digest(bundle_dir())


def installed_fingerprint(home: str | Path) -> str:
    """The fingerprint an installed copy was created from; empty when no stamp exists."""
    stamp = installed_root(home) / BUNDLE_STAMP
    try:
        data = json.loads(stamp.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    value = data.get("fingerprint") if isinstance(data, dict) else ""
    return value if isinstance(value, str) else ""


def plugin_changed(home: str | Path) -> bool:
    """Whether the source bundle differs from the one installed in ``home``."""
    return installed_fingerprint(home) != bundle_fingerprint()


def _source_version() -> str:
    try:
        data = json.loads((bundle_dir() / "package.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    value = data.get("version") if isinstance(data, dict) else ""
    return value if isinstance(value, str) else ""


def install_plugin(
    *,
    home: str | Path,
    profile: str = "web",
    binary: str | Path | None = None,
    init: bool = False,
    if_changed: bool = False,
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

    `if_changed=True` makes the copy conditional on the content fingerprint: a bundle whose
    bytes already match is left alone, and nothing is re-verified. That is the mode a node uses
    at every start — an upgrade is picked up, a no-op start costs one hash.

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

    if if_changed and not plugin_changed(home):
        return installed_patch(home)

    if init:
        if binary is None:
            raise RuntimeError("init=True needs a dsh binary: provisioning a home is dsh's job")
        _provision(binary=Path(binary), home=Path(home), profile=profile)

    target = installed_root(home)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)
    _write_stamp(target)

    patch = installed_patch(home)
    if binary is not None:
        _verify(binary=Path(binary), home=Path(home), profile=profile, patch=patch)
    return patch


def refresh_plugin(
    *, home: str | Path, profile: str = "web", binary: str | Path | None = None
) -> bool:
    """Install only if the source changed; returns whether a copy happened."""
    changed = plugin_changed(home)
    if changed:
        install_plugin(home=home, profile=profile, binary=binary, if_changed=False)
    return changed


def _write_stamp(target: Path) -> None:
    """Record what this copy is a copy of — content hash, not a hand-maintained number."""
    stamp = target / BUNDLE_STAMP
    stamp.write_text(
        json.dumps(
            {"fingerprint": _digest(target), "version": _source_version()},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


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
