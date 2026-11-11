"""Installing a plugin bundle into a dsh home — copy, then prove it loaded.

A bundle is not published and not built. It is a small directory of plain files that ships
inside a wheel, and installing it is: copy the directory into the home's shared
`node_modules`, then let dsh apply the bundle's patch layer at startup. No registry, no
pnpm, no TypeScript toolchain — a Python project should not need a JavaScript build to
deliver its own capability.

Three conditions make a bundle loadable, and all three are placement problems, not
packaging problems:

1. **Resolvable by name.** The patch entries say `name: '@local/<bundle>'`, and node
   resolves that by walking up from the importing file. Copying the bundle into
   `<home>/profiles/node_modules/@local/<bundle>/` puts it beside dsh's own packages
   (which dsh hoists to `profiles/node_modules`), so one copy serves every profile under
   that home.
2. **Present in the patch layer.** dsh applies `--patch <file>` after the profile layer.
   The bundle ships its own patch file, so starting with
   `--patch <copy>/cordis.patch.yml` is the whole registration — the user's profile
   config is never edited.
3. **Plain JavaScript under `node_modules`.** Node refuses to strip TypeScript for files
   inside `node_modules` (`ERR_UNSUPPORTED_NODE_MODULES_TYPE_STRIPPING`), so a bundle
   entry pointing at a `.ts` file there can never load. A bundle whose source is
   TypeScript hands `install_bundle` a `transform` that rewrites the copied tree before
   the stamp is written; the installed copy is always the thing dsh will actually import.

Copy, not symlink: a symlink goes stale the moment the package is reinstalled somewhere
else, and its failure mode is "the plugin silently stops loading".

## The bundle carries its own fingerprint

An installed bundle is only as fresh as the last copy, and nothing about the old one looks
wrong — a missing verb surfaces as a silent failure much later. So every copy writes
``dsh-bundle.json`` beside the plugin: the content hash of the **source** files plus the
package version. The hash is taken from the source, not from the copy, because a copy may
be transformed on the way in; ``bundle_changed`` compares that stamp with the source and
copies only when the bytes differ. The check names no verb, so a new plugin capability can
never require a matching new check.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Callable

__all__ = [
    "BUNDLE_NAME",
    "BUNDLE_STAMP",
    "BRIDGE_BUNDLE_NAME",
    "bridge_dir",
    "bundle_changed",
    "bundle_dir",
    "bundle_fingerprint",
    "install_bundle",
    "install_plugin",
    "installed_fingerprint",
    "installed_patch",
    "installed_root",
    "plugin_changed",
    "refresh_bundle",
    "refresh_plugin",
]

#: This generation's own bundle name — also its path under `node_modules`.
BUNDLE_NAME = "@local/ghoshell-dsh"

#: The page bridge's bundle name — a second package, installed only where a page is driven.
BRIDGE_BUNDLE_NAME = "@local/ghoshell-dsh-bridge"

#: The marker that must appear in `dsh --dump-config` once this bundle's patch is applied.
BUNDLE_ENTRY_ID = "ghoshell"

#: The stamp written into every installed copy — the fingerprint of what it was copied from.
BUNDLE_STAMP = "dsh-bundle.json"

#: Rewrites a freshly copied bundle in place (e.g. TypeScript → JavaScript).
BundleTransform = Callable[[Path], None]


def bundle_dir() -> Path:
    """Where this generation's service bundle lives inside the installed package.

    The service bundle is the one every consumer wants: the RPC verbs over sessions.
    """
    return Path(__file__).resolve().parent / "plugin" / "service"


def bridge_dir() -> Path:
    """Where the page bridge bundle lives — installed only where a dsh page is meant to be driven.

    It is a separate package on purpose: the service answers questions about sessions, the
    bridge holds a queue a browser polls. A ghost home wants the first and not the second.
    """
    return Path(__file__).resolve().parent / "plugin" / "bridge"


# ---- placement ---- #

def installed_root(home: str | Path, name: str = BUNDLE_NAME) -> Path:
    """Where a bundle goes inside a dsh home — shared by every profile under it."""
    return Path(home) / "profiles" / "node_modules" / name


def installed_patch(home: str | Path, name: str = BUNDLE_NAME) -> Path:
    """The patch file to hand dsh as `--patch` once the bundle is installed."""
    return installed_root(home, name) / "cordis.patch.yml"


# ---- fingerprint ---- #

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


def bundle_fingerprint(source: str | Path | None = None) -> str:
    """The fingerprint of a bundle as it ships in its package (default: this generation's)."""
    return _digest(Path(source) if source is not None else bundle_dir())


def installed_fingerprint(home: str | Path, name: str = BUNDLE_NAME) -> str:
    """The fingerprint an installed copy was created from; empty when no stamp exists."""
    stamp = installed_root(home, name) / BUNDLE_STAMP
    try:
        data = json.loads(stamp.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    value = data.get("fingerprint") if isinstance(data, dict) else ""
    return value if isinstance(value, str) else ""


def bundle_changed(
    home: str | Path, source: str | Path | None = None, name: str = BUNDLE_NAME
) -> bool:
    """Whether the source bundle differs from the one installed in ``home``."""
    return installed_fingerprint(home, name) != bundle_fingerprint(source)


def plugin_changed(home: str | Path) -> bool:
    """Whether this generation's own bundle differs from the one installed in ``home``."""
    return bundle_changed(home)


# ---- install ---- #

def _source_version(source: Path) -> str:
    try:
        data = json.loads((source / "package.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    value = data.get("version") if isinstance(data, dict) else ""
    return value if isinstance(value, str) else ""


def install_bundle(
    *,
    source: str | Path,
    name: str,
    home: str | Path,
    profile: str = "web",
    binary: str | Path | None = None,
    init: bool = False,
    if_changed: bool = False,
    transform: BundleTransform | None = None,
    verify_marker: str | None = None,
) -> Path:
    """Copy a bundle into `home` and return its patch path; optionally prove dsh sees it.

    ``transform`` runs on the **installed copy** before the stamp is written — it exists so
    a TypeScript source can become the JavaScript a `node_modules` entry must be. The stamp
    records the source fingerprint, so a transformed copy still reports "unchanged" on the
    next start.

    ``verify_marker`` is a string the composed tree must contain once the patch is applied —
    the bundle's package name is the honest choice, because the loader entry id can be one
    an older layer also carries. Together with ``binary`` it catches a patch that was
    ignored, which is otherwise only visible as a plugin that never mounted.

    Two scopes, because they are different promises:

    - ``init=False`` (default) — **bundle only**. The home is somebody else's; we drop our
      files into it and touch nothing else. Re-running is safe, and it is the right mode
      for a home that is already in service.
    - ``init=True`` — **prepare the home too**. `dsh --profile <p> --dump-config` is run
      first, which is how dsh provisions a home and a profile without booting anything;
      only then is the bundle copied in. Use it when the directory may be empty, e.g. from
      a ghost's own initialisation.

    ``if_changed=True`` makes the copy conditional on the content fingerprint: a bundle
    whose bytes already match is left alone, and nothing is re-verified. That is the mode a
    node uses at every start — an upgrade is picked up, a no-op start costs one hash.

    Booting is never used for either: `--dump-config` composes the same tree a boot would,
    costs a second, and cannot leave a process behind.
    """
    source_path = Path(source)
    if not source_path.is_dir():
        raise RuntimeError(f"bundle missing from its package: {source_path}")

    if if_changed and not bundle_changed(home, source_path, name):
        return installed_patch(home, name)

    binary_path = Path(binary) if binary is not None else None
    if init:
        if binary_path is None:
            raise RuntimeError(
                f"init=True needs a dsh binary: provisioning {home} is dsh's job, not ours"
            )
        _provision(binary=binary_path, home=Path(home), profile=profile)

    target = installed_root(home, name)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source_path, target)
    if transform is not None:
        transform(target)
    _write_stamp(target, fingerprint=_digest(source_path), version=_source_version(source_path))

    patch = installed_patch(home, name)
    if binary_path is not None and verify_marker is not None:
        _verify(binary=binary_path, home=Path(home), profile=profile, patch=patch, marker=verify_marker)
    return patch


def refresh_bundle(
    *,
    source: str | Path,
    name: str,
    home: str | Path,
    profile: str = "web",
    binary: str | Path | None = None,
    init: bool = False,
    transform: BundleTransform | None = None,
    verify_marker: str | None = None,
) -> bool:
    """Install only if the source changed; returns whether a copy happened."""
    changed = bundle_changed(home, source, name)
    provision = init and not _profile_provisioned(home, profile)
    if changed or provision:
        install_bundle(
            source=source,
            name=name,
            home=home,
            profile=profile,
            binary=binary,
            init=provision,
            if_changed=False,
            transform=transform,
            verify_marker=verify_marker,
        )
    return changed


def install_plugin(
    *,
    home: str | Path,
    profile: str = "web",
    binary: str | Path | None = None,
    init: bool = False,
    if_changed: bool = False,
) -> Path:
    """Install this generation's own bundle — the thin wrapper over `install_bundle`."""
    return install_bundle(
        source=bundle_dir(),
        name=BUNDLE_NAME,
        home=home,
        profile=profile,
        binary=binary,
        init=init,
        if_changed=if_changed,
        verify_marker=BUNDLE_ENTRY_ID,
    )


def refresh_plugin(
    *, home: str | Path, profile: str = "web", binary: str | Path | None = None
) -> bool:
    """Install this generation's own bundle only if the source changed."""
    return refresh_bundle(
        source=bundle_dir(),
        name=BUNDLE_NAME,
        home=home,
        profile=profile,
        binary=binary,
        verify_marker=BUNDLE_ENTRY_ID,
    )


def _write_stamp(target: Path, *, fingerprint: str, version: str) -> None:
    """Record what this copy is a copy of — content hash, not a hand-maintained number."""
    stamp = target / BUNDLE_STAMP
    stamp.write_text(
        json.dumps({"fingerprint": fingerprint, "version": version}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )


# ---- provisioning ---- #

def _profile_dir(home: str | Path, profile: str) -> Path:
    return Path(home) / "profiles" / profile


def _profile_provisioned(home: str | Path, profile: str) -> bool:
    """A profile exists once dsh has written its manifest — nothing else counts as one."""
    return (_profile_dir(home, profile) / "package.json").is_file()


def _dsh_env(home: Path) -> dict[str, str]:
    """`DSH_HOME` is how dsh knows which home to work on — `cwd` alone means `~/.dsh`.

    Both subprocess calls below are about *this* home; without this, a fresh ghost silently
    provisions the user's global home instead, and the copy lands next to a profile that
    does not exist.
    """
    import os

    env = dict(os.environ)
    env["DSH_HOME"] = str(home)
    return env


def _provision(*, binary: Path, home: Path, profile: str) -> None:
    """Let dsh create the home and the profile — `--dump-config` provisions without booting."""
    home.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [str(binary), "--profile", profile, "--dump-config"],
        cwd=str(home),
        env=_dsh_env(home),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if result.returncode != 0:
        tail = f"{result.stdout}\n{result.stderr}"[-2000:]
        raise RuntimeError(f"dsh could not provision profile {profile!r}:\n{tail}")


def _verify(*, binary: Path, home: Path, profile: str, patch: Path, marker: str) -> None:
    """Compose the profile with our patch layer and check the marker is in it."""
    home.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [str(binary), "--profile", profile, "--patch", str(patch), "--dump-config"],
        cwd=str(home),
        env=_dsh_env(home),
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
    if marker not in output:
        raise RuntimeError(
            f"dsh composed a config without {marker!r} — the patch layer was "
            f"ignored. Tail:\n{output[-2000:]}"
        )
