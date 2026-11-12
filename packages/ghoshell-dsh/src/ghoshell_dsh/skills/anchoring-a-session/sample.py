#!/usr/bin/env python3
"""Take an anchor from a live dsh session — the sample for `anchoring-a-session`.

    python src/ghoshell_dsh/skills/anchoring-a-session/sample.py
    python src/ghoshell_dsh/skills/anchoring-a-session/sample.py --keep-home

What it proves, in order:

1. **the verb answers at all** — `ghoshell/anchor` is a Remote on the installed plugin, called
   through the same envelope every other dsh verb uses, not a hand-rolled HTTP route;
2. **the coordinate is the log's own** — `seq` is the last event's seq, and the title rides
   along because it was read from the log, not asserted;
3. **an earlier point is honoured, an impossible one is refused loudly** — `seq=0` lands on the
   first event; `seq=10**6` and an unknown session both raise instead of returning something
   plausible;
4. **an anchor leaves nothing behind** — the session count in `DSH_HOME/sessions` is unchanged
   by taking one.

No model credentials are needed. The session is given one durable event by `rename` (a
`session/title` event), which is enough to anchor to. A turn is not required: an anchor is a
coordinate, and a coordinate does not need a conversation to happen first.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import socket
import sys
import tempfile
from pathlib import Path

from ghoshell_dsh.v0_2_0 import DSH_VERSION, DshLauncher, DshLauncherConfig
from ghoshell_dsh.v0_2_0.client import DshRpcException
from ghoshell_dsh.v0_2_0.install import install_plugin
from ghoshell_dsh.v0_2_0.types import sessions

TITLE = "anchor probe"
PRESET = "minimal"


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def session_dirs(home: Path) -> int:
    """How many session directories dsh is holding — the thing an anchor must not add to."""
    root = home / "sessions"
    if not root.is_dir():
        return 0
    return sum(1 for path in root.rglob("*") if path.is_dir())


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"{'PASS' if condition else 'FAIL'}  {label}{f' — {detail}' if detail else ''}")
    return condition


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="web")
    parser.add_argument("--home", default=None, help="DSH_HOME to use (default: a throwaway dir)")
    parser.add_argument("--keep-home", action="store_true")
    parser.add_argument("--timeout", type=float, default=45.0)
    args = parser.parse_args()

    home = Path(args.home) if args.home else Path(tempfile.mkdtemp(prefix="dsh-anchor-"))
    os.environ["DSH_WEB_NO_OPEN"] = "1"

    # The plugin must be in place *before* the process starts: the patch layer is read at boot.
    patch = install_plugin(home=home, profile=args.profile)
    print(f"dsh_version_declared: {DSH_VERSION}")
    print(f"home: {home}")
    print(f"plugin_patch: {patch}")

    ok = True
    launcher = DshLauncher(
        DshLauncherConfig(
            home=home,
            profile=args.profile,
            port=free_port(),
            readiness_timeout=args.timeout,
            plugin_patch=patch,
        )
    )
    try:
        async with launcher:
            client = launcher.client
            created = await client.session_create(
                sessions.SessionCreateParams(cwd=str(home), agentPreset=PRESET)
            )
            session_id = created.sessionId
            print(f"session: {session_id} (preset {created.agentPreset})")
            ok &= check("the plugin is installed and the session is created", session_id != "")

            session = launcher.attach_session(session_id)
            renamed = await session.rename(title=TITLE)
            print(f"renamed at seq {renamed.seq}")

            before = session_dirs(home)
            anchor = await session.anchor(note="skill sample")

            ok &= check("session_id is the one asked for", anchor.session_id == session_id, anchor.session_id)
            has_seq = anchor.seq is not None and anchor.seq >= renamed.seq
            ok &= check("seq is the log's last position", has_seq, str(anchor.seq))
            ok &= check("title was read from the log", anchor.title == TITLE, repr(anchor.title))
            ok &= check("agent_preset came from the session header", anchor.agent_preset == PRESET,
                        repr(anchor.agent_preset))
            same_cwd = Path(anchor.cwd or "").resolve() == home.resolve()
            ok &= check("cwd came from the session header", same_cwd, str(anchor.cwd))
            ok &= check("note is carried as given", anchor.note == "skill sample", repr(anchor.note))
            ok &= check("turn is absent without a turn, not invented", anchor.turn is None, repr(anchor.turn))
            after = session_dirs(home)
            ok &= check("taking an anchor creates no session", after == before, f"{before} -> {after}")

            first = await session.anchor(seq=0)
            ok &= check("an earlier seq is honoured", first.seq == 0, str(first.seq))

            try:
                await session.anchor(seq=10**6)
                ok &= check("an out-of-range seq is refused", False, "it returned instead")
            except DshRpcException as exc:
                ok &= check("an out-of-range seq is refused", "outside the log" in str(exc), str(exc))

            try:
                await client.remote("ghoshell/anchor", {"sessionId": "no-such-session"})
                ok &= check("an unknown session is refused", False, "it returned instead")
            except DshRpcException as exc:
                ok &= check("an unknown session is refused", "no live session" in str(exc), str(exc))

            dumped = anchor.dump(home / f"{session_id}.anchor.yml")
            ok &= check("an anchor can leave the process", dumped.is_file(), str(dumped))
            ok &= check("...and load back as itself", type(anchor).from_file(dumped) == anchor)

            print(f"anchor: {anchor.describe()}")
    except TimeoutError as exc:
        print(f"NOT READY: {exc}", file=sys.stderr)
        return 1
    finally:
        if not args.keep_home:
            shutil.rmtree(home, ignore_errors=True)

    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
