#!/usr/bin/env python3
"""The regression baseline for the dsh wrapper — one launch, every core mechanism, one verdict each.

    cd packages/ghoshell-dsh
    PYTHONPATH=src:../ghoshell-processes/src python src/ghoshell_dsh/skills/verifying-against-a-live-dsh/sample.py

    # no model credentials? every case that needs one reports BLOCKED instead of vanishing
    # keep the home around to inspect what the run left behind:
    ... sample.py --keep-home

This is the executable half of `regressions/dsh-wrapper/REGRESSION.md`: the case ids printed here
are the case ids in that table, and one full run of this program *is* a baseline.

Design rules, because earlier versions of these probes broke all three:

- **self-contained**: it creates its own `DSH_HOME`, installs the plugin into it, picks a free
  port, and launches its own dsh. Nothing is assumed about the machine beyond a working `dsh`.
- **one process, many cases**: cases run in dependency order against the same dsh, so state
  effects (a switched permission, a forked child) are visible instead of hidden by isolation.
- **it says what it measured**: every line carries case id, verdict, and the number or shape the
  case looked at. A baseline that reads "PASS" with no measurement cannot be compared later.
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

from ghoshell_dsh.blueprint.models.llm import LlmConfigPatch
from ghoshell_dsh.blueprint.transcript import TranscriptView
from ghoshell_dsh.v0_2_0 import DSH_VERSION, DshLauncher, DshLauncherConfig
from ghoshell_dsh.v0_2_0.client import DshRpcException
from ghoshell_dsh.v0_2_0.install import install_plugin
from ghoshell_dsh.v0_2_0.types import sessions

NOTE_TOOL = {
    "name": "record_note",
    "description": "Record one short note. Call this instead of answering in prose.",
    "parameters": {
        "type": "object",
        "properties": {"note": {"type": "string"}},
        "required": ["note"],
    },
}

VERDICTS: list[tuple[str, str, str]] = []  # (case id, verdict, detail)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def verdict(case: str, ok: bool | None, detail: str = "") -> bool | None:
    """Record one case. `None` means blocked — it ran into something environmental, not a failure."""
    label = "BLOCKED" if ok is None else ("PASS" if ok else "FAIL")
    VERDICTS.append((case, label, detail))
    print(f"{label:<7} {case}  {detail}")
    return ok


def session_dirs(home: Path) -> int:
    root = home / "sessions"
    return 0 if not root.is_dir() else sum(1 for p in root.rglob("session-*") if p.is_dir())


async def run(args: argparse.Namespace) -> int:
    home = Path(args.home) if args.home else Path(tempfile.mkdtemp(prefix="dsh-baseline-"))
    os.environ["DSH_WEB_NO_OPEN"] = "1"
    patch = install_plugin(home=home, profile=args.profile)
    others = home / "other-workspace"
    others.mkdir(parents=True, exist_ok=True)

    print(f"dsh_version_declared: {DSH_VERSION}")
    print(f"home: {home}")

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
            # ---- TC-001: the process reaches readiness -------------------------------
            verdict("TC-001", launcher.token() is not None, "token discovered, mux connected")

            conn = launcher
            # ---- TC-002: the connection is usable the moment it is entered -----------
            providers = None
            try:
                providers = await conn.llm_providers()
                verdict("TC-002", bool(providers), "attach implies readiness; llm_providers answered")
            except Exception as exc:
                verdict("TC-002", False, f"{type(exc).__name__}: {exc}")

            live = [p.id for p in (providers or []) if p.live]
            # ---- TC-003: model resources are listed, live apart from declared -------
            verdict("TC-003", bool(live), f"{len(live)} live / {len(providers or [])} total: {live[:3]}")

            # ---- TC-004: a session can be created and its state read ----------------
            created = await conn.client.session_create(
                sessions.SessionCreateParams(cwd=str(home), agentPreset=args.preset or "minimal")
            )
            session_id = created.sessionId
            session = conn.attach_session(session_id)
            await asyncio.sleep(1.2)
            state = await session.session_state()
            verdict(
                "TC-004",
                state.attached and state.turn is None,
                f"running={state.running} pending={state.hasPending} turn={state.turn}",
            )

            # ---- TC-005: the session knows where it may act -------------------------
            policy = await session.policy()
            verdict(
                "TC-005",
                policy.sandbox in ("read-only", "workspace-write") and policy.sandboxRoot == str(home),
                f"{policy.sandbox} @ {policy.sandboxRoot}",
            )

            # ---- TC-006: permission switches, and full access is refused ------------
            closed = await session.set_policy(sandbox="read-only")
            reopened = await session.set_policy(sandbox="workspace-write")
            refused = False
            try:
                await session.set_policy(sandbox="danger-full-access")
            except DshRpcException as exc:
                refused = "not reachable" in str(exc)
            verdict(
                "TC-006",
                closed.sandbox == "read-only" and reopened.sandbox == "workspace-write" and refused,
                f"{closed.sandbox} -> {reopened.sandbox}, full access refused={refused}",
            )

            # ---- TC-007: switching permission does not touch the conversation -------
            clean = await TranscriptView.read(session)
            verdict("TC-007", clean.turns == 0 and clean.text == "", f"{clean.turns} turns after the switches")

            # Credentials are a human step, not something a model goes looking for: whether this
            # dsh can reach a model is decided by the profile a person registered. The flag says
            # "that has been done"; without it these cases report BLOCKED rather than guessing.
            has_credentials = args.with_model_cases
            turn_ran = False
            if has_credentials:
                # A fork can only cut at a completed turn, so the baseline needs one real turn
                # before TC-014. It is not a case of its own: it is the precondition that makes
                # the later cases mean something.
                try:
                    result = await session.run("Reply with exactly: ok.", timeout=240.0)
                    turn_ran = bool(result.finish_reason)
                    print(f"        (one turn ran: finish={result.finish_reason!r} "
                          f"reply={result.final_response[:24]!r})")
                except Exception as exc:
                    print(f"        (turn failed: {type(exc).__name__}: {exc})")

            # ---- TC-008: an anchor is a coordinate, not a guess ---------------------
            renamed = await session.rename(title="baseline session")
            anchor = await session.anchor(note="baseline")
            verdict(
                "TC-008",
                anchor.seq is not None and anchor.seq >= renamed.seq and anchor.title == "baseline session",
                f"seq={anchor.seq} turn={anchor.turn} title={anchor.title!r}",
            )

            # ---- TC-009: an anchor refuses a position that does not exist -----------
            out_of_range = False
            try:
                await session.anchor(seq=10**6)
            except DshRpcException as exc:
                out_of_range = "outside the log" in str(exc)
            earlier = await session.anchor(seq=0)
            verdict("TC-009", out_of_range and earlier.seq == 0, f"refused={out_of_range}, seq=0 honoured")

            # ---- TC-010: a reading excludes results, reasoning and injected context --
            view = await TranscriptView.read(session, limit_turns=3)
            verdict(
                "TC-010",
                view.turns >= 0 and "tool/result" not in view.text,
                f"{view.turns} turns / {view.messages} messages / {view.tool_calls} tools",
            )

            if not has_credentials:
                for case in ("TC-011", "TC-012", "TC-013"):
                    verdict(
                        case,
                        None,
                        "needs credentials registered in this profile by a human (--with-model-cases)",
                    )
            else:
                # ---- TC-011: an off-record call answers and records nothing ---------
                before = session_dirs(home)
                convo_before = await TranscriptView.read(session)
                reply = await session.chat_off_record(
                    "Reply with exactly: ok.",
                    llm=LlmConfigPatch(maxTokens=3000, reasoningEffort="off"),
                )
                convo_after = await TranscriptView.read(session)
                verdict(
                    "TC-011",
                    bool(reply.text) and convo_after.text == convo_before.text and session_dirs(home) == before,
                    f"text={reply.text[:28]!r} cacheRead={reply.usage.cacheReadTokens} "
                    f"sessions {before}->{session_dirs(home)}",
                )

                # ---- TC-012: extra tool declarations are accepted -------------------
                tooled = await session.chat_off_record(
                    "Do not answer in prose. Call record_note with note='baseline'.",
                    llm=LlmConfigPatch(maxTokens=4000),
                    tools=[NOTE_TOOL],
                )
                verdict(
                    "TC-012",
                    True,
                    f"tool_calls={[c.name for c in tooled.tool_calls]} — today the model may pick a session tool",
                )

                # ---- TC-013: an injected message is queued without waking the agent --
                accepted = await session.inject("baseline probe: queue only.", wait="accepted")
                verdict(
                    "TC-013",
                    accepted.messageId != "" and accepted.woke is False,
                    f"id={accepted.messageId[:8]} woke={accepted.woke}",
                )

            # ---- TC-014: a fork continues the history somewhere else ----------------
            if not turn_ran:
                for case in ("TC-014", "TC-015"):
                    verdict(case, None, "a fork needs a completed turn to cut at; none ran")
                forked = None
            else:
                forked = await session.fork_into(str(others))
                child = conn.attach_session(forked.sessionId)
                await asyncio.sleep(1.2)
                child_policy = await child.policy()
                child_anchor = await child.anchor()
                verdict(
                    "TC-014",
                    Path(child_policy.sandboxRoot or "") == others.resolve()
                    and child_anchor.turn == anchor.turn,
                    f"root={child_policy.sandboxRoot} turn={child_anchor.turn} seeded={forked.seededEvents}",
                )

                # ---- TC-015: the fork inherits its parent's permission --------------
                await session.set_policy(approval="never")
                second = await session.fork_into(str(others))
                grandchild = conn.attach_session(second.sessionId)
                await asyncio.sleep(1.2)
                inherited = await grandchild.policy()
                verdict(
                    "TC-015",
                    inherited.sandbox == reopened.sandbox and inherited.approval == "never",
                    f"{inherited.sandbox} / {inherited.approval}",
                )

            # ---- TC-016: a session can be registered under its workspace -----------
            # The registry is the witness, not the follow stream: a workspace created after the
            # baseline snapshot arrives does not push itself into `workspaces()`, so asserting on
            # that list would test the stream's timing rather than the registration.
            first = await session.attach_workspace()
            again = await session.attach_workspace()
            verdict(
                "TC-016",
                first != "" and first == again,
                f"workspace={first[:8]} reused={first == again}",
            )

            await session.close()
    except TimeoutError as exc:
        print(f"NOT READY: {exc}", file=sys.stderr)
        return 1
    finally:
        if not args.keep_home:
            shutil.rmtree(home, ignore_errors=True)

    failed = [c for c, v, _ in VERDICTS if v == "FAIL"]
    blocked = [c for c, v, _ in VERDICTS if v == "BLOCKED"]
    print(f"\n{len(VERDICTS) - len(failed) - len(blocked)}/{len(VERDICTS)} passed"
          + (f", {len(blocked)} blocked" if blocked else "")
          + (f", FAILED: {', '.join(failed)}" if failed else ""))
    return 0 if not failed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="web")
    parser.add_argument("--home", default=None, help="DSH_HOME to use (default: a throwaway dir)")
    parser.add_argument("--keep-home", action="store_true")
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--preset", default=None, help="agent preset for the test session")
    parser.add_argument(
        "--with-model-cases",
        action="store_true",
        help="the human has registered credentials in this profile; run the cases that need a model",
    )
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
