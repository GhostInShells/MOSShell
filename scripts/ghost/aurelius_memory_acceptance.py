"""Deterministic AureliusMemory persistence smoke test (no LLM/network required)."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from ghoshell_moss.core.blueprint.mindflow import Moment
from ghoshell_moss.ghosts.aurelius import AureliusMemory
from ghoshell_moss.message import Message


def _moment(user: str, assistant: str) -> Moment:
    return Moment(
        percepts={"acceptance": [Message.new().with_content(user)]},
        logos=assistant,
    )


def _history_text(memory: AureliusMemory) -> str:
    return "\n".join(str(getattr(part, "content", "")) for message in memory.model_history() for part in message.parts)


def verify(root: Path) -> None:
    token = "moss-memory-acceptance-731"
    first = AureliusMemory(root, "acceptance", auto_commit_every=2)
    first.remember(_moment(f"请记住验证码 {token}", "已记录。"))
    commit = first.remember(_moment("现在关闭并重新打开。", "可以。"))
    assert commit is not None, "mechanical commit was not created"
    first.close()

    reopened = AureliusMemory(root, "acceptance", auto_commit_every=2)
    text = _history_text(reopened)
    assert token in text, "persisted user fact was not restored"
    assert "已记录" in text, "persisted Ghost response was not restored"
    state = reopened.inspect()
    assert state["commit_count"] == 1, state
    reopened.close()

    print("PASS: AureliusMemory write -> commit -> close -> reopen -> render")
    print(f"root={root}")
    print(f"commit_count={state['commit_count']} staging_count={state['staging_count']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, help="Keep and inspect a chosen memento root")
    args = parser.parse_args()
    if args.root:
        verify(args.root.expanduser().resolve())
        return
    with tempfile.TemporaryDirectory(prefix="moss-aurelius-memory-") as directory:
        verify(Path(directory))


if __name__ == "__main__":
    main()
