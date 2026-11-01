"""on_commit — move the finished plan and tasks into the anchor that just landed, then speak.

The protocol hands over the branch directory, the commit directory and the coordinate via
MementoHookEnv. Two things happen, both on the bypass: files whose status says the work is
over move into the new anchor, and the ghost is told what moved and what is still open — a
file disappearing without a word looks exactly like a file being lost, and a finished anchor
with three stale tasks is exactly the thing nobody remembers by the next turn.

The anchor's directory exists because the move needs somewhere to land: no file moves, no
directory. Nothing here writes a receipt — the message below is the receipt, and it travels
as a signal rather than as a file in the node space.

Nothing here waits on the kernel, and the kernel waits on nothing here.
"""

import asyncio
import datetime
import pathlib
import shutil

import frontmatter

from ghoshell_moss.memento.hooks import MementoHookEnv

FINISHED = {"done", "dropped", "closed", "completed"}


def metadata(path: pathlib.Path) -> dict:
    """The frontmatter through the library that reads it everywhere else.

    Hand-rolled parsing was wrong once and cost an anchor its archive: a file that opens with a
    blank line still has frontmatter, and only the library knows that.
    """
    try:
        return dict(frontmatter.load(path).metadata)
    except Exception:  # unreadable means no frontmatter; moving a file must not die on a parse
        return {}


def status_of(path: pathlib.Path) -> str:
    return str(metadata(path).get("status", "")).strip()


def age_days(path: pathlib.Path) -> str:
    """How long since this file said it was touched — the question a stale task asks."""
    raw = str(metadata(path).get("updated", "")).strip()
    if not raw:
        return "?"
    try:
        when = datetime.datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return "?"
    now = datetime.datetime.now(when.tzinfo)
    return f"{max((now - when).days, 0)}d"


def archive(branch: pathlib.Path, commit: pathlib.Path) -> list[str]:
    """Move finished work into the anchor — the only thing here that touches the disk.

    The anchor's directory is created by this move and by nothing else: an anchor that takes
    nothing in gets no directory, which is what keeps the node space small enough to walk.
    """
    moved: list[str] = []
    plan = branch / "plan.md"
    targets = [plan] if plan.is_file() else []
    targets += sorted(branch.glob("*.task.md"))
    for path in targets:
        if status_of(path) not in FINISHED:
            continue
        commit.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(commit / path.name))
        moved.append(path.name)
    return moved


def still_open(branch: pathlib.Path) -> tuple[str, list[str]]:
    """What the branch still holds: the plan's status, and every task with its age."""
    plan = branch / "plan.md"
    plan_state = status_of(plan) if plan.is_file() else "—"
    tasks = [
        f"{path.name[: -len('.task.md')]} ({status_of(path) or 'no status'}, {age_days(path)})"
        for path in sorted(branch.glob("*.task.md"))
    ]
    return plan_state, tasks


def notify(message: str, coord: str) -> None:
    """Say it from the bypass: our own process joins the matrix and pushes one notify.

    The message travels as a **container**, not as a bare string: a raw line lands on the same
    layer as the human's own speech and nothing can tell the two apart. A tag is the boundary —
    whatever sits inside ``memento_notice`` was said by this hook, never by him.

    No return value travels back to the kernel — the hook speaks for itself, which is why a
    slow or broken hook can never hold up the commit that already landed.
    """
    from ghoshell_moss.core.blueprint.matrix import Matrix
    from ghoshell_moss.core.blueprint.mindflow import Priority
    from ghoshell_moss.message import Message
    from ghoshell_moss.signals import NotifySignalMeta

    async def _send(matrix) -> None:
        signal = NotifySignalMeta(next=False).to_signal(
            Message.new(
                tag="memento_notice", attributes={"kind": "commit", "coord": coord}
            ).with_content(message),
            description="memento on_commit: what the anchor took, what is still open",
            # A receipt, not a distress call: which files moved, which tasks are still open.
            # No reason to win an attention in flight, none to reserve the next turn.
            # BACKGROUND can never preempt, and a notify that cannot preempt is buffered
            # rather than dropped — exactly the right weight for this.
            priority=Priority.BACKGROUND,
        )
        matrix.session.add_signal(signal)
        await asyncio.sleep(0.5)

    Matrix.new("memento_on_commit").run(_send)


def main() -> None:
    branch = MementoHookEnv.BRANCH_DIR.path()
    commit = MementoHookEnv.COMMIT_DIR.path()
    if branch is None or commit is None:
        return
    coord = MementoHookEnv.COORD.get() or commit.name

    moved = archive(branch, commit)
    plan_state, tasks = still_open(branch)

    # Nothing to report is silence: a notice that says nothing costs the reader an attention.
    if not moved and not tasks and plan_state in ("", "—"):
        return

    lines = [
        "moved into this anchor: " + (", ".join(moved) if moved else "nothing finished"),
        f"plan: {plan_state}",
        "still open: " + ("; ".join(tasks) if tasks else "no task left on the branch"),
    ]
    message = f"[memento] {coord}: " + " · ".join(lines)
    try:
        notify(message, coord)
    except Exception:  # moving the files is the job; failing to speak must not undo it
        pass


if __name__ == "__main__":
    main()