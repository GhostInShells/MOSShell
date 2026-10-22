"""on_commit — move the finished plan and tasks into the anchor that just landed, then speak.

The protocol hands over MEMENTO_BRANCH_DIR, MEMENTO_COMMIT_DIR and MEMENTO_COORD. Two things
happen, both on the bypass: files whose status says the work is over move into the new anchor,
and the ghost is told what moved and what is still open — a file disappearing without a word
looks exactly like a file being lost, and a finished anchor with three stale tasks is exactly
the thing nobody remembers by the next turn.

Every line this prints is a hint the ghost wrote for itself. Nothing here waits on the kernel,
and the kernel waits on nothing here.
"""

import asyncio
import datetime
import os
import pathlib
import shutil

import frontmatter

FINISHED = {"done", "dropped", "closed", "completed"}


def metadata(path: pathlib.Path) -> dict:
    """The frontmatter through the library that reads it everywhere else.

    Hand-rolled parsing was wrong once and cost an anchor its archive: a file that opens with a
    blank line still has frontmatter, and only the library knows that.
    """
    try:
        return dict(frontmatter.load(path).metadata)
    except Exception:  # 读不了就当没有 frontmatter; 搬文件不该因为解析失败而炸
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


def record(commit: pathlib.Path, coord: str, lines: list[str]) -> None:
    """Leave the verdict in the anchor's own directory — a reader sees it without asking."""
    commit.mkdir(parents=True, exist_ok=True)
    (commit / "on_commit.md").write_text(
        f"# on_commit — {coord}\n\n" + "\n".join(lines) + "\n",
        encoding="utf-8",
    )


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
            # 这是回执，不是呼救。它说的是「搬走了哪些文件、还开着哪些任务」——
            # 没有任何理由让它赢下一次正在进行的注意力，也没有理由让它预约下一轮。
            # BACKGROUND 永远挑战不成功；notify 抢不到时会缓冲而不是丢弃，正好。
            priority=Priority.BACKGROUND,
        )
        matrix.session.add_signal(signal)
        await asyncio.sleep(0.5)

    Matrix.new("memento_on_commit").run(_send)


def main() -> None:
    branch = pathlib.Path(os.environ["MEMENTO_BRANCH_DIR"])
    commit = pathlib.Path(os.environ["MEMENTO_COMMIT_DIR"])
    coord = os.environ.get("MEMENTO_COORD") or commit.name

    moved = archive(branch, commit)
    plan_state, tasks = still_open(branch)

    # 没事可报就不吭声: 一条什么都不说的通知, 比没有通知更费人一次注意力.
    if not moved and not tasks and plan_state in ("", "—"):
        return

    lines = [
        "moved into this anchor: " + (", ".join(moved) if moved else "nothing finished"),
        f"plan: {plan_state}",
        "still open: " + ("; ".join(tasks) if tasks else "no task left on the branch"),
    ]
    record(commit, coord, lines)
    message = f"[memento] {coord}: " + " · ".join(lines)
    print(message)
    try:
        notify(message, coord)
    except Exception:  # 说不出去也得把文件搬对; 旁路不许因为说话失败而挡人
        pass


if __name__ == "__main__":
    main()
