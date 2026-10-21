
"""The default shape of an ego trajectory — grounds and hooks, written once per ghost home.

A trajectory is a set of directories, and two of them carry a cognitive field: the branch
directory answers "what is this line now", the commit directory answers "what did it leave
behind". One more file decides what a commit leaves behind. All three are conventions, not
configuration: they are written when missing and never overwritten afterwards — the files
belong to whoever edits them next, which is the ghost itself.

This is the one place those defaults are spelled out. Whatever is already on disk wins.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["ensure_memento_layout", "BRANCH_GROUND", "COMMITS_GROUND", "ON_COMMIT_HOOK"]


BRANCH_GROUND = """---
name: branch
description: What this line is right now — its plan, its tasks, and how they get archived
pins:
- label: plan
  verb: file
  arguments: {path: plan.md, budget: 1200}
  description: The plan for this stretch; its status lives in the frontmatter
- label: tasks
  verb: frontmatter
  arguments: {path: "*.task.md", keys: [name, status, updated], limit: 15}
  description: Which tasks exist, where they stand, and when they last moved
---

# The working surface of this line

Branch view's ground point: rendering stops here and renders this directory only.

**plan.md** — one per line, answering "what are we doing". **`slug.task.md`** — one per task.

```md
---
status: active          # active | done | dropped  (closed / completed also count)
created: 2026-10-21T16:00+08:00
updated: 2026-10-21T16:05+08:00
---
```

Tasks add `name`; nothing else is required.

`updated` is part of the convention: touch the file, bump it. The pins show it, so a task's
age is visible before its text is — a week-stale task is a question, not work in progress.

`done` / `dropped` get archived; anything else counts as active — better to keep a file than
to move one by mistake.

## Archiving is not my job

When an anchor lands, `on_commit` moves the finished plan and tasks into the new commit
directory (`commits/…/cmt_{coord}/`). Two things are mine: **write the status correctly**,
then **commit as usual**. Moving files must not depend on my remembering — remembering fails,
a file does not.

## Reading discipline

The pins above give existence and status only. A line with ten tasks would fill the screen:
**open the file when you need its text**, never expect the ground to unfold it for you.
"""


COMMITS_GROUND = """---
name: commits
description: What one anchor was — what it left behind
pins:
- label: contents
  verb: ls
  arguments: {path: "$CWD", depth: 2, limit: 40}
  description: What this anchor's directory holds
---

# The anchor directory

This is the commit view's ground point. It is rendered only when one commit is read in
detail: resolution walks up to `commits/` and stops, and **only the current anchor's
directory** is rendered — there is no per-commit copy of this ground.

## What is inside

- The plan and tasks whose status had ended in that stretch, moved in by `on_commit`.
- Whatever a hook wrote itself.
- Not every anchor has a directory — empty is not broken.
"""


ON_COMMIT_HOOK = '''"""on_commit — move the finished plan and tasks into the anchor that just landed.

The protocol hands over MEMENTO_BRANCH_DIR, MEMENTO_COMMIT_DIR and MEMENTO_COORD. This does
one thing: if a status says the work is over, the file is archived. One criterion only —
write the status correctly and it moves.
"""

import os
import pathlib
import shutil

FINISHED = {"done", "dropped", "closed", "completed"}


def status_of(path: pathlib.Path) -> str:
    """Read `status` from the frontmatter; no frontmatter means no status."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return ""
    if not text.startswith("---"):
        return ""
    end = text.find("\\n---", 3)
    if end < 0:
        return ""
    for line in text[3:end].splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition(":")
        if key.strip() == "status":
            return value.strip().strip("'\\"")
    return ""


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


def main() -> None:
    branch = pathlib.Path(os.environ["MEMENTO_BRANCH_DIR"])
    commit = pathlib.Path(os.environ["MEMENTO_COMMIT_DIR"])
    moved = archive(branch, commit)
    if moved:
        print(f"archived into {commit.name}: " + ", ".join(moved))


if __name__ == "__main__":
    main()
'''


def _write_if_missing(path: Path, text: str) -> bool:
    """A default is written once. After that the file belongs to whoever edited it."""
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return True


def ensure_memento_layout(root: Path, branch_dir: Path) -> list[str]:
    """Put the two ground points and the default hook in place; missing ones only.

    Returns the paths that were created — enough for a caller to say what it did without
    reading the disk back.
    """
    created: list[tuple[Path, str]] = [
        (branch_dir / "GROUND.md", BRANCH_GROUND),
        (root / "commits" / "GROUND.md", COMMITS_GROUND),
        (root / "hooks" / "on_commit.py", ON_COMMIT_HOOK),
    ]
    written: list[str] = []
    for path, text in created:
        if _write_if_missing(path, text):
            written.append(str(path))
    return written
