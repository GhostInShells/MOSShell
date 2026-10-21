
"""The default trajectory layout — written once, never overwritten.

The whole point of these files is that the ghost can edit them afterwards, so the test that
matters most is the second run: a default must not undo an edit.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ghoshell_moss.ghosts.dolores._memento_layout import (
    BRANCH_GROUND,
    COMMITS_GROUND,
    ON_COMMIT_HOOK,
    ensure_memento_layout,
)


@pytest.fixture()
def dirs(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "ego"
    branch = root / "branches" / "b1"
    branch.mkdir(parents=True)
    root.mkdir(exist_ok=True)
    return root, branch


def test_writes_the_two_ground_points_and_the_hook(dirs: tuple[Path, Path]) -> None:
    root, branch = dirs
    written = ensure_memento_layout(root, branch)

    assert (branch / "GROUND.md").read_text() == BRANCH_GROUND
    assert (root / "commits" / "GROUND.md").read_text() == COMMITS_GROUND
    assert (root / "hooks" / "on_commit.py").read_text() == ON_COMMIT_HOOK
    assert len(written) == 3


def test_a_second_run_touches_nothing(dirs: tuple[Path, Path]) -> None:
    root, branch = dirs
    ensure_memento_layout(root, branch)
    (branch / "GROUND.md").write_text("edited by the ghost\n", encoding="utf-8")

    assert ensure_memento_layout(root, branch) == []

    assert (branch / "GROUND.md").read_text() == "edited by the ghost\n"


def test_the_hook_is_executable_python(dirs: tuple[Path, Path]) -> None:
    """它会被 memento 自己的解释器当脚本跑 —— 语法错了就是开机才发现."""
    root, branch = dirs
    ensure_memento_layout(root, branch)
    source = (root / "hooks" / "on_commit.py").read_text()
    compile(source, "on_commit.py", "exec")
