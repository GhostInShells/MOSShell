
"""Hook discovery and execution — the four things that must not silently rot.

Nothing here mocks the mechanism: a real file is written and a real subprocess runs it.
The tests are slow by nature (one interpreter spawn each) and that is the point — the
contract is "write a file, it runs", and a fake would not prove it.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from ghoshell_moss.memento.hooks import (
    HOOK_DIR_NAME,
    HOOK_SLOTS,
    hook_env,
    hook_paths,
    render_view_text,
    run_slot,
)


def _write_hook(root: Path, slot: str, body: str) -> Path:
    directory = root / HOOK_DIR_NAME
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{slot}.py"
    path.write_text(body, encoding="utf-8")
    return path


@pytest.fixture()
def roots(tmp_path: Path) -> tuple[Path, Path]:
    """(branch dir, ego root) — the two places a slot can hang."""
    branch = tmp_path / "branches" / "b1"
    branch.mkdir(parents=True)
    ego = tmp_path / "ego"
    ego.mkdir()
    return branch, ego


def test_only_mounted_slots_are_found(roots: tuple[Path, Path]) -> None:
    branch, ego = roots
    assert hook_paths([branch, ego], "on_commit") == []

    mounted = _write_hook(ego, "on_commit", "print('hi')\n")
    assert hook_paths([branch, ego], "on_commit") == [mounted]
    # 没挂的槽位仍然空着 —— 文件在不在, 就是开没开.
    assert hook_paths([branch, ego], "commit_view") == []


def test_branch_file_does_not_shadow_the_ego_one(roots: tuple[Path, Path]) -> None:
    branch, ego = roots
    own = _write_hook(branch, "branch_view", "print('own')\n")
    default = _write_hook(ego, "branch_view", "print('default')\n")

    found = hook_paths([branch, ego], "branch_view")
    assert found == [own, default], "覆盖是叠加, 不是替换"


def test_protocol_env_carries_the_promised_directory(roots: tuple[Path, Path], tmp_path: Path) -> None:
    branch, ego = roots
    commit_dir = tmp_path / "commits" / "cmt_1-9"
    env = hook_env(
        "on_commit",
        root=ego,
        branch="main",
        branch_dir=branch,
        coord="1-9",
        commit_dir=commit_dir,
    )
    assert env["MEMENTO_HOOK_SLOT"] == "on_commit"
    assert env["MEMENTO_COMMIT_DIR"] == str(commit_dir.resolve())
    assert env["MEMENTO_BRANCH_DIR"] == str(branch.resolve())
    assert env["MEMENTO_COORD"] == "1-9"
    assert env["MEMENTO_BRANCH"] == "main"


def test_a_hook_can_write_where_the_protocol_says(roots: tuple[Path, Path], tmp_path: Path) -> None:
    branch, ego = roots
    commit_dir = tmp_path / "commits" / "cmt_1-9"
    commit_dir.mkdir(parents=True)
    _write_hook(
        ego,
        "on_commit",
        "import os, pathlib\n"
        "pathlib.Path(os.environ['MEMENTO_COMMIT_DIR'], 'left-behind.md').write_text('x')\n",
    )

    results = asyncio.run(
        run_slot(
            "on_commit",
            [branch, ego],
            env_base=hook_env("on_commit", root=ego, branch="main", branch_dir=branch, coord="1-9", commit_dir=commit_dir),
            log_root=ego,
        )
    )

    assert [r.ok for r in results] == [True]
    assert (commit_dir / "left-behind.md").read_text() == "x"


def test_a_failure_is_recorded_and_does_not_stop_the_other(roots: tuple[Path, Path]) -> None:
    branch, ego = roots
    _write_hook(branch, "commit_view", "raise SystemExit(3)\n")
    _write_hook(ego, "commit_view", "print('survived')\n")

    results = asyncio.run(run_slot("commit_view", [branch, ego], env_base={}, log_root=ego))

    assert [r.ok for r in results] == [False, True]
    # 失败的那一条不进页面, 但页面照样出 —— 读面不因旁路而死.
    assert render_view_text(results) == "survived\n\n[hook] commit_view: exit 3"

    log = (ego / HOOK_DIR_NAME / ".log.jsonl").read_text().strip().splitlines()
    assert len(log) == 2


def test_a_slow_hook_is_cut_at_its_deadline(roots: tuple[Path, Path]) -> None:
    branch, ego = roots
    _write_hook(ego, "branch_view", "import time\ntime.sleep(5)\nprint('too late')\n")

    results = asyncio.run(run_slot("branch_view", [branch, ego], env_base={}, timeout=0.5))

    assert results[0].ok is False
    assert "timeout" in results[0].error


def test_every_slot_name_is_a_file_name() -> None:
    """槽位的名字就是文件名; 改名等于换协议, 所以把它钉住."""
    assert HOOK_SLOTS == ("on_commit", "commit_view", "branch_view")
