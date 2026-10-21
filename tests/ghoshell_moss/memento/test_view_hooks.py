"""View hooks belong to the view, not to whoever renders it.

D21: a plan written into the branch directory never reached the boot memory, because only
``memento:view`` knew to append the ``branch_view`` slot — the generation itself did not.
These tests pin the fix where the contract now lives: ``aview_with_hooks`` /
``aget_commit_with_hooks`` run the slot and carry the result in ``ground``, and they re-run
it on every call (a view is a snapshot of a moving directory).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ghoshell_moss.memento import new_local_memento


def _mount(branch_dir: Path, slot: str, body: str) -> None:
    hooks = branch_dir / "hooks"
    hooks.mkdir(parents=True, exist_ok=True)
    (hooks / f"{slot}.py").write_text(body, encoding="utf-8")


@pytest.fixture()
def branch(tmp_path: Path):
    memento = new_local_memento(tmp_path / "owner")
    memento.create_branch("main")
    return memento.get_branch("main")


@pytest.mark.asyncio
async def test_branch_view_carries_what_the_directory_renders(branch) -> None:
    (branch.path / "plan.md").write_text("# the plan\n", encoding="utf-8")
    _mount(
        branch.path,
        "branch_view",
        "import os, pathlib\n"
        "d = pathlib.Path(os.environ['MEMENTO_BRANCH_DIR'])\n"
        "print((d / 'plan.md').read_text())\n",
    )
    branch.commit(message="anchor")

    view = await branch.aview_with_hooks()

    assert "# the plan" in view.ground


@pytest.mark.asyncio
async def test_plain_view_stays_hook_free(branch) -> None:
    _mount(branch.path, "branch_view", "print('PLAN')\n")
    branch.commit(message="anchor")

    assert (await branch.aview()).ground == "", "裸投影不该跑 hook"
    assert branch.view().ground == ""


@pytest.mark.asyncio
async def test_every_generation_reruns_the_hook(branch) -> None:
    plan = branch.path / "plan.md"
    plan.write_text("first\n", encoding="utf-8")
    _mount(
        branch.path,
        "branch_view",
        "import os, pathlib\n"
        "d = pathlib.Path(os.environ['MEMENTO_BRANCH_DIR'])\n"
        "print((d / 'plan.md').read_text())\n",
    )
    branch.commit(message="anchor")

    assert "first" in (await branch.aview_with_hooks()).ground
    plan.write_text("second\n", encoding="utf-8")
    assert "second" in (await branch.aview_with_hooks()).ground, "缓存下来的 plan 是错的 plan"


@pytest.mark.asyncio
async def test_commit_view_carries_its_directory_rendering(branch) -> None:
    anchor = branch.commit(message="anchor")
    _mount(
        branch.path,
        "commit_view",
        "import os\nprint('COORD ' + os.environ['MEMENTO_COORD'])\n",
    )

    view = await branch.aget_commit_with_hooks(1)

    assert view is not None and "COORD 1-1" in view.ground


@pytest.mark.asyncio
async def test_a_broken_hook_does_not_take_the_view_with_it(branch) -> None:
    _mount(branch.path, "branch_view", "raise SystemExit(2)\n")
    branch.commit(message="anchor")

    view = await branch.aview_with_hooks()

    assert view.commits_total == 1, "hook 失败不该让 view 生成不出来"
