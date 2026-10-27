"""One ghost, several homes — the shape the manager exists to hold.

Pins three properties the single-root path could not have:

* each mounted home appears as **one content child**, named by its slot key;
* the home's own ``groundset:`` declarations appear as **its** children (two levels);
* there is exactly **one command surface**, on the root channel — content does not
  come with a second copy of ``open/close/refresh``;
* with several homes, ``ground=`` is required: an unqualified call **errors** and
  lists the homes instead of silently picking a primary one;
* a home without a ``GROUND.md`` is **not mounted**, and says so in the notice.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from ghoshell_moss.channels.ground_channel import new_ground_channel  # noqa: E402
from ghoshell_moss.core.concepts.errors import CommandError, CommandErrorCode  # noqa: E402
from ghoshell_moss.ground._grounds import DefaultGroundSet  # noqa: E402
from ghoshell_moss.ground._manager import GroundSlot, GroundsManager  # noqa: E402

L0 = "---\nname: {name}\ndescription: {name}\n{extra}---\n\n{body}\n"


def _ground(path: Path, name: str, *, groundset: list[str] | None = None, body: str = "body") -> Path:
    path.mkdir(parents=True, exist_ok=True)
    extra = ""
    if groundset:
        extra = "groundset:\n" + "".join(f"- {g}\n" for g in groundset)
    (path / "GROUND.md").write_text(
        L0.format(name=name, extra=extra, body=body), encoding="utf-8")
    return path


def _manager(tmp_path: Path) -> GroundsManager:
    ghost = _ground(tmp_path / "ghost", "ghost-home", groundset=["notes", "journal"])
    _ground(ghost / "notes", "notes")
    _ground(ghost / "journal", "journal")
    project = _ground(tmp_path / "project", "project", groundset=["conventions"])
    _ground(project / "conventions", "conventions")
    mode = tmp_path / "mode"
    mode.mkdir()
    (mode / "HOST.md").write_text("not a ground\n", encoding="utf-8")
    return GroundsManager([
        GroundSlot("ghost_home", ghost),
        GroundSlot("mode_home", mode),
        GroundSlot("project_home", project),
    ])


@pytest.mark.asyncio
async def test_each_home_is_one_content_child_with_its_own_children(tmp_path: Path):
    manager = await _manager(tmp_path).mount(lambda slot: DefaultGroundSet(workspace_root=slot.home, root_label=slot.key))
    chan = new_ground_channel(manager)

    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        names = set(runtime.metas())

    assert "ghost_home" in names and "project_home" in names
    # the home's own `groundset:` is materialized under it — two levels, not a flat pool
    assert "ghost_home.notes" in names
    assert "ghost_home.journal" in names
    assert "project_home.conventions" in names
    # a home without GROUND.md never becomes a channel
    assert "mode_home" not in names


@pytest.mark.asyncio
async def test_absence_is_content_not_silence(tmp_path: Path):
    manager = await _manager(tmp_path).mount(lambda slot: DefaultGroundSet(workspace_root=slot.home, root_label=slot.key))
    chan = new_ground_channel(manager)

    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        notice = runtime.self_meta().notice or ""

    assert "mode_home: absent" in notice
    assert "ghost_home: ok" in notice and "project_home: ok" in notice


@pytest.mark.asyncio
async def test_the_command_surface_has_exactly_one_copy(tmp_path: Path):
    manager = await _manager(tmp_path).mount(lambda slot: DefaultGroundSet(workspace_root=slot.home, root_label=slot.key))
    chan = new_ground_channel(manager)

    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        root_commands = {c.name for c in runtime.self_meta().commands}
        child_commands = {
            name: [c.name for c in meta.commands]
            for name, meta in runtime.metas().items()
            if name.startswith("ghost_home") or name.startswith("project_home")
        }

    assert {"open", "close", "refresh", "stats", "ground_run"} <= root_commands
    # 读侧归 CLI, 命令面不重复实现
    assert {"render", "validate", "templates", "spec"}.isdisjoint(root_commands)
    assert all(not cmds for cmds in child_commands.values()), child_commands


@pytest.mark.asyncio
async def test_unqualified_call_never_picks_a_primary_home(tmp_path: Path):
    """Two homes mounted → an unqualified path has two answers, so it must not guess."""
    manager = await _manager(tmp_path).mount(lambda slot: DefaultGroundSet(workspace_root=slot.home, root_label=slot.key))
    chan = new_ground_channel(manager)

    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        with pytest.raises(CommandError) as err:
            await runtime.execute_command("open", kwargs={"directory": "notes"})
        # "你少传了一个参数"不该升级成致命 (505): 它自己的 code 要说清是哪一类.
        assert err.value.code == CommandErrorCode.INVALID_USAGE.value
        assert "ground=" in str(err.value)
        assert "ghost_home" in str(err.value) and "project_home" in str(err.value)


@pytest.mark.asyncio
async def test_naming_a_home_resolves_without_ambiguity(tmp_path: Path):
    manager = await _manager(tmp_path).mount(lambda slot: DefaultGroundSet(workspace_root=slot.home, root_label=slot.key))
    chan = new_ground_channel(manager)

    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command(
            "open", kwargs={"directory": "conventions", "ground": "project_home"}
        )
        assert "opened conventions" in result


@pytest.mark.asyncio
async def test_stats_gives_each_home_one_name_and_its_path(tmp_path: Path):
    """一个 home 一个身份: 行首是 slot key, 行内是绝对路径; 根场不再另起一行."""
    manager = await _manager(tmp_path).mount(lambda slot: DefaultGroundSet(workspace_root=slot.home, root_label=slot.key))
    chan = new_ground_channel(manager)

    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command("stats")

    assert f"ghost_home: {tmp_path / 'ghost'}" in result
    assert str(tmp_path / "project") in result
    # 缺席那层也在, 且带原因 (不是静默消失)
    assert f"mode_home: {tmp_path / 'mode'} — mode_home: absent" in result
    # 根场不以目录名再占一行
    assert "\n  ghost:" not in result
