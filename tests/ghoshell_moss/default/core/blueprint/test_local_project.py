"""LocalProject 单元测试 — ghosts / modes 发现、cells 构造、缓存."""

import sys
from pathlib import Path

import pytest

from ghoshell_moss.core.blueprint.environment import (
    Environment, DEFAULT_WORKSPACE_DIR_NAME, MossMeta, MOSS_META_FILE,
)
from ghoshell_moss.core.blueprint.project import HostModeMeta, HOST_MODE_FILE
from ghoshell_moss.core.blueprint.ghost import GhostMeta
from ghoshell_moss.project.local_project import LocalProject
from ghoshell_moss.project.local_host_mode import LocalHostMode


def _minimal_project(tmp_path: Path) -> tuple[Path, Environment]:
    ws = tmp_path / DEFAULT_WORKSPACE_DIR_NAME
    ws.mkdir()
    env = Environment(workspace=ws)
    return ws, env


class TestLocalProjectModes:

    def test_list_modes_empty_when_no_modes_dir(self, tmp_path):
        ws, env = _minimal_project(tmp_path)
        project = LocalProject(env)
        assert list(project.list_modes()) == []

    def test_list_modes_discovers_from_hierarchy(self, tmp_path):
        ws, env = _minimal_project(tmp_path)
        desktop = ws / 'modes' / 'desktop'
        desktop.mkdir(parents=True)
        (desktop / HOST_MODE_FILE).write_text(
            '---\nname: desktop\ndescription: desktop mode\n---\n# desktop\n'
        )
        robot = ws / 'modes' / 'robot'
        robot.mkdir(parents=True)
        (robot / HOST_MODE_FILE).write_text(
            '---\nname: robot\n---\n# robot\n'
        )

        project = LocalProject(env)
        modes = list(project.list_modes())
        assert len(modes) == 2
        names = {m.name() for _, m in modes}
        assert names == {'desktop', 'robot'}

    def test_list_modes_directory_name_wins(self, tmp_path):
        """HOST.md name 与目录名不一致时, 目录名获胜."""
        ws, env = _minimal_project(tmp_path)
        mode_dir = ws / 'modes' / 'real_name'
        mode_dir.mkdir(parents=True)
        (mode_dir / HOST_MODE_FILE).write_text(
            '---\nname: wrong_name\n---\n# content\n'
        )

        project = LocalProject(env)
        modes = list(project.list_modes())
        assert len(modes) == 1
        assert modes[0][1].name() == 'real_name'

    def test_list_modes_skips_directories_without_host_md(self, tmp_path):
        ws, env = _minimal_project(tmp_path)
        empty_dir = ws / 'modes' / 'not_a_mode'
        empty_dir.mkdir(parents=True)

        project = LocalProject(env)
        assert list(project.list_modes()) == []

    def test_get_mode_returns_local_host_mode(self, tmp_path):
        ws, env = _minimal_project(tmp_path)
        mode_dir = ws / 'modes' / 'desktop'
        mode_dir.mkdir(parents=True)
        (mode_dir / HOST_MODE_FILE).write_text(
            '---\nname: desktop\n---\n# desktop mode\n'
        )

        project = LocalProject(env)
        mode = project.get_mode('desktop')
        assert isinstance(mode, LocalHostMode)
        assert mode.name == 'desktop'
        assert mode.workspace_dir == mode_dir

    def test_get_mode_lookup_error(self, tmp_path):
        ws, env = _minimal_project(tmp_path)
        project = LocalProject(env)
        with pytest.raises(LookupError, match='ghost_mode'):
            project.get_mode('ghost_mode')


class TestLocalProjectGhosts:

    def test_ghosts_empty_when_package_not_available(self, tmp_path):
        ws, env = _minimal_project(tmp_path)
        project = LocalProject(env)
        ghosts = list(project.ghosts())
        assert ghosts == []

    def test_get_ghost_lookup_error(self, tmp_path):
        ws, env = _minimal_project(tmp_path)
        project = LocalProject(env)
        with pytest.raises(LookupError, match='nonexistent'):
            project.get_ghost('nonexistent')


class TestLocalProjectCells:

    def test_cells_returns_registry(self, tmp_path):
        ws, env = _minimal_project(tmp_path)
        project = LocalProject(env)
        cells = project.cells
        assert cells is not None
        cells2 = project.cells
        assert cells is cells2


class TestLocalProjectProperties:

    def test_env_and_workspace(self, tmp_path):
        ws, env = _minimal_project(tmp_path)
        project = LocalProject(env)
        assert project.env is env
        assert project.workspace is not None
