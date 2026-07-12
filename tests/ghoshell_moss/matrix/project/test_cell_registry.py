"""
ProjectCellRegistry inventory 单测.

覆盖只读发现: scan / 匹配过滤 / installed 过滤 / 缓存.
不测 spawn/kill/runtime file — 那些已彻底离开 registry (§UU-6/UU-9).
"""

import textwrap
from pathlib import Path
from unittest.mock import Mock

import pytest

from ghoshell_moss.core.blueprint.cell import CellManifest
from ghoshell_moss.project.cell_registry import ProjectCellRegistry


def _write_cell(directory: Path, name: str, taxonomy: str = '', extra: str = '') -> None:
    directory.mkdir(parents=True, exist_ok=True)
    frontmatter = f"name: {name}\n"
    if taxonomy:
        frontmatter += f"taxonomy: {taxonomy}\n"
    frontmatter += "run: python main.py\n"
    frontmatter += extra
    (directory / 'CELL.md').write_text(
        f"---\n{frontmatter}---\n{name} body\n",
    )


@pytest.fixture
def project(tmp_path: Path):
    apps = tmp_path / 'apps'
    _write_cell(apps / 'sensors' / 'audio_capture', 'audio_capture', 'sensors')
    _write_cell(apps / 'sensors' / 'vision', 'vision', 'sensors')
    _write_cell(apps / 'bodies' / 'g1', 'g1_body', 'bodies')
    _write_cell(apps / 'tools' / 'screen', 'screen', 'tools')
    # 未安装的 cell — INSTALL.md 存在但 .installed 不存在.
    d = apps / 'tools' / 'not_ready'
    _write_cell(d, 'not_ready', 'tools')
    (d / 'INSTALL.md').write_text('pip install foo')
    # cell 目录内部的子目录不再递归扫描 (walk 里的 continue 保证).
    (apps / 'sensors' / 'audio_capture' / 'nested').mkdir()
    _write_cell(apps / 'sensors' / 'audio_capture' / 'nested', 'nested_should_not_appear')

    env = Mock()
    env.project_path = tmp_path
    return env, [apps]


class TestList:

    def test_scans_all_cell_manifests(self, project):
        env, dirs = project
        reg = ProjectCellRegistry(env, dirs)
        result = reg.list_cell_manifests()
        keys = set(result.keys())
        assert 'apps/sensors/audio_capture' in keys
        assert 'apps/sensors/vision' in keys
        assert 'apps/bodies/g1' in keys
        assert 'apps/tools/screen' in keys
        # 未安装的仍然出现 (installed=None 默认返回全部, WW-5 故事 3).
        assert 'apps/tools/not_ready' in keys

    def test_does_not_recurse_into_cell_dir(self, project):
        env, dirs = project
        reg = ProjectCellRegistry(env, dirs)
        result = reg.list_cell_manifests()
        assert not any('nested' in k for k in result.keys())

    def test_installed_filter_true(self, project):
        env, dirs = project
        reg = ProjectCellRegistry(env, dirs)
        result = reg.list_cell_manifests(installed=True)
        assert 'apps/tools/not_ready' not in result
        assert 'apps/sensors/vision' in result

    def test_installed_filter_false(self, project):
        env, dirs = project
        reg = ProjectCellRegistry(env, dirs)
        result = reg.list_cell_manifests(installed=False)
        assert list(result.keys()) == ['apps/tools/not_ready']

    def test_include_pattern(self, project):
        env, dirs = project
        reg = ProjectCellRegistry(env, dirs)
        result = reg.list_cell_manifests(include=['apps/sensors/*'])
        assert set(result.keys()) == {'apps/sensors/audio_capture', 'apps/sensors/vision'}

    def test_exclude_pattern(self, project):
        env, dirs = project
        reg = ProjectCellRegistry(env, dirs)
        result = reg.list_cell_manifests(exclude=['apps/tools/*'])
        assert 'apps/tools/screen' not in result
        assert 'apps/sensors/audio_capture' in result

    def test_cache_reused_without_refresh(self, project):
        env, dirs = project
        reg = ProjectCellRegistry(env, dirs)
        _ = reg.list_cell_manifests()
        # 添加新 cell, 但不 refresh → 不应出现.
        _write_cell(dirs[0] / 'tools' / 'new_cell', 'new_cell', 'tools')
        result = reg.list_cell_manifests(refresh=False)
        assert 'apps/tools/new_cell' not in result

        result_refreshed = reg.list_cell_manifests(refresh=True)
        assert 'apps/tools/new_cell' in result_refreshed


class TestGet:

    def test_get_by_relative_path(self, project):
        env, dirs = project
        reg = ProjectCellRegistry(env, dirs)
        m = reg.get_cell_manifest('apps/sensors/audio_capture')
        assert m is not None
        assert m.name == 'audio_capture'
        assert m.taxonomy == 'sensors'

    def test_get_missing_returns_none(self, project):
        env, dirs = project
        reg = ProjectCellRegistry(env, dirs)
        assert reg.get_cell_manifest('apps/nonexistent') is None


class TestRegistryHasNoRuntimeMethods:
    """§UU-6/UU-9: spawn/kill/runtime 全部离开 registry."""

    @pytest.mark.parametrize('banned', [
        'spawn_cell',
        'kill_all_runtime_cells',
        'dump_spawn_env',
        'cell_runtimes_dir',
        'local_runtime_cells',
        'recursively_kill_process',
    ])
    def test_no_process_methods(self, project, banned):
        env, dirs = project
        reg = ProjectCellRegistry(env, dirs)
        assert not hasattr(reg, banned), (
            f"ProjectCellRegistry leaked runtime method {banned!r} — "
            "spawn/kill/runtime should live at run_cell throat / CLI ledger reader."
        )
