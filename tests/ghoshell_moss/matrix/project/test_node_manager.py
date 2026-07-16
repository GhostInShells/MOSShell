"""
ProjectNodeManager inventory 单测.

覆盖只读发现: scan / 匹配过滤 / installed 过滤 / 缓存 / launcher 装配.
不测 spawn/kill/runtime file — 那些已彻底离开 registry (§UU-6/UU-9).
"""

import textwrap
from pathlib import Path
from unittest.mock import Mock

import pytest

from ghoshell_moss.core.blueprint.cell import NodeManifest
from ghoshell_moss.project.node_manager import ProjectNodeManager


def _write_cell(directory: Path, name: str, category: str = '', extra: str = '') -> None:
    directory.mkdir(parents=True, exist_ok=True)
    frontmatter = f"name: {name}\n"
    if category:
        frontmatter += f"category: {category}\n"
    frontmatter += "run: python main.py\n"
    frontmatter += extra
    (directory / NodeManifest.MANIFEST_FILENAME).write_text(
        f"---\n{frontmatter}---\n{name} body\n",
    )


@pytest.fixture
def project(tmp_path: Path):
    apps = tmp_path / 'apps'
    _write_cell(apps / 'sensors' / 'audio_capture', 'audio_capture', 'sensors')
    _write_cell(apps / 'sensors' / 'vision', 'vision', 'sensors')
    _write_cell(apps / 'bodies' / 'g1', 'g1_body', 'bodies')
    _write_cell(apps / 'tools' / 'screen', 'screen', 'tools')
    # 未安装的 cell — 安装说明文件存在但完成标记不存在.
    d = apps / 'tools' / 'not_ready'
    _write_cell(d, 'not_ready', 'tools')
    (d / NodeManifest.INSTALL_FILENAME).write_text('pip install foo')
    # cell 目录内部的子目录不再递归扫描 (walk 里的 continue 保证).
    (apps / 'sensors' / 'audio_capture' / 'nested').mkdir()
    _write_cell(apps / 'sensors' / 'audio_capture' / 'nested', 'nested_should_not_appear')

    env = Mock()
    env.project_path = tmp_path
    env.project_id = 'proj-x'
    env.project_name = 'projx'
    env.this_cell_address = ''
    env.dump_cell_env = Mock(return_value={})
    return env, [apps]


class TestList:

    def test_scans_all_cell_manifests(self, project):
        env, dirs = project
        reg = ProjectNodeManager(env, dirs)
        result = reg.list_nodes()
        keys = set(result.keys())
        assert 'apps/sensors/audio_capture' in keys
        assert 'apps/sensors/vision' in keys
        assert 'apps/bodies/g1' in keys
        assert 'apps/tools/screen' in keys
        # 未安装的仍然出现 (installed=None 默认返回全部, WW-5 故事 3).
        assert 'apps/tools/not_ready' in keys

    def test_does_not_recurse_into_cell_dir(self, project):
        env, dirs = project
        reg = ProjectNodeManager(env, dirs)
        result = reg.list_nodes()
        assert not any('nested' in k for k in result.keys())

    def test_installed_filter_true(self, project):
        env, dirs = project
        reg = ProjectNodeManager(env, dirs)
        result = reg.list_nodes(installed=True)
        assert 'apps/tools/not_ready' not in result
        assert 'apps/sensors/vision' in result

    def test_installed_filter_false(self, project):
        env, dirs = project
        reg = ProjectNodeManager(env, dirs)
        result = reg.list_nodes(installed=False)
        assert list(result.keys()) == ['apps/tools/not_ready']

    def test_include_pattern(self, project):
        env, dirs = project
        reg = ProjectNodeManager(env, dirs)
        result = reg.list_nodes(include=['apps/sensors/*'])
        assert set(result.keys()) == {'apps/sensors/audio_capture', 'apps/sensors/vision'}

    def test_exclude_pattern(self, project):
        env, dirs = project
        reg = ProjectNodeManager(env, dirs)
        result = reg.list_nodes(exclude=['apps/tools/*'])
        assert 'apps/tools/screen' not in result
        assert 'apps/sensors/audio_capture' in result

    def test_cache_reused_without_refresh(self, project):
        env, dirs = project
        reg = ProjectNodeManager(env, dirs)
        _ = reg.list_nodes()
        # 添加新 cell, 但不 refresh → 不应出现.
        _write_cell(dirs[0] / 'tools' / 'new_cell', 'new_cell', 'tools')
        result = reg.list_nodes(refresh=False)
        assert 'apps/tools/new_cell' not in result

        result_refreshed = reg.list_nodes(refresh=True)
        assert 'apps/tools/new_cell' in result_refreshed

    def test_paths_override_default_dirs(self, project, tmp_path):
        # 默认根目录 apps/ 之外单独造一个 cell, 用 paths= override.
        env, dirs = project
        extra = tmp_path / 'extra'
        _write_cell(extra / 'foo', 'foo', 'sensors')
        reg = ProjectNodeManager(env, dirs)
        # 默认扫描不含 extra/.
        assert 'extra/foo' not in reg.list_nodes()
        # override 后只扫 extra/, 默认 apps/ 里的不应出现.
        result = reg.list_nodes(paths=[extra])
        assert 'extra/foo' in result
        assert 'apps/sensors/audio_capture' not in result

    def test_paths_override_bypasses_cache(self, project, tmp_path):
        # override 分支不写默认缓存, 也不读默认缓存.
        env, dirs = project
        reg = ProjectNodeManager(env, dirs)
        _ = reg.list_nodes()  # 预热默认缓存
        extra = tmp_path / 'extra'
        _write_cell(extra / 'foo', 'foo', 'sensors')
        # 即使 refresh=False, override 也应重新扫描 (paths 优先).
        result = reg.list_nodes(refresh=False, paths=[extra])
        assert 'extra/foo' in result


class TestGet:

    def test_get_by_relative_path(self, project):
        env, dirs = project
        reg = ProjectNodeManager(env, dirs)
        m = reg.get_node('apps/sensors/audio_capture')
        assert m is not None
        assert m.name == 'audio_capture'
        assert m.category == 'sensors'

    def test_get_missing_returns_none(self, project):
        env, dirs = project
        reg = ProjectNodeManager(env, dirs)
        assert reg.get_node('apps/nonexistent') is None


class TestLauncher:
    """get_node_launcher 是 NodeLauncher.from_manifest 的薄组合."""

    def test_get_launcher_returns_ready_launcher(self, project):
        env, dirs = project
        reg = ProjectNodeManager(env, dirs)
        launcher = reg.get_node_launcher('apps/sensors/audio_capture')
        assert launcher is not None
        # cwd = 声明文件所在目录 (NodeManifest.cwd 承诺).
        assert launcher.cwd == (dirs[0] / 'sensors' / 'audio_capture').resolve()
        # run = [command, *arguments]; fixture 声明 `run: python main.py`,
        # ExecSpec 默认 command='python' → NodeLauncher 里 hardcode 替换 sys.executable.
        assert len(launcher.run) >= 1
        assert launcher.run[0].endswith('python') or 'python' in launcher.run[0]
        # runtime 是待回填的 CellRuntimeInfo, pid/pgid 由 spawner 填.
        assert launcher.runtime.pid == 0
        assert launcher.runtime.cell.name == 'audio_capture'
        # dump_cell_env 被调, 返回值用作 launcher.env.
        env.dump_cell_env.assert_called_once()
        assert launcher.env == {}

    def test_get_launcher_missing_returns_none(self, project):
        env, dirs = project
        reg = ProjectNodeManager(env, dirs)
        assert reg.get_node_launcher('apps/nonexistent') is None


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
        reg = ProjectNodeManager(env, dirs)
        assert not hasattr(reg, banned), (
            f"ProjectNodeManager leaked runtime method {banned!r} — "
            "spawn/kill/runtime should live at run_node throat / CLI ledger reader."
        )
