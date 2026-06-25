"""ProjectCellRegistry 单元测试 — 发现、读写、缓存、过滤.

只测文件系统操作，不测 spawn / kill 等进程副作用。
"""

import os
import tempfile
from pathlib import Path

import pytest

from ghoshell_moss.core.blueprint.cell import (
    CellRegistry,
    CellManifest,
    CellMetadata,
    CellLauncher,
    Cell,
    normalize,
)
from ghoshell_moss.core.blueprint.environment import Environment
from ghoshell_moss.project.cell_registry import ProjectCellRegistry


# ==================================================================
# helpers
# ==================================================================

def _write_cell_md(directory: Path, name: str, **kwargs) -> Path:
    """在 directory 下写一个 CELL.md，返回 directory path."""
    directory.mkdir(parents=True, exist_ok=True)
    manifest = CellManifest(
        type=kwargs.pop('type', 'worker'),
        name=name,
        description=kwargs.pop('description', f'{name} description'),
        launcher=CellLauncher(),
        **kwargs,
    )
    manifest.write_file(directory)
    return directory


def _make_env(tmp_path: Path, **kwargs) -> Environment:
    """创建一个测试用 Environment，workspace 在 tmp_path 内."""
    ws = tmp_path / '.moss_ws'
    ws.mkdir(parents=True, exist_ok=True)
    return Environment(workspace=ws, project=tmp_path, **kwargs)


# ==================================================================
# 构造
# ==================================================================

class TestConstruction:
    def test_basic(self, tmp_path):
        env = _make_env(tmp_path)
        reg = ProjectCellRegistry(env, cell_dirs=[])
        assert reg.cell_runtimes_dir == env.cell_runtimes_dir

    def test_cell_runtimes_dir(self, tmp_path):
        env = _make_env(tmp_path)
        reg = ProjectCellRegistry(env, cell_dirs=[])
        assert reg.cell_runtimes_dir == env.workspace_path / 'runtime' / 'cells'


# ==================================================================
# list_cell_manifests — 发现
# ==================================================================

class TestListCellManifests:
    def test_empty_dirs_returns_empty(self, tmp_path):
        env = _make_env(tmp_path)
        cells_dir = tmp_path / 'cells'
        cells_dir.mkdir()
        reg = ProjectCellRegistry(env, cell_dirs=[cells_dir])
        assert reg.list_cell_manifests() == {}

    def test_cell_dir_not_exists(self, tmp_path):
        env = _make_env(tmp_path)
        reg = ProjectCellRegistry(env, cell_dirs=[tmp_path / 'nonexistent'])
        assert reg.list_cell_manifests() == {}

    def test_single_cell(self, tmp_path):
        env = _make_env(tmp_path)
        cells_dir = tmp_path / 'cells'
        _write_cell_md(cells_dir / 'tools' / 'web-fetch', name='web-fetch')

        reg = ProjectCellRegistry(env, cell_dirs=[cells_dir])
        result = reg.list_cell_manifests()

        assert len(result) == 1
        key = 'cells/tools/web-fetch'
        assert key in result
        assert result[key].name == 'web-fetch'

    def test_multiple_cells(self, tmp_path):
        env = _make_env(tmp_path)
        cells_dir = tmp_path / 'cells'
        _write_cell_md(cells_dir / 'tools' / 'web-fetch', name='web-fetch')
        _write_cell_md(cells_dir / 'tools' / 'image-process', name='image-process')
        _write_cell_md(cells_dir / 'robots' / 'unitree' / 'g1', name='g1')

        reg = ProjectCellRegistry(env, cell_dirs=[cells_dir])
        result = reg.list_cell_manifests()

        assert len(result) == 3
        assert 'cells/tools/web-fetch' in result
        assert 'cells/tools/image-process' in result
        assert 'cells/robots/unitree/g1' in result

    def test_skips_hidden_dirs(self, tmp_path):
        env = _make_env(tmp_path)
        cells_dir = tmp_path / 'cells'
        _write_cell_md(cells_dir / 'tools' / 'web-fetch', name='web-fetch')
        _write_cell_md(cells_dir / '.venv' / 'lib' / 'fake-cell', name='fake')

        reg = ProjectCellRegistry(env, cell_dirs=[cells_dir])
        result = reg.list_cell_manifests()

        assert len(result) == 1
        assert 'cells/tools/web-fetch' in result

    def test_skips_pycache_and_node_modules(self, tmp_path):
        env = _make_env(tmp_path)
        cells_dir = tmp_path / 'cells'
        _write_cell_md(cells_dir / 'tools' / 'web-fetch', name='web-fetch')
        _write_cell_md(cells_dir / '__pycache__' / 'fake', name='fake')
        _write_cell_md(cells_dir / 'node_modules' / 'pkg' / 'fake', name='fake')

        reg = ProjectCellRegistry(env, cell_dirs=[cells_dir])
        result = reg.list_cell_manifests()

        assert len(result) == 1
        assert 'cells/tools/web-fetch' in result

    def test_stops_recursing_at_cell_dir(self, tmp_path):
        """cell 目录内部有子目录时不应该被当作其他 cell 扫描."""
        env = _make_env(tmp_path)
        cells_dir = tmp_path / 'cells'
        cell_dir = cells_dir / 'tools' / 'web-fetch'
        _write_cell_md(cell_dir, name='web-fetch')
        # 在 cell 内部创建嵌套目录 + CELL.md — 不应被发现
        nested = cell_dir / 'scripts' / 'hidden-cell'
        _write_cell_md(nested, name='hidden')

        reg = ProjectCellRegistry(env, cell_dirs=[cells_dir])
        result = reg.list_cell_manifests()

        assert len(result) == 1
        assert 'cells/tools/web-fetch' in result

    def test_skips_invalid_cell_md(self, tmp_path):
        env = _make_env(tmp_path)
        cells_dir = tmp_path / 'cells'
        _write_cell_md(cells_dir / 'tools' / 'good', name='good')
        # 写一个损坏的 CELL.md
        bad_dir = cells_dir / 'tools' / 'bad'
        bad_dir.mkdir(parents=True)
        (bad_dir / 'CELL.md').write_text('not: valid: yaml: [[[')

        reg = ProjectCellRegistry(env, cell_dirs=[cells_dir])
        result = reg.list_cell_manifests()

        assert len(result) == 1
        assert 'cells/tools/good' in result

    def test_skips_intermediate_dirs_without_cell_md(self, tmp_path):
        """group 目录 (如 cells/tools/) 没有 CELL.md 时不应被当作 cell."""
        env = _make_env(tmp_path)
        cells_dir = tmp_path / 'cells'
        (cells_dir / 'tools').mkdir(parents=True)
        _write_cell_md(cells_dir / 'tools' / 'web-fetch', name='web-fetch')

        reg = ProjectCellRegistry(env, cell_dirs=[cells_dir])
        result = reg.list_cell_manifests()

        assert len(result) == 1
        assert 'cells/tools/web-fetch' in result

    def test_multiple_cell_dirs(self, tmp_path):
        """从 project 和 workspace 两个目录同时发现."""
        env = _make_env(tmp_path)
        project_cells = tmp_path / 'cells'
        workspace_cells = tmp_path / '.moss_ws' / 'cells'
        _write_cell_md(project_cells / 'tools' / 'web-fetch', name='web-fetch')
        _write_cell_md(workspace_cells / 'internal' / 'debug', name='debug')

        reg = ProjectCellRegistry(env, cell_dirs=[project_cells, workspace_cells])
        result = reg.list_cell_manifests()

        assert len(result) == 2
        assert 'cells/tools/web-fetch' in result
        assert '.moss_ws/cells/internal/debug' in result


# ==================================================================
# list_cell_manifests — 过滤
# ==================================================================

class TestListCellManifestsFilter:
    def test_installed_true_filters_out_not_installed(self, tmp_path):
        """有 INSTALL.md 但无 .installed 文件 → installed=False → 被过滤."""
        env = _make_env(tmp_path)
        cells_dir = tmp_path / 'cells'
        _write_cell_md(cells_dir / 'tools' / 'default', name='default')
        # 未安装的 cell
        not_installed_dir = cells_dir / 'tools' / 'not-ready'
        _write_cell_md(not_installed_dir, name='not-ready')
        (not_installed_dir / 'INSTALL.md').write_text('# install')

        reg = ProjectCellRegistry(env, cell_dirs=[cells_dir])
        result = reg.list_cell_manifests(installed=True)

        assert len(result) == 1
        assert 'cells/tools/default' in result

    def test_installed_false_returns_all(self, tmp_path):
        env = _make_env(tmp_path)
        cells_dir = tmp_path / 'cells'
        _write_cell_md(cells_dir / 'tools' / 'default', name='default')
        not_installed_dir = cells_dir / 'tools' / 'not-ready'
        _write_cell_md(not_installed_dir, name='not-ready')
        (not_installed_dir / 'INSTALL.md').write_text('# install')

        reg = ProjectCellRegistry(env, cell_dirs=[cells_dir])
        result = reg.list_cell_manifests(installed=False)

        assert len(result) == 2

    def test_include_pattern(self, tmp_path):
        env = _make_env(tmp_path)
        cells_dir = tmp_path / 'cells'
        _write_cell_md(cells_dir / 'tools' / 'web-fetch', name='web-fetch')
        _write_cell_md(cells_dir / 'robots' / 'g1', name='g1')

        reg = ProjectCellRegistry(env, cell_dirs=[cells_dir])
        result = reg.list_cell_manifests(include=['cells/tools/*'])

        assert len(result) == 1
        assert 'cells/tools/web-fetch' in result

    def test_exclude_pattern(self, tmp_path):
        env = _make_env(tmp_path)
        cells_dir = tmp_path / 'cells'
        _write_cell_md(cells_dir / 'tools' / 'web-fetch', name='web-fetch')
        _write_cell_md(cells_dir / 'tools' / 'deprecated', name='deprecated')

        reg = ProjectCellRegistry(env, cell_dirs=[cells_dir])
        result = reg.list_cell_manifests(exclude=['*/deprecated'])

        assert len(result) == 1
        assert 'cells/tools/web-fetch' in result


# ==================================================================
# 缓存
# ==================================================================

class TestCache:
    def test_refresh_false_uses_cache(self, tmp_path):
        env = _make_env(tmp_path)
        cells_dir = tmp_path / 'cells'
        _write_cell_md(cells_dir / 'tools' / 'web-fetch', name='web-fetch')

        reg = ProjectCellRegistry(env, cell_dirs=[cells_dir])
        first = reg.list_cell_manifests(refresh=True)
        # 删除磁盘上的 CELL.md
        (cells_dir / 'tools' / 'web-fetch' / 'CELL.md').unlink()
        second = reg.list_cell_manifests(refresh=False)

        assert len(first) == 1
        assert len(second) == 1  # 缓存未更新

    def test_refresh_true_rebuilds_cache(self, tmp_path):
        env = _make_env(tmp_path)
        cells_dir = tmp_path / 'cells'
        _write_cell_md(cells_dir / 'tools' / 'web-fetch', name='web-fetch')

        reg = ProjectCellRegistry(env, cell_dirs=[cells_dir])
        first = reg.list_cell_manifests(refresh=True)
        # 删除磁盘上的 CELL.md
        (cells_dir / 'tools' / 'web-fetch' / 'CELL.md').unlink()
        second = reg.list_cell_manifests(refresh=True)

        assert len(first) == 1
        assert len(second) == 0


# ==================================================================
# get_cell_manifest
# ==================================================================

class TestGetCellManifest:
    def test_found(self, tmp_path):
        env = _make_env(tmp_path)
        cells_dir = tmp_path / 'cells'
        _write_cell_md(cells_dir / 'tools' / 'web-fetch', name='web-fetch')

        reg = ProjectCellRegistry(env, cell_dirs=[cells_dir])
        manifest = reg.get_cell_manifest('cells/tools/web-fetch')

        assert manifest is not None
        assert manifest.name == 'web-fetch'

    def test_not_found(self, tmp_path):
        env = _make_env(tmp_path)
        cells_dir = tmp_path / 'cells'
        cells_dir.mkdir()

        reg = ProjectCellRegistry(env, cell_dirs=[cells_dir])
        assert reg.get_cell_manifest('cells/nonexistent') is None

    def test_not_a_directory(self, tmp_path):
        """路径是文件而非目录时返回 None."""
        env = _make_env(tmp_path)
        cells_dir = tmp_path / 'cells'
        cells_dir.mkdir()
        (cells_dir / 'README.md').write_text('hello')

        reg = ProjectCellRegistry(env, cell_dirs=[cells_dir])
        assert reg.get_cell_manifest('cells/README.md') is None

    def test_accepts_path_object(self, tmp_path):
        env = _make_env(tmp_path)
        cells_dir = tmp_path / 'cells'
        _write_cell_md(cells_dir / 'tools' / 'web-fetch', name='web-fetch')

        reg = ProjectCellRegistry(env, cell_dirs=[cells_dir])
        manifest = reg.get_cell_manifest(Path('cells/tools/web-fetch'))

        assert manifest is not None
        assert manifest.name == 'web-fetch'


# ==================================================================
# dump_spawn_env
# ==================================================================

class TestDumpSpawnEnv:
    def test_returns_dict_with_cell_address(self, tmp_path):
        env = _make_env(tmp_path, cell_address='host/main')
        reg = ProjectCellRegistry(env, cell_dirs=[])
        data = reg.dump_spawn_env('worker/test-cell')

        assert data['MOSS_CELL_ADDRESS'] == 'worker/test-cell'
        assert data['MOSS_PARENT_CELL_ADDRESS'] == 'host/main'
        assert data['MOSS_WORKSPACE'] == str(env.workspace_path)

    def test_with_os_env_is_false(self, tmp_path):
        """dump_spawn_env 固定传 with_os_env=False."""
        env = _make_env(tmp_path, cell_address='host/main')
        # 在 os.environ 中放一个值
        os.environ['TEST_EXTRA'] = 'should_not_appear'
        try:
            reg = ProjectCellRegistry(env, cell_dirs=[])
            data = reg.dump_spawn_env('worker/test')
            assert 'TEST_EXTRA' not in data
        finally:
            os.environ.pop('TEST_EXTRA', None)


# ==================================================================
# local_runtime_cells — 读写运行时文件 (无 spawn)
# ==================================================================

class TestLocalRuntimeCells:
    def test_empty_when_no_runtime_files(self, tmp_path):
        env = _make_env(tmp_path)
        reg = ProjectCellRegistry(env, cell_dirs=[])
        assert reg.local_runtime_cells() == []

    def test_reads_manually_written_runtime_file(self, tmp_path):
        env = _make_env(tmp_path)
        reg = ProjectCellRegistry(env, cell_dirs=[])
        runtimes_dir = env.cell_runtimes_dir
        runtimes_dir.mkdir(parents=True)

        cell = Cell(
            meta=CellMetadata(type='worker', name='test-cell'),
            launcher=CellLauncher(),
        )
        cell.set_alive(pid=os.getpid())
        cell.write_runtime_file(runtimes_dir)

        cells = reg.local_runtime_cells()
        assert len(cells) == 1
        assert cells[0].meta.name == 'test-cell'
        assert cells[0].status.state == 'alive'

    def test_multiple_runtime_cells_sorted(self, tmp_path):
        env = _make_env(tmp_path)
        reg = ProjectCellRegistry(env, cell_dirs=[])
        runtimes_dir = env.cell_runtimes_dir
        runtimes_dir.mkdir(parents=True)

        for name in ['cell-b', 'cell-a']:
            cell = Cell(
                meta=CellMetadata(type='worker', name=name),
                launcher=CellLauncher(),
            )
            cell.set_alive(pid=os.getpid())
            cell.write_runtime_file(runtimes_dir)

        cells = reg.local_runtime_cells()
        names = {c.meta.name for c in cells}
        assert names == {'cell-a', 'cell-b'}


# ==================================================================
# Cell.read_from_runtime_file / write_runtime_file 纯读写契约
# ==================================================================

class TestCellRuntimeFileRoundtrip:
    def test_write_and_read_back(self, tmp_path):
        env = _make_env(tmp_path)
        runtimes_dir = env.cell_runtimes_dir
        runtimes_dir.mkdir(parents=True)

        cell = Cell(
            meta=CellMetadata(type='worker', name='roundtrip', description='test'),
            launcher=CellLauncher(cwd='/tmp'),
        )
        cell.set_alive(pid=12345)
        cell.write_runtime_file(runtimes_dir)

        filename = Cell.make_runtime_filename(cell.address)
        file = runtimes_dir / filename
        assert file.exists()

        restored = Cell.read_from_runtime_file(file)
        assert restored.meta.name == 'roundtrip'
        assert restored.meta.type == 'worker'
        assert restored.status.pid == 12345
        assert restored.status.state == 'alive'
        assert restored.launcher.cwd == '/tmp'

    def test_runtime_filename_is_filesystem_safe(self):
        address = 'worker/my-cell/test_uid'
        filename = Cell.make_runtime_filename(address)
        assert ' ' not in filename
        assert filename.startswith('cell-')
        assert filename.endswith('.json')


# ==================================================================
# CellRegistry.match_cells — 静态过滤
# ==================================================================

class TestMatchCells:
    def test_include_narrows(self):
        manifests = {
            'cells/tools/a': CellManifest(type='worker', name='a', launcher=CellLauncher()),
            'cells/tools/b': CellManifest(type='worker', name='b', launcher=CellLauncher()),
            'cells/robots/c': CellManifest(type='worker', name='c', launcher=CellLauncher()),
        }
        result = dict(CellRegistry.match_cells(manifests, include=['cells/tools/*']))
        assert set(result.keys()) == {'cells/tools/a', 'cells/tools/b'}

    def test_exclude_removes(self):
        manifests = {
            'cells/tools/a': CellManifest(type='worker', name='a', launcher=CellLauncher()),
            'cells/tools/b': CellManifest(type='worker', name='b', launcher=CellLauncher()),
        }
        result = dict(CellRegistry.match_cells(manifests, exclude=['*/b']))
        assert set(result.keys()) == {'cells/tools/a'}

    def test_empty_include_returns_all(self):
        manifests = {
            'cells/x': CellManifest(type='worker', name='x', launcher=CellLauncher()),
        }
        result = dict(CellRegistry.match_cells(manifests))
        assert len(result) == 1
