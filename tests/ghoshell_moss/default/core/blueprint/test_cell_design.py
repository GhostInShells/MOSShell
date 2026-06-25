"""Cell 蓝图抽象层单元测试 — 数据模型 + 寻址函数 + 文件读写契约.

不依赖网络 / Zenoh / 子进程，只验证数据结构与纯函数行为。
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

from ghoshell_moss.core.blueprint.cell import (
    CellType,
    CellMetadata,
    CellLauncher,
    CellManifest,
    CellStatus,
    Cell,
    CellRegistry,
    CellNetwork,
    make_address,
    normalize,
)


# ==================================================================
# CellType
# ==================================================================

class TestCellType:
    def test_values(self):
        assert CellType.host == 'host'
        assert CellType.worker == 'worker'

    def test_only_host_and_worker(self):
        vals = {e.value for e in CellType}
        assert vals == {'host', 'worker'}


# ==================================================================
# make_address
# ==================================================================

class TestMakeAddress:
    def test_two_parts(self):
        assert make_address('host', 'default') == 'host/default'

    def test_variadic(self):
        assert make_address('a', 'b', 'c') == 'a/b/c'

    def test_single_part(self):
        assert make_address('only') == 'only'

    def test_empty_parts(self):
        assert make_address() == ''


# ==================================================================
# normalize
# ==================================================================

class TestNormalize:
    def test_slashes_to_underscores(self):
        assert normalize('host/default') == 'host_default'

    def test_backslashes(self):
        assert normalize(r'host\name') == 'host_name'

    def test_dots_and_dashes(self):
        assert normalize('node.tools-web') == 'node_tools_web'


# ==================================================================
# CellMetadata
# ==================================================================

class TestCellMetadata:
    def test_defaults(self):
        m = CellMetadata(name='test')
        assert m.type == 'worker'
        assert m.singleton is True
        assert m.description == ''
        assert m.channel is False

    def test_channel_true(self):
        m = CellMetadata(name='srv', channel=True)
        assert m.channel is True

    def test_host_type(self):
        m = CellMetadata(type='host', name='main')
        assert m.type == 'host'

    def test_from_proc_returns_worker_non_singleton(self):
        m = CellMetadata.from_proc()
        assert m.type == CellType.worker
        assert m.singleton is False

    def test_from_proc_custom_name(self):
        m = CellMetadata.from_proc(name='my-cam')
        assert m.name == 'my-cam'


# ==================================================================
# CellLauncher
# ==================================================================

class TestCellLauncher:
    def test_defaults_all_empty(self):
        launcher = CellLauncher()
        assert launcher.interpreter == ''
        assert launcher.cmd == ''
        assert launcher.cwd == ''
        assert launcher.arguments == ''

    def test_cwd_path_explicit(self):
        launcher = CellLauncher(cwd='subdir')
        assert launcher.cwd_path.name == 'subdir'

    def test_cwd_path_empty_defaults_to_cwd(self):
        launcher = CellLauncher(cwd='')
        assert launcher.cwd_path == Path.cwd()

    def test_new_empty(self):
        launcher = CellLauncher.new_empty()
        assert launcher.interpreter == ''
        assert launcher.cmd == ''
        assert launcher.cwd == ''
        assert launcher.arguments == ''

    def test_from_proc(self):
        launcher = CellLauncher.from_proc()
        assert launcher.interpreter != ''
        assert launcher.cmd != ''
        assert launcher.cwd != ''


# ==================================================================
# CellManifest — 文件读写与 installed 推导
# ==================================================================

_LAUNCHER_FLAT = """---
type: worker
name: web-fetch
description: "抓取网页内容"
singleton: true
launcher:
  interpreter: python
  cmd: main.py
  cwd: nodes/web-fetch
---
instruction body
"""


class TestCellManifestReadWrite:

    # -- read -----------------------------------------------------------

    def test_read_flat_launcher(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / 'CELL.md').write_text(_LAUNCHER_FLAT)
            m = CellManifest.read_from_file(d / 'CELL.md')

        assert m.name == 'web-fetch'
        assert m.type == 'worker'
        assert m.singleton is True
        assert m.launcher.interpreter == 'python'
        assert m.launcher.cmd == 'main.py'
        assert m.instruction == 'instruction body'
        assert m.installed is True

    def test_cwd_absolute_after_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / 'nodes' / 'web-fetch'
            d.mkdir(parents=True)
            (d / 'CELL.md').write_text(_LAUNCHER_FLAT)
            m = CellManifest.read_from_file(d / 'CELL.md')
            assert Path(m.launcher.cwd).is_absolute()

    def test_installed_true_no_install_md(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / 'CELL.md').write_text(_LAUNCHER_FLAT)
            m = CellManifest.read_from_file(d / 'CELL.md')
        assert m.installed is True

    def test_installed_false_when_install_md_missing_dotfile(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / 'CELL.md').write_text(_LAUNCHER_FLAT)
            (d / 'INSTALL.md').touch()
            m = CellManifest.read_from_file(d / 'CELL.md')
        assert m.installed is False

    def test_installed_true_when_dotfile_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / 'CELL.md').write_text(_LAUNCHER_FLAT)
            (d / 'INSTALL.md').touch()
            (d / '.installed').touch()
            m = CellManifest.read_from_file(d / 'CELL.md')
        assert m.installed is True

    def test_empty_instruction(self):
        manifest = CellManifest(
            type='worker',
            name='empty',
            launcher=CellLauncher(),
            instruction='',
        )
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            manifest.write_file(d)
            m = CellManifest.read_from_file(d / 'CELL.md')
        assert m.instruction == ''

    # -- write ----------------------------------------------------------

    def test_write_and_roundtrip(self):
        manifest = CellManifest(
            type='worker',
            name='test-node',
            singleton=True,
            launcher=CellLauncher(interpreter='python', cmd='main.py'),
            instruction='hello world',
        )
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            manifest.write_file(d)
            assert (d / 'CELL.md').exists()

            m2 = CellManifest.read_from_file(d / 'CELL.md')
            assert m2.name == 'test-node'
            assert m2.type == 'worker'
            assert m2.launcher.interpreter == 'python'
            assert m2.launcher.cmd == 'main.py'
            assert m2.instruction == 'hello world'

    def test_write_excludes_installed_from_frontmatter(self):
        manifest = CellManifest(
            type='worker',
            name='test',
            launcher=CellLauncher(),
            instruction='x',
            installed=True,
        )
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            manifest.write_file(d)
            content = (d / 'CELL.md').read_text()
            yaml_block = content.split('---')[1]
            assert 'installed' not in yaml_block

    # -- meta extraction ------------------------------------------------

    def test_meta_returns_pure_cell_metadata(self):
        manifest = CellManifest(
            type='worker',
            name='test',
            singleton=False,
            launcher=CellLauncher(interpreter='python'),
            instruction='hi',
        )
        meta = manifest.meta()
        assert isinstance(meta, CellMetadata)
        assert meta.name == 'test'
        assert meta.type == 'worker'
        assert meta.singleton is False
        assert not hasattr(meta, 'launcher')
        assert not hasattr(meta, 'instruction')

    # -- read_from_directory -------------------------------------------

    def test_read_from_directory_returns_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / 'CELL.md').write_text(_LAUNCHER_FLAT)
            m = CellManifest.read_from_directory(d)
            assert m is not None
            assert m.name == 'web-fetch'

    def test_read_from_directory_returns_none_when_no_cell_md(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            assert CellManifest.read_from_directory(d) is None

    # -- CellManifest.new ----------------------------------------------

    def test_new_from_meta_and_launcher(self):
        meta = CellMetadata(name='cam', type='host', channel=True)
        launcher = CellLauncher(interpreter='python', cmd='cam.py')
        manifest = CellManifest.new(meta=meta, launcher=launcher, instruction='camera driver')
        assert manifest.name == 'cam'
        assert manifest.type == 'host'
        assert manifest.launcher.interpreter == 'python'
        assert manifest.instruction == 'camera driver'
        assert manifest.installed is True


# ==================================================================
# CellStatus
# ==================================================================

class TestCellStatus:
    def test_default_uid_is_generated(self):
        s = CellStatus()
        assert len(s.uid) > 0

    def test_default_version_zero(self):
        s = CellStatus()
        assert s.version == 0

    def test_default_updated_is_set(self):
        s = CellStatus()
        assert s.updated > 0

    def test_default_state_stopped(self):
        s = CellStatus()
        assert s.state == 'stopped'

    def test_default_pid_zero(self):
        s = CellStatus()
        assert s.pid == 0

    def test_default_project_id_empty(self):
        s = CellStatus()
        assert s.project_id == ''

    def test_from_proc_is_starting(self):
        s = CellStatus.from_proc()
        assert s.state == 'starting'
        assert s.pid == os.getpid()
        assert s.failure == ''

    def test_uid_unique_per_instance(self):
        s1 = CellStatus()
        s2 = CellStatus()
        assert s1.uid != s2.uid


# ==================================================================
# Cell — 寻址、命名、构造
# ==================================================================

class TestCellNaming:
    def test_singleton_name_is_normalized(self):
        meta = CellMetadata(type='worker', name='MyCam', singleton=True)
        cell = Cell.new(meta=meta)
        assert cell.name == 'mycam'

    def test_non_singleton_name_includes_uid(self):
        meta = CellMetadata(type='worker', name='cam', singleton=False)
        cell = Cell.new(meta=meta)
        assert cell.name.startswith('cam_')
        assert len(cell.name) > 4

    def test_normalized_name(self):
        meta = CellMetadata(name='My-Cell')
        cell = Cell.new(meta=meta)
        assert cell.normalized_name == 'my_cell'

    def test_unique_name_format(self):
        meta = CellMetadata(name='cam', singleton=False)
        cell = Cell.new(meta=meta)
        uid_prefix = cell.status.uid[:8]
        assert cell.unique_name == f'cam_{uid_prefix}'

    def test_type_property_from_enum(self):
        meta = CellMetadata(type=CellType.host, name='main')
        cell = Cell.new(meta=meta)
        assert cell.type == 'host'

    def test_type_property_from_str(self):
        meta = CellMetadata(type='robot', name='r1')
        cell = Cell.new(meta=meta)
        assert cell.type == 'robot'

    def test_address_always_includes_uid(self):
        meta = CellMetadata(type='worker', name='cam', singleton=True)
        cell = Cell.new(meta=meta)
        # address = type/normalized_name/uid
        parts = cell.address.split('/')
        assert parts[0] == 'worker'
        assert parts[1] == 'cam'
        assert len(parts[2]) > 0


class TestCellLockerName:
    def test_singleton_same_meta_gives_same_locker(self):
        """singleton: 相同 type+name 产生相同的锁名 (去重)."""
        meta = CellMetadata(type='worker', name='robot-arm', singleton=True)
        c1 = Cell.new(meta=meta)
        c2 = Cell.new(meta=meta)
        assert c1.cell_locker_name == c2.cell_locker_name

    def test_non_singleton_same_meta_gives_different_locker(self):
        """非 singleton: 每个实例有唯一锁名 (uid 去重)."""
        meta = CellMetadata(type='worker', name='worker', singleton=False)
        c1 = Cell.new(meta=meta)
        c2 = Cell.new(meta=meta)
        assert c1.cell_locker_name != c2.cell_locker_name


class TestCellIsHost:
    def test_host_type_is_host(self):
        meta = CellMetadata(type='host', name='main')
        cell = Cell.new(meta=meta)
        assert cell.is_host is True

    def test_worker_is_not_host(self):
        meta = CellMetadata(type='worker', name='w1')
        cell = Cell.new(meta=meta)
        assert cell.is_host is False

    def test_custom_type_not_host(self):
        meta = CellMetadata(type='fractal', name='f1')
        cell = Cell.new(meta=meta)
        assert cell.is_host is False


# ==================================================================
# Cell — 生命周期
# ==================================================================

class TestCellLifecycle:
    def test_set_alive(self):
        cell = Cell.new(meta=CellMetadata(name='x'))
        cell.set_alive(pid=99999)
        assert cell.status.state == 'alive'
        assert cell.status.pid == 99999
        assert cell.status.failure == ''
        # pid 99999 不存在
        assert cell.is_alive() is False

    def test_set_alive_default_pid(self):
        cell = Cell.new(meta=CellMetadata(name='x'))
        cell.set_alive()
        assert cell.status.pid == os.getpid()
        assert cell.status.state == 'alive'
        assert cell.is_alive() is True

    def test_set_alive_increments_version(self):
        cell = Cell.new(meta=CellMetadata(name='x'))
        v0 = cell.status.version
        cell.set_alive()
        assert cell.status.version == v0 + 1

    def test_set_failed(self):
        cell = Cell.new(meta=CellMetadata(name='x'))
        cell.set_alive()
        cell.set_failed('connection refused')
        assert cell.status.state == 'stopped'
        assert cell.status.failure == 'connection refused'

    def test_set_failed_increments_version(self):
        cell = Cell.new(meta=CellMetadata(name='x'))
        cell.set_alive()
        v1 = cell.status.version
        cell.set_failed('error')
        assert cell.status.version == v1 + 1

    def test_is_alive_false_when_stopped(self):
        cell = Cell.new(meta=CellMetadata(name='x'))
        assert cell.is_alive() is False

    def test_update(self):
        cell = Cell.new(meta=CellMetadata(name='x'))
        v0 = cell.status.version
        t0 = cell.status.updated
        cell.update()
        assert cell.status.version == v0 + 1
        assert cell.status.updated >= t0


# ==================================================================
# Cell — 构造
# ==================================================================

class TestCellConstruction:
    def test_new_minimal(self):
        meta = CellMetadata(name='test')
        cell = Cell.new(meta=meta)
        assert cell.meta.name == 'test'
        assert cell.launcher.cmd == ''
        assert cell.status.state == 'stopped'
        assert cell.is_alive() is False

    def test_from_proc(self):
        cell = Cell.from_proc()
        assert cell.meta.type == CellType.worker
        assert cell.meta.singleton is False
        assert cell.status.state == 'starting'

    def test_from_manifest_stopped(self):
        manifest = CellManifest(
            type='worker',
            name='test',
            launcher=CellLauncher(interpreter='python'),
        )
        cell = Cell.from_manifest(manifest, status_from_proc=False)
        assert cell.meta.name == 'test'
        assert cell.launcher.interpreter == 'python'
        assert cell.status.state == 'stopped'

    def test_as_manifest_roundtrip(self):
        manifest = CellManifest(
            type='host',
            name='default',
            launcher=CellLauncher(interpreter='python'),
            instruction='the host',
        )
        cell = Cell.from_manifest(manifest)
        m2 = cell.as_manifest(instruction='the host')
        assert m2.name == 'default'
        assert m2.type == 'host'
        assert m2.launcher.interpreter == 'python'
        assert m2.instruction == 'the host'


# ==================================================================
# Cell — 启动参数解析
# ==================================================================

class TestCellLaunch:
    def test_launch_program_python(self):
        launcher = CellLauncher(interpreter='python', cmd='main.py')
        cell = Cell.new(meta=CellMetadata(name='x'), launcher=launcher)
        assert cell.launch_program() == sys.executable

    def test_launch_program_custom_interpreter(self):
        launcher = CellLauncher(interpreter='/usr/bin/python3', cmd='main.py')
        cell = Cell.new(meta=CellMetadata(name='x'), launcher=launcher)
        assert cell.launch_program() == '/usr/bin/python3'

    def test_launch_program_no_interpreter_falls_back_to_cmd(self):
        launcher = CellLauncher(interpreter='', cmd='./my_binary')
        cell = Cell.new(meta=CellMetadata(name='x'), launcher=launcher)
        assert cell.launch_program() == './my_binary'

    def test_launch_args_interpreter_and_cmd(self):
        launcher = CellLauncher(interpreter='python', cmd='main.py', arguments='--port 8080')
        cell = Cell.new(meta=CellMetadata(name='x'), launcher=launcher)
        assert cell.launch_args() == ['main.py', '--port', '8080']

    def test_launch_args_no_interpreter(self):
        launcher = CellLauncher(interpreter='', cmd='./my_binary', arguments='--verbose')
        cell = Cell.new(meta=CellMetadata(name='x'), launcher=launcher)
        assert cell.launch_args() == ['--verbose']

    def test_launch_args_empty(self):
        launcher = CellLauncher()
        cell = Cell.new(meta=CellMetadata(name='x'), launcher=launcher)
        assert cell.launch_args() == []

    def test_launch_cwd_explicit(self):
        launcher = CellLauncher(cwd='sub/dir')
        cell = Cell.new(meta=CellMetadata(name='x'), launcher=launcher)
        cwd = Path('/base')
        result = cell.launch_cwd(cwd=cwd)
        assert result == (cwd / 'sub/dir').resolve()

    def test_launch_cwd_empty_defaults_to_arg(self):
        launcher = CellLauncher(cwd='')
        cell = Cell.new(meta=CellMetadata(name='x'), launcher=launcher)
        cwd = Path('/base')
        assert cell.launch_cwd(cwd=cwd) == cwd


# ==================================================================
# Cell — 运行时文件读写
# ==================================================================

class TestCellRuntimeFile:
    def test_runtime_filename(self):
        fname = Cell.make_runtime_filename('worker/cam')
        assert fname == 'cell-worker_cam.json'

    def test_runtime_filepath(self):
        cell = Cell.new(meta=CellMetadata(name='test', type='worker'))
        d = Path('/tmp/fake')
        fp = cell.runtime_filepath(d)
        assert fp.parent == d
        assert fp.name.startswith('cell-')
        assert fp.name.endswith('.json')

    def test_write_and_read_runtime_file(self):
        cell = Cell.new(meta=CellMetadata(name='test', type='worker'))
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            cell.write_runtime_file(d)
            expected = Cell.make_runtime_filename(cell.address)
            assert (d / expected).exists()

            loaded = Cell.read_from_runtime_file(d / expected)
            assert loaded.meta.name == 'test'
            assert loaded.address == cell.address
            assert loaded.status.uid == cell.status.uid

    def test_find_runtime_cells(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            c1 = Cell.new(meta=CellMetadata(name='a'))
            c2 = Cell.new(meta=CellMetadata(name='b'))
            c1.write_runtime_file(d)
            c2.write_runtime_file(d)
            (d / 'other.json').write_text('{}')

            cells = list(Cell.find_runtime_cells(d))
            assert len(cells) == 2
            names = {c.meta.name for c in cells}
            assert names == {'a', 'b'}

    def test_find_runtime_cells_throw(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / 'cell-corrupt.json').write_text('not json')
            with pytest.raises(Exception):
                list(Cell.find_runtime_cells(d, throw=True))


# ==================================================================
# Cell — to_json
# ==================================================================

class TestCellToJson:
    def test_to_json_with_cell_type_enum(self):
        meta = CellMetadata(type=CellType.host, name='main')
        cell = Cell.new(meta=meta)
        raw = cell.to_json()
        assert '"host"' in raw or 'host' in raw

    def test_to_json_roundtrip(self):
        cell = Cell.new(meta=CellMetadata(name='test'))
        cell.set_alive(pid=os.getpid())
        raw = cell.to_json()
        loaded = Cell.model_validate_json(raw)
        assert loaded.meta.name == 'test'
        assert loaded.status.state == 'alive'


# ==================================================================
# CellRegistry — ABC 静态方法契约
# ==================================================================

class TestCellRegistryMatchCells:
    def test_match_all(self):
        manifests = {
            'tools/a': CellManifest(type='worker', name='a', launcher=CellLauncher()),
            'tools/b': CellManifest(type='worker', name='b', launcher=CellLauncher()),
        }
        result = list(CellRegistry.match_cells(manifests))
        assert len(result) == 2

    def test_match_include_filter(self):
        manifests = {
            'tools/a': CellManifest(type='worker', name='a', launcher=CellLauncher()),
            'tools/b': CellManifest(type='worker', name='b', launcher=CellLauncher()),
            'other/c': CellManifest(type='worker', name='c', launcher=CellLauncher()),
        }
        result = list(CellRegistry.match_cells(manifests, include=['tools/*']))
        assert len(result) == 2
        paths = {p for p, _ in result}
        assert paths == {'tools/a', 'tools/b'}

    def test_match_exclude_filter(self):
        manifests = {
            'tools/a': CellManifest(type='worker', name='a', launcher=CellLauncher()),
            'tools/b': CellManifest(type='worker', name='b', launcher=CellLauncher()),
        }
        result = list(CellRegistry.match_cells(manifests, exclude=['tools/b']))
        assert len(result) == 1
        path, _ = result[0]
        assert path == 'tools/a'

    def test_match_empty(self):
        result = list(CellRegistry.match_cells({}))
        assert result == []


# ==================================================================
# CellNetwork — ABC 契约验证
# ==================================================================

class TestCellNetworkABC:
    """验证 ABC abstractmethod 的存在。子类必须实现。"""

    def test_required_properties(self):
        import inspect
        abstract = set()
        for name, method in inspect.getmembers(CellNetwork, predicate=inspect.isfunction):
            if getattr(method, '__isabstractmethod__', False):
                abstract.add(name)
        assert 'get_host' in abstract
        assert 'all_hosts' in abstract
        assert 'get_live_cells' in abstract
        assert 'create_provider' in abstract
        assert 'create_proxy' in abstract
        assert 'update_cell' in abstract
        assert 'revoke_cell' in abstract

    def test_required_abstract_properties(self):
        abstract = set()
        for name, prop in CellNetwork.__dict__.items():
            if isinstance(prop, property) and getattr(prop.fget, '__isabstractmethod__', False):
                abstract.add(name)
        assert 'name' in abstract
        assert 'description' in abstract
        assert 'scope' in abstract


# ==================================================================
# Integration: CELL.md → manifest → cell → runtime file
# ==================================================================

class TestCellIntegration:
    """模拟 host 发现 → manifest → cell → runtime file 的完整链路."""

    def test_static_discovery_to_runtime_registration(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / 'CELL.md').write_text("""---
type: worker
name: camera
singleton: true
launcher:
  interpreter: python
  cmd: capture.py
  cwd: nodes/sensors/camera
---
实时摄像头采集节点
""")
            # 1. Registry 读 manifest
            manifest = CellManifest.read_from_file(d / 'CELL.md')
            assert manifest.name == 'camera'

            # 2. Host 构造 Cell (模拟 spawn 前)
            cell = Cell.from_manifest(manifest, status_from_proc=False)
            assert cell.address.startswith('worker/camera/')
            assert cell.status.state == 'stopped'

            # 3. 模拟 spawn 后 — cell 进程标记 alive 并写 runtime 文件
            cell.set_alive(pid=os.getpid())
            runtime_dir = d / 'runtime' / 'cells'
            runtime_dir.mkdir(parents=True)
            cell.write_runtime_file(runtime_dir)

            # 4. Registry 从 runtime 文件反查
            found = list(Cell.find_runtime_cells(runtime_dir))
            assert len(found) == 1
            assert found[0].address == cell.address
            assert found[0].is_alive()

    def test_set_alive_and_set_failed_cycle(self):
        cell = Cell.new(meta=CellMetadata(name='x', type='worker'))
        assert cell.is_alive() is False

        cell.set_alive(pid=99999)
        assert cell.status.state == 'alive'
        assert cell.is_alive() is False

        cell.set_alive()
        assert cell.is_alive() is True
        assert cell.status.failure == ''

        cell.set_failed('crash at startup')
        assert cell.status.state == 'stopped'
        assert cell.status.failure == 'crash at startup'
        assert cell.is_alive() is False