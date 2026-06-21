"""Cell 蓝图抽象层单元测试 — 数据模型 + 寻址函数 + 文件读写契约.

不依赖网络 / Zenoh / 子进程，只验证数据结构与纯函数行为。
"""

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
    split_address,
    make_default_logger_name,
    make_bridge_address,
    split_bridge_address,
    normalize,
    CellAddress,
    CellBridgeAddress,
)


# ==================================================================
# CellType
# ==================================================================

class TestCellType:
    def test_values(self):
        assert CellType.host == 'host'
        assert CellType.fractal == 'fractal'
        assert CellType.worker == 'worker'

    def test_no_app_type(self):
        """app 类型已删除，不再作为框架保留类型."""
        assert 'app' not in [e.value for e in CellType]

    def test_str_coercion(self):
        assert CellType('host') == CellType.host
        assert CellType('worker') == CellType.worker


# ==================================================================
# make_address / split_address
# ==================================================================

class TestMakeAddress:
    def test_basic(self):
        assert make_address('host', 'default') == 'host/default'

    def test_strips_whitespace(self):
        assert make_address(' worker ', ' name ') == 'worker/name'

    def test_rejects_empty_type(self):
        with pytest.raises(ValueError, match='Invalid address parts'):
            make_address('', 'name')

    def test_rejects_empty_name(self):
        with pytest.raises(ValueError, match='Invalid address parts'):
            make_address('type', '')

    def test_rejects_slash_in_type(self):
        with pytest.raises(ValueError, match="must not contain '/'"):
            make_address('a/b', 'name')

    def test_with_celltype_enum(self):
        assert make_address(CellType.worker, 'cam') == 'worker/cam'


class TestSplitAddress:
    def test_basic(self):
        assert split_address('host/default') == ('host', 'default')

    def test_nested_name(self):
        assert split_address('node/tools/web-fetch') == ('node', 'tools/web-fetch')

    def test_rejects_flat_string(self):
        with pytest.raises(ValueError, match='Invalid cell address'):
            split_address('no_slash')


# ==================================================================
# make_bridge_address / split_bridge_address
# ==================================================================

class TestBridgeAddress:
    def test_make(self):
        assert make_bridge_address('worker/cam', 'uid001') == 'worker/cam/uid001'


# ==================================================================
# normalize
# ==================================================================

class TestNormalize:
    def test_slashes(self):
        assert normalize('host/default') == 'host_default'
        assert normalize('node/tools.web-fetch') == 'node_tools_web_fetch'


# ==================================================================
# make_default_logger_name
# ==================================================================

class TestDefaultLoggerName:
    def test_slashes_to_dots(self):
        assert make_default_logger_name('host/default') == 'host.default'

    def test_backslashes_to_dots(self):
        assert make_default_logger_name(r'host\default') == 'host.default'


# ==================================================================
# CellMetadata
# ==================================================================

class TestCellMetadata:
    def test_defaults(self):
        m = CellMetadata(name='test')
        assert m.type == 'worker'
        assert m.singleton is True
        assert m.description == ''

    def test_from_proc_is_worker_non_singleton(self):
        m = CellMetadata.from_proc()
        assert m.type == CellType.worker
        assert m.singleton is False

    def test_from_proc_custom_name(self):
        m = CellMetadata.from_proc(name='my-worker')
        assert m.name == 'my-worker'
        assert m.singleton is False


# ==================================================================
# CellLauncher
# ==================================================================

class TestCellLauncher:
    def test_defaults(self):
        launcher = CellLauncher()
        assert launcher.interpreter == 'python'
        assert launcher.cmd == 'main.py'
        assert launcher.cwd == './'

    def test_cwd_path(self):
        launcher = CellLauncher(cwd='subdir')
        assert launcher.cwd_path.name == 'subdir'

    def test_cwd_path_empty_defaults_to_cwd(self):
        launcher = CellLauncher(cwd='')
        import os
        assert launcher.cwd_path == Path(os.getcwd())

    def test_from_proc(self):
        launcher = CellLauncher.from_proc()
        assert launcher.interpreter != ''
        assert launcher.cmd != ''
        assert launcher.cwd != ''


# ==================================================================
# CellManifest — 文件读写与 installed 推导
# ==================================================================

_LAUNCHER_NESTED = """
type: worker
name: web-fetch
description: "抓取网页内容"
singleton: true
launcher:
  interpreter: python
  cmd: main.py
  cwd: nodes/tools/web-fetch
"""


class TestCellManifestReadWrite:

    # -- read -----------------------------------------------------------

    def test_read_nested_launcher(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / 'CELL.md').write_text(f'---{_LAUNCHER_NESTED}---\ninstruction body\n')
            m = CellManifest.read_from_file(d / 'CELL.md')

        assert m.name == 'web-fetch'
        assert m.type == 'worker'
        assert m.singleton is True
        assert m.launcher.interpreter == 'python'
        assert m.launcher.cmd == 'main.py'
        assert m.launcher.cwd is not None
        assert m.instruction == 'instruction body'
        assert m.installed is True  # 无 INSTALL.md，默认已安装

    def test_cwd_is_absolute_after_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / 'nodes' / 'tools' / 'web-fetch'
            d.mkdir(parents=True)
            (d / 'CELL.md').write_text(f'---{_LAUNCHER_NESTED}---\n')
            m = CellManifest.read_from_file(d / 'CELL.md')
            assert Path(m.launcher.cwd).is_absolute()

    def test_installed_true_when_no_install_md(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / 'CELL.md').write_text(f'---{_LAUNCHER_NESTED}---\n')
            m = CellManifest.read_from_file(d / 'CELL.md')
        assert m.installed is True

    def test_installed_false_when_install_md_exists_but_not_installed(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / 'CELL.md').write_text(f'---{_LAUNCHER_NESTED}---\n')
            (d / 'INSTALL.md').touch()
            m = CellManifest.read_from_file(d / 'CELL.md')
        assert m.installed is False

    def test_installed_true_when_installed_file_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / 'CELL.md').write_text(f'---{_LAUNCHER_NESTED}---\n')
            (d / 'INSTALL.md').touch()
            (d / '.installed').touch()
            m = CellManifest.read_from_file(d / 'CELL.md')
        assert m.installed is True

    def test_empty_instruction(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / 'CELL.md').write_text(f'---{_LAUNCHER_NESTED}---')
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
            name='test-node',
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
            (d / 'CELL.md').write_text(f'---{_LAUNCHER_NESTED}---\n')
            m = CellManifest.read_from_directory(d)
            assert m is not None
            assert m.name == 'web-fetch'

    def test_read_from_directory_returns_none_when_no_cell_md(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            assert CellManifest.read_from_directory(d) is None


# ==================================================================
# CellStatus + Cell.set_alive / set_failed
# ==================================================================

class TestCellStatus:
    def test_default_uid_is_generated(self):
        s = CellStatus()
        assert len(s.uid) > 0

    def test_from_proc_is_starting(self):
        s = CellStatus.from_proc()
        assert s.state == 'starting'
        assert isinstance(s.pid, int)
        assert s.failure == ''

    def test_uid_unique_per_instance(self):
        s1 = CellStatus()
        s2 = CellStatus()
        assert s1.uid != s2.uid


class TestCellLifecycleMethods:
    def test_set_alive(self):
        cell = Cell.new(meta=CellMetadata(name='x'))
        cell.set_alive(pid=12345)
        assert cell.status.state == 'alive'
        assert cell.status.pid == 12345
        assert cell.status.failure == ''
        assert cell.is_alive() is False  # pid 12345 不存在

    def test_set_alive_default_pid(self):
        import os
        cell = Cell.new(meta=CellMetadata(name='x'))
        cell.set_alive()
        assert cell.status.pid == os.getpid()
        assert cell.status.state == 'alive'
        assert cell.is_alive() is True

    def test_set_failed(self):
        cell = Cell.new(meta=CellMetadata(name='x'))
        cell.set_alive()
        cell.set_failed('connection refused')
        assert cell.status.state == 'stopped'
        assert cell.status.failure == 'connection refused'

    def test_is_alive_checks_psutil(self):
        cell = Cell.new(meta=CellMetadata(name='x'))
        cell.set_alive()
        assert cell.is_alive() is True

    def test_is_alive_returns_false_when_stopped(self):
        cell = Cell.new(meta=CellMetadata(name='x'))
        assert cell.is_alive() is False
        cell.set_alive()
        cell.set_failed('done')
        assert cell.is_alive() is False


# ==================================================================
# Cell — 寻址、构造、序列化
# ==================================================================

class TestCellAddress:
    """address 与 bridge_address 行为."""

    def test_singleton_address_is_type_name(self):
        meta = CellMetadata(type='worker', name='camera', singleton=True)
        cell = Cell.new(meta=meta)
        assert cell.address == 'worker/camera'

    def test_non_singleton_address_is_type_uid(self):
        meta = CellMetadata(type='worker', name='cam', singleton=False)
        cell = Cell.new(meta=meta)
        assert cell.address.startswith('worker/')
        assert cell.address != 'worker/cam'

    def test_bridge_address_appends_uid(self):
        meta = CellMetadata(type='worker', name='cam', singleton=True)
        cell = Cell.new(meta=meta)
        assert cell.bridge_address.endswith(cell.status.uid)

    def test_non_singleton_bridge_has_double_uid(self):
        """non-singleton 的 address 和 bridge_address 都含 uid，重复可接受."""
        meta = CellMetadata(type='worker', name='cam', singleton=False)
        cell = Cell.new(meta=meta)
        parts = cell.bridge_address.split('/')
        assert len(parts) == 2
        assert parts[0] == 'worker'

    def test_type_property(self):
        meta = CellMetadata(type=CellType.host, name='main')
        cell = Cell.new(meta=meta)
        assert cell.type == 'host'


class TestCellConstruction:
    def test_new(self):
        meta = CellMetadata(name='test')
        cell = Cell.new(meta=meta)
        assert cell.meta.name == 'test'
        assert cell.status.state == 'stopped'
        assert cell.is_alive() is False

    def test_new_with_launcher(self):
        launcher = CellLauncher(interpreter='python', cmd='main.py')
        cell = Cell.new(meta=CellMetadata(name='test'), launcher=launcher)
        assert cell.launcher.interpreter == 'python'

    def test_from_proc(self):
        cell = Cell.from_proc()
        assert cell.meta.type == CellType.worker
        assert cell.meta.singleton is False
        assert cell.status.state == 'starting'

    def test_from_manifest_default_starting(self):
        manifest = CellManifest(
            type='worker',
            name='test',
            launcher=CellLauncher(interpreter='python'),
        )
        cell = Cell.from_manifest(manifest)
        assert cell.meta.name == 'test'
        assert cell.launcher.interpreter == 'python'
        assert cell.status.state == 'starting'

    def test_from_manifest_stopped(self):
        manifest = CellManifest(
            type='worker',
            name='test',
            launcher=CellLauncher(interpreter='python'),
        )
        cell = Cell.from_manifest(manifest, status_from_proc=False)
        assert cell.status.state == 'stopped'
        assert cell.is_alive() is False

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


class TestCellLaunch:
    def test_launch_program_python(self):
        launcher = CellLauncher(interpreter='python', cmd='main.py')
        cell = Cell.new(meta=CellMetadata(name='x'), launcher=launcher)
        import sys
        assert cell.launch_program() == sys.executable

    def test_launch_program_full_path(self):
        launcher = CellLauncher(interpreter='/usr/bin/python3', cmd='main.py')
        cell = Cell.new(meta=CellMetadata(name='x'), launcher=launcher)
        assert cell.launch_program() == '/usr/bin/python3'

    def test_launch_program_no_interpreter(self):
        launcher = CellLauncher(interpreter='', cmd='./my_binary')
        cell = Cell.new(meta=CellMetadata(name='x'), launcher=launcher)
        assert cell.launch_program() == './my_binary'

    def test_launch_args(self):
        launcher = CellLauncher(interpreter='python', cmd='main.py', args='--port 8080')
        cell = Cell.new(meta=CellMetadata(name='x'), launcher=launcher)
        assert cell.launch_args() == ['main.py', '--port', '8080']

    def test_launch_args_no_interpreter(self):
        launcher = CellLauncher(interpreter='', cmd='./my_binary', args='--verbose')
        cell = Cell.new(meta=CellMetadata(name='x'), launcher=launcher)
        assert cell.launch_args() == ['--verbose']


class TestCellRuntimeFile:
    def test_normalized_address(self):
        cell = Cell.new(meta=CellMetadata(name='test', type='host'))
        assert cell.normalized_address() == 'host_test'

    def test_normalized_name(self):
        cell = Cell.new(meta=CellMetadata(name='MyCell', type='worker'))
        assert cell.normalized_name() == 'mycell'

    def test_runtime_filename(self):
        fname = Cell.make_runtime_filename('worker/cam')
        assert fname.startswith('cell-')
        assert fname.endswith('.json')
        assert 'worker_cam' in fname

    def test_write_and_read_runtime_file(self):
        cell = Cell.new(meta=CellMetadata(name='test', type='worker'))
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            cell.write_runtime_file(d)
            fname = Cell.make_runtime_filename(cell.address)
            assert (d / fname).exists()

            loaded = Cell.read_from_runtime_file(d / fname)
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

    def test_runtime_filepath(self):
        cell = Cell.new(meta=CellMetadata(name='test', type='worker'))
        d = Path('/tmp/fake')
        fp = cell.runtime_filepath(d)
        assert fp.parent == d
        assert fp.name.startswith('cell-')
        assert fp.name.endswith('.json')


# ==================================================================
# CellRegistry — ABC 契约验证
# ==================================================================

class TestCellRegistryABC:
    """验证 ABC 方法的参数签名与返回类型声明."""

    def test_match_cells_include_all(self):
        manifests = {
            'group_a/app1': CellManifest(type='worker', name='app1', launcher=CellLauncher()),
            'group_b/app2': CellManifest(type='worker', name='app2', launcher=CellLauncher()),
        }
        result = list(CellRegistry.match_cells(manifests))
        assert len(result) == 2

    def test_match_cells_include_filter(self):
        manifests = {
            'tools/a': CellManifest(type='worker', name='a', launcher=CellLauncher()),
            'tools/b': CellManifest(type='worker', name='b', launcher=CellLauncher()),
            'other/c': CellManifest(type='worker', name='c', launcher=CellLauncher()),
        }
        result = list(CellRegistry.match_cells(manifests, include=['tools/*']))
        assert len(result) == 2
        paths = {p for p, _ in result}
        assert paths == {'tools/a', 'tools/b'}

    def test_match_cells_exclude_filter(self):
        manifests = {
            'tools/a': CellManifest(type='worker', name='a', launcher=CellLauncher()),
            'tools/b': CellManifest(type='worker', name='b', launcher=CellLauncher()),
        }
        result = list(CellRegistry.match_cells(manifests, exclude=['tools/b']))
        assert len(result) == 1
        path, _ = result[0]
        assert path == 'tools/a'

    def test_match_cells_empty_cells(self):
        result = list(CellRegistry.match_cells({}))
        assert result == []


# ==================================================================
# Integration: 完整的从 CELL.md 到 Cell 网络注册的路径
# ==================================================================

class TestCellLifecycle:
    """模拟 host 发现 → manifest → cell → runtime file → network 查询的完整链路."""

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
            assert cell.address == 'worker/camera'
            assert cell.status.state == 'stopped'

            # 3. 模拟 spawn 后 — cell 进程标记 alive 并写 runtime 文件
            cell.set_alive()
            runtime_dir = d / 'runtime' / 'cells'
            runtime_dir.mkdir(parents=True)
            cell.write_runtime_file(runtime_dir)

            # 4. Registry 从 runtime 文件反查
            found = list(Cell.find_runtime_cells(runtime_dir))
            assert len(found) == 1
            assert found[0].address == 'worker/camera'
            assert found[0].bridge_address != "worker/camera"
            assert found[0].is_alive()

            # 5. Network announce 用的 bridge_address 可反查
            bridge = found[0].bridge_address
            typ, uid = split_bridge_address(bridge)
            assert typ == 'worker'
            assert uid == found[0].status.uid

    def test_set_alive_and_set_failed_cycle(self):
        cell = Cell.new(meta=CellMetadata(name='x', type='worker'))
        assert cell.is_alive() is False

        cell.set_alive(pid=99999)
        assert cell.status.state == 'alive'
        # pid 99999 不存在, is_alive 用 psutil 检查
        assert cell.is_alive() is False

        import os
        cell.set_alive()  # 用当前进程 pid
        assert cell.is_alive() is True
        assert cell.status.failure == ''

        cell.set_failed('crash at startup')
        assert cell.status.state == 'stopped'
        assert cell.status.failure == 'crash at startup'
        assert cell.is_alive() is False
