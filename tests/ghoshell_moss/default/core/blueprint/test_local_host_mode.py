"""LocalHostMode 单元测试 — 构造、bootstrap、manifests 懒加载、缓存.

测试用最小模式目录结构，不依赖 stubs workspace 内的实际 manifest 文件。
"""

import logging
import sys
from pathlib import Path

import pytest

from ghoshell_moss.core.blueprint.environment import (
    Environment, DEFAULT_WORKSPACE_DIR_NAME, MOSS_NAME_PATTERN,
)
from ghoshell_moss.core.blueprint.project import (
    HostModeMeta, HOST_MODE_FILE,
)
from ghoshell_moss.project.local_host_mode import LocalHostMode


def _minimal_mode(tmp_path: Path, *, name: str = 'testmode'):
    """创建最小可用的模式目录结构.

    返回 (mode_dir, env, meta).
    """
    ws = tmp_path / DEFAULT_WORKSPACE_DIR_NAME
    ws.mkdir()

    mode_dir = tmp_path / 'modes' / name
    mode_dir.mkdir(parents=True)

    # 最小 HOST.md — 全用默认值，name 与目录名一致
    host_md = mode_dir / HOST_MODE_FILE
    host_md.write_text(f'---\nname: {name}\n---\n# {name} instruction\n')

    # 最小 src/HOST 包 — 让 ScannedModeManifests 可以 import
    src_host = mode_dir / 'src' / 'HOST'
    src_host.mkdir(parents=True)
    (src_host / '__init__.py').touch()
    manifests_dir = src_host / 'manifests'
    manifests_dir.mkdir()
    (manifests_dir / '__init__.py').touch()

    env = Environment(workspace=ws)
    meta = HostModeMeta.from_file(host_md)
    return mode_dir, env, meta


class TestLocalHostModeConstruct:

    def test_construct_with_explicit_params(self, tmp_path):
        mode_dir, env, meta = _minimal_mode(tmp_path, name='desktop')
        custom_logger = logging.getLogger('test_custom')
        mode = LocalHostMode(
            env=env, meta=meta, workspace_dir=mode_dir, logger=custom_logger,
        )
        assert mode.name == 'desktop'
        assert mode.env is env
        assert mode.meta is meta
        assert mode.workspace_dir == mode_dir
        assert mode.logger is custom_logger

    def test_logger_falls_back_to_moss(self, tmp_path):
        mode_dir, env, meta = _minimal_mode(tmp_path)
        mode = LocalHostMode(env=env, meta=meta, workspace_dir=mode_dir)
        assert mode.logger.name == 'moss'


class TestLocalHostModeBootstrapGuard:

    def test_manifests_raises_before_bootstrap(self, tmp_path):
        mode_dir, env, meta = _minimal_mode(tmp_path)
        mode = LocalHostMode(env=env, meta=meta, workspace_dir=mode_dir)
        with pytest.raises(RuntimeError, match='not bootstrapped'):
            mode.manifests()

    def test_nodes_raises_before_bootstrap(self, tmp_path):
        # HostMode.cells() 已在 §YY-2 删除, 只剩 nodes_discover_paths().
        # 该方法不受 bootstrap 守卫 (纯 meta 配置解析, 无副作用).
        mode_dir, env, meta = _minimal_mode(tmp_path)
        mode = LocalHostMode(env=env, meta=meta, workspace_dir=mode_dir)
        # nodes_discover_paths 不依赖 bootstrap, 应可直接调用
        paths = mode.nodes_discover_paths()
        assert isinstance(paths, list)

    def test_bootstrap_makes_manifest_and_nodes_available(self, tmp_path):
        mode_dir, env, meta = _minimal_mode(tmp_path)
        mode = LocalHostMode(env=env, meta=meta, workspace_dir=mode_dir)
        mode.bootstrap()
        manifests = mode.manifests()
        assert manifests is not None

    def test_bootstrap_idempotent(self, tmp_path):
        mode_dir, env, meta = _minimal_mode(tmp_path)
        mode = LocalHostMode(env=env, meta=meta, workspace_dir=mode_dir)
        mode.bootstrap()
        mode.bootstrap()
        mode.bootstrap()
        assert mode.manifests() is not None


class TestLocalHostModeCache:

    def test_manifests_returns_same_instance(self, tmp_path):
        mode_dir, env, meta = _minimal_mode(tmp_path)
        mode = LocalHostMode(env=env, meta=meta, workspace_dir=mode_dir)
        mode.bootstrap()
        a = mode.manifests()
        b = mode.manifests()
        assert a is b

    def test_nodes_returns_same_instance(self, tmp_path):
        # nodes_discover_paths 每次读 meta.node_dirs — 不做缓存 (纯投影视图,
        # 缓存是漂移种子). meta 配置就是权威, list 每次新构造.
        mode_dir, env, meta = _minimal_mode(tmp_path)
        mode = LocalHostMode(env=env, meta=meta, workspace_dir=mode_dir)
        mode.bootstrap()
        a = mode.nodes_discover_paths()
        b = mode.nodes_discover_paths()
        assert a == b


class TestLocalHostModeBootstrapSideEffects:

    def test_bootstrap_adds_src_to_sys_path(self, tmp_path):
        mode_dir, env, meta = _minimal_mode(tmp_path)
        mode = LocalHostMode(env=env, meta=meta, workspace_dir=mode_dir)
        src_path = str((mode_dir / 'src').absolute())
        assert src_path not in sys.path
        mode.bootstrap()
        assert src_path in sys.path
        # 清理 — 避免污染后续测试
        sys.path.remove(src_path)

    def test_bootstrap_loads_dotenv_when_exists(self, tmp_path):
        mode_dir, env, meta = _minimal_mode(tmp_path)
        (mode_dir / '.env').write_text('MOSS_TEST_VAR=hello')
        mode = LocalHostMode(env=env, meta=meta, workspace_dir=mode_dir)
        mode.bootstrap()
        import os
        assert os.environ.get('MOSS_TEST_VAR') == 'hello'
        # 清理
        os.environ.pop('MOSS_TEST_VAR', None)
        # 清理 sys.path
        src_path = str((mode_dir / 'src').absolute())
        if src_path in sys.path:
            sys.path.remove(src_path)


class TestLocalHostModeNameConstraint:

    def test_name_from_host_md_can_differ_from_directory(self, tmp_path):
        """HOST.md 内的 name 可能与目录名不同 — 由 LocalProject 层决定用哪个."""
        mode_dir, env, meta = _minimal_mode(tmp_path, name='dir_name')
        # meta.name 来自 HOST.md, 覆盖为目录名是 list_modes 的职责
        assert meta.name == 'dir_name'
        # 构造不受影响
        mode = LocalHostMode(env=env, meta=meta, workspace_dir=mode_dir)
        mode.bootstrap()
        assert mode.name == 'dir_name'

    def test_name_must_match_pattern(self):
        for bad in ['bad-name', 'bad.name', 'bad name', '2bad', 'bad/name']:
            with pytest.raises(Exception):
                HostModeMeta(name=bad)
