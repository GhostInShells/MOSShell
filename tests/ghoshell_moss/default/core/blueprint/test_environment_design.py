"""Environment 蓝图抽象层单元测试 — RuntimeScope + MossMeta 数据结构.

仅测试数据模型行为、env var 序列化/反序列化、文件读写。
不测 Environment.discover 和具体路径约定。
"""

import os
import tempfile
from pathlib import Path

import pytest

from ghoshell_moss.core.blueprint.environment import (
    RuntimeScope,
    MossMeta,
    DEFAULT_SESSION_SCOPE,
    DEFAULT_MOSS_MODE,
    DEFAULT_GHOST_NAME,
    ENV_MOSS_MODE_KEY,
    ENV_GHOST_NAME_KEY,
    ENV_MOSS_HOST_PID_KEY,
    ENV_SESSION_SCOPE_KEY,
    ENV_SESSION_ID_KEY,
)


# ==================================================================
# RuntimeScope — 默认值与构造
# ==================================================================

class TestRuntimeScope:
    def test_defaults(self):
        scope = RuntimeScope()
        assert scope.session_scope == DEFAULT_SESSION_SCOPE
        assert scope.mode_name == DEFAULT_MOSS_MODE
        assert scope.ghost_name == DEFAULT_GHOST_NAME
        assert scope.host_pid == 0
        assert scope.source == ''

    def test_session_id_auto_generated(self):
        s1 = RuntimeScope()
        s2 = RuntimeScope()
        assert len(s1.session_id) > 0
        assert s1.session_id != s2.session_id


# ==================================================================
# RuntimeScope.new() — 显式参数 + env fallback
# ==================================================================

class TestRuntimeScopeNew:
    def test_explicit_params_win(self, monkeypatch):
        """显式传入的参数完全覆盖 env."""
        monkeypatch.setenv(ENV_MOSS_MODE_KEY, 'env-mode')
        scope = RuntimeScope.new(
            mode_name='cli-mode',
            ghost_name='echo',
            host_pid=42,
            session_scope='lab',
            session_id='fixed',
        )
        assert scope.mode_name == 'cli-mode'
        assert scope.ghost_name == 'echo'
        assert scope.host_pid == 42
        assert scope.session_scope == 'lab'
        assert scope.session_id == 'fixed'

    def test_fallback_to_env(self, monkeypatch):
        monkeypatch.setenv(ENV_MOSS_MODE_KEY, 'test-mode')
        monkeypatch.setenv(ENV_GHOST_NAME_KEY, 'test-ghost')
        monkeypatch.setenv(ENV_SESSION_SCOPE_KEY, 'test-scope')
        monkeypatch.setenv(ENV_SESSION_ID_KEY, 'test-sid')
        monkeypatch.setenv(ENV_MOSS_HOST_PID_KEY, '12345')

        scope = RuntimeScope.new()
        assert scope.mode_name == 'test-mode'
        assert scope.ghost_name == 'test-ghost'
        assert scope.session_scope == 'test-scope'
        assert scope.session_id == 'test-sid'
        assert scope.host_pid == 12345


# ==================================================================
# RuntimeScope.create_from_env() — 纯 env 读取
# ==================================================================

class TestRuntimeScopeFromEnv:
    def test_reads_all_keys(self, monkeypatch):
        monkeypatch.setenv(ENV_MOSS_MODE_KEY, 'desktop')
        monkeypatch.setenv(ENV_GHOST_NAME_KEY, 'echo')
        monkeypatch.setenv(ENV_MOSS_HOST_PID_KEY, '9999')
        monkeypatch.setenv(ENV_SESSION_SCOPE_KEY, 'my-scope')
        monkeypatch.setenv(ENV_SESSION_ID_KEY, 'sid-001')

        scope = RuntimeScope.create_from_env()
        assert scope.mode_name == 'desktop'
        assert scope.host_pid == 9999
        assert scope.session_scope == 'my-scope'
        assert scope.session_id == 'sid-001'
        assert scope.source == 'env'

    def test_accepts_explicit_dict(self):
        scope = RuntimeScope.create_from_env(env_data={
            ENV_MOSS_MODE_KEY: 'custom',
            ENV_SESSION_ID_KEY: 'custom-sid',
        })
        assert scope.mode_name == 'custom'
        assert scope.session_id == 'custom-sid'


# ==================================================================
# RuntimeScope — env 序列化/反序列化
# ==================================================================

class TestRuntimeScopeDumpEnv:
    def test_dump_roundtrip(self, monkeypatch):
        """dump_env_data → create_from_env 形成对称."""
        original = RuntimeScope.new(
            mode_name='m', ghost_name='g', host_pid=1,
            session_scope='s', session_id='id',
        )
        data = original.dump_env_data()
        restored = RuntimeScope.create_from_env(env_data=data)
        assert restored.mode_name == original.mode_name
        assert restored.ghost_name == original.ghost_name
        assert restored.host_pid == original.host_pid
        assert restored.session_scope == original.session_scope
        assert restored.session_id == original.session_id


# ==================================================================
# RuntimeScope — 文件序列化/反序列化
# ==================================================================

class TestRuntimeScopeFileIO:
    def test_write_and_read_roundtrip(self):
        scope = RuntimeScope(mode_name='test', ghost_name='echo', host_pid=12345)
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            scope.write_to_directory(d)
            loaded = RuntimeScope.read_from_directory(d)
            assert loaded is not None
            assert loaded.mode_name == 'test'
            assert loaded.ghost_name == 'echo'
            assert loaded.host_pid == 12345

    def test_read_from_empty_dir_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            assert RuntimeScope.read_from_directory(Path(tmp)) is None

    def test_corrupt_file_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / 'runtime_scope.json').write_text('not json')
            assert RuntimeScope.read_from_directory(d) is None


# ==================================================================
# MossMeta — MOSS.md 解析
# ==================================================================

class TestMossMeta:
    def test_defaults(self):
        m = MossMeta()
        assert m.name == 'moss'
        assert m.default_mode == 'default'

    def test_from_file_reads_frontmatter(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / 'MOSS.md').write_text("""---
name: test-env
ctml_version: "1.0"
default_mode: desktop
---
This is the system prompt.
""")
            m = MossMeta.from_file(d / 'MOSS.md')
            assert m.name == 'test-env'
            assert m.ctml_version == '1.0'
            assert m.default_mode == 'desktop'
            assert m.system_prompt == 'This is the system prompt.'