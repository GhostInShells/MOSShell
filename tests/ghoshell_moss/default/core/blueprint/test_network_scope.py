"""network scope 的解析规则.

判据只有一条: **没声明 scope 就落在本 project 的 project id 上, 于是不同 project
天然隔离在同一 network 的不同子网里; 只有显式声明同一个 scope, 两者才进同一个子网.**

scope 的优先级是 运行时声明 > network 自己的配置 > project id, 最终不允许为空.
"""

import json
from pathlib import Path

import pytest

from ghoshell_moss.core.blueprint.environment import (
    Environment,
    DEFAULT_WORKSPACE_DIR_NAME,
    MossMeta,
)
from ghoshell_moss.core.blueprint.project import NetworkMetadata
from ghoshell_moss.project.local_project import LocalProject
from ghoshell_moss.factory import resolve_network

MOSS_ENV_KEYS = [
    'MOSS_PROJECT_ID',
    'MOSS_NETWORK_SCOPE',
    'MOSS_NETWORK',
    'MOSS_WORKSPACE',
    'MOSS_PROJECT_DIR',
    'MOSS_MODE_NAME',
    'MOSS_GHOST_NAME',
    'MOSS_CELL_ADDRESS',
    'MOSS_PARENT_CELL_ADDRESS',
]


@pytest.fixture(autouse=True)
def clean_moss_env(monkeypatch):
    """测试进程可能本身就跑在一个 MOSS 实例里 — 先摘掉它下传的 runtime scope."""
    for key in MOSS_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def _make_env(tmp_path: Path, name: str = 'a', **kwargs) -> tuple[Path, Environment]:
    ws = tmp_path / name / DEFAULT_WORKSPACE_DIR_NAME
    ws.mkdir(parents=True, exist_ok=True)
    return ws, Environment(workspace=ws, **kwargs)


def _write_network(ws: Path, name: str = 'local', scope: str | None = None) -> None:
    data = {'name': name, 'driver': 'zenoh', 'config': {}}
    if scope is not None:
        data['scope'] = scope
    directory = ws / 'networks'
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f'{name}.json').write_text(json.dumps(data))


class TestEnvironmentDeclaredScope:

    def test_undeclared_scope_falls_back_to_project_id(self, tmp_path):
        """空 scope 是常态 —— 它意味着"不进任何共享子网"."""
        _, env = _make_env(tmp_path)
        assert env.declared_network_scope == ''
        assert env.network_scope == env.project_id
        assert env.network_scope != 'default'

    def test_two_projects_do_not_share_a_scope_by_default(self, tmp_path):
        _, a = _make_env(tmp_path, 'a')
        _, b = _make_env(tmp_path, 'b')
        assert a.project_id != b.project_id
        assert a.network_scope != b.network_scope

    def test_explicit_scope_is_shared_between_projects(self, tmp_path):
        _, a = _make_env(tmp_path, 'a', scope='lab')
        _, b = _make_env(tmp_path, 'b', scope='lab')
        assert a.declared_network_scope == 'lab'
        assert a.network_scope == b.network_scope == 'lab'

    def test_moss_md_scope_wins_over_project_id(self, tmp_path):
        ws, _ = _make_env(tmp_path)
        MossMeta(default_network_scope='swarm_a').write_to_directory(ws)
        env = Environment(workspace=ws)
        assert env.declared_network_scope == 'swarm_a'
        assert env.network_scope == 'swarm_a'

    def test_project_id_scope_is_a_legal_declaration(self, tmp_path):
        """远程接入要写成对方的 project id —— 它不以字母开头, 不能被名字规则挡掉."""
        _, first = _make_env(tmp_path, 'a')
        _, second = _make_env(tmp_path, 'b', scope=first.project_id)
        assert second.declared_network_scope == first.project_id
        assert second.network_scope == first.network_scope


class TestResolveNetwork:

    def test_network_config_scope_used_when_runtime_declares_nothing(self, tmp_path):
        ws, env = _make_env(tmp_path)
        _write_network(ws, 'local', scope='lan_a')
        project = LocalProject(env)
        assert resolve_network(env, project).scope == 'lan_a'

    def test_runtime_declaration_overrides_network_config(self, tmp_path):
        ws, env = _make_env(tmp_path, scope='mine')
        _write_network(ws, 'local', scope='lan_a')
        project = LocalProject(env)
        assert resolve_network(env, project).scope == 'mine'

    def test_unpinned_network_falls_through_to_project_id(self, tmp_path):
        ws, env = _make_env(tmp_path)
        _write_network(ws, 'local', scope=None)
        project = LocalProject(env)
        assert resolve_network(env, project).scope == env.project_id

    def test_missing_network_config_falls_through_to_project_id(self, tmp_path):
        _, env = _make_env(tmp_path)
        project = LocalProject(env)
        metadata = resolve_network(env, project)
        assert isinstance(metadata, NetworkMetadata)
        assert metadata.scope == env.project_id
        assert metadata.driver == 'zenoh'

    def test_resolved_scope_is_never_empty(self, tmp_path):
        """三处声明都为空时也要有个确定的值 — 空白命名空间会把所有 project 混在一起."""
        ws, env = _make_env(tmp_path)
        _write_network(ws, 'local', scope='')
        project = LocalProject(env)
        assert resolve_network(env, project).scope == env.project_id
