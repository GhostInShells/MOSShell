"""Environment 蓝图抽象层单元测试 — 进程级环境发现与变量治理.

测试 Environment 的数据模型行为、env var 序列化、路径发现逻辑、seal 生命周期.
不测完整的 discover() 链路 (需要真实 .moss_ws 目录结构).
"""

import os
from pathlib import Path

import pytest

from ghoshell_moss.core.blueprint.environment import (
    Environment,
    EnvironmentSealedError,
    EnvironmentNotSealedError,
    DEFAULT_WORKSPACE_DIR_NAME,
    ENV_WORKSPACE_DIR_KEY,
    ENV_PROJECT_DIR_KEY,
    ENV_PROJECT_ID_KEY,
    ENV_NETWORK_KEY,
    ENV_NETWORK_SCOPE_KEY,
    ENV_SESSION_ID_KEY,
    ENV_MOSS_MODE_KEY,
    ENV_GHOST_NAME_KEY,
    ENV_CELL_ADDRESS_KEY,
    ENV_PARENT_CELL_ADDRESS_KEY,
    MOSS_META_FILE,
    MossMeta,
    WORKSPACE_PROJECT_ID_FILE,
    NODE_PATH_GHOST_KEY,
    NODE_PATH_MODE_KEY,
    resolve_node_dir,
)


@pytest.fixture(autouse=True)
def _reset_env_singleton():
    """每个测试前后清理 seal 的全局副作用 (单例 + os.environ 快照)."""
    with Environment.fixture():
        yield


# ==================================================================
# Environment — 构造 (显式参数)
# ==================================================================

class TestEnvironmentInitExplicit:
    def test_all_params_explicit(self, tmp_path):
        """所有参数显式传入时, 不从 env 读取."""
        ws = tmp_path / DEFAULT_WORKSPACE_DIR_NAME
        ws.mkdir()
        project = tmp_path / 'my_project'
        project.mkdir()

        env = Environment(
            workspace=ws,
            project=project,
            mode='desktop',
            ghost='echo',
            network='lab',
            scope='swarm-1',
            cell_address='worker/cam/uid001',
            parent_cell_address='host/main/uid000',
        )
        assert env.workspace_path == ws
        assert env.project_path == project
        assert env.mode_name == 'desktop'
        assert env.ghost_name == 'echo'
        assert env.network == 'lab'
        assert env.network_scope == 'swarm-1'
        assert env.this_cell_address == 'worker/cam/uid001'
        assert env.parent_cell_address == 'host/main/uid000'
        assert env.no_mode is False
        assert env.no_ghost is False

    def test_fallback_to_env_vars(self, tmp_path, monkeypatch):
        """未传参时走环境变量."""
        ws = tmp_path / DEFAULT_WORKSPACE_DIR_NAME
        ws.mkdir()
        project = tmp_path / 'my_project'
        project.mkdir()

        monkeypatch.setenv(ENV_MOSS_MODE_KEY, 'robot')
        monkeypatch.setenv(ENV_GHOST_NAME_KEY, 'pilot')
        monkeypatch.setenv(ENV_NETWORK_KEY, 'prod')
        monkeypatch.setenv(ENV_NETWORK_SCOPE_KEY, 'group-a')
        monkeypatch.setenv(ENV_CELL_ADDRESS_KEY, 'host/main/uid')
        monkeypatch.setenv(ENV_PARENT_CELL_ADDRESS_KEY, 'host/parent/uid')
        monkeypatch.setenv(ENV_PROJECT_DIR_KEY, str(project))

        env = Environment(workspace=ws)
        assert env.mode_name == 'robot'
        assert env.ghost_name == 'pilot'
        assert env.network == 'prod'
        assert env.network_scope == 'group-a'
        assert env.this_cell_address == 'host/main/uid'
        assert env.parent_cell_address == 'host/parent/uid'


# ==================================================================
# Environment — project_id 生成
# ==================================================================

class TestEnvironmentProjectId:
    def test_from_env_var(self, tmp_path, monkeypatch):
        ws = tmp_path / DEFAULT_WORKSPACE_DIR_NAME
        ws.mkdir()
        monkeypatch.setenv(ENV_PROJECT_ID_KEY, 'proj-env')
        env = Environment(workspace=ws)
        assert env.project_id == 'proj-env'

    def test_auto_generated_and_persisted_to_sidecar(self, tmp_path):
        """首次 discover 无 env var, project_id 自生成并写 sidecar; 二次读同 id."""
        ws = tmp_path / DEFAULT_WORKSPACE_DIR_NAME
        ws.mkdir()
        env1 = Environment(workspace=ws)
        pid = env1.project_id
        assert pid  # 非空
        sidecar = ws / WORKSPACE_PROJECT_ID_FILE
        assert sidecar.exists()
        assert sidecar.read_text().strip() == pid
        # 二次实例化必须读回同一个 id
        env2 = Environment(workspace=ws)
        assert env2.project_id == pid

    def test_env_var_priority_over_sidecar(self, tmp_path, monkeypatch):
        ws = tmp_path / DEFAULT_WORKSPACE_DIR_NAME
        ws.mkdir()
        (ws / WORKSPACE_PROJECT_ID_FILE).write_text('file-id')
        monkeypatch.setenv(ENV_PROJECT_ID_KEY, 'env-id')
        env = Environment(workspace=ws)
        assert env.project_id == 'env-id'


# ==================================================================
# Environment — project auto-discovery
# ==================================================================

class TestEnvironmentProjectAuto:
    def test_project_defaults_to_workspace_parent(self, tmp_path):
        project = tmp_path / 'my_project'
        project.mkdir()
        ws = project / DEFAULT_WORKSPACE_DIR_NAME
        ws.mkdir()
        env = Environment(workspace=ws)
        assert env.project_path == project

    def test_project_from_env_var(self, tmp_path, monkeypatch):
        ws = tmp_path / DEFAULT_WORKSPACE_DIR_NAME
        ws.mkdir()
        alt = tmp_path / 'alt_project'
        alt.mkdir()
        monkeypatch.setenv(ENV_PROJECT_DIR_KEY, str(alt))
        env = Environment(workspace=ws)
        assert env.project_path == alt


# ==================================================================
# Environment — session_id
# ==================================================================

class TestEnvironmentSessionId:
    def test_auto_generated_when_not_in_env(self, tmp_path):
        ws = tmp_path / DEFAULT_WORKSPACE_DIR_NAME
        ws.mkdir()
        env1 = Environment(workspace=ws)
        env2 = Environment(workspace=ws)
        assert len(env1.session_id) > 0
        assert env1.session_id != env2.session_id

    def test_from_env_var(self, tmp_path, monkeypatch):
        ws = tmp_path / DEFAULT_WORKSPACE_DIR_NAME
        ws.mkdir()
        monkeypatch.setenv(ENV_SESSION_ID_KEY, 'explicit-sid')
        env = Environment(workspace=ws)
        assert env.session_id == 'explicit-sid'


# ==================================================================
# Environment — seal
# ==================================================================

class TestEnvironmentSeal:
    def test_seal_writes_os_environ_snapshot(self, tmp_path):
        """seal 一次性把 runtime scope 写入 os.environ (子进程继承通道)."""
        ws = tmp_path / DEFAULT_WORKSPACE_DIR_NAME
        ws.mkdir()
        env = Environment(workspace=ws, mode='test-mode', network='my-net')
        env.seal()
        assert os.environ[ENV_WORKSPACE_DIR_KEY] == str(ws)
        assert os.environ[ENV_MOSS_MODE_KEY] == 'test-mode'
        assert os.environ[ENV_NETWORK_KEY] == 'my-net'
        assert os.environ[ENV_PROJECT_ID_KEY] == env.project_id

    def test_seal_twice_raises(self, tmp_path):
        """seal 是一次性跃迁, 二次调用抛 EnvironmentSealedError."""
        ws = tmp_path / DEFAULT_WORKSPACE_DIR_NAME
        ws.mkdir()
        env = Environment(workspace=ws, mode='m1')
        env.seal()
        with pytest.raises(EnvironmentSealedError):
            env.seal()

    def test_seal_registers_singleton(self, tmp_path):
        """seal 后 discover 返回同一实例."""
        ws = tmp_path / DEFAULT_WORKSPACE_DIR_NAME
        ws.mkdir()
        env = Environment(workspace=ws)
        env.seal()
        discovered = Environment.discover()
        assert discovered is env

    def test_is_sealed_flag(self, tmp_path):
        ws = tmp_path / DEFAULT_WORKSPACE_DIR_NAME
        ws.mkdir()
        env = Environment(workspace=ws)
        assert env.is_sealed is False
        env.seal()
        assert env.is_sealed is True

    def test_getter_reads_self_not_os_environ(self, tmp_path, monkeypatch):
        """seal 后 getter 一律回读 self._*, 不受 os.environ 后续变更影响.

        Environment 是权威信源, os.environ 只是 seal 瞬间的镜像输出.
        """
        ws = tmp_path / DEFAULT_WORKSPACE_DIR_NAME
        ws.mkdir()
        env = Environment(workspace=ws, mode='original', ghost='echo-orig')
        env.seal()
        # 恶意外部改动 os.environ 不应干扰 env 的信源身份
        monkeypatch.setenv(ENV_MOSS_MODE_KEY, 'hacked')
        monkeypatch.setenv(ENV_GHOST_NAME_KEY, 'hacked')
        assert env.mode_name == 'original'
        assert env.ghost_name == 'echo-orig'


# ==================================================================
# Environment — dump_runtime_scope
# ==================================================================

class TestDumpRuntimeScope:
    def test_contains_all_keys(self, tmp_path):
        ws = tmp_path / DEFAULT_WORKSPACE_DIR_NAME
        ws.mkdir()
        env = Environment(workspace=ws, mode='test', ghost='echo', network='lab')
        scope = env.dump_runtime_scope()
        assert scope['MOSS_WORKSPACE'] == str(ws)
        assert scope['MOSS_MODE_NAME'] == 'test'
        assert scope['MOSS_GHOST_NAME'] == 'echo'
        assert scope['MOSS_NETWORK'] == 'lab'
        assert 'MOSS_PROJECT_ID' in scope
        assert 'MOSS_CELL_ADDRESS' in scope
        assert 'MOSS_PARENT_CELL_ADDRESS' in scope

    def test_does_not_contain_session_id(self, tmp_path):
        """session_id 不传递给子进程 — 子进程自己生成."""
        ws = tmp_path / DEFAULT_WORKSPACE_DIR_NAME
        ws.mkdir()
        env = Environment(workspace=ws)
        scope = env.dump_runtime_scope()
        assert 'MOSS_SESSION_ID' not in scope


# ==================================================================
# Environment — dump_cell_env
# ==================================================================

class TestDumpCellEnv:
    def test_includes_cell_address(self, tmp_path):
        ws = tmp_path / DEFAULT_WORKSPACE_DIR_NAME
        ws.mkdir()
        env = Environment(workspace=ws)
        data = env.dump_cell_env(
            cell_address='worker/test/uid',
            parent_cell_address='host/main/uid',
            with_os_env=False,
        )
        assert data['MOSS_CELL_ADDRESS'] == 'worker/test/uid'
        assert data['MOSS_PARENT_CELL_ADDRESS'] == 'host/main/uid'

    def test_parent_falls_back_to_this(self, tmp_path):
        ws = tmp_path / DEFAULT_WORKSPACE_DIR_NAME
        ws.mkdir()
        env = Environment(workspace=ws, cell_address='host/main/uid000')
        data = env.dump_cell_env(with_os_env=False)
        assert data['MOSS_PARENT_CELL_ADDRESS'] == 'host/main/uid000'

    def test_with_os_env_includes_parent_env(self, tmp_path, monkeypatch):
        ws = tmp_path / DEFAULT_WORKSPACE_DIR_NAME
        ws.mkdir()
        monkeypatch.setenv('CUSTOM_VAR', 'hello')
        env = Environment(workspace=ws)
        data = env.dump_cell_env(cell_address='a', with_os_env=True)
        assert data['CUSTOM_VAR'] == 'hello'
        assert data['MOSS_CELL_ADDRESS'] == 'a'

    def test_empty_values_removed(self, tmp_path):
        ws = tmp_path / DEFAULT_WORKSPACE_DIR_NAME
        ws.mkdir()
        env = Environment(workspace=ws, mode='none', ghost='none')
        data = env.dump_cell_env(cell_address='', with_os_env=False)
        # mode=none 和 ghost=none 应该保留 (它们是合法值)
        # 但 cell_address='' 传入会覆盖 MOSS_CELL_ADDRESS 为空, 然后被过滤
        assert 'MOSS_CELL_ADDRESS' not in data

    def test_cell_address_override_priority(self, tmp_path):
        """dump_cell_env 的 cell_address 优先于 dump_runtime_scope 的值."""
        ws = tmp_path / DEFAULT_WORKSPACE_DIR_NAME
        ws.mkdir()
        env = Environment(workspace=ws, cell_address='host/main/uid000')
        data = env.dump_cell_env(cell_address='worker/new/uid', with_os_env=False)
        assert data['MOSS_CELL_ADDRESS'] == 'worker/new/uid'


# ==================================================================
# Environment — find_workspace_path
# ==================================================================

class TestFindWorkspacePath:
    def test_from_env_var(self, tmp_path, monkeypatch):
        ws = tmp_path / 'custom_ws'
        ws.mkdir()
        monkeypatch.setenv(ENV_WORKSPACE_DIR_KEY, str(ws))
        result = Environment.find_workspace_path()
        assert result == ws

    def test_env_var_raises_if_not_exists(self, monkeypatch):
        monkeypatch.setenv(ENV_WORKSPACE_DIR_KEY, '/nonexistent/ws')
        with pytest.raises(EnvironmentError, match='does not exist'):
            Environment.find_workspace_path()

    def test_from_cwd(self, tmp_path, monkeypatch):
        ws = tmp_path / DEFAULT_WORKSPACE_DIR_NAME
        ws.mkdir()
        monkeypatch.chdir(tmp_path)
        result = Environment.find_workspace_path()
        assert result == ws

    def test_from_parent_dir(self, tmp_path, monkeypatch):
        ws = tmp_path / DEFAULT_WORKSPACE_DIR_NAME
        ws.mkdir()
        subdir = tmp_path / 'a' / 'b' / 'c'
        subdir.mkdir(parents=True)
        monkeypatch.chdir(subdir)
        result = Environment.find_workspace_path()
        assert result == ws

    def test_fallback_to_cwd_expect(self, tmp_path, monkeypatch):
        """找不到时返回 cwd/.moss_ws 预期路径."""
        empty = tmp_path / 'empty'
        empty.mkdir()
        monkeypatch.chdir(empty)
        result = Environment.find_workspace_path()
        assert result == empty / DEFAULT_WORKSPACE_DIR_NAME

    def test_stops_at_home(self, tmp_path, monkeypatch):
        """超过 home 目录后不再向上搜索."""
        home_ws = Path.home() / DEFAULT_WORKSPACE_DIR_NAME
        # 确保 home 下没有 .moss_ws (避免干扰)
        existed = home_ws.exists()
        if existed:
            # skip this test to avoid false negative
            pytest.skip("home .moss_ws exists, can't test stop-at-home")
        leaf = tmp_path / 'x' / 'y' / 'z'
        leaf.mkdir(parents=True)
        monkeypatch.chdir(leaf)
        result = Environment.find_workspace_path()
        # 找到 tmp_path 以下或 cwd 预期
        assert result.name == DEFAULT_WORKSPACE_DIR_NAME


# ==================================================================
# Environment — discover (singleton + bootstrap flag)
# ==================================================================

class TestEnvironmentDiscover:
    def test_discover_returns_same_instance_after_seal(self, tmp_path):
        ws = tmp_path / DEFAULT_WORKSPACE_DIR_NAME
        ws.mkdir()
        env = Environment(workspace=ws)
        env.seal()
        d1 = Environment.discover()
        d2 = Environment.discover()
        assert d1 is d2
        assert d1 is env

    def test_discover_default_bootstraps_when_no_singleton(self, tmp_path, monkeypatch):
        """discover() 默认 bootstrap=True: 无单例时自动 Environment().seal() 兜底.

        子 cell main.py 场景: 父进程注入 os.environ, 子进程一行 discover() 完成入网.
        """
        ws = tmp_path / DEFAULT_WORKSPACE_DIR_NAME
        ws.mkdir()
        monkeypatch.chdir(tmp_path)
        env = Environment.discover()
        assert env.workspace_path == ws
        assert env.is_sealed is True

    def test_discover_bootstrap_false_returns_singleton_when_sealed(self, tmp_path):
        """discover(bootstrap=False) 有单例时正常返回, 不管 flag."""
        ws = tmp_path / DEFAULT_WORKSPACE_DIR_NAME
        ws.mkdir()
        env = Environment(workspace=ws)
        env.seal()
        assert Environment.discover(bootstrap=False) is env


# ==================================================================
# Environment — 目录路径属性
# ==================================================================

class TestEnvironmentDirPaths:

    def test_node_dirs_under_project(self, tmp_path):
        project = tmp_path / 'proj'
        project.mkdir()
        ws = project / DEFAULT_WORKSPACE_DIR_NAME
        ws.mkdir()
        env = Environment(workspace=ws)
        assert env.default_project_nodes_dir.relative_to(env.project_path)
        assert env.default_workspace_nodes_dir.relative_to(env.workspace_path)

    def test_resolve_node_dir_four_addresses(self, tmp_path):
        # 四地址组合: 无前缀→project, $MOSS_WORKSPACE→workspace,
        # $MODE→mode home, $GHOST→ghost home.
        ws = tmp_path / DEFAULT_WORKSPACE_DIR_NAME
        ws.mkdir()
        meta = MossMeta(name='testproj', default_mode='desktop', default_ghost='echo')
        meta.write_to_directory(ws)
        env = Environment(workspace=ws)

        assert resolve_node_dir('nodes', env) == env.project_path / 'nodes'
        assert resolve_node_dir(f'${ENV_WORKSPACE_DIR_KEY}/nodes', env) == env.workspace_path / 'nodes'
        assert resolve_node_dir(f'${NODE_PATH_MODE_KEY}/nodes',
                                env) == env.workspace_path / 'modes' / 'desktop' / 'nodes'
        assert resolve_node_dir(f'${NODE_PATH_GHOST_KEY}/nodes',
                                env) == env.workspace_path / 'ghosts' / 'echo' / 'nodes'

    def test_runtime_and_log_dirs_under_workspace(self, tmp_path):
        ws = tmp_path / DEFAULT_WORKSPACE_DIR_NAME
        ws.mkdir()
        env = Environment(workspace=ws)
        assert env.cell_runtimes_dir.relative_to(ws)
        assert env.log_config_file.relative_to(ws)


# ==================================================================
# MossMeta — 默认值来源
# ==================================================================

class TestMossMetaDefaults:
    """MossMeta 在场时, Environment 的默认值来自 MOSS.md."""

    def test_defaults_from_moss_meta(self, tmp_path):
        ws = tmp_path / DEFAULT_WORKSPACE_DIR_NAME
        ws.mkdir()
        meta = MossMeta(
            name='testproj',
            default_mode='desktop',
            default_ghost='echo',
            default_network='lab',
            default_network_scope='swarm_a',
        )
        meta.write_to_directory(ws)
        env = Environment(workspace=ws)
        assert env.mode_name == 'desktop'
        assert env.ghost_name == 'echo'
        assert env.network == 'lab'
        assert env.network_scope == 'swarm_a'


# ==================================================================
# MossMeta — 环境变量优先级
# ==================================================================

class TestMossMetaEnvVarPriority:

    def test_env_var_overrides_moss_meta(self, tmp_path, monkeypatch):
        ws = tmp_path / DEFAULT_WORKSPACE_DIR_NAME
        ws.mkdir()
        meta = MossMeta(default_mode='desktop', default_network='lab')
        meta.write_to_directory(ws)
        monkeypatch.setenv(ENV_MOSS_MODE_KEY, 'robot')
        monkeypatch.setenv(ENV_NETWORK_KEY, 'prod')
        env = Environment(workspace=ws)
        assert env.mode_name == 'robot'
        assert env.network == 'prod'

    def test_explicit_kwarg_overrides_everything(self, tmp_path, monkeypatch):
        ws = tmp_path / DEFAULT_WORKSPACE_DIR_NAME
        ws.mkdir()
        MossMeta(default_mode='from_meta').write_to_directory(ws)
        monkeypatch.setenv(ENV_MOSS_MODE_KEY, 'from_env')
        env = Environment(workspace=ws, mode='from_kwarg')
        assert env.mode_name == 'from_kwarg'

    def test_env_var_overrides_moss_meta_ghost(self, tmp_path, monkeypatch):
        ws = tmp_path / DEFAULT_WORKSPACE_DIR_NAME
        ws.mkdir()
        MossMeta(default_ghost='echo').write_to_directory(ws)
        monkeypatch.setenv(ENV_GHOST_NAME_KEY, 'pilot')
        env = Environment(workspace=ws)
        assert env.ghost_name == 'pilot'


# ==================================================================
# MossMeta / HostModeMeta — 命名语法校验
# ==================================================================

class TestMossNamePattern:
    """MOSS_NAME_PATTERN 校验: ^[a-zA-Z_][a-zA-Z0-9_]*$"""

    def test_valid_names(self):
        meta = MossMeta(name='my_project', default_mode='desktop',
                        default_ghost='echo_v2', default_network='lab_01')
        assert meta.name == 'my_project'
        assert meta.default_mode == 'desktop'

    def test_slash_rejected(self):
        with pytest.raises(Exception):
            MossMeta(default_mode='desktop/evil')

    def test_dot_rejected(self):
        with pytest.raises(Exception):
            MossMeta(default_ghost='echo.v2')

    def test_hyphen_rejected(self):
        with pytest.raises(Exception):
            MossMeta(default_network='my-net')

    def test_leading_digit_rejected(self):
        with pytest.raises(Exception):
            MossMeta(default_mode='2bad')

    def test_space_rejected(self):
        with pytest.raises(Exception):
            MossMeta(name='my project')

    def test_empty_string_rejected(self):
        with pytest.raises(Exception):
            MossMeta(name='')

    def test_host_mode_meta_name_validation(self):
        from ghoshell_moss.core.blueprint.project import HostModeMeta
        with pytest.raises(Exception):
            HostModeMeta(name='bad-name')


# ==================================================================
# Environment — init_workspace
# ==================================================================

class TestInitWorkspace:
    def test_creates_dir_and_stubs(self, tmp_path):
        ws = tmp_path / 'new_ws'
        Environment.init_workspace(ws)
        assert ws.exists()

    def test_force_overwrites_existing(self, tmp_path):
        ws = tmp_path / 'ws'
        Environment.init_workspace(ws)
        # second call should not raise
        Environment.init_workspace(ws, force=True)


# ==================================================================
# MossMeta.write_to_file
# ==================================================================

class TestMossMetaWriteToFile:
    def test_create_meta_file_sets_self_file(self, tmp_path):
        """create_meta_file 创建后 self.file 必须指向 MOSS.md 绝对路径."""
        ws = tmp_path / DEFAULT_WORKSPACE_DIR_NAME
        ws.mkdir()
        meta = Environment.create_meta_file(ws)
        assert meta.file == str((ws / MOSS_META_FILE).absolute())

    def test_create_meta_file_then_rewrite_works(self, tmp_path):
        """create_meta_file 后, 后续 write_to_file() 无参也能 dump (self.file 已就位)."""
        ws = tmp_path / DEFAULT_WORKSPACE_DIR_NAME
        ws.mkdir()
        meta = Environment.create_meta_file(ws)
        meta.default_mode = 'desktop'
        meta.write_to_file()  # 不传 file 参数, 必须能成功
        reloaded = MossMeta.read_from_file(ws / MOSS_META_FILE)
        assert reloaded.default_mode == 'desktop'
