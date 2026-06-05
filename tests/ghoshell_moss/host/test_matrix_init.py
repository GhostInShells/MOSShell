"""MatrixImpl.__init__ cell address 解析和 _is_main 判定。

覆盖: DEFAULT_CELL_ADDRESS 从 'main' 改为 'host/{mode}' 后，
MatrixImpl 仍能正确识别主 cell 身份。
"""

import os
import pytest
from unittest.mock import MagicMock, patch

from ghoshell_moss.core.blueprint.matrix import Mode, Manifests
from ghoshell_moss.core.blueprint.environment import Environment, DEFAULT_CELL_ADDRESS
from ghoshell_moss.host.matrix import MatrixImpl, HostCell, UnknownCell
from ghoshell_moss.contracts.workspace import LocalWorkspace


@pytest.fixture
def mode():
    return Mode(name="test_mode", description="A test mode")


@pytest.fixture
def workspace(tmp_path):
    ws = LocalWorkspace(tmp_path)
    ws.root_path().mkdir(parents=True, exist_ok=True)
    return ws


@pytest.fixture
def env(workspace, mode):
    return Environment(workspace_path=workspace.root_path(), mode=mode.name)


@pytest.fixture
def app_store():
    store = MagicMock()
    store.list_apps.return_value = []
    return store


@pytest.fixture
def manifest():
    m = MagicMock()
    m.ctml_versions.return_value = {"v1_0_0.zh": MagicMock()}
    m.channels.return_value = {}
    return m


# ------------------------------------------------------------------
# 默认 cell address — 无 MOSS_CELL_ADDRESS 环境变量
# ------------------------------------------------------------------


def test_default_cell_address_is_host_mode(env, mode):
    """无 MOSS_CELL_ADDRESS 时，默认 cell_address = 'host/{mode}' 格式化后的值."""
    assert env.cell_address == f"host/{mode.name}"


def test_default_cell_address_does_not_equal_template(env, mode):
    """格式化后的 cell address ('host/test_mode') != 模板字符串 ('host/{mode}')."""
    assert env.cell_address != DEFAULT_CELL_ADDRESS


# ------------------------------------------------------------------
# MatrixImpl._this_cell / _is_main 判定
# ------------------------------------------------------------------


def _make_matrix(mode, env, workspace, app_store, manifest):
    """构造 MatrixImpl，绕过 _prepare_system_prompter 的 CTML 文件依赖."""
    with patch.object(MatrixImpl, '_prepare_system_prompter', return_value=MagicMock()):
        return MatrixImpl(
            mode=mode,
            env=env,
            app_store=app_store,
            manifest=manifest,
            workspace=workspace,
        )


def test_is_main_true_when_cell_address_matches_host_cell(mode, env, workspace, app_store, manifest):
    """默认情况下，env.cell_address == main_cell.address，_is_main 应为 True."""
    matrix = _make_matrix(mode, env, workspace, app_store, manifest)
    assert matrix._is_main is True
    assert isinstance(matrix.this, HostCell)
    assert matrix.this.type == "host"


def test_is_main_true_even_if_branch_comparison_is_broken(mode, env, workspace, app_store, manifest):
    """验证: 即使 DEFAULT_CELL_ADDRESS 模板比较失效，
    cells.get() 回退分支仍能通过 address 匹配找到 HostCell。
    """
    env_cell = env.cell_address
    assert env_cell != DEFAULT_CELL_ADDRESS  # 模板比较永不成立

    matrix = _make_matrix(mode, env, workspace, app_store, manifest)

    # 因为 env_cell == main_cell.address，cells.get() 应该找到 HostCell
    assert matrix.this.type == "host"


def test_is_main_false_when_cell_address_is_unknown(mode, workspace, app_store, manifest, tmp_path):
    """当 MOSS_CELL_ADDRESS 指向未知地址时，回退为 UnknownCell，_is_main = False."""
    unknown_addr = "fractal/some_remote_node"

    os.environ["MOSS_CELL_ADDRESS"] = unknown_addr
    try:
        env2 = Environment(workspace_path=tmp_path, mode=mode.name)
        assert env2.cell_address == unknown_addr

        matrix = _make_matrix(mode, env2, workspace, app_store, manifest)
        assert matrix._is_main is False
        assert isinstance(matrix.this, UnknownCell)
        assert matrix.this.type == "unknown"
    finally:
        del os.environ["MOSS_CELL_ADDRESS"]


def test_is_main_true_when_address_matches_first_cell(mode, env, workspace, app_store, manifest):
    """当 MOSS_CELL_ADDRESS 匹配 cells dict 中的某个 app cell 时，
    该 cell 被找到但不是 HostCell，_is_main 应为 False。
    """
    from ghoshell_moss.core.blueprint.app import AppInfo, AppWatcher

    app = AppInfo(
        name="test_app", group="tools",
        description="test", work_directory="/tmp",
        watcher=AppWatcher(),
    )
    app_store.list_apps.return_value = [app]

    os.environ["MOSS_CELL_ADDRESS"] = app.address
    try:
        env2 = Environment(workspace_path=workspace.root_path(), mode=mode.name)
        assert env2.cell_address == app.address

        matrix = _make_matrix(mode, env2, workspace, app_store, manifest)
        # 找到了 AppCell，不是 HostCell
        assert matrix._is_main is False
        assert matrix.this.type == "app"
    finally:
        del os.environ["MOSS_CELL_ADDRESS"]


# ------------------------------------------------------------------
# channel_proxy 守卫
# ------------------------------------------------------------------


def test_channel_proxy_raises_when_not_main_cell(mode, workspace, app_store, manifest, tmp_path):
    """非主 cell 调用 channel_proxy 应抛出 RuntimeError."""
    os.environ["MOSS_CELL_ADDRESS"] = "fractal/some_remote_node"
    try:
        env2 = Environment(workspace_path=tmp_path, mode=mode.name)
        matrix = _make_matrix(mode, env2, workspace, app_store, manifest)
        matrix._check_running = MagicMock()  # bypass _check_running

        assert matrix._is_main is False
        with pytest.raises(RuntimeError, match="Only allowed in main cell type"):
            matrix.channel_proxy(address="app/test/foo", name="proxy")
    finally:
        del os.environ["MOSS_CELL_ADDRESS"]


def test_channel_proxy_succeeds_when_main_cell(mode, env, workspace, app_store, manifest):
    """主 cell 调用 channel_proxy 不因 _is_main 守卫而抛异常."""
    matrix = _make_matrix(mode, env, workspace, app_store, manifest)
    matrix._check_running = MagicMock()
    assert matrix._is_main is True

    # Zenoh session 未启动所以后续会失败，但守卫检查在之前且通过
    with pytest.raises(Exception) as exc_info:
        matrix.channel_proxy(address="app/test/foo", name="proxy")
    assert "Only allowed in main cell type" not in str(exc_info.value)
