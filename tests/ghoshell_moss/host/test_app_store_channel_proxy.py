"""
AppStoreChannel proxy 连接有效性验证。

场景: provider 启动了远程 channel (App 已打开)
     → AppStoreChannelState.get_virtual_children() 返回 ChannelProxy
     → bootstrap proxy 并调用命令。
"""
import asyncio
import pytest
from unittest.mock import MagicMock

from ghoshell_moss.core.py_channel import PyChannel
from ghoshell_moss.core.blueprint.app import AppInfo, AppWatcher
from ghoshell_moss.bridges.zenoh_bridge import ZenohChannelProvider, ZenohProxyChannel
from ghoshell_moss.host.app_store_channel import AppStoreChannelState
from ghoshell_moss.depends import depend_zenoh

depend_zenoh()
import zenoh


@pytest.fixture
def zenoh_session():
    session = zenoh.open(zenoh.Config())
    yield session
    if not session.is_closed():
        session.close()


@pytest.fixture
def session_scope():
    from ghoshell_common.helpers import uuid
    return uuid()


@pytest.fixture
def mock_app():
    return AppInfo(
        name="test_app",
        group="test_group",
        description="A test app",
        work_directory="/tmp",
        watcher=AppWatcher(),
    )


# -- helpers --


def _build_state(mock_app, zenoh_session, session_scope, **store_kw):
    """构建一个带 mock AppStore 和真实 ZenohProxyChannel 的 AppStoreChannelState."""
    mock_store = MagicMock()
    mock_store.list_apps.return_value = [mock_app]
    for attr, value in store_kw.items():
        setattr(mock_store, attr, value)

    mock_matrix = MagicMock()
    mock_matrix.is_running.return_value = True

    def _channel_proxy(*, address, name, description='', id=None, **kwargs):
        return ZenohProxyChannel(
            name=name,
            description=description,
            address=address,
            session_scope=session_scope,
            zenoh_session=zenoh_session,
            uid=id,
        )

    mock_matrix.channel_proxy = _channel_proxy

    return AppStoreChannelState(
        app_store=mock_store,
        matrix=mock_matrix,
        name="apps",
        description="App Store",
    )


# ------------------------------------------------------------------
# own commands
# ------------------------------------------------------------------


def test_own_commands(zenoh_session, session_scope, mock_app):
    """AppStoreChannelState 自身的命令: start / stop / list_apps."""
    state = _build_state(mock_app, zenoh_session, session_scope)
    cmds = state.own_commands()
    assert "start" in cmds
    assert "stop" in cmds
    assert "list_apps" in cmds


# ------------------------------------------------------------------
# virtual children: provider 未启动
# ------------------------------------------------------------------


def test_virtual_children_without_provider(zenoh_session, session_scope, mock_app):
    """provider 未启动时, get_virtual_children() 仍然返回 proxy 对象."""
    state = _build_state(mock_app, zenoh_session, session_scope)
    children = state.get_virtual_children()

    proxy_name = mock_app.fullname.replace('/', '_')
    assert proxy_name in children
    assert children[proxy_name].name() == proxy_name


# ------------------------------------------------------------------
# virtual children: provider 已启动
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_virtual_children_with_provider_running(
    zenoh_session, session_scope, mock_app,
):
    """provider 已启动时, virtual child proxy 可 bootstrap, 连接, 并调用命令."""
    # 1. 创建 App 侧的 channel
    app_chan = PyChannel(name="test_group_test_app")

    @app_chan.build.command()
    async def ping() -> str:
        return "pong"

    # 2. 启动 provider (模拟 App 调用了 matrix.provide_channel)
    provider = ZenohChannelProvider(
        zenoh_session=zenoh_session,
        address=mock_app.address,
        session_scope=session_scope,
    )

    state = _build_state(mock_app, zenoh_session, session_scope)

    async with provider.arun(app_chan):
        # 3. 获取 virtual children
        children = state.get_virtual_children()
        proxy_name = mock_app.fullname.replace('/', '_')
        assert proxy_name in children

        channel_proxy = children[proxy_name]

        # 4. bootstrap → 连接 → 调用
        async with channel_proxy.bootstrap() as runtime:
            await runtime.wait_connected()
            assert runtime.has_own_command("ping")
            result = await runtime.execute_command("ping")
            assert result == "pong"


# ------------------------------------------------------------------
# 核心 bug 场景: provider 已启动, 但 proxy 要先获取再 bootstrap
# (Shell 的实际路径: channel tree 在 refresh 时 add → bootstrap)
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_proxy_bootstrap_after_provider_started(
    zenoh_session, session_scope, mock_app,
):
    """
    模拟 Shell 路径: 先从 get_virtual_children() 拿到 proxy,
    之后再 bootstrap. 验证 proxy 的 bootstrap 时机不影响连通.
    """
    # 1. 先创建 proxy (此时 provider 未启动)
    state = _build_state(mock_app, zenoh_session, session_scope)
    children = state.get_virtual_children()
    proxy_name = mock_app.fullname.replace('/', '_')
    proxy = children[proxy_name]

    # 2. 启动 provider
    app_chan = PyChannel(name="test_group_test_app")

    @app_chan.build.command()
    async def greet(name: str) -> str:
        return f"Hello, {name}!"

    provider = ZenohChannelProvider(
        zenoh_session=zenoh_session,
        address=mock_app.address,
        session_scope=session_scope,
    )

    async with provider.arun(app_chan):
        # 3. 现在 bootstrap proxy (Shell 的 tree.add 会做这一步)
        async with proxy.bootstrap() as runtime:
            await runtime.wait_connected()
            assert runtime.has_own_command("greet")
            result = await runtime.execute_command("greet", kwargs={"name": "MOSS"})
            assert result == "Hello, MOSS!"

    # 4. provider 退出后, proxy 再次 bootstrap — 连不上
    async with proxy.bootstrap() as runtime:
        assert runtime.is_running()
        assert not runtime.is_connected()


# ------------------------------------------------------------------
# start 命令 timeout 参数
# ------------------------------------------------------------------


def _async_return(value):
    async def _inner(*args, **kwargs):
        return value
    return _inner


@pytest.mark.asyncio
async def test_start_timeout_default_no_wait(
    zenoh_session, session_scope, mock_app,
):
    """timeout=-1 (默认): 调用 start_app 后立即返回，不等待连接。"""
    state = _build_state(
        mock_app, zenoh_session, session_scope,
        start_app=_async_return("App started"),
    )

    cmd = state.get_own_command("start")
    result = await cmd(fullname="test_group/test_app", timeout=-1)
    assert "App started" in result
    assert "[OK]" not in result
    assert "[WARN]" not in result


@pytest.mark.asyncio
async def test_start_without_channel_runtime(
    zenoh_session, session_scope, mock_app,
):
    """timeout>=0 但不在 ChannelCtx 中: 返回 WARN。"""
    state = _build_state(
        mock_app, zenoh_session, session_scope,
        start_app=_async_return("App started"),
    )

    cmd = state.get_own_command("start")
    result = await cmd(fullname="test_group/test_app", timeout=0)
    assert "App started" in result
    assert "Not in channel runtime" in result


@pytest.mark.asyncio
async def test_start_timeout_zero_connects_via_tree(
    zenoh_session, session_scope, mock_app,
):
    """timeout=0 且 provider 运行中: tree 路径 refresh → fetch → wait_connected 全通。"""
    from ghoshell_moss.core.concepts.channel import ChannelCtx
    from ghoshell_moss.host.app_store_channel import build_apps_channel

    app_chan = PyChannel(name="test_group_test_app")

    @app_chan.build.command()
    async def ping() -> str:
        return "pong"

    provider = ZenohChannelProvider(
        zenoh_session=zenoh_session,
        address=mock_app.address,
        session_scope=session_scope,
    )

    mock_store = MagicMock()
    mock_store.list_apps.return_value = [mock_app]
    mock_store.start_app = _async_return("App started")

    mock_matrix = MagicMock()
    mock_matrix.is_running.return_value = True
    mock_matrix.channel_proxy = lambda *, address, name, description='', id=None, **kw: ZenohProxyChannel(
        name=name, description=description,
        address=address, session_scope=session_scope,
        zenoh_session=zenoh_session, uid=id,
    )

    chan = build_apps_channel(store=mock_store, matrix=mock_matrix, name="apps")

    async with provider.arun(app_chan):
        runtime = chan.bootstrap()  # 自动创建 tree, 自身为根节点
        await runtime.start()
        try:
            cmd = runtime.get_own_command("start")
            result = await ChannelCtx(runtime=runtime).run(
                cmd, fullname="test_group/test_app", timeout=0,
            )
            assert "App started" in result
            assert "[OK] App channel connected and ready" in result
        finally:
            await runtime.close()


@pytest.mark.asyncio
async def test_start_timeout_positive_expires(
    zenoh_session, session_scope, mock_app,
):
    """timeout>0 且 provider 未启动: 超时返回 WARN。"""
    from ghoshell_moss.core.concepts.channel import ChannelCtx
    from ghoshell_moss.host.app_store_channel import build_apps_channel

    mock_store = MagicMock()
    mock_store.list_apps.return_value = [mock_app]
    mock_store.start_app = _async_return("App started")

    mock_matrix = MagicMock()
    mock_matrix.is_running.return_value = True
    mock_matrix.channel_proxy = lambda *, address, name, description='', id=None, **kw: ZenohProxyChannel(
        name=name, description=description,
        address=address, session_scope=session_scope,
        zenoh_session=zenoh_session, uid=id,
    )

    chan = build_apps_channel(store=mock_store, matrix=mock_matrix, name="apps")

    runtime = chan.bootstrap()
    await runtime.start()
    try:
        cmd = runtime.get_own_command("start")
        result = await ChannelCtx(runtime=runtime).run(
            cmd, fullname="test_group/test_app", timeout=0.5,
        )
        assert "App started" in result
        assert "[WARN] App started but channel not connected" in result
    finally:
        await runtime.close()
