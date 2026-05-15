"""
验证 ZenohSessionFractalHub 和 ZenohSessionFractalNodeProvider 的关键联通性：

1. Hub 启动 → NodeProvider expose channel → Hub 发现 → proxy 连接 → 执行命令
2. NodeProvider 重复 __aenter__ 抛出 RuntimeError
3. Hub 重复 __aenter__ 抛出 RuntimeError
"""
import asyncio
import contextlib
import logging
import tempfile
from pathlib import Path

import pytest

from ghoshell_moss.core.py_channel import PyChannel
from ghoshell_moss.bridges.zenoh_bridge import ZenohProxyChannel
from ghoshell_moss.host.fractal import FRACTAL_SESSION_SCOPE, FractalKeyExpressions
from ghoshell_moss.host.fractal.zenoh_fractal import (
    ZenohSessionFractalHub,
    ZenohSessionFractalNodeProvider,
)
from ghoshell_moss.depends import depend_zenoh

depend_zenoh()
import zenoh

FRACTAL_CONFIG_CONTENT = """
{
  mode: "peer",
  listen: {
    endpoints: ["tcp/0.0.0.0:0"]
  }
}
"""


@pytest.fixture
def zenoh_config_file():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json5', delete=False) as f:
        f.write(FRACTAL_CONFIG_CONTENT)
        f.flush()
        yield Path(f.name)
    import os
    try:
        os.unlink(f.name)
    except OSError:
        pass


@pytest.fixture
def logger():
    logger = logging.getLogger("test_fractal")
    logger.setLevel(logging.DEBUG)
    return logger


# ------------------------------------------------------------------
# NodeProvider 联通性
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_node_provider_connectivity(zenoh_config_file, logger):
    """NodeProvider expose channel → proxy 连接 → 执行命令"""
    hub_name = "test_hub"
    node_name = "test_provider"

    provider = ZenohSessionFractalNodeProvider(
        hub_name=hub_name,
        zenoh_conf_file=zenoh_config_file,
        logger=logger,
        node_name=node_name,
    )

    chan = PyChannel(name="test_chan")

    @chan.build.command(return_command=True)
    async def ping() -> str:
        return "pong"

    test_sub_chan = PyChannel(name="test_sub_chan")
    chan.build.import_channels(test_sub_chan)

    async with provider:
        cp = provider.channel_provider(hub_name)
        assert cp is not None
        task = asyncio.create_task(cp.arun_until_closed(chan))

        await asyncio.sleep(0.5)

        key_expr = FractalKeyExpressions(hub_name=hub_name)
        proxy_address = key_expr.provider_node_address(node_name=node_name)

        child_session = zenoh.open(zenoh.Config())
        try:
            proxy = ZenohProxyChannel(
                name="proxy",
                description="",
                address=proxy_address,
                session_scope=FRACTAL_SESSION_SCOPE,
                zenoh_session=child_session,
            )

            async with proxy.bootstrap() as runtime:
                await runtime.wait_connected()
                assert runtime.is_running()
                result = await runtime.execute_command("ping")
                assert result == "pong"
                assert len(runtime.metas()) == 2
                for key, meta in runtime.metas().items():
                    assert meta.available
                    assert meta.virtual
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            if not child_session.is_closed():
                child_session.close()


# ------------------------------------------------------------------
# Hub subscriber 发现
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hub_discovers_provider(zenoh_config_file, logger):
    """Hub subscriber 接收 Provider re-put → Hub 发现节点"""
    hub_name = "test_hub"

    hub = ZenohSessionFractalHub(
        hub_name=hub_name,
        zenoh_conf_file=zenoh_config_file,
        logger=logger,
        refresh_interval=0.3,
        auto_approve_connecting=True,
    )

    provider = ZenohSessionFractalNodeProvider(
        hub_name=hub_name,
        zenoh_conf_file=zenoh_config_file,
        logger=logger,
        node_name="test_node",
        reput_interval=0.3,
    )

    async with hub:
        async with provider:
            # 等待: 初始 put + 至少一次 re-put 到达 subscriber
            await asyncio.sleep(1.0)

            connected = hub.get_connected()
            names = [c.name for c in connected]
            assert "test_node" in names, f"Expected test_node in connected, got {names}"

        # 退出 Provider → re-put 停止 → Hub 应 stale prune
        # 等待 stale_timeout (refresh_interval * 3 = 0.9s) + 一次 refresh loop tick
        await asyncio.sleep(1.5)
        connected = hub.get_connected()
        names = [c.name for c in connected]
        assert "test_node" not in names, f"Expected test_node pruned, got {names}"


# ------------------------------------------------------------------
# Hub + NodeProvider 共存
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hub_and_provider_coexist(zenoh_config_file, logger):
    """Hub 和 NodeProvider 可以同时运行，生命周期独立，互不干扰。"""
    hub_name = "test_hub"

    hub = ZenohSessionFractalHub(
        hub_name=hub_name,
        zenoh_conf_file=zenoh_config_file,
        logger=logger,
        refresh_interval=0.5,
        auto_approve_connecting=True,
    )

    provider = ZenohSessionFractalNodeProvider(
        hub_name=hub_name,
        zenoh_conf_file=zenoh_config_file,
        logger=logger,
        node_name="test_provider",
    )

    async with contextlib.AsyncExitStack() as stack:
        await stack.enter_async_context(hub)
        await stack.enter_async_context(provider)

        assert hub.is_running()
        assert provider.is_running()

        # Hub 可以创建 channel_hub
        hub_channel = hub.as_channel()
        assert hub_channel is not None

    # 退出后两者都停止
    assert not hub.is_running()
    assert not provider.is_running()


# ------------------------------------------------------------------
# 生命周期保护
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_node_provider_double_enter_rejected(zenoh_config_file, logger):
    """重复 __aenter__ 抛出 RuntimeError"""
    provider = ZenohSessionFractalNodeProvider(
        hub_name="test_hub",
        zenoh_conf_file=zenoh_config_file,
        logger=logger,
    )

    async with provider:
        with pytest.raises(RuntimeError, match="already started"):
            await provider.__aenter__()


@pytest.mark.asyncio
async def test_hub_double_enter_rejected(zenoh_config_file, logger):
    """重复 __aenter__ 抛出 RuntimeError"""
    hub = ZenohSessionFractalHub(
        hub_name="test_hub",
        zenoh_conf_file=zenoh_config_file,
        logger=logger,
    )

    async with hub:
        with pytest.raises(RuntimeError, match="already started"):
            await hub.__aenter__()
