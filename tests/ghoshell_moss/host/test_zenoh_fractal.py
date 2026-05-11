"""
验证 ZenohSessionFractal 的关键联通性：
1. provide_channel → child proxy 连接 → 执行命令
2. 重复 provide_channel 抛出 RuntimeError
"""
import asyncio
import logging
import tempfile
from pathlib import Path

import pytest

from ghoshell_moss.core.blueprint.environment import MossMeta
from ghoshell_moss.core.py_channel import PyChannel
from ghoshell_moss.bridges.zenoh_bridge import ZenohProxyChannel
from ghoshell_moss.host.zenoh_fractal import ZenohSessionFractal
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
    # cleanup
    import os
    try:
        os.unlink(f.name)
    except OSError:
        pass


@pytest.mark.asyncio
async def test_fractal_connectivity(zenoh_config_file):
    """父节点 provide channel → 子节点 proxy 连接 → 执行命令"""
    parent_meta = MossMeta(name="test_parent")
    logger = logging.getLogger("test_fractal")
    logger.setLevel(logging.DEBUG)

    parent_fractal = ZenohSessionFractal(
        meta=parent_meta,
        zenoh_conf_file=zenoh_config_file,
        logger=logger,
    )

    chan = PyChannel(name="parent")

    @chan.build.command(return_command=True)
    async def ping() -> str:
        return "pong"

    async with parent_fractal:
        parent_fractal.provide_channel(chan)

        # 等待 zenoh 网络就绪
        await asyncio.sleep(0.5)

        child_session = zenoh.open(zenoh.Config())
        try:
            proxy = ZenohProxyChannel(
                name="proxy",
                description="",
                address="test_parent",
                session_scope=ZenohSessionFractal.FRACTAL_SESSION_SCOPE,
                zenoh_session=child_session,
            )

            async with proxy.bootstrap() as runtime:
                await runtime.wait_connected()
                assert runtime.is_running()
                result = await runtime.execute_command("ping")
                assert result == "pong"
        finally:
            if not child_session.is_closed():
                child_session.close()


@pytest.mark.asyncio
async def test_fractal_double_provide_rejected(zenoh_config_file):
    """重复 provide_channel 应该抛出 RuntimeError"""
    parent_meta = MossMeta(name="test_parent")
    logger = logging.getLogger("test_fractal")

    parent_fractal = ZenohSessionFractal(
        meta=parent_meta,
        zenoh_conf_file=zenoh_config_file,
        logger=logger,
    )

    chan1 = PyChannel(name="chan1")
    chan2 = PyChannel(name="chan2")

    async with parent_fractal:
        parent_fractal.provide_channel(chan1)

        with pytest.raises(RuntimeError, match="Channel already provided"):
            parent_fractal.provide_channel(chan2)
