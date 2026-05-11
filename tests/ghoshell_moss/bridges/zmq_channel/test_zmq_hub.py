"""
ZMQHub 单元测试 — 验证动态注册/发现/注销。
"""
import asyncio
import logging
import tempfile
from pathlib import Path

import pytest

from ghoshell_moss.core.py_channel import PyChannel
from ghoshell_moss.bridges.zmq_channel.zmq_hub import (
    ZMQHub,
    zmq_register,
    zmq_unregister,
    zmq_query,
)
from ghoshell_moss.bridges.zmq_channel.zmq_channel import (
    ZMQChannelProvider,
    ZMQSocketType,
)


def _ipc_addr(name: str) -> str:
    """生成测试用 IPC 地址。"""
    tmp = Path(tempfile.gettempdir()) / f"moss-test-zmq-hub-{name}.sock"
    # 清理可能的残留
    import os
    try:
        os.unlink(str(tmp))
    except OSError:
        pass
    return f"ipc://{tmp}"


# ------------------------------------------------------------------
# 注册/查询/注销
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_register_and_discover():
    """节点注册后 hub.registered_nodes() 可发现。"""
    hub_addr = _ipc_addr("test-register")

    hub = ZMQHub(name="test-hub", registry_address=hub_addr)
    async with hub:
        # 初始为空
        assert len(hub.registered_nodes()) == 0

        # 注册一个节点
        resp = await zmq_register(
            hub_addr, "node-a",
            channel_address="tcp://127.0.0.1:19991",
            description="Test node A",
        )
        assert resp["status"] == "ok"

        # 让 registry loop 处理
        await asyncio.sleep(0.05)

        nodes = hub.registered_nodes()
        assert len(nodes) == 1
        assert "node-a" in nodes
        assert nodes["node-a"].channel_address == "tcp://127.0.0.1:19991"
        assert nodes["node-a"].description == "Test node A"


@pytest.mark.asyncio
async def test_multiple_registrations():
    """多个节点注册。"""
    hub_addr = _ipc_addr("test-multi")

    hub = ZMQHub(name="test-hub", registry_address=hub_addr)
    async with hub:
        await zmq_register(hub_addr, "a", channel_address="tcp://127.0.0.1:10001")
        await zmq_register(hub_addr, "b", channel_address="tcp://127.0.0.1:10002")
        await zmq_register(hub_addr, "c", channel_address="tcp://127.0.0.1:10003")

        await asyncio.sleep(0.05)

        nodes = hub.registered_nodes()
        assert len(nodes) == 3
        assert set(nodes.keys()) == {"a", "b", "c"}


@pytest.mark.asyncio
async def test_unregister():
    """节点注销后 hub 移除。"""
    hub_addr = _ipc_addr("test-unregister")

    hub = ZMQHub(name="test-hub", registry_address=hub_addr)
    async with hub:
        await zmq_register(hub_addr, "node-x", channel_address="tcp://127.0.0.1:10004")
        await asyncio.sleep(0.05)
        assert "node-x" in hub.registered_nodes()

        await zmq_unregister(hub_addr, "node-x")
        await asyncio.sleep(0.05)

        assert "node-x" not in hub.registered_nodes()


@pytest.mark.asyncio
async def test_query():
    """zmq_query 返回节点列表。"""
    hub_addr = _ipc_addr("test-query")

    hub = ZMQHub(name="test-hub", registry_address=hub_addr)
    async with hub:
        # 空查询
        nodes = await zmq_query(hub_addr)
        assert nodes == []

        await zmq_register(hub_addr, "n1", channel_address="tcp://127.0.0.1:10005")
        await asyncio.sleep(0.05)

        nodes = await zmq_query(hub_addr)
        assert len(nodes) == 1
        assert nodes[0]["name"] == "n1"
        assert nodes[0]["channel_address"] == "tcp://127.0.0.1:10005"


@pytest.mark.asyncio
async def test_node_info():
    """node_info 查询单个节点。"""
    hub_addr = _ipc_addr("test-nodeinfo")

    hub = ZMQHub(name="test-hub", registry_address=hub_addr)
    async with hub:
        await zmq_register(hub_addr, "target", channel_address="tcp://127.0.0.1:10006")
        await asyncio.sleep(0.05)

        info = hub.node_info("target")
        assert info is not None
        assert info.name == "target"
        assert info.channel_address == "tcp://127.0.0.1:10006"

        assert hub.node_info("nonexistent") is None


# ------------------------------------------------------------------
# create_proxy
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_proxy():
    """为已注册节点创建 ZMQChannelProxy。"""
    hub_addr = _ipc_addr("test-proxy")

    hub = ZMQHub(name="test-hub", registry_address=hub_addr)
    async with hub:
        await zmq_register(hub_addr, "node-p", channel_address="tcp://127.0.0.1:10007")
        await asyncio.sleep(0.05)

        proxy = hub.create_proxy("node-p")
        assert proxy is not None
        assert proxy.name() == "node-p"

        # 未注册节点应抛出
        with pytest.raises(KeyError, match="not registered"):
            hub.create_proxy("nonexistent")


# ------------------------------------------------------------------
# as_channel
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_as_channel_commands():
    """as_channel() 生成的 Channel 有 list_nodes/open_node/close_node。"""
    hub_addr = _ipc_addr("test-aschannel")

    hub = ZMQHub(name="test-hub", registry_address=hub_addr)
    async with hub:
        channel = hub.as_channel()
        runtime = channel.bootstrap()

        async with runtime:
            # 验证命令存在
            cmd_list = runtime.get_command("list_nodes")
            assert cmd_list is not None
            cmd_open = runtime.get_command("open_node")
            assert cmd_open is not None
            cmd_close = runtime.get_command("close_node")
            assert cmd_close is not None

            # list_nodes 初始为空
            result = await cmd_list()
            assert "No registered nodes" in result

            # 注册节点后 list_nodes
            await zmq_register(hub_addr, "node-1", channel_address="tcp://127.0.0.1:10010")
            await asyncio.sleep(0.05)
            result = await cmd_list()
            assert "node-1" in result

            # open_node
            result = await cmd_open("node-1")
            assert "opened" in result

            # 重复 open
            result = await cmd_open("node-1")
            assert "already open" in result

            # open 不存在的节点
            result = await cmd_open("nobody")
            assert "not found" in result

            # close_node
            result = await cmd_close("node-1")
            assert "closed" in result


# ------------------------------------------------------------------
# 端到端: provider + proxy 通过 hub 连接
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_e2e_provider_registers_via_hub():
    """完整流程: provider 启动 → 注册到 hub → proxy 通过 hub 发现并连接 → 执行命令。"""
    import random
    port = random.randint(10000, 20000)
    channel_addr = f"tcp://127.0.0.1:{port}"
    hub_addr = _ipc_addr("test-e2e")

    # 1. 启动 hub
    hub = ZMQHub(name="e2e-hub", registry_address=hub_addr)
    async with hub:
        # 2. 注册到 hub
        await zmq_register(hub_addr, "server-node", channel_address=channel_addr)
        await asyncio.sleep(0.05)

        # 3. 启动 provider
        provider = ZMQChannelProvider(address=channel_addr, socket_type=ZMQSocketType.PAIR)
        server_channel = PyChannel(name="server")

        @server_channel.build.command()
        async def echo(msg: str) -> str:
            return f"echo: {msg}"

        provider.run_in_thread(server_channel)

        try:
            # 4. 通过 hub 创建 proxy
            proxy = hub.create_proxy("server-node")

            async with proxy.bootstrap() as runtime:
                await runtime.wait_connected()
                assert runtime.is_running()

                cmd = runtime.get_command("echo")
                result = await cmd("hello")
                assert result == "echo: hello"
        finally:
            provider.close()
