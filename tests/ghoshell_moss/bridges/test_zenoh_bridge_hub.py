import pytest
import asyncio
import zenoh

from ghoshell_moss.message import unique_id
from ghoshell_moss.core.py_channel import PyChannel
from ghoshell_moss.bridges.zenoh_bridge import ZenohChannelHub


@pytest.mark.asyncio
async def test_hub_provider_proxy_connect_and_execute():
    """Hub 创建的 provider 和 proxy 能够连接并执行命令。"""
    session = zenoh.open(zenoh.Config())
    scope = unique_id()
    hub = ZenohChannelHub(zenoh_session=session, scope=scope)

    try:
        address = "test/hub_node"
        provider = hub.provider(address)
        proxy = hub.proxy(address, name="hub_proxy")

        chan = PyChannel(name="provider")

        @chan.build.command()
        async def foo() -> int:
            return 42

        async with provider.arun(chan):
            async with proxy.bootstrap() as runtime:
                await runtime.wait_connected()
                assert runtime.is_running()
                result = await runtime.execute_command("foo")
                assert result == 42
    finally:
        if not session.is_closed():
            session.close()


@pytest.mark.asyncio
async def test_hub_discover_provider_liveness():
    """proxy wait_connected 后，hub 可通过 liveliness 发现 provider address。"""
    session = zenoh.open(zenoh.Config())
    scope = unique_id()
    hub = ZenohChannelHub(zenoh_session=session, scope=scope)

    try:
        address = "test/hub_discover"
        provider = hub.provider(address)
        proxy = hub.proxy(address, name="hub_discover_proxy")

        chan = PyChannel(name="provider")

        @chan.build.command()
        async def foo() -> int:
            return 99

        async with provider.arun(chan):
            async with proxy.bootstrap() as runtime:
                await runtime.wait_connected()

                addresses = hub.get_liveness_provider_address()
                assert address in addresses, f"expected {address} in {addresses}"
    finally:
        if not session.is_closed():
            session.close()


@pytest.mark.asyncio
async def test_hub_context_manager_auto_discover():
    """__aenter__ 后 provider 上线时 hub 自动创建 proxy 并记录 online record。"""
    session = zenoh.open(zenoh.Config())
    scope = unique_id()
    hub = ZenohChannelHub(zenoh_session=session, scope=scope)

    try:
        address = "test/hub_auto"
        chan = PyChannel(name="provider")

        @chan.build.command()
        async def foo() -> int:
            return 7

        async with hub:
            # hub 进入上下文，开始监听 provider liveness
            provider = hub.provider(address)

            async with provider.arun(chan):
                # provider 宣告 liveness 后，hub 应自动创建 proxy
                await asyncio.sleep(0.3)

                assert address in hub.proxies
                auto_proxy = hub.proxies[address]
                assert auto_proxy is not None

                # 验证自动创建的 proxy 可以正常连接
                async with auto_proxy.bootstrap() as runtime:
                    await runtime.wait_connected()
                    assert runtime.is_running()
                    result = await runtime.execute_command("foo")
                    assert result == 7

            # provider 退出后，应收到 offline record
            await asyncio.sleep(0.3)
            records = hub.records
            assert len(records) >= 2

            online_records = [r for r in records if r.status == "online" and r.address == address]
            offline_records = [r for r in records if r.status == "offline" and r.address == address]
            assert len(online_records) >= 1
            assert len(offline_records) >= 1
    finally:
        if not session.is_closed():
            session.close()


@pytest.mark.asyncio
async def test_hub_max_records():
    """records 数量不超过 max_records 配置。"""
    session = zenoh.open(zenoh.Config())
    scope = unique_id()
    max_records = 4
    hub = ZenohChannelHub(zenoh_session=session, scope=scope, max_records=max_records)

    try:
        async with hub:
            for i in range(6):
                address = f"test/hub_rec_{i}"
                provider = hub.provider(address)
                chan = PyChannel(name="provider")
                async with provider.arun(chan):
                    await asyncio.sleep(0.1)

            await asyncio.sleep(0.3)

            records = hub.records
            assert len(records) <= max_records, f"records={len(records)} exceeds max={max_records}"
            # 应该保留最新的记录
            assert all(r.address.startswith("test/hub_rec_") for r in records)
            # 最早的两个地址应该被淘汰
            addresses = {r.address for r in records}
            assert "test/hub_rec_0" not in addresses
            assert "test/hub_rec_1" not in addresses
    finally:
        if not session.is_closed():
            session.close()
