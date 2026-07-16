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
        proxy = hub.proxy(address, name_hint="hub_proxy")

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
        proxy = hub.proxy(address, name_hint="hub_discover_proxy")

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
async def test_hub_context_manager_liveness_and_explicit_proxy():
    """__aenter__ opens liveness listener; provider online produces record + fires
    callback; proxy is NOT auto-built (§UU-8: proxy = accept-on-create belongs to
    the upper CellNetwork layer, hub only fans out notifications).
    """
    session = zenoh.open(zenoh.Config())
    scope = unique_id()
    hub = ZenohChannelHub(zenoh_session=session, scope=scope)

    online_callback_fired: list[str] = []
    offline_callback_fired: list[str] = []
    hub.on_provider_online(lambda a: online_callback_fired.append(a))
    hub.on_provider_offline(lambda a: offline_callback_fired.append(a))

    try:
        address = "test/hub_auto"
        chan = PyChannel(name="provider")

        @chan.build.command()
        async def foo() -> int:
            return 7

        async with hub:
            provider = hub.provider(address)

            async with provider.arun(chan):
                # provider announces liveness; hub records + fires callback,
                # but does NOT auto-build a proxy (§UU-8).
                await asyncio.sleep(0.3)

                assert address not in hub.proxies, (
                    "hub must not auto-build proxy on provider online (§UU-8)"
                )
                assert address in online_callback_fired, (
                    f"on_provider_online should have fired for {address}"
                )

                # explicit proxy build works and can connect to the provider.
                proxy = hub.proxy(address, name_hint="explicit_proxy")
                assert hub.proxies.get(address) is proxy

                async with proxy.bootstrap() as runtime:
                    await runtime.wait_connected()
                    assert runtime.is_running()
                    result = await runtime.execute_command("foo")
                    assert result == 7

            # provider exits: hub records offline, fires callback, and drops proxy.
            await asyncio.sleep(0.3)
            assert address in offline_callback_fired
            assert address not in hub.proxies, (
                "hub should drop proxy on provider offline (see _on_provider_offline)"
            )

            records = hub.records
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
