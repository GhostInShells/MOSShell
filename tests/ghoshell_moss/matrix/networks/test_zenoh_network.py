"""ZenohCellNetwork 基础单测 — announce / discover / host dedup / cross-session / channel bridge.

测试锚点:
- announce 后 on_change callback 能感知上线
- get_host() 发现 scope 内 host
- 两个独立 session 互相发现 (跨实例通讯)
- host 重复宣告 → DuplicatedError
- create_provider + create_proxy 跨 session channel 桥接
"""

import asyncio
import threading
import time

import pytest
from ghoshell_moss.depends import depend_zenoh

depend_zenoh()
import zenoh

from ghoshell_moss.core.blueprint.cell import (
    Cell,
    CellAddress,
    CellMetadata,
    CellLauncher,
    CellStatus,
    CellType,
    DuplicatedError,
)
from ghoshell_moss.core.py_channel import PyChannel
from ghoshell_moss.matrix.networks.zenoh_cell_network import ZenohCellNetwork
from ghoshell_moss.tools.zenoh_helper import MatrixNamespace
from ghoshell_moss.contracts.logger import get_moss_logger

_logger = get_moss_logger()


def _mk_cell(
        cell_type: CellType | str = CellType.worker,
        name: str = "test-cell",
        *,
        channel: bool = False,
) -> Cell:
    return Cell(
        meta=CellMetadata(type=cell_type, name=name, channel=channel),
        launcher=CellLauncher(),
        status=CellStatus(state='alive'),
    )


def _mk_host(name: str = "test-host", *, channel: bool = False) -> Cell:
    return _mk_cell(CellType.host, name, channel=channel)


# ==================================================================
# announce + on_change 发现
# ==================================================================


@pytest.mark.asyncio
async def test_announce_and_discover_via_on_change():
    """announce 一个 worker cell, observer 通过 on_change callback 感知上线."""
    scope = "test-on-change"

    with zenoh.open(zenoh.Config()) as session:
        ns = MatrixNamespace(network_scope=scope)

        observer = ZenohCellNetwork(
            session=session, logger=_logger, namespace=ns, scope=scope,
        )
        announcer = ZenohCellNetwork(
            session=session, logger=_logger, namespace=ns, scope=scope,
        )

        received: list[tuple[Cell, bool]] = []
        event = threading.Event()

        def _on_change(cell: Cell, online: bool):
            received.append((cell, online))
            event.set()

        async with observer:
            observer.on_change(_on_change)

            cell = _mk_cell(CellType.worker, "worker-a")
            await announcer.update_cell(cell)

            assert event.wait(timeout=3.0), "on_change not fired within timeout"

            assert len(received) == 1
            discovered, online = received[0]
            assert online is True
            assert discovered.address == cell.address

            await announcer.revoke_cell(cell)


@pytest.mark.asyncio
async def test_announce_and_discover_multiple_cells():
    """announce 多个 cell, observer 分别感知."""
    scope = "test-multi-on-change"

    with zenoh.open(zenoh.Config()) as session:
        ns = MatrixNamespace(network_scope=scope)

        observer = ZenohCellNetwork(
            session=session, logger=_logger, namespace=ns, scope=scope,
        )
        announcer = ZenohCellNetwork(
            session=session, logger=_logger, namespace=ns, scope=scope,
        )

        received: list[CellAddress] = []
        count_event = threading.Event()

        def _on_change(cell: Cell, online: bool):
            if online:
                received.append(cell.address)
                if len(received) >= 2:
                    count_event.set()

        async with observer:
            observer.on_change(_on_change)

            cell_a = _mk_cell(CellType.worker, "multi-a")
            cell_b = _mk_cell(CellType.worker, "multi-b")
            await announcer.update_cell(cell_a)
            await announcer.update_cell(cell_b)

            assert count_event.wait(timeout=3.0), f"only got {len(received)}/2 cells"
            assert cell_a.address in received
            assert cell_b.address in received

            await announcer.revoke_cell(cell_a)
            await announcer.revoke_cell(cell_b)


# ==================================================================
# host 发现
# ==================================================================


@pytest.mark.asyncio
async def test_get_host_discovers_host():
    """announce 一个 host, get_host() 能发现并返回完整 Cell."""
    scope = "test-get-host"

    with zenoh.open(zenoh.Config()) as session:
        ns = MatrixNamespace(network_scope=scope)

        host_net = ZenohCellNetwork(
            session=session, logger=_logger, namespace=ns, scope=scope,
        )
        client_net = ZenohCellNetwork(
            session=session, logger=_logger, namespace=ns, scope=scope,
        )

        async with client_net:
            host_cell = _mk_host("main-host")
            await host_net.update_cell(host_cell)

            await asyncio.sleep(0.2)

            found = await client_net.get_host()
            assert found is not None, "get_host() returned None"
            assert found.address == host_cell.address
            assert found.meta.type == CellType.host

            await host_net.revoke_cell(host_cell)


@pytest.mark.asyncio
async def test_get_host_returns_none_when_no_host():
    """scope 内没有 host 时 get_host() 返回 None."""
    scope = "test-no-host"

    with zenoh.open(zenoh.Config()) as session:
        ns = MatrixNamespace(network_scope=scope)
        net = ZenohCellNetwork(
            session=session, logger=_logger, namespace=ns, scope=scope,
        )

        async with net:
            found = await net.get_host()
            assert found is None


# ==================================================================
# 跨 session 通讯
# ==================================================================


@pytest.mark.asyncio
async def test_two_sessions_discover_host():
    """两个独立 zenoh session 之间可以发现对方宣告的 host."""
    scope = "test-cross-session"

    session_a = zenoh.open(zenoh.Config())
    session_b = zenoh.open(zenoh.Config())

    try:
        ns_a = MatrixNamespace(network_scope=scope)
        ns_b = MatrixNamespace(network_scope=scope)

        host_net = ZenohCellNetwork(
            session=session_a, logger=_logger, namespace=ns_a, scope=scope,
        )
        client_net = ZenohCellNetwork(
            session=session_b, logger=_logger, namespace=ns_b, scope=scope,
        )

        async with client_net:
            host_cell = _mk_host("cross-host")
            await host_net.update_cell(host_cell)

            await asyncio.sleep(0.3)

            found = await client_net.get_host()
            assert found is not None, "cross-session get_host() returned None"
            assert found.address == host_cell.address
            assert found.meta.name == "cross-host"

            await host_net.revoke_cell(host_cell)

    finally:
        session_a.close()
        session_b.close()


@pytest.mark.asyncio
async def test_two_sessions_on_change():
    """跨 session — observer 通过 on_change callback 感知其他 session 的 cell 上线."""
    scope = "test-cross-on-change"

    session_a = zenoh.open(zenoh.Config())
    session_b = zenoh.open(zenoh.Config())

    try:
        ns_a = MatrixNamespace(network_scope=scope)
        ns_b = MatrixNamespace(network_scope=scope)

        announcer = ZenohCellNetwork(
            session=session_a, logger=_logger, namespace=ns_a, scope=scope,
        )
        observer = ZenohCellNetwork(
            session=session_b, logger=_logger, namespace=ns_b, scope=scope,
        )

        event = threading.Event()
        received: list[CellAddress] = []

        def _on_change(cell: Cell, online: bool):
            if online:
                received.append(cell.address)
                event.set()

        async with observer:
            observer.on_change(_on_change)

            cell = _mk_cell(CellType.worker, "cross-worker")
            await announcer.update_cell(cell)

            assert event.wait(timeout=3.0), "cross-session on_change not fired"
            assert cell.address in received

            await announcer.revoke_cell(cell)

    finally:
        session_a.close()
        session_b.close()


# ==================================================================
# host DuplicatedError
# ==================================================================


@pytest.mark.asyncio
async def test_host_duplicated_error():
    """scope 内已有 host 时, 宣告第二个 host → DuplicatedError."""
    scope = "test-host-dup"

    with zenoh.open(zenoh.Config()) as session:
        ns = MatrixNamespace(network_scope=scope)
        net = ZenohCellNetwork(
            session=session, logger=_logger, namespace=ns, scope=scope,
        )

        host_a = _mk_host("host-first")
        await net.update_cell(host_a)

        try:
            host_b = _mk_host("host-second")
            with pytest.raises(DuplicatedError):
                await net.update_cell(host_b)
        finally:
            await net.revoke_cell(host_a)


@pytest.mark.asyncio
async def test_host_duplicated_cross_session():
    """跨 session — session A 宣告 host 后, session B 宣告另一个 host → DuplicatedError."""
    scope = "test-host-dup-cross"

    session_a = zenoh.open(zenoh.Config())
    session_b = zenoh.open(zenoh.Config())

    try:
        ns_a = MatrixNamespace(network_scope=scope)
        ns_b = MatrixNamespace(network_scope=scope)

        net_a = ZenohCellNetwork(
            session=session_a, logger=_logger, namespace=ns_a, scope=scope,
        )
        net_b = ZenohCellNetwork(
            session=session_b, logger=_logger, namespace=ns_b, scope=scope,
        )

        host_a = _mk_host("host-first")
        await net_a.update_cell(host_a)

        try:
            host_b = _mk_host("host-second")
            with pytest.raises(DuplicatedError):
                await net_b.update_cell(host_b)
        finally:
            await net_a.revoke_cell(host_a)

    finally:
        session_a.close()
        session_b.close()


@pytest.mark.asyncio
async def test_worker_no_duplicated_error():
    """worker cell 不触发 DuplicatedError — 仅 host 有此约束."""
    scope = "test-worker-no-dup"

    with zenoh.open(zenoh.Config()) as session:
        ns = MatrixNamespace(network_scope=scope)
        net = ZenohCellNetwork(
            session=session, logger=_logger, namespace=ns, scope=scope,
        )

        worker_a = _mk_cell(CellType.worker, "worker-x")
        worker_b = _mk_cell(CellType.worker, "worker-y")

        await net.update_cell(worker_a)
        try:
            await net.update_cell(worker_b)
        except DuplicatedError:
            pytest.fail("worker cells should not trigger DuplicatedError")
        finally:
            await net.revoke_cell(worker_a)
            await net.revoke_cell(worker_b)


# ==================================================================
# channel bridge — provider / proxy 跨 session 通讯
# ==================================================================


@pytest.mark.asyncio
async def test_channel_bridge_cross_session():
    """跨 session: A create_provider, B create_proxy, 桥接通讯.

    验证 CellNetwork 返回的 provider/proxy 能完成基本的通道联通 —
    proxy 端拿到 provider 端 channel 的命令元数据.
    """
    scope = "test-channel-bridge"

    session_a = zenoh.open(zenoh.Config())
    session_b = zenoh.open(zenoh.Config())

    try:
        ns_a = MatrixNamespace(network_scope=scope)
        ns_b = MatrixNamespace(network_scope=scope)

        net_a = ZenohCellNetwork(
            session=session_a, logger=_logger, namespace=ns_a, scope=scope,
        )
        net_b = ZenohCellNetwork(
            session=session_b, logger=_logger, namespace=ns_b, scope=scope,
            allow_create_proxy=True,
        )

        # -- provider 端: 构建 channel 并启动 provider --
        provider_main = PyChannel(name="provider_main")

        @provider_main.build.command()
        async def echo(msg: str = "hello") -> str:
            return f"echo: {msg}"

        provider_cell = _mk_host("provider-host")
        await net_a.update_cell(provider_cell)

        provider = net_a.create_provider(provider_cell.address, provider_main)

        # -- proxy 端: 等待 cell 上线, 创建 proxy --
        async with net_b:
            # 等 host 被发现
            await asyncio.sleep(0.3)
            host = await net_b.get_host()
            assert host is not None, "provider host not discovered"

            proxy = net_b.create_proxy(
                host.address,
                name="bridge_proxy",
                description="proxy to provider_main channel",
            )

            # provider + proxy 启动, 验证联通
            async with provider.arun(provider_main):
                # 给 liveness + pub/sub 一点时间建立
                await asyncio.sleep(0.3)

                # proxy bootstrap — 连接到 provider 获取 channel 元数据
                from ghoshell_moss.core.ctml.shell import new_ctml_shell
                shell = new_ctml_shell("channel_bridge_test")
                shell.main_channel.import_channels(proxy)
                async with shell:
                    await asyncio.wait_for(
                        shell.wait_connected("bridge_proxy"), timeout=5.0,
                    )
                    metas = shell.channel_metas()
                    assert "bridge_proxy" in metas
                    assert metas["bridge_proxy"].proxy is True

                    commands = shell.commands()
                    assert "bridge_proxy" in commands
                    assert "echo" in commands["bridge_proxy"]

            await net_a.revoke_cell(provider_cell)

    finally:
        session_a.close()
        session_b.close()
