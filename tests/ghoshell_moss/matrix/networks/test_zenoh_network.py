"""ZenohCellNetwork 基础单测 — announce / discover / host dedup / cross-session / channel bridge.

测试锚点:
- announce 后 on_change callback 能感知上线
- get_host() 发现 scope 内 host
- 两个独立 session 互相发现 (跨实例通讯)
- host 重复宣告 → DuplicatedError
- §SS-3 新 API: broadcast_log, check_unique, wait_connected, auto build proxy
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
    HOST_TYPE,
    WORKER_TYPE,
    DuplicatedError,
)
from ghoshell_moss.core.py_channel import PyChannel
from ghoshell_moss.matrix.networks.zenoh_cell_network import ZenohCellNetwork
from ghoshell_moss.tools.zenoh_helper import MatrixNamespace
from ghoshell_moss.contracts.logger import get_moss_logger

_logger = get_moss_logger()


def _mk_cell(
        cell_type: str = WORKER_TYPE,
        name: str = "test-cell",
) -> Cell:
    return Cell(
        meta=CellMetadata(type=cell_type, name=name),
        launcher=CellLauncher(),
        status=CellStatus(state='alive'),
    )


def _mk_host(name: str = "test-host") -> Cell:
    return _mk_cell(HOST_TYPE, name)


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

            cell = _mk_cell(WORKER_TYPE, "worker-a")
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

            cell_a = _mk_cell(WORKER_TYPE, "multi-a")
            cell_b = _mk_cell(WORKER_TYPE, "multi-b")
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
            assert found.meta.type == HOST_TYPE

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

            cell = _mk_cell(WORKER_TYPE, "cross-worker")
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

        worker_a = _mk_cell(WORKER_TYPE, "worker-x")
        worker_b = _mk_cell(WORKER_TYPE, "worker-y")

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
    """跨 session 桥接 (§SS-3 新 API): A provide(), B 自动 build proxy.

    验证 §SS-5 自动 proxy 链 — hub liveness PUT → auto_build_proxy → broadcast log.
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

        async with net_a:
            await net_a.update_cell(provider_cell)
            provider = await net_a.provide(
                provider_main, address=provider_cell.address,
            )

            # -- proxy 端: 等 auto build proxy 完成 --
            async with net_b:
                # provider + proxy 启动, 验证联通
                async with provider.arun(provider_main):
                    # 等远端 hub 看到 channel liveness, 自动 build proxy
                    ready = await net_b.wait_connected(
                        provider_cell.address, timeout=5.0,
                    )
                    assert ready, "auto build proxy timed out"

                    proxy = net_b.get_proxy(provider_cell.address)
                    assert proxy is not None, "proxy not present after wait_connected"
                    proxy_name = proxy.name()

                    # bootstrap proxy 到 ctml shell 验证命令元数据可达
                    from ghoshell_moss.core.ctml.shell import new_ctml_shell
                    shell = new_ctml_shell("channel_bridge_test")
                    shell.main_channel.import_channels(proxy)
                    async with shell:
                        await asyncio.wait_for(
                            shell.wait_connected(proxy_name), timeout=5.0,
                        )
                        metas = shell.channel_metas()
                        assert proxy_name in metas
                        assert metas[proxy_name].proxy is True

                        commands = shell.commands()
                        assert proxy_name in commands
                        assert "echo" in commands[proxy_name]

                await net_a.revoke_cell(provider_cell)

    finally:
        session_a.close()
        session_b.close()


# ==================================================================
# §SS-3 — singleton 唯一性 (使用者视角)
# ==================================================================


@pytest.mark.asyncio
async def test_singleton_same_identity_cannot_be_announced_twice():
    """同 identity 的 singleton cell 不能在同一 network 上共存.

    使用者视角: 我用同一 type/name 起两次 (uid 不同, 但 identity 一致), 第二次必失败.
    Identity 是 cell 的稳定 id, network 用它做 singleton 约束 — 实现层走哪个 key
    不在测试关心范围内.
    """
    scope = "test-singleton-identity"
    with zenoh.open(zenoh.Config()) as session:
        ns = MatrixNamespace(network_scope=scope)
        net = ZenohCellNetwork(
            session=session, logger=_logger, namespace=ns, scope=scope,
        )

        async with net:
            first = _mk_cell(WORKER_TYPE, "sole-cam")
            second = _mk_cell(WORKER_TYPE, "sole-cam")
            assert first.identity == second.identity
            assert first.address != second.address  # uid 不同, address 也不同

            await net.update_cell(first)
            with pytest.raises(DuplicatedError):
                await net.update_cell(second)

            await net.revoke_cell(first)


# ==================================================================
# §SS-3 — broadcast_log / recent_logs
# ==================================================================


async def _wait_until(predicate, timeout: float = 3.0, interval: float = 0.05) -> bool:
    """轮询等待 predicate 为真. 返回 True=ready, False=超时."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return predicate()


@pytest.mark.asyncio
async def test_broadcast_log_received_remotely():
    """net_a broadcast_log → net_b 的 log subscriber 收到 → recent_logs 含此条."""
    scope = "test-log-broadcast"
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

        async with net_b, net_a:
            cell = _mk_cell(WORKER_TYPE, "log-src")
            await net_a.update_cell(cell)

            await net_a.broadcast_log(cell.address, "hello-from-a")

            ok = await _wait_until(
                lambda: any(
                    log.address == cell.address and log.content == "hello-from-a"
                    for log in net_b.recent_logs(limit=20)
                ),
                timeout=3.0,
            )
            assert ok, f"log not received on net_b, got: {net_b.recent_logs()}"

            await net_a.revoke_cell(cell)
    finally:
        session_a.close()
        session_b.close()


@pytest.mark.asyncio
async def test_broadcast_log_terminal_pops_remote_cache():
    """terminal=True log 触发远端 cache pop + on_change(online=False).

    与 liveness DELETE 路径语义同向: cell 下线观察者必感知, 不管走哪条路径.
    """
    scope = "test-log-terminal"
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

        offline_received: list[CellAddress] = []

        def _on_change(cell: Cell, online: bool):
            if not online:
                offline_received.append(cell.address)

        async with net_b, net_a:
            net_b.on_change(_on_change)

            cell = _mk_cell(WORKER_TYPE, "term-cell")
            await net_a.update_cell(cell)

            cache_ready = await _wait_until(
                lambda: cell.address in net_b.live_cells(), timeout=3.0,
            )
            assert cache_ready, "cell not in remote cache before terminal broadcast"

            await net_a.broadcast_log(cell.address, "going-down", terminal=True)

            offline_fired = await _wait_until(
                lambda: cell.address in offline_received, timeout=3.0,
            )
            assert offline_fired, f"on_change(offline) not fired, got: {offline_received}"

            cache_cleared = await _wait_until(
                lambda: cell.address not in net_b.live_cells(), timeout=2.0,
            )
            assert cache_cleared, "cell still in remote cache after terminal log"

            await net_a.revoke_cell(cell)
    finally:
        session_a.close()
        session_b.close()


@pytest.mark.asyncio
async def test_broadcast_log_non_terminal_refreshes_cache_without_online_event():
    """非 terminal log 让远端刷新 cell snapshot, 但不重复 fire on_change(True).

    使用者视角: on_change 是上下线状态变化, snapshot 改不该当成"再次上线".
    需要新 snapshot 的调用方走 live_cells / get_live_cells.
    """
    scope = "test-log-non-terminal"
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

        change_calls: list[tuple[CellAddress, bool]] = []

        def _on_change(cell: Cell, online: bool):
            change_calls.append((cell.address, online))

        async with net_b, net_a:
            net_b.on_change(_on_change)

            cell = _mk_cell(WORKER_TYPE, "snapshot-cell")
            await net_a.update_cell(cell)

            cache_ready = await _wait_until(
                lambda: cell.address in net_b.live_cells(), timeout=3.0,
            )
            assert cache_ready

            # 此刻应仅有一次 (cell.address, True) — 上线
            online_events_before = [
                c for c in change_calls if c == (cell.address, True)
            ]
            assert len(online_events_before) == 1

            await net_a.broadcast_log(cell.address, "status-tick")

            # 等远端 log 消费完毕
            log_received = await _wait_until(
                lambda: any(
                    log.content == "status-tick" and log.address == cell.address
                    for log in net_b.recent_logs(limit=20)
                ),
                timeout=3.0,
            )
            assert log_received

            # 再给 consumer task 一点喘息时间确保不会延迟 fire
            await asyncio.sleep(0.2)

            online_events_after = [
                c for c in change_calls if c == (cell.address, True)
            ]
            assert len(online_events_after) == 1, (
                f"on_change(True) fired {len(online_events_after)} times "
                f"after non-terminal log, expected 1"
            )

            await net_a.revoke_cell(cell)
    finally:
        session_a.close()
        session_b.close()
    """broadcast_log 非本 network announce 的 address 应抛 LookupError."""
    scope = "test-log-not-owned"
    with zenoh.open(zenoh.Config()) as session:
        ns = MatrixNamespace(network_scope=scope)
        net = ZenohCellNetwork(
            session=session, logger=_logger, namespace=ns, scope=scope,
        )
        async with net:
            with pytest.raises(LookupError):
                await net.broadcast_log("worker/ghost/never-existed", "noise")


# ==================================================================
# §SS-3 — wait_connected
# ==================================================================


@pytest.mark.asyncio
async def test_wait_connected_returns_false_on_timeout():
    """wait_connected 对不存在的 address 在 timeout 后返回 False."""
    scope = "test-wait-connected-timeout"
    with zenoh.open(zenoh.Config()) as session:
        ns = MatrixNamespace(network_scope=scope)
        net = ZenohCellNetwork(
            session=session, logger=_logger, namespace=ns, scope=scope,
            allow_create_proxy=True,
        )
        async with net:
            ready = await net.wait_connected(
                "worker/never/uid", timeout=0.3,
            )
            assert ready is False


@pytest.mark.asyncio
async def test_wait_connected_returns_true_when_proxy_already_built():
    """provider arun + auto build proxy 完成后, wait_connected 立即返回 True.

    覆盖 wait_connected 的 short-circuit 分支 (proxy 已存在不等 callback).
    """
    scope = "test-wait-connected-ready"
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

        provider_main = PyChannel(name="wait_provider")

        @provider_main.build.command()
        async def ping() -> str:
            return "pong"

        provider_cell = _mk_host("wait-host")

        async with net_a:
            await net_a.update_cell(provider_cell)
            provider = await net_a.provide(
                provider_main, address=provider_cell.address,
            )

            async with net_b:
                async with provider.arun(provider_main):
                    # 第一次 wait — 走 callback 路径
                    ready = await net_b.wait_connected(
                        provider_cell.address, timeout=5.0,
                    )
                    assert ready is True

                    # 第二次 wait — 走 short-circuit (proxy 已 ready)
                    ready_again = await net_b.wait_connected(
                        provider_cell.address, timeout=0.1,
                    )
                    assert ready_again is True

                await net_a.revoke_cell(provider_cell)
    finally:
        session_a.close()
        session_b.close()
