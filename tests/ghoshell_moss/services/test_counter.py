"""Counter service — operator-level unit tests, no Matrix harness.

Exercises ServiceOperator provide / discovery / get / pub-sub / lifecycle
across two ZenohOperator instances on one shared zenoh session. This isolates
the operator from the Matrix cell layer — the V1 system-test `get` failure
(empty replies) is reproducible here or confirmed to be cell-layer wiring.

Run: pytest tests/ghoshell_moss/services/test_counter.py
"""

import asyncio
import logging

import pytest

from ghoshell_moss.depends import depend_matrix

depend_matrix()
import zenoh

from ghoshell_moss.core.blueprint.service import Sample, ServiceMeta
from ghoshell_moss.matrix.operator.zenoh_operator import ZenohOperator
from ghoshell_moss.message import unique_id
from ghoshell_moss.services.counter import CounterServer


@pytest.fixture
def scope() -> str:
    return unique_id()


@pytest.fixture
def session():
    s = zenoh.open(zenoh.Config())
    yield s
    if not s.is_closed():
        s.close()


@pytest.fixture
def logger() -> logging.Logger:
    return logging.getLogger("test_services")


@pytest.fixture
def network_ns(scope: str) -> str:
    return f"MOSS/{scope}"


def make_operator(session, network_ns: str, address: str) -> ZenohOperator:
    return ZenohOperator(
        session=session,
        network_ns=network_ns,
        this_address=address,
        logger=logging.getLogger("test_operator"),
    )


async def _wait_for(
        predicate, timeout: float = 5.0, step: float = 0.05,
) -> bool:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(step)
    return False


# ══════════════════════════════════════════════════════════════════
# 1. Discovery + query (the V1 failure path, operator-isolated)
# ══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_discovery_and_query(session, network_ns):
    server_op = make_operator(session, network_ns, "node/counter_service/abc")
    client_op = make_operator(session, network_ns, "node/counter_caller/def")
    server = CounterServer.from_operator(server_op)

    async with server_op, client_op, server:
        # -- discovery: client sees the counter service --
        services: list[ServiceMeta] = []
        for _ in range(100):
            services = await client_op.get_services_by_kind("counter")
            if services:
                break
            await asyncio.sleep(0.05)
        assert services, "client did not discover counter service"
        meta = services[0]
        assert meta["kind"] == "counter"
        assert meta["address"].startswith("node/counter_service/")

        # -- query: inc (directed to the discovered service) --
        replies = await client_op.get("counter", "inc", None, meta)
        assert replies, "inc: no reply"
        assert replies[0]["payload"].decode() == "1"

        # -- stateful: inc again increments --
        replies = await client_op.get("counter", "inc", None, meta)
        assert replies[0]["payload"].decode() == "2"

        # -- echo (params round-trip) --
        replies = await client_op.get("counter", "echo", b"hello world", meta)
        assert replies, "echo: no reply"
        assert replies[0]["payload"].decode() == "hello world"

        # -- aggregate get: no services vararg → queries all counters --
        replies = await client_op.get("counter", "echo", b"all")
        assert replies, "aggregate get: no reply"
        assert {r["address"] for r in replies} == {meta["address"]}


# ══════════════════════════════════════════════════════════════════
# 2. Pub/sub transport (the wire the webview badge/state streams use)
# ══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_pub_sub_transport(session, network_ns):
    server_op = make_operator(session, network_ns, "node/counter_service/pubsub")
    client_op = make_operator(session, network_ns, "node/counter_caller/pubsub")
    server = CounterServer.from_operator(server_op)

    async with server_op, client_op, server:
        received: list[Sample] = []

        async def _on_ping(sample: Sample):
            received.append(sample)

        handle = client_op.sub("counter", "ping", _on_ping)
        try:
            server.provider.pub("ping", b"hello")
            assert await _wait_for(lambda: bool(received)), (
                "subscriber never received the published sample"
            )
            assert received[0]["payload"] == b"hello"
            # sample envelope carries the publishing service's address
            assert received[0]["address"] == "node/counter_service/pubsub"
            assert received[0]["key"] == "ping"
        finally:
            handle.close()


# ══════════════════════════════════════════════════════════════════
# 3. Lifecycle callbacks (presence — the drifting-layer float/dim)
# ══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_lifecycle_start_stop_callbacks(session, network_ns):
    server_op = make_operator(session, network_ns, "node/counter_service/lifecycle")
    client_op = make_operator(session, network_ns, "node/counter_caller/lifecycle")

    started: list[ServiceMeta] = []
    stopped: list[ServiceMeta] = []
    h_start = client_op.on_service_start("counter", started.append)
    h_stop = client_op.on_service_stop("counter", stopped.append)

    try:
        async with client_op:
            server = CounterServer.from_operator(server_op)
            async with server_op, server:
                assert await _wait_for(lambda: bool(started)), (
                    "on_service_start never fired"
                )
                assert started[0]["kind"] == "counter"

            # server exited — token undeclared → on_service_stop
            assert await _wait_for(lambda: bool(stopped)), (
                "on_service_stop never fired"
            )
    finally:
        h_start.close()
        h_stop.close()
