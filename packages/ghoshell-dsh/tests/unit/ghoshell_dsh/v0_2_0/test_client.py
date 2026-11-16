"""DshClient — each RPC returns a value or a typed failure the caller can perceive.

The promise under test: a Remote call never ends in silence. A server "no" is a
``DshRpcException``; a missing answer (transport / HTTP / envelope) is a
``DshTransportException`` carrying the method and rpcId; only success returns a value.
"""

import logging

import httpx
import pytest
from ghoshell_dsh.v0_2_0.client import (
    DshClient,
    DshRpcException,
    DshTransportException,
)


def _client(handler) -> DshClient:
    return DshClient(
        "http://dsh.test",
        logging.getLogger("test.dsh-client"),
        transport=httpx.MockTransport(handler),
    )


def _response(*, value: dict | None = None, error: dict | None = None) -> httpx.Response:
    if error is None:
        result = {"ok": True, "value": value if value is not None else {}}
    else:
        result = {"ok": False, "error": error}
    return httpx.Response(
        200, json={"type": "server-response", "rpcId": "rpc-1", "result": result}
    )


@pytest.mark.asyncio
async def test_remote_returns_the_value():
    client = _client(lambda request: _response(value={"answer": 42}))
    try:
        answer = await client.remote("ghoshell/chatOffRecord", {"sessionId": "s"})
    finally:
        await client.close()
    assert answer == {"answer": 42}


@pytest.mark.asyncio
async def test_business_failure_raises_with_the_rpc_error():
    client = _client(lambda request: _response(
        error={"code": "gateway/failed", "message": "boom", "details": {}}
    ))
    try:
        with pytest.raises(DshRpcException) as caught:
            await client.remote("ghoshell/chatOffRecord", {})
    finally:
        await client.close()
    assert caught.value.method == "ghoshell/chatOffRecord"
    assert caught.value.error["code"] == "gateway/failed"


@pytest.mark.asyncio
async def test_transport_failure_carries_method_and_rpc_id():
    def handler(request):
        raise httpx.ConnectError("dsh is down", request=request)

    client = _client(handler)
    try:
        with pytest.raises(DshTransportException) as caught:
            await client.remote("ghoshell/chatOffRecord", {})
    finally:
        await client.close()
    assert caught.value.method == "ghoshell/chatOffRecord"
    assert caught.value.rpc_id.startswith("rpc-")
    assert isinstance(caught.value.__cause__, httpx.ConnectError)


@pytest.mark.asyncio
async def test_raw_rpc_surfaces_a_business_failure_instead_of_returning_it():
    client = _client(lambda request: _response(
        error={"code": "gateway/failed", "message": "boom", "details": {}}
    ))
    try:
        with pytest.raises(DshRpcException):
            await client.rpc("$events/result", {"args": {}})
    finally:
        await client.close()
