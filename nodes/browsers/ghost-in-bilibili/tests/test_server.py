"""WS transport edge: hello/session, cmd round-trip, input signal, disconnect."""

from __future__ import annotations

import asyncio
import json

import pytest
from websockets.asyncio.client import connect

from ghoshell_ghost_in_bilibili.model import BridgeModel
from ghoshell_ghost_in_bilibili.server import BilibiliServer


async def _wait_until(pred, timeout: float = 2.0):
    for _ in range(100):
        if pred():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("condition not met within timeout")


async def _hello_content(ws, session: str = "s1", tab: int = 1, bvid: str = "BV1"):
    await ws.send(json.dumps({"type": "hello", "session": session, "boot": "x", "ua": "t"}))
    await ws.send(json.dumps({"type": "content", "tab": tab, "bvid": bvid, "title": "T", "url": "u"}))


@pytest.mark.asyncio
async def test_hello_and_content_create_page():
    model = BridgeModel()
    server = BilibiliServer(model)
    await server.start()
    try:
        async with connect(server.url) as ws:
            await _hello_content(ws)
            await _wait_until(lambda: model.label_of("s1", 1) == "p1")
            assert model.page_by_label("p1").bvid == "BV1"
            assert model.page_by_label("p1").reachable is True
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_send_action_roundtrip():
    model = BridgeModel()
    server = BilibiliServer(model)
    await server.start()
    try:
        async with connect(server.url) as ws:
            await _hello_content(ws)
            await _wait_until(lambda: model.page_by_label("p1") is not None)

            async def answer():
                raw = await ws.recv()
                msg = json.loads(raw)
                assert msg["type"] == "cmd"
                await ws.send(json.dumps({"type": "result", "cid": msg["cid"], "ok": True, "result": "seeked"}))

            task = asyncio.create_task(answer())
            result = await server.send_action("p1", "seek", 60)
            await task
            assert result["ok"] is True
            assert result["result"] == "seeked"
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_send_action_offline_page():
    model = BridgeModel()
    model.update_content("s1", 1, "BV1", "T", "u")  # page exists, no connection
    server = BilibiliServer(model)
    await server.start()
    try:
        result = await server.send_action("p1", "play")
        assert result["ok"] is False
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_input_raises_signal():
    signals = []
    model = BridgeModel()
    server = BilibiliServer(model, send_signal=signals.append)
    await server.start()
    try:
        async with connect(server.url) as ws:
            await _hello_content(ws)
            await _wait_until(lambda: model.label_of("s1", 1) == "p1")
            await ws.send(json.dumps({"type": "input", "tab": 1, "text": "你能看到吗"}))
            await _wait_until(lambda: len(signals) == 1)
        assert signals[0].name == "input"
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_disconnect_marks_unreachable():
    model = BridgeModel()
    server = BilibiliServer(model)
    await server.start()
    try:
        async with connect(server.url) as ws:
            await _hello_content(ws)
            await _wait_until(lambda: model.label_of("s1", 1) == "p1")
            assert model.page_by_label("p1").reachable is True
        await _wait_until(lambda: model.page_by_label("p1").reachable is False)
    finally:
        await server.stop()
