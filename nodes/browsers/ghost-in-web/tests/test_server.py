"""WS edge: hello/content, cmd round-trip, the two human pushes, disconnect."""

from __future__ import annotations

import asyncio
import base64
import io
import json

import pytest
from PIL import Image
from websockets.asyncio.client import connect

from ghoshell_ghost_in_web.model import PageModel
from ghoshell_ghost_in_web.server import WebServer, decode_data_url, _downscale_jpeg


def _png_data_url(color=(255, 0, 0)) -> str:
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), color).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


async def _wait_until(pred, timeout: float = 2.0):
    for _ in range(100):
        if pred():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("condition not met within timeout")


async def _hello_content(ws, session: str = "s1", tab: int = 1):
    await ws.send(json.dumps({"type": "hello", "session": session, "boot": "x", "ua": "t"}))
    await ws.send(json.dumps({"type": "content", "tab": tab, "title": "T", "url": "https://x.example"}))


def test_decode_data_url_roundtrip():
    img = decode_data_url(_png_data_url((10, 20, 30)))
    assert img.size == (4, 4)
    assert img.getpixel((0, 0))[:3] == (10, 20, 30)


def test_decode_data_url_rejects_garbage():
    with pytest.raises(ValueError):
        decode_data_url("https://not-a-data-url")


def test_downscale_jpeg_bounds_dimension_and_format():
    img = Image.new("RGB", (2880, 964), (200, 100, 50))
    jpeg = _downscale_jpeg(img)
    back = Image.open(io.BytesIO(jpeg))
    assert back.format == "JPEG"
    assert max(back.size) <= 1280
    assert abs(back.size[0] / back.size[1] - 2880 / 964) < 0.02


@pytest.mark.asyncio
async def test_hello_and_content_create_page():
    model = PageModel()
    server = WebServer(model)
    await server.start()
    try:
        async with connect(server.url) as ws:
            await _hello_content(ws)
            await _wait_until(lambda: model.label_of("s1", 1) == "p1")
            assert model.page_by_label("p1").url == "https://x.example"
            assert model.page_by_label("p1").reachable is True
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_auth_toggles_perception():
    model = PageModel()
    server = WebServer(model)
    await server.start()
    try:
        async with connect(server.url) as ws:
            await _hello_content(ws)
            await _wait_until(lambda: model.page_by_label("p1") is not None)
            await ws.send(json.dumps({"type": "auth", "tab": 1, "on": True}))
            await _wait_until(lambda: model.page_by_label("p1").perceived)
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_send_action_roundtrip():
    model = PageModel()
    server = WebServer(model)
    await server.start()
    try:
        async with connect(server.url) as ws:
            await _hello_content(ws)
            await _wait_until(lambda: model.page_by_label("p1") is not None)

            async def answer():
                raw = await ws.recv()
                msg = json.loads(raw)
                assert msg["type"] == "cmd"
                assert msg["action"] == "read"
                await ws.send(json.dumps({
                    "type": "result", "cid": msg["cid"], "ok": True, "result": "hello body",
                }))

            task = asyncio.create_task(answer())
            result = await server.send_action("p1", "read")
            await task
            assert result["ok"] is True and result["result"] == "hello body"
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_send_action_offline_page():
    model = PageModel()
    model.update_content("s1", 1, "T", "u")  # page exists, no connection
    server = WebServer(model)
    await server.start()
    try:
        assert (await server.send_action("p1", "read"))["ok"] is False
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_screenshot_event_pushes_image_signal():
    signals = []
    model = PageModel()
    server = WebServer(model, send_signal=signals.append)
    await server.start()
    try:
        async with connect(server.url) as ws:
            await _hello_content(ws)
            await _wait_until(lambda: model.label_of("s1", 1) == "p1")
            await ws.send(json.dumps({
                "type": "event", "tab": 1, "kind": "screenshot", "data": _png_data_url(),
            }))
            await _wait_until(lambda: len(signals) == 1)
        assert signals[0].name == "aside"
        entry = next(e for e in model.audit() if e.kind == "event" and e.ok)
        assert entry.image_b64 is not None  # 审计页要能渲染出这张图
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_bad_screenshot_is_logged_not_raised():
    signals = []
    model = PageModel()
    server = WebServer(model, send_signal=signals.append)
    await server.start()
    try:
        async with connect(server.url) as ws:
            await _hello_content(ws)
            await _wait_until(lambda: model.label_of("s1", 1) == "p1")
            await ws.send(json.dumps({"type": "event", "tab": 1, "kind": "screenshot", "data": "junk"}))
            await _wait_until(lambda: any(e.detail.startswith("截图解码失败") for e in model.audit()))
        assert signals == []
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_input_raises_signal_and_logs():
    signals = []
    model = PageModel()
    server = WebServer(model, send_signal=signals.append)
    await server.start()
    try:
        async with connect(server.url) as ws:
            await _hello_content(ws)
            await _wait_until(lambda: model.label_of("s1", 1) == "p1")
            await ws.send(json.dumps({"type": "input", "tab": 1, "text": "你能看到吗"}))
            await _wait_until(lambda: len(signals) == 1)
        assert signals[0].name == "input"
        assert any(e.kind == "dialog" and e.detail.startswith("←") for e in model.audit())
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_disconnect_marks_unreachable():
    model = PageModel()
    server = WebServer(model)
    await server.start()
    try:
        async with connect(server.url) as ws:
            await _hello_content(ws)
            await _wait_until(
                lambda: model.page_by_label("p1") is not None
                and model.page_by_label("p1").reachable is True
            )
        await _wait_until(
            lambda: model.page_by_label("p1") is not None
            and model.page_by_label("p1").reachable is False
        )
    finally:
        await server.stop()
