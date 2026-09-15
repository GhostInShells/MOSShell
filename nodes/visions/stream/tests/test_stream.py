"""Stream node channel tests — injected fake source, no ffmpeg/stream needed.

Run from node root:
    uv run pytest tests/ -v        (node shared venv)  or
    .venv/bin/pytest tests/ -v     (main venv — controller is source-agnostic)
"""
import io
import os
import sys
import time
from unittest.mock import MagicMock

import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ghoshell_moss.core.concepts.command import Observe
from ghoshell_moss.message import Base64Image

from stream_node.source import split_latest_jpeg
from stream_node.stream import StreamController

_ADDR = "rtmp://127.0.0.1/live/desk"


def _jpeg(w=64, h=48, color=(120, 160, 200)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, "JPEG", quality=85)
    return buf.getvalue()


class FakeSource:
    def __init__(self):
        self.failed = None
        self._latest = None

    def set_frame(self, jpeg: bytes, ts: float | None = None):
        self._latest = (ts if ts is not None else time.time(), jpeg)

    def latest(self):
        return self._latest


def make_controller(**kw):
    src = FakeSource()
    ctrl = StreamController(
        source=src, address=_ADDR, label="desk", logger=MagicMock(), **kw,
    )
    return ctrl, src


async def run(chan, ctml):
    from ghoshell_moss.core.ctml import ctml_shell_test

    tasks = await ctml_shell_test(chan, ctml=ctml)
    assert len(tasks) == 1
    return await tasks[0]


def _has_image(messages) -> bool:
    for m in messages:
        for c in m.as_contents():
            if Base64Image.from_content(c) is not None:
                return True
    return False


def _image_of(result) -> Image.Image:
    for m in result.messages:
        for c in m.as_contents():
            b = Base64Image.from_content(c)
            if b is not None:
                return b.to_pil_image()
    raise AssertionError("no image in observe result")


# ---- pure: JPEG framing ---- #

def test_split_latest_jpeg_single():
    jpeg = _jpeg()
    out, rest = split_latest_jpeg(jpeg)
    assert out == jpeg
    assert rest == b""


def test_split_latest_jpeg_concatenated():
    a, b = _jpeg(color=(10, 10, 10)), _jpeg(color=(200, 200, 200))
    out, rest = split_latest_jpeg(a + b + b"\xff\xd8")  # trailing partial SOI
    assert out == b
    assert rest == b"\xff\xd8"


def test_split_latest_jpeg_incomplete():
    buf = b"\xff\xd8" + b"partial"
    out, rest = split_latest_jpeg(buf)
    assert out is None
    assert rest == buf


# ---- threshold gate (through the public capture command) ---- #

@pytest.mark.asyncio
async def test_capture_resamples_large_frame():
    ctrl, src = make_controller(max_edge=64)
    src.set_frame(_jpeg(w=320, h=180))
    result = await run(ctrl.as_channel(), "<stream:capture />")
    assert isinstance(result, Observe)
    assert max(_image_of(result).size) <= 64


@pytest.mark.asyncio
async def test_capture_passthrough_small_frame():
    ctrl, src = make_controller(max_edge=64)
    src.set_frame(_jpeg(w=64, h=48))
    result = await run(ctrl.as_channel(), "<stream:capture />")
    text = result.messages[0].to_content_string()
    assert "resampled" not in text


# ---- command surface ---- #

@pytest.mark.asyncio
async def test_status_shape():
    ctrl, _ = make_controller()
    result = await run(ctrl.as_channel(), "<stream:status />")
    assert result["address"] == _ADDR
    assert result["watch_on"] is False


@pytest.mark.asyncio
async def test_watch_toggle():
    ctrl, _ = make_controller()
    r_on = await run(ctrl.as_channel(), '<stream:watch on="true" />')
    assert "watch:on" in r_on
    r_off = await run(ctrl.as_channel(), '<stream:watch on="false" />')
    assert "watch:off" in r_off


@pytest.mark.asyncio
async def test_capture_returns_image():
    ctrl, src = make_controller()
    src.set_frame(_jpeg())
    result = await run(ctrl.as_channel(), "<stream:capture />")
    assert isinstance(result, Observe)
    assert _has_image(result.messages)


@pytest.mark.asyncio
async def test_capture_no_frame():
    ctrl, _ = make_controller()
    result = await run(ctrl.as_channel(), "<stream:capture />")
    assert "no frame" in result.messages[0].to_content_string()


# ---- context ---- #

def test_context_no_image_when_watch_off():
    ctrl, src = make_controller()
    src.set_frame(_jpeg())
    msgs = ctrl._context()
    assert not _has_image(msgs)


def test_context_image_when_watch_on_and_fresh():
    ctrl, src = make_controller()
    ctrl._watch_on = True
    src.set_frame(_jpeg())
    msgs = ctrl._context()
    assert _has_image(msgs)


def test_context_no_address_leak():
    ctrl, src = make_controller()
    ctrl._watch_on = True
    src.set_frame(_jpeg())
    msgs = ctrl._context()
    text = msgs[0].to_content_string()
    assert "<stream:" not in text
    assert "nodes/" not in text
