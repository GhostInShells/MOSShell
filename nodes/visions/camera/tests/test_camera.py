"""Camera node channel tests — injected fake source, no camera hardware.

Run from node root:
    uv run pytest tests/ -v        (node shared venv)  or
    .venv/bin/pytest tests/ -v     (main venv — controller is cv2-agnostic)
"""
import os
import sys
import time
from unittest.mock import MagicMock

import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ghoshell_moss.core.concepts.command import Observe
from ghoshell_moss.message import Base64Image

from camera_node.camera import CameraController

_FACE = {"x": 0.1, "y": 0.1, "w": 0.3, "h": 0.3, "cx": 0.25, "cy": 0.25}


class FakeSource:
    def __init__(self):
        self._opened = True
        self.frame = Image.new("RGB", (64, 48), (120, 160, 200))

    def open(self, index=None, width=None, height=None):
        self._opened = True
        return True

    def set_resolution(self, width=None, height=None):
        return None

    def grab(self):
        return self.frame if self._opened else None

    def is_opened(self):
        return self._opened

    def close(self):
        self._opened = False


def make_controller():
    src = FakeSource()
    ctrl = CameraController(
        None,
        source=src,
        list_cameras=lambda: [{"index": 0, "name": "facetime-hd"}],
        detect_faces=lambda frame: [_FACE],
        logger=MagicMock(),
    )
    return ctrl


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


@pytest.mark.asyncio
async def test_status_shape():
    ctrl = make_controller()
    result = await run(ctrl.as_channel(), "<camera:status />")
    assert result["camera"] == 0
    assert result["watch_on"] is False
    assert result["resolution"] == [640, 480]


@pytest.mark.asyncio
async def test_watch_toggle():
    ctrl = make_controller()
    r_on = await run(ctrl.as_channel(), '<camera:watch on="true" />')
    assert "watch:on" in r_on
    assert ctrl._watch_on is True
    r_off = await run(ctrl.as_channel(), '<camera:watch on="false" />')
    assert "watch:off" in r_off


@pytest.mark.asyncio
async def test_capture_returns_image():
    ctrl = make_controller()
    result = await run(ctrl.as_channel(), "<camera:capture />")
    assert isinstance(result, Observe)
    assert _has_image(result.messages)


@pytest.mark.asyncio
async def test_set_config_bounds():
    ctrl = make_controller()
    chan = ctrl.as_channel()
    bad = await run(chan, "<camera:set_config fps=\"99\" />")
    assert "out of bounds" in bad
    bad_res = await run(chan, '<camera:set_config resolution="500x500" />')
    assert "not allowed" in bad_res


@pytest.mark.asyncio
async def test_set_config_valid():
    ctrl = make_controller()
    result = await run(ctrl.as_channel(), '<camera:set_config fps="5.0" resolution="1280x720" />')
    assert "fps=5.0" in result
    assert "res=1280x720" in result


@pytest.mark.asyncio
async def test_list_cameras():
    ctrl = make_controller()
    result = await run(ctrl.as_channel(), "<camera:list_cameras />")
    assert result[0]["index"] == 0


def test_context_no_image_when_watch_off():
    ctrl = make_controller()
    ctrl._latest = (time.time(), Image.new("RGB", (8, 8)))
    msgs = ctrl._context()
    assert not _has_image(msgs)


def test_context_image_when_watch_on_and_fresh():
    ctrl = make_controller()
    ctrl._watch_on = True
    ctrl._latest = (time.time(), Image.new("RGB", (8, 8)))
    msgs = ctrl._context()
    assert _has_image(msgs)


def test_context_no_address_leak():
    ctrl = make_controller()
    ctrl._watch_on = True
    ctrl._latest = (time.time(), Image.new("RGB", (8, 8)))
    msgs = ctrl._context()
    text = msgs[0].to_content_string()
    assert "<camera:" not in text
    assert "nodes/" not in text
