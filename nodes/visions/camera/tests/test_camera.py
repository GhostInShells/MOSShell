"""Camera node channel tests — injected fake source, no camera hardware.

Run from node root:
    uv run pytest tests/ -v        (node shared venv)  or
    .venv/bin/pytest tests/ -v     (main venv — controller is cv2-agnostic)
"""
import os
import sys
from unittest.mock import MagicMock

import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from camera_node.camera import CameraController

_FACE = {"x": 0.1, "y": 0.1, "w": 0.3, "h": 0.3, "cx": 0.25, "cy": 0.25}


class FakeSource:
    def __init__(self):
        self._opened = True
        self.frame = Image.new("RGB", (64, 48), (120, 160, 200))

    def open(self, index=None, width=None, height=None):
        self._opened = True
        return True

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
async def test_capture():
    ctrl = make_controller()
    result = await run(ctrl.as_channel(), "<camera:capture />")
    assert "captured 64x48" in result
    assert len(ctrl._cache) == 1


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


@pytest.mark.asyncio
async def test_detect_faces():
    ctrl = make_controller()
    result = await run(ctrl.as_channel(), "<camera:detect_faces />")
    assert result[0]["cx"] == 0.25


@pytest.mark.asyncio
async def test_context_has_frame():
    ctrl = make_controller()
    chan = ctrl.as_channel()
    await run(chan, "<camera:capture />")
    msgs = await ctrl._context()
    names = {m.name for m in msgs}
    assert "__camera_frame__" in names
    assert "__camera_status__" in names
