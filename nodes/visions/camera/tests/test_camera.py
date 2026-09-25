"""Camera producer node tests — injected fake source, no camera hardware.

Run from node root:
    uv run pytest tests/ -v        (node shared venv)  or
    .venv/bin/pytest tests/ -v     (main venv — producer is cv2-agnostic)
"""
import asyncio
import contextlib
import os
import sys
from unittest.mock import MagicMock

import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from camera_node.camera import CameraProducer

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

    def close(self):
        self._opened = False


def make_producer(detect_faces=lambda frame: [_FACE]):
    src = FakeSource()
    matrix = MagicMock()
    producer = CameraProducer(
        matrix,
        source=src,
        detect_faces=detect_faces,
        logger=MagicMock(),
        camera_index=0,
        fps=10.0,
        resolution=(64, 48),
    )
    return producer, src, matrix


async def run_loop_once(producer):
    """Run the capture loop for ~one iteration, then cancel cleanly."""
    task = asyncio.create_task(producer.run_loop())
    await asyncio.sleep(0.15)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


def test_open_close_lifecycle():
    producer, src, _ = make_producer()
    assert producer.open() is True
    producer.close()
    assert src._opened is False


@pytest.mark.asyncio
async def test_run_loop_produces_frame_and_face():
    producer, _, matrix = make_producer()
    producer.open()
    await run_loop_once(producer)
    assert producer.latest_jpeg() is not None
    assert matrix.session.topics.pub.called


@pytest.mark.asyncio
async def test_no_face_no_publish():
    producer, _, matrix = make_producer(detect_faces=lambda frame: [])
    producer.open()
    await run_loop_once(producer)
    assert producer.latest_jpeg() is not None
    assert not matrix.session.topics.pub.called
