"""Screen Capture App — screenshot via mss on each Ghost think cycle.

Captures the screen on demand via context_messages so the Ghost always sees
the freshest frame, with zero overhead when nobody is looking.
"""
from __future__ import annotations

import asyncio

from pathlib import Path

import mss
from PIL import Image

from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.core.blueprint.channel_builder import (
    new_channel,
    Message,
)

_APP_DIR = Path(__file__).parent

channel = new_channel(
    name="screen_capture",
    description=(
        "Screen capture via mss — captures the current screen on each think cycle. "
        "Ghost sees the latest screenshot via context_messages. "
        "IMPORTANT: The screenshot is ALREADY injected into your context every round — "
        "you can see the screen without calling anything. Do NOT call exec or any capture "
        "command to get a screenshot; you already have it."
    ),
)

_active_monitor: int = 0


def _get_physical_monitors(all_monitors: list) -> tuple[list[dict], list[int]]:
    """Filter out virtual combined screen (macOS monitor 0), return physical monitors
    and their original indices."""
    if len(all_monitors) <= 1:
        return list(all_monitors), list(range(len(all_monitors)))

    # Compute bounding box of monitors 1..n. If monitor 0 matches it exactly,
    # it's the virtual combined screen (macOS).
    rest = all_monitors[1:]
    min_left = min(m["left"] for m in rest)
    min_top = min(m["top"] for m in rest)
    max_right = max(m["left"] + m["width"] for m in rest)
    max_bottom = max(m["top"] + m["height"] for m in rest)
    combined_w = max_right - min_left
    combined_h = max_bottom - min_top

    m0 = all_monitors[0]
    if (m0["left"] == min_left and m0["top"] == min_top
            and m0["width"] == combined_w and m0["height"] == combined_h):
        # Monitor 0 is virtual — skip it.
        return rest, list(range(1, len(all_monitors)))

    return list(all_monitors), list(range(len(all_monitors)))


@channel.build.context_messages
async def context() -> list:
    """Capture current screen and provide as context for Ghost with monitor info."""
    img, monitors_text = await _capture_with_info(_active_monitor)
    if img is not None:
        img.save(_APP_DIR / "tmp.png", format="PNG")
        return [
            Message.new().with_content(
                f"[screen_capture] Monitor {_active_monitor} ({img.size[0]}x{img.size[1]})"
            ),
            Message.new().with_content(monitors_text),
            Message.new().with_content(img),
        ]
    return [
        Message.new().with_content(
            "[screen_capture] Failed to capture screen."
        )
    ]


async def _capture_with_info(monitor_index: int) -> tuple[Image.Image | None, str]:
    """Capture screenshot and build monitor list in one mss session."""

    def _grab():
        with mss.MSS() as sct:
            physical, _orig_indices = _get_physical_monitors(sct.monitors)
            lines = []
            for i, m in enumerate(physical):
                lines.append(f"  [{i}] {m['width']}x{m['height']} at ({m['left']},{m['top']})")
            monitors_text = "[screen_capture] Available monitors:\n" + "\n".join(lines)

            if monitor_index >= len(physical):
                return None, monitors_text
            mon = physical[monitor_index]
            sct_img = sct.grab(mon)
            img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
            return img, monitors_text

    return await asyncio.to_thread(_grab)


async def main(matrix: Matrix):
    matrix.logger.info("screen_capture started")
    await matrix.provide_channel(channel)

    quit_event = asyncio.Event()
    try:
        await quit_event.wait()
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    Matrix.discover().run(main)
