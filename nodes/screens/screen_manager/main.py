"""MOSS node cell entry point.

Start:  moss nodes run nodes/screens/screen_manager
Debug:  python main.py

Port:   --port N, else MOSS_SCREEN_MANAGER_PORT, else 20381 (fixed).

A screen is a *place*, not a service: the human's window onto the ghost has to be
findable at a known address, so this node claims a fixed port from the 2038 series
(20380 is the zenoh mesh; this is its neighbour) instead of an ephemeral one. Pass
``--port`` to run a second screen beside the first.

Open:   the surface opens in the system browser on start (``--no-open`` for a
headless run). A windowless screen is one nobody can see; and because the port is
fixed, a restart re-points an already-open tab rather than stranding it.

One process, two faces over one store: the channel (the ghost's control surface) and
the web surface (the human's window view and steering). The human's moves reach the
ghost through ``matrix.send_signal_to_ghost``.

On the mesh, the screen is also a **consumer**: it watches the ``webview`` service
kind and adopts every live view onto the desktop, so a node that serves a page
appears on screen with no manual ``open``.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import webbrowser
from pathlib import Path

_NODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_NODE_DIR / "src"))

from ghoshell_moss.core.blueprint.matrix import Matrix  # noqa: E402
from ghoshell_moss.services.webview import WebViewClient  # noqa: E402

from ghoshell_screen_manager.audio import (  # noqa: E402
    AudioSource,
    MockAudioSource,
    TopicAudioSource,
)
from ghoshell_screen_manager.bridge import WebViewBridge  # noqa: E402
from ghoshell_screen_manager.channel import build_screen_channel  # noqa: E402
from ghoshell_screen_manager.model import ScreenModel  # noqa: E402
from ghoshell_screen_manager.surface import ScreenSurface  # noqa: E402

_INDEX_HTML = _NODE_DIR / "index.html"

HOST = "127.0.0.1"
DEFAULT_PORT = 20381
"""2038 series: 20380 is the zenoh mesh endpoint, so the screen sits next to the
thing that carries it. Fixed, not ephemeral — the human's window must live at an
address that survives a restart (and a second screen opts in via ``--port``)."""


def resolve_port() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--port", type=int, default=0)
    args, _ = parser.parse_known_args()
    if args.port:
        return args.port
    return int(os.getenv("MOSS_SCREEN_MANAGER_PORT", DEFAULT_PORT))


def wants_open() -> bool:
    """The browser opens by default — pass ``--no-open`` for a headless run."""
    return "--no-open" not in sys.argv[1:]


def wants_mock_audio() -> bool:
    """``--mock-audio`` swaps the live bus for the fabricated demo feed."""
    return "--mock-audio" in sys.argv[1:]


async def main(matrix: Matrix) -> None:
    model = ScreenModel()

    # The live bus is the default: the background is driven by real speech, and
    # speech arriving *is* the switch. ``--mock-audio`` is the demo/headless escape.
    audio: AudioSource
    if wants_mock_audio():
        audio = MockAudioSource()
        print("[webview_screen] audio: mock (--mock-audio)", flush=True)
    else:
        audio = TopicAudioSource(matrix.session.topics)
        await audio.start()
        print("[webview_screen] audio: live (AudioSampleTopic)", flush=True)

    surface = ScreenSurface(
        model,
        audio,
        send_signal=matrix.send_signal_to_ghost,
        self_identity=matrix.this.address,
        host=HOST,
        port=resolve_port(),
        html_path=_INDEX_HTML,
    )
    await surface.start()
    print(f"[webview_screen] surface at {surface.url}", flush=True)

    if wants_open():
        # Open just after the surface is listening, so the first request finds a
        # live server. A failure here is a fact to report, never a reason to die:
        # the screen is still usable from the url in the notice.
        try:
            opened = await asyncio.to_thread(webbrowser.open, surface.url)
            if not opened:
                print(
                    f"[webview_screen] no browser accepted the url — "
                    f"open {surface.url} by hand",
                    flush=True,
                )
        except Exception as e:  # noqa: BLE001 — never let a window kill the body
            print(f"[webview_screen] could not open a browser: {e!r}", flush=True)

    # Mesh consumer: adopt live web views onto the desktop. Optional — a screen
    # with no mesh peers just never adopts anything.
    client = await matrix.connect_service(WebViewClient)
    bridge = WebViewBridge(model, client, emit=surface.broadcast)
    await bridge.start()

    channel = build_screen_channel(
        model,
        surface=surface,
        audio=audio,
        surface_url=lambda: surface.url,
    )
    await matrix.provide_channel(channel)


if __name__ == "__main__":
    Matrix.discover().run(main)
