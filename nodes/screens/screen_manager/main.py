"""MOSS node cell entry point.

Start:  moss nodes run nodes/screens/screen_manager
Debug:  python main.py
Port:   --port N, else MOSS_SCREEN_MANAGER_PORT, else an ephemeral port (0)

One process, two faces over one store: the channel (the ghost's control surface) and
the web surface (the human's window view and steering). The human's moves reach the
ghost through ``matrix.send_signal_to_ghost``.

On the mesh, the screen is also a **consumer**: it watches the ``webview`` service
kind and adopts every live view onto the desktop, so a node that serves a page
appears on screen with no manual ``open``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_NODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_NODE_DIR / "src"))

from ghoshell_moss.core.blueprint.matrix import Matrix  # noqa: E402
from ghoshell_moss.services.webview import WebViewClient  # noqa: E402

from ghoshell_screen_manager.audio import MockAudioSource  # noqa: E402
from ghoshell_screen_manager.bridge import WebViewBridge  # noqa: E402
from ghoshell_screen_manager.channel import build_screen_channel  # noqa: E402
from ghoshell_screen_manager.model import ScreenModel  # noqa: E402
from ghoshell_screen_manager.surface import ScreenSurface  # noqa: E402

_INDEX_HTML = _NODE_DIR / "index.html"

HOST = "127.0.0.1"
DEFAULT_PORT = 0
"""0 = bind an ephemeral port; the real port is reported back to the model as a
warm ``url`` notice fragment. Web-surface nodes never claim a fixed port unless
``--port`` / ``MOSS_SCREEN_MANAGER_PORT`` asks them to, so siblings never collide."""


def resolve_port() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--port", type=int, default=0)
    args, _ = parser.parse_known_args()
    if args.port:
        return args.port
    return int(os.getenv("MOSS_SCREEN_MANAGER_PORT", DEFAULT_PORT))


async def main(matrix: Matrix) -> None:
    model = ScreenModel()
    audio = MockAudioSource()

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
        views_notice=bridge.notice,
    )
    await matrix.provide_channel(channel)


if __name__ == "__main__":
    Matrix.discover().run(main)
