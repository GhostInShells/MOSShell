"""MOSS node cell entry point — co_browser.

Start:  moss nodes run nodes/browsers/co_browser
Debug:  python main.py
Port:   --port N, else MOSS_CO_BROWSER_PORT, else an ephemeral port (0).

Two things pop open on startup: the playwright headed browser (the model's
canvas) and this node's own surface (the human's observation deck) in the
user's default browser.
"""

from __future__ import annotations

import argparse
import os
import sys
import webbrowser
from pathlib import Path

_NODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_NODE_DIR / "src"))

from ghoshell_moss.core.blueprint.matrix import Matrix  # noqa: E402

from ghoshell_co_browser.channel import build_co_browser_channel  # noqa: E402
from ghoshell_co_browser.store import FrameStore  # noqa: E402
from ghoshell_co_browser.surface import CoBrowserSurface  # noqa: E402

_DOMAIN = _NODE_DIR / "domains" / "playwright.py"
_INDEX_HTML = _NODE_DIR / "index.html"

HOST = "127.0.0.1"
DEFAULT_PORT = 0


def resolve_port() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--no-open", action="store_true",
                        help="don't auto-open the surface page in a browser")
    args, _ = parser.parse_known_args()
    if args.port:
        return args.port
    return int(os.getenv("MOSS_CO_BROWSER_PORT", DEFAULT_PORT))


def _auto_open() -> bool:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--no-open", action="store_true")
    args, _ = parser.parse_known_args()
    return not args.no_open


async def main(matrix: Matrix) -> None:
    store = FrameStore()
    surface = CoBrowserSurface(
        store,
        host=HOST,
        port=resolve_port(),
        html_path=_INDEX_HTML,
    )
    channel = build_co_browser_channel(
        str(_DOMAIN),
        matrix.processes,
        store=store,
        surface=surface,
        surface_url=lambda: surface.url,
    )

    await surface.start()
    print(f"[co_browser] frames at {surface.url}", flush=True)
    if _auto_open():
        try:
            webbrowser.open(surface.url)
        except Exception as e:
            print(f"[co_browser] auto-open failed: {e}", flush=True)
    await matrix.provide_channel(channel)


if __name__ == "__main__":
    Matrix.discover().run(main)
