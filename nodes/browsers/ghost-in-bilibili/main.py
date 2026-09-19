"""ghost-in-bilibili node entry point.

Start:  moss nodes run nodes/browsers/ghost-in-bilibili
Debug:  python main.py              # full: Matrix + channel + HTTP server
        python main.py --standalone  # HTTP server only (no Matrix)

Port:   --port N, else MOSS_GHOST_IN_BILIBILI_PORT, else 23880

The default is a fixed, uncommon port: the Chrome extension hardcodes the node URL,
so an ephemeral port would desync.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
from pathlib import Path

_NODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_NODE_DIR / "src"))

HOST = "127.0.0.1"
DEFAULT_PORT = 23880


def resolve_port() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--port", type=int, default=0)
    args, _ = parser.parse_known_args()
    if args.port:
        return args.port
    return int(os.getenv("MOSS_GHOST_IN_BILIBILI_PORT", DEFAULT_PORT))


def _standalone() -> None:
    from ghoshell_ghost_in_bilibili.model import BridgeModel
    from ghoshell_ghost_in_bilibili.server import run_server
    run_server(HOST, resolve_port(), _NODE_DIR / "index.html", BridgeModel())


async def main(matrix) -> None:
    from ghoshell_ghost_in_bilibili.model import BridgeModel
    from ghoshell_ghost_in_bilibili.server import run_server
    from ghoshell_ghost_in_bilibili.channel import build_channel

    # TODO: ghost_name 从 matrix 的 env/project 拿;先占位 "moss"。
    model = BridgeModel()
    threading.Thread(
        target=run_server, args=(HOST, resolve_port(), _NODE_DIR / "index.html", model),
        daemon=True,
    ).start()
    await matrix.provide_channel(build_channel(model))


if __name__ == "__main__":
    if "--standalone" in sys.argv:
        _standalone()
    else:
        from ghoshell_moss.core.blueprint.matrix import Matrix
        Matrix.discover().run(main)
