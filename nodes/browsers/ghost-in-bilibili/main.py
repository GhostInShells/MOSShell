"""ghost-in-bilibili node entry point.

Start:  moss nodes run nodes/browsers/ghost-in-bilibili
Debug:  python main.py

Port:   --port N, else MOSS_GHOST_IN_BILIBILI_PORT, else 23880 (fixed — the
        extension hardcodes the node URL, so an ephemeral port would desync).

One process, two faces over one store: the channel (the ghost's control surface)
and the WS server (the extension's edge). Human panel input reaches the ghost via
``matrix.send_signal_to_ghost``; subtitle tracks persist under ``matrix.home``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_NODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_NODE_DIR / "src"))

from ghoshell_moss.core.blueprint.matrix import Matrix  # noqa: E402

from ghoshell_ghost_in_bilibili.channel import build_channel  # noqa: E402
from ghoshell_ghost_in_bilibili.model import BridgeModel  # noqa: E402
from ghoshell_ghost_in_bilibili.server import BilibiliServer  # noqa: E402
from ghoshell_ghost_in_bilibili.subtitle import SubtitleStore  # noqa: E402

HOST = "127.0.0.1"
DEFAULT_PORT = 23880


def resolve_port() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--port", type=int, default=0)
    args, _ = parser.parse_known_args()
    if args.port:
        return args.port
    return int(os.getenv("MOSS_GHOST_IN_BILIBILI_PORT", DEFAULT_PORT))


def resolve_origins() -> list[str] | None:
    """Pin the extension id (comma-separated) to keep other web pages out of the
    localhost WS. Empty = allow all (dev); the id is known only after install."""
    raw = os.getenv("MOSS_GHOST_IN_BILIBILI_ORIGINS", "")
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    return origins or None


async def main(matrix: Matrix) -> None:
    model = BridgeModel()
    subtitles = SubtitleStore(matrix.home / "subtitles")

    server = BilibiliServer(
        model,
        send_signal=matrix.send_signal_to_ghost,
        host=HOST,
        port=resolve_port(),
        origins=resolve_origins(),
    )
    await server.start()
    print(f"[ghost-in-bilibili] ws at {server.url}", flush=True)

    await matrix.provide_channel(build_channel(model, subtitles, dispatch=server))


if __name__ == "__main__":
    Matrix.discover().run(main)
