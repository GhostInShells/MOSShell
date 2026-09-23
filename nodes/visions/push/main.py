"""MOSS node cell entry point — the unified local visual push node.

Start:  moss nodes run nodes/visions/push
Debug:  ../.venv/bin/python main.py
Port:   --port N, else MOSS_PUSH_PORT, else an ephemeral port (0)

One process, two faces over one store: the channel (the ghost's request/stop
surface) and the web surface (the human's approval and watch surface). A live
stream is an ffmpeg child the node owns and serves as MJPEG; the human's actions
reach the ghost through ``matrix.send_signal_to_ghost``.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_NODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_NODE_DIR / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_NODE_DIR / ".env")

from ghoshell_moss.core.blueprint.matrix import Matrix  # noqa: E402

from push_node.channel import build_push_channel  # noqa: E402
from push_node.store import AcceptAll, PushStore  # noqa: E402
from push_node.surface import PushHandles, PushSurface  # noqa: E402

_INDEX_HTML = _NODE_DIR / "index.html"

HOST = "127.0.0.1"
DEFAULT_PORT = 0
"""0 = bind an ephemeral port; the real port is reported back to the model as a
warm ``url`` notice fragment. Web-surface nodes never claim a fixed port unless
``--port`` / ``MOSS_PUSH_PORT`` asks them to, so siblings never collide."""


def resolve_port() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--port", type=int, default=0)
    args, _ = parser.parse_known_args()
    if args.port:
        return args.port
    return int(os.getenv("MOSS_PUSH_PORT", DEFAULT_PORT))


async def main(matrix: Matrix) -> None:
    store = PushStore()
    accept_all = AcceptAll()
    handles = PushHandles()

    surface = PushSurface(
        store,
        accept_all,
        send_signal=matrix.send_signal_to_ghost,
        self_identity=matrix.this.address,
        host=HOST,
        port=resolve_port(),
        html_path=_INDEX_HTML,
        handles=handles,
    )

    channel = build_push_channel(
        store,
        accept_all,
        matrix.processes,
        surface=surface,
        handles=handles,
        signaler=matrix.send_signal_to_ghost,
        surface_url=lambda: surface.url,
    )

    await surface.start()
    print(f"[push] surface at {surface.url}", flush=True)
    await matrix.provide_channel(channel)


if __name__ == "__main__":
    Matrix.discover().run(main)
