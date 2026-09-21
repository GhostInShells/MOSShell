"""MOSS node cell entry point.

Start:  moss nodes run nodes/webview_apps/zhihu
Debug:  python main.py
Port:   --port N, else MOSS_ZHIHU_PORT, else an ephemeral port (0)

One process, two faces over one store: the channel (the ghost's control surface)
and the web surface (the human's auth + action verdicts). Verdicts reach the ghost
through the channel's waiter tasks + aside signals. The zhihu-cli subprocess is the
data chassis; every command is short-lived.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_NODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_NODE_DIR / "src"))

from ghoshell_moss.core.blueprint.matrix import Matrix  # noqa: E402

from ghoshell_zhihu.channel import build_channel  # noqa: E402
from ghoshell_zhihu.cli import ZhihuCli  # noqa: E402
from ghoshell_zhihu.store import ZhihuStore  # noqa: E402
from ghoshell_zhihu.surface import ZhihuSurface  # noqa: E402

_INDEX_HTML = _NODE_DIR / "index.html"
_SKILL_DIR = _NODE_DIR / "skill"

HOST = "127.0.0.1"
DEFAULT_PORT = 0


def resolve_port() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--port", type=int, default=0)
    args, _ = parser.parse_known_args()
    if args.port:
        return args.port
    return int(os.getenv("MOSS_ZHIHU_PORT", DEFAULT_PORT))


async def main(matrix: Matrix) -> None:
    store = ZhihuStore()
    cli = ZhihuCli(_SKILL_DIR)

    surface = ZhihuSurface(
        store,
        cli,
        host=HOST,
        port=resolve_port(),
        html_path=_INDEX_HTML,
    )

    channel = build_channel(
        store,
        cli,
        signaler=matrix.send_signal_to_ghost,
        broadcast=surface.broadcast,
        identity=matrix.this.unique_name,
    )

    await surface.start()
    print(f"[zhihu] surface at {surface.url}", flush=True)

    status = await cli.run("auth", "status")
    if status.get("ok"):
        store.auth_configured = True
        store.identity = status.get("masked", "authorized")

    await matrix.provide_channel(channel)


if __name__ == "__main__":
    Matrix.discover().run(main)
