"""MOSS node cell entry point for the decision-tree node.

Start:  moss nodes run nodes/webview_apps/decision_tree
Debug:  python main.py
Port:   --port N, else MOSS_DECISION_TREE_PORT, else an ephemeral port (0)

One process, two faces over one store: the channel (the model's control surface
over the graph) and the web surface (the human's tree + detail pane). The human's
moves — select a node, confirm an action, open a path — reach the model through
``matrix.send_signal_to_ghost``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_NODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_NODE_DIR / "src"))

from ghoshell_moss.core.blueprint.matrix import Matrix  # noqa: E402

from ghoshell_decision_tree.channel import build_decision_tree_channel  # noqa: E402
from ghoshell_decision_tree.surface import DecisionTreeSurface  # noqa: E402
from ghoshell_decision_tree.store import DecisionTreeStore  # noqa: E402

_INDEX_HTML = _NODE_DIR / "index.html"

HOST = "127.0.0.1"
DEFAULT_PORT = 0
"""0 = bind an ephemeral port; the real port is reported back to the model as a
warm ``url`` notice fragment. Web-surface nodes never claim a fixed port unless
``--port`` / ``MOSS_DECISION_TREE_PORT`` asks them to, so siblings never collide."""


def resolve_port() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--port", type=int, default=0)
    args, _ = parser.parse_known_args()
    if args.port:
        return args.port
    return int(os.getenv("MOSS_DECISION_TREE_PORT", DEFAULT_PORT))


async def main(matrix: Matrix) -> None:
    store = DecisionTreeStore(root=matrix.home / "trees")
    surface = DecisionTreeSurface(
        host=HOST,
        port=resolve_port(),
        html_path=_INDEX_HTML,
        get_state=lambda: {"trees": []},
        on_action=lambda frame: None,
    )
    channel = build_decision_tree_channel(
        store,
        surface=surface,
        signaler=matrix.send_signal_to_ghost,
    )
    await surface.start()
    print(f"[decision_tree] at {surface.url}", flush=True)
    await matrix.provide_channel(channel)


if __name__ == "__main__":
    Matrix.discover().run(main)
