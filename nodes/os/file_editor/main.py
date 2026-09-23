"""MOSS node cell entry point.

Start:  moss nodes run nodes/os/file_editor
Debug:  python main.py
Port:   --port N, else MOSS_FILE_EDITOR_PORT, else an ephemeral port (0)

One process, two faces over one store: the channel (the ghost's control surface,
projected onto the network) and the web surface (the human's card stream,
verdicts and questions). The human's moves reach the ghost through
``matrix.send_signal_to_ghost``.

Durability is a mirror, not a log: every action that moves the text rewrites the
thread's draft under ``runtime/drafts``, so a node crash loses the history but
not the text. At startup the drafts directory is swept — work nobody claims
within the TTL is removed, the rest are offered for adoption in the notice.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_NODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_NODE_DIR / "src"))

from ghoshell_moss.core.blueprint.matrix import Matrix  # noqa: E402

from ghoshell_file_editor.channel import build_file_editor_channel  # noqa: E402
from ghoshell_file_editor.store import DocStore  # noqa: E402
from ghoshell_file_editor.surface import FileEditorSurface  # noqa: E402

_INDEX_HTML = _NODE_DIR / "index.html"

HOST = "127.0.0.1"
DEFAULT_PORT = 0
"""0 = bind an ephemeral port; the real port is reported back to the model as a
warm ``url`` notice fragment. Web-surface nodes never claim a fixed port unless
``--port`` / ``MOSS_FILE_EDITOR_PORT`` asks them to, so siblings never collide."""


def resolve_port() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--port", type=int, default=0)
    args, _ = parser.parse_known_args()
    if args.port:
        return args.port
    return int(os.getenv("MOSS_FILE_EDITOR_PORT", DEFAULT_PORT))


class _Gate:
    """The 'temporarily disabled' switch shared by channel and surface."""

    def __init__(self) -> None:
        self.enabled = True


async def main(matrix: Matrix) -> None:
    gate = _Gate()

    store = DocStore(
        drafts_dir=matrix.home / "runtime" / "drafts",
        root=matrix.project_home,
    )
    store.sweep()

    surface = FileEditorSurface(
        store,
        send_signal=matrix.send_signal_to_ghost,
        self_identity=matrix.this.address,
        on_toggle=lambda enabled: setattr(gate, "enabled", enabled),
        host=HOST,
        port=resolve_port(),
        html_path=_INDEX_HTML,
    )
    channel = build_file_editor_channel(
        store,
        surface=surface,
        signaler=matrix.send_signal_to_ghost,
        enabled=lambda: gate.enabled,
        surface_url=lambda: surface.url,
    )

    await surface.start()
    print(f"[file_editor] cards at {surface.url} — drafts {store.drafts_dir}", flush=True)
    await matrix.provide_channel(channel)


if __name__ == "__main__":
    Matrix.discover().run(main)
