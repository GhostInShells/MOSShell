"""MOSS node cell entry point.

Start:  moss nodes run nodes/os/file_editor
Debug:  python main.py

The node is one process serving two faces over one store: the channel (the
ghost's control surface, projected onto the network) and the web surface (the
human's stream + verdicts). Human events signal the ghost via
``matrix.send_signal_to_ghost``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_NODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_NODE_DIR / "src"))

from ghoshell_moss.core.blueprint.matrix import Matrix  # noqa: E402

from ghoshell_file_editor.channel import build_file_editor_channel  # noqa: E402
from ghoshell_file_editor.store import ThreadStore  # noqa: E402
from ghoshell_file_editor.surface import FileEditorSurface  # noqa: E402

_INDEX_HTML = _NODE_DIR / "index.html"

HOST = "127.0.0.1"
PORT = 8767
_LOG_NAME = "file_editor.jsonl"


class _Gate:
    """The 'temporarily disabled' switch shared by channel and surface."""

    def __init__(self) -> None:
        self.enabled = True


async def main(matrix: Matrix):
    gate = _Gate()

    log_path = matrix.home / _LOG_NAME
    store = (
        ThreadStore.replay(log_path)
        if log_path.exists()
        else ThreadStore(log_path=log_path)
    )

    surface = FileEditorSurface(
        store,
        send_signal=matrix.send_signal_to_ghost,
        self_identity=matrix.this.unique_name,
        on_toggle=lambda enabled: setattr(gate, "enabled", enabled),
        host=HOST,
        port=PORT,
        html_path=_INDEX_HTML,
    )
    channel = build_file_editor_channel(
        store, surface=surface, enabled=lambda: gate.enabled,
    )

    await surface.start()
    await matrix.provide_channel(channel)


if __name__ == "__main__":
    Matrix.discover().run(main)
