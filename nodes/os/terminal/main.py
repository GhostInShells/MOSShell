"""MOSS node cell entry point.

Start:  moss nodes run nodes/os/terminal
Debug:  python main.py
Port:   --port N, else MOSS_TERMINAL_PORT, else 8768

One process, two faces over one store: the channel (the ghost's control surface,
projected onto the network) and the web surface (the human's card stream and
verdicts). The human's verdicts reach the ghost through
``matrix.send_signal_to_ghost``.

Subprocesses come from ``matrix.processes``, so their lifetimes belong to the
node process — a command outlives the card that asked for it, but never the node.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_NODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_NODE_DIR / "src"))

from ghoshell_moss.contracts.llms import LLMFuncs  # noqa: E402
from ghoshell_moss.core.blueprint.matrix import Matrix  # noqa: E402
from ghoshell_moss.ground import DefaultGroundSet  # noqa: E402

from ghoshell_terminal.channel import build_terminal_channel  # noqa: E402
from ghoshell_terminal.store import CardStore, Mode  # noqa: E402
from ghoshell_terminal.surface import StopHandles, TerminalSurface  # noqa: E402

_INDEX_HTML = _NODE_DIR / "index.html"

HOST = "127.0.0.1"
DEFAULT_PORT = 8768


def resolve_port() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--port", type=int, default=0)
    args, _ = parser.parse_known_args()
    if args.port:
        return args.port
    return int(os.getenv("MOSS_TERMINAL_PORT", DEFAULT_PORT))


async def main(matrix: Matrix) -> None:
    stops = StopHandles()
    store = CardStore(
        root=matrix.project_home,
        outputs_dir=matrix.home / "runtime" / "outputs",
        log_dir=matrix.home / "runtime" / "cards",
    )
    groundset = DefaultGroundSet(workspace_root=matrix.project_home, materialize=False)

    def get_llm_funcs():
        """Lazy — resolve LLMFuncs only when an analyze request actually arrives."""
        try:
            return matrix.container.get(LLMFuncs)
        except Exception:
            return None

    surface = TerminalSurface(
        store,
        send_signal=matrix.send_signal_to_ghost,
        self_identity=matrix.this.unique_name,
        host=HOST,
        port=resolve_port(),
        html_path=_INDEX_HTML,
        stops=stops,
        llm_funcs=get_llm_funcs,
        groundset=groundset,
    )
    channel = build_terminal_channel(
        store,
        matrix.processes,
        surface=surface,
        signaler=matrix.send_signal_to_ghost,
        stops=stops,
        groundset=groundset,
        enabled=lambda: store.mode != Mode.DISABLED,
    )

    await surface.start()
    print(f"[terminal] cards at {surface.url} — root {store.root}", flush=True)
    await matrix.provide_channel(channel)


if __name__ == "__main__":
    Matrix.discover().run(main)
