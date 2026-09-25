"""Calendar node entry point.

Start:  moss nodes run nodes/webview_apps/calendar
Debug:  python main.py

One process, two faces over one sqlite file: the channel is the ghost's control surface,
the web page is the human's. The page binds an ephemeral port by default — read the live
URL from the channel's ``url`` notice rather than assuming one. The database lives in this
cell's ``home`` (override with ``MOSS_CALENDAR_DB``).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_NODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_NODE_DIR / "src"))

from ghoshell_moss.core.blueprint.matrix import Matrix  # noqa: E402

from ghoshell_calendar.channel import new_calendar_channel  # noqa: E402
from ghoshell_calendar.reminders import ReminderLoop, human_edit_signal  # noqa: E402
from ghoshell_calendar.store import CalendarStore  # noqa: E402
from ghoshell_calendar.surface import CalendarSurface  # noqa: E402

HOST = os.getenv("MOSS_CALENDAR_HOST", "127.0.0.1")


def resolve_port() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--port", type=int, default=0)
    args, _ = parser.parse_known_args()
    return args.port or int(os.getenv("MOSS_CALENDAR_PORT", "0"))


def resolve_db(matrix: Matrix) -> Path:
    override = os.getenv("MOSS_CALENDAR_DB", "")
    return Path(override) if override else matrix.home / "calendar.db"


async def main(matrix: Matrix) -> None:
    store = CalendarStore(resolve_db(matrix))
    surface = CalendarSurface(
        store,
        html_path=_NODE_DIR / "index.html",
        host=HOST,
        port=resolve_port(),
        on_human_edit=lambda text: matrix.send_signal_to_ghost(human_edit_signal(text)),
    )
    loop = ReminderLoop(
        store,
        send_signal=matrix.send_signal_to_ghost,
        broadcast=surface.broadcast,
    )
    await matrix.provide_channel(new_calendar_channel(store, surface, loop))


if __name__ == "__main__":
    Matrix.discover().run(main)
