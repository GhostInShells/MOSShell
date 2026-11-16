"""The position every beside container carries — when it is asked, and which span it is about.

A forked Emanation inherits the source session's history, and that history's last clock is the
last frame it saw — not the moment it was asked. Its own view cannot say "now is later than my
view"; the request is the only thing that can. Without it, a question issued hours (or months)
after the anchor reads as if it had been asked at the anchor.

A coordinate, not a line name: after a checkout a name points at whatever runs on that line
next, while ``{index}-{seq}`` keeps pointing at the span that was actually asked about.

The instant is a **full date** on the body's clock: the reader's view may be months old, so a
bare ``D17`` would be ambiguous, while the offset keeps a bare clock unambiguous. It is the
same offset the moment echoes carry — one clock, one way of saying it.
"""

from __future__ import annotations

from datetime import datetime

from ghoshell_moss.message import format_zone

__all__ = ["emanation_position"]


def emanation_position(*, coord: str = "", at: datetime | None = None) -> str:
    """Render `` at="2026-11-17 00:04:53+8" coord="1-756"`` for a beside container.

    ``at`` defaults to now. An aware value is converted to this body's zone; a naive one is
    read as local — the same rule the rest of the ghost's stamps follow. ``coord`` is omitted
    when empty (a note that has no commit, a path with no address), never rendered blank.
    """
    moment = (at or datetime.now()).astimezone()
    position = f' at="{moment:%Y-%m-%d %H:%M:%S}{format_zone(moment)}"'
    if coord:
        position += f' coord="{coord}"'
    return position
