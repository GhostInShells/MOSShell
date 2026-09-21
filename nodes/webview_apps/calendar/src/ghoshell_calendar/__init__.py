"""ghoshell_calendar — a shared calendar body for the ghost.

The sqlite file is the truth; the web page and the channel are its two faces. See
``store.py`` for why the file can be shared, ``channel.py`` for the ghost's half,
``surface.py`` for the human's, and ``reminders.py`` for how a due event reaches the
ghost's mindflow without anyone polling.
"""

from .channel import new_calendar_channel, parse_remind, parse_when
from .reminders import ReminderLoop, human_edit_signal
from .store import LEVEL_NAMES, MAX_LEVEL, CalendarStore, level_name
from .surface import CalendarSurface

__all__ = [
    "CalendarStore",
    "CalendarSurface",
    "ReminderLoop",
    "new_calendar_channel",
    "human_edit_signal",
    "parse_when",
    "parse_remind",
    "level_name",
    "LEVEL_NAMES",
    "MAX_LEVEL",
]
