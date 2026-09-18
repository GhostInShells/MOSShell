"""Terminal node internals.

The model's commands and the human's verdicts meet at the :class:`CardStore`;
everything else is a face over it. :mod:`card` is the data model, :mod:`channel`
the model-facing surface, :mod:`surface` the human-facing one, :mod:`poller` the
bridge to live subprocess output.
"""

from .card import Card, CardState, CardType, Interaction, Thread
from .channel import build_terminal_channel
from .store import CardStore, Mode
from .surface import StopHandles, TerminalSurface

__all__ = [
    "Card",
    "CardState",
    "CardType",
    "CardStore",
    "Interaction",
    "Mode",
    "StopHandles",
    "TerminalSurface",
    "Thread",
    "build_terminal_channel",
]
