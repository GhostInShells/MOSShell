"""beside — a parallel thinking group the ego issues now and reads later.

`ghost.beside.think` is the first `beside` timing primitive: the ego keeps interacting, and a
group of pure-thinking directions runs off-record on the same session context, one step behind.
The receipt is immediate; the conclusions arrive as one `mind/delivery` signal.

Config lives in the ghost's own store (`.dolores/configs/ego_beside_mind.yml`), because the
switch and budgets are runtime knobs, not ghost-wide assembly. `EgoBesideChannel.factory`
reads it: a disabled organ mounts nothing at all.
"""

from .channel import EgoBesideChannel, build_beside_channel
from .config import EgoBesideMindConfig
from .runtime import BesideRuntime, BesideTicket

__all__ = [
    "BesideRuntime",
    "BesideTicket",
    "EgoBesideChannel",
    "EgoBesideMindConfig",
    "build_beside_channel",
]
