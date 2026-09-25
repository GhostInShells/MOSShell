"""Startup document — parsed shape of ``startup/{mode}.startup.yml``.

Fed by :meth:`Dolores._load_startup` at ``__aenter__``; consumed by
:meth:`Dolores.startup` (command + instruction → self-wake signal) and by
:meth:`Dolores.channel` (frame → the frame channel's ``init_frame``). Every field is
optional: an empty startup doc is a silent boot with no injected frame.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ghoshell_moss.channels.frame_channel import Frame

__all__ = ["StartupDoc"]


class StartupDoc(BaseModel):
    """One yaml file, one dataclass — the ghost's boot-time payload.

    :param description: Free-form label of what this startup is for; not consumed by
        the runtime, kept as a self-explanatory hint for whoever edits the file.
    :param command: A CTML fragment executed as the ghost's first action on boot.
        Empty = no forced first action.
    :param instruction: Boot-time instruction prepended to the ego's first turn.
        Empty = no boot instruction.
    :param frame: A :class:`Frame` handed to the frame channel as its ``init_frame``.
        None = no seeded frame; the channel starts empty and the model loads/defines
        one when needed.
    """

    description: str = Field(default="", description="Free-form label.")
    command: str = Field(default="", description="Forced first CTML action.")
    instruction: str = Field(default="", description="Boot-time instruction.")
    frame: Frame | None = Field(default=None, description="Seed frame for the frame channel.")
