"""Startup document — parsed shape of ``startup/{mode}.startup.yml``.

Fed by :meth:`Dolores._load_startup` at ``__aenter__``; consumed by
:meth:`Dolores.startup` (command + instruction → self-wake signal). Every field is
optional: an empty startup doc is a silent boot with no injected frame.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

__all__ = ["StartupDoc"]


class StartupDoc(BaseModel):
    """One yaml file, one dataclass — the ghost's boot-time payload.

    :param description: Free-form label of what this startup is for; not consumed by
        the runtime, kept as a self-explanatory hint for whoever edits the file.
    :param command: A CTML fragment executed as the ghost's first action on boot.
        Empty = no forced first action.
    :param instruction: Boot-time instruction prepended to the ego's first turn.
        Empty = no boot instruction.
    :param frame: Kept as raw config only. The frame channel was dropped (its
        implementation never matched the design); the block stays so an existing
        startup file still validates, and nothing reads it.
    :param default_thinking_effort: The depth the ghost declares at boot (``off`` /
        ``low`` / ``high`` / ``max``). Empty = declare nothing, leaving the depth to
        dsh/UI (the deployment's default model selection). A declared depth is written
        once and then lives in the request header — it does not pin the session, so
        the UI can still change it later.
    """

    description: str = Field(default="", description="Free-form label.")
    command: str = Field(default="", description="Forced first CTML action.")
    instruction: str = Field(default="", description="Boot-time instruction.")
    frame: dict[str, Any] | None = Field(
        default=None, description="Retained config; the frame channel is dropped."
    )
    default_thinking_effort: str = Field(
        default="",
        description="Boot-time thinking depth declaration (off/low/high/max). Empty = leave it to dsh/UI.",
    )

    @field_validator("default_thinking_effort", mode="before")
    @classmethod
    def _plain_off_is_a_depth(cls, value: object) -> object:
        """YAML 1.1 reads an unquoted ``off`` as ``False``. Accept it as the depth, not as an error.

        Guards the whole doc: a validation error here would be caught upstream and drop the entire
        startup (frame + instruction included) over one hand-written scalar.
        """
        return "off" if value is False else value
