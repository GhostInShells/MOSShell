"""Journal channel — ghost.journal | 集成 | alpha

A debug surface, not an organ the ghost reaches for in ordinary work. The journal's read face
belongs in memory; this channel exists so the mechanism can be inspected while that wiring is
being built, and it is mounted only on request: ``ghost`` gates it as a virtual child.

Read-only on purpose. Opening a day, attaching anchors, exporting the root ``AGENTS.md`` — the
writes and governance the mechanism performs — are not commands here: they are CLI-callable and
belong to the lifecycle, not to a surface the model sees.
"""

from __future__ import annotations

import json
from datetime import date

from ghoshell_moss.core.blueprint.channel_builder import MutableChannel, new_channel

from .contract import DoloresJournal

__all__ = ["build_journal_channel"]


def _date_or_none(text: str) -> date | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        raise ValueError(f"not an ISO date: {text!r}") from None


def build_journal_channel(
        journal: DoloresJournal,
        *,
        name: str = "journal",
        description: str | None = None,
) -> MutableChannel:
    """Assemble the journal's debug face.

    :param journal: the contract; the channel never sees the implementation.
    :param name: CTML tag name (``ghost.journal``, mounted on request).
    :param description: overrides the default description.
    """
    if description is None:
        description = (
            "The journal's debug face — what memory will carry, for inspection while the organ "
            "is being built. Mounted on request; not an ordinary control surface."
        )
    chan = new_channel(name=name, description=description)

    @chan.build.instruction
    async def _instruction() -> str:
        return (
            "## Journal (debug)\n"
            "The journal's read face while its memory wiring is being built: `render` is the "
            "block memory will carry, `status` reports presence, budgets and unreadable "
            "documents, `commits` lists a day's anchors. Writes and governance are not commands "
            "here — they belong to the lifecycle."
        )

    @chan.build.command(name="render", always_observe=True)
    async def render(at: str = "") -> str:
        """Render the journal block — the face the memory surface will carry.

        :param at: an ISO date to anchor the window, e.g. `2026-11-19`; empty = today.
        """
        try:
            day = _date_or_none(at)
        except ValueError as error:
            return f"[journal] {error}"
        return await journal.render(day)

    @chan.build.command(name="status", always_observe=True)
    async def status(at: str = "") -> str:
        """Where the day stands: presence, budgets, and problems (JSON).

        :param at: an ISO date, e.g. `2026-11-19`; empty = today.
        """
        try:
            day = _date_or_none(at)
        except ValueError as error:
            return f"[journal] {error}"
        return (await journal.status(day)).model_dump_json(indent=2)

    @chan.build.command(name="commits", always_observe=True)
    async def commits(at: str = "") -> str:
        """The anchors attached to a day, oldest first (JSON list).

        :param at: an ISO date, e.g. `2026-11-19`; empty = today.
        """
        try:
            day = _date_or_none(at)
        except ValueError as error:
            return f"[journal] {error}"
        return json.dumps(await journal.commits(day))

    return chan
