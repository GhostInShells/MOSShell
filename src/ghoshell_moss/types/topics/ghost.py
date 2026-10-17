"""Global ghost protocol topics — what a ghost broadcasts about *how it is*.

Distinct from the capability topics (audio / vision): those carry what the ghost
perceives or says. These carry what is happening *to* it — thinking, idle, a tool
call — so a surface can follow a ghost without being in the loop. A screen, a
status bar, a debug panel.

The schema is not redefined here. ``GhostEvent`` is defined once, in the ghost
abstraction, and this gives it a second identity: the same object, also allowed to
travel on the bus. Two definitions of one event would be two truths to keep in
step.
"""

from __future__ import annotations

from ghoshell_moss.core.blueprint.ghost import GhostEvent
from ghoshell_moss.core.concepts.topic import TopicModel

__all__ = [
    "GhostEventTopic",
    "GHOST_THINKING",
    "GHOST_IDLE",
    "GHOST_TOOL_CALL",
    "GHOST_TOOL_RESULT",
]

# Namespaced, so consumers dispatch by name and ghosts never collide. These four
# are the lifecycle events the runtime itself emits; a ghost may add its own under
# the same prefix.
GHOST_THINKING = "ghost/lifecycle/thinking"
GHOST_IDLE = "ghost/lifecycle/idle"
GHOST_TOOL_CALL = "ghost/tool/call"
GHOST_TOOL_RESULT = "ghost/tool/result"


class GhostEventTopic(GhostEvent, TopicModel):
    """A ``GhostEvent`` that is also a topic.

    The event's own fields are the schema — ``event`` (namespaced name),
    ``content`` (the one line a surface shows) and ``payload`` (opaque) — and
    ``TopicModel`` adds only the envelope. Inheritance rather than a copy, so a
    change to what an event *is* cannot leave the two drifting.

    Consumers that only narrate read ``content`` and never touch the payload; that
    is the whole reason the summary field exists.
    """

    @classmethod
    def topic_type(cls) -> str:
        return "ghost_event"

    @classmethod
    def default_topic_name(cls) -> str:
        return "ghost_event"
