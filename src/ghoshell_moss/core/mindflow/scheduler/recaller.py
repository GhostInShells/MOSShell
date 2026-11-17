"""SchedulerRecaller — talk to the self that wrote a plan.

The scheduler stores an :class:`~.models.AnchorRef` with every record and never
interprets it. This is the one seam where a ghost prototype plugs in: it knows what its
anchors mean, how to render one for the model, and how to reopen that context for a
conversation.

Two rules keep the seam honest:

- **Unbound means absent.** When no implementation is bound in IoC, the ``chat`` command
  never reaches the channel surface and the firing message drops its pointer line. A
  capability that cannot be honoured must not be advertised.
- **The scheduler stays ignorant.** Nothing in this package imports memento or dsh; the
  dependency runs one way.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

from .models import AnchorRef

__all__ = ["SchedulerRecaller", "decode_anchor", "encode_anchor"]


def encode_anchor(metatype: str, **fields: Any) -> AnchorRef:
    """Serialize a prototype's own fields into the opaque payload."""
    return AnchorRef(metatype=metatype, metadata=json.dumps(fields, ensure_ascii=False))


def decode_anchor(anchor: AnchorRef) -> dict[str, Any]:
    """Read the payload back; an empty or broken payload yields an empty mapping."""
    if not anchor.metadata:
        return {}
    try:
        data = json.loads(anchor.metadata)
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


class SchedulerRecaller(ABC):
    """Reconstruct and reopen the context behind one anchor."""

    @abstractmethod
    def metatype(self) -> str:
        """The anchor convention this implementation owns, e.g. ``dsh-session``."""

    @abstractmethod
    def stamp(self) -> AnchorRef | None:
        """The anchor of the self that is writing *now*; None when it cannot be resolved.

        Called once per arming. Returning None is legal — a plan without a pointer is
        still a plan; it just cannot be asked why it exists.
        """

    @abstractmethod
    def describe(self, anchor: AnchorRef) -> str:
        """One line the model can read: where and when the plan was written."""

    @abstractmethod
    async def chat(self, anchor: AnchorRef, prompt: str) -> str:
        """One round with that past self. Blocking by contract.

        The channel is the caller: it awaits this and hands the answer back as the
        command result. Returning a receipt instead would be a different mechanism
        (``memento.chat``); this one is deliberately a question with an answer.
        """
