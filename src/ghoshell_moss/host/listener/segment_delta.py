"""Segment delta — turn a recognition event stream into packets that never repeat.

The listener's default contract is one delivery per segment: the tail packet carries the
whole segment text. A conversation etiquette wants something else — every clause reaches
the model, while the ASR segment keeps running (cutting the segment per clause restarts
the engine turn and drops the continuity the recognizer needs to correct itself).

That combination is only safe if the packets are incremental, and the one rule that makes
it safe is this: **a packet never repeats text an earlier packet already delivered.**

Concretely:

- a **clause** packet carries only what is new since the last delivered packet;
- a **segment** (tail) packet carries only the part no clause has delivered yet — the
  clause text it would otherwise re-send is never included;
- when the recognizer revised text it already delivered (a tail may correct an earlier
  clause), the packet says so (``revised``) and carries the authoritative full text, so
  the correction is not lost and is not mistaken for more content.

The delta is computed as "everything after the longest common prefix with the text last
delivered", never by subtracting strings by length: ASR punctuation and spacing shift
between results, and any length-based subtraction cuts a sentence in half.

Nothing here talks to the ASR, the controller or the nucleus: it is a pure function of the
event stream, so the rules above can be pinned by tests before anything is wired.
"""

from __future__ import annotations

from dataclasses import dataclass

from ghoshell_moss.contracts.asr import RecognitionEvent, RecognitionPhase

__all__ = ["DeltaPacket", "SegmentDelta"]


def _common_prefix_len(left: str, right: str) -> int:
    """Length of the longest common prefix — the boundary nothing new can start before."""
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


@dataclass
class DeltaPacket:
    """One incremental packet — what to hand over, and how it relates to what came before.

    ``delta`` is the new text (empty when the packet only carries a correction, or when
    there is nothing to say). ``text`` is the segment's full text at this instant — the
    authoritative version whenever ``revised`` is set.

    ``final`` marks the segment's last packet (a tail); ``unit`` tells the consumer which
    kind of boundary produced it, so a model can tell "he finished a sentence" from "he
    finished the whole thing".
    """

    segment_id: str
    index: int
    unit: str  # "clause" | "segment"
    delta: str
    text: str
    revised: bool = False
    final: bool = False
    clause_text: str = ""


class SegmentDelta:
    """Feed recognition events in order; get back the packets that carry only new text.

    Usage is one instance per listening session: ``feed()`` returns the packets an event
    produced (usually zero or one), and the caller delivers them. The instance holds the
    cursor — the text it has already delivered within the current segment — and resets it
    when a new ``segment_id`` appears.

    Phases: ``CLAUSE`` produces a clause packet. ``TAIL`` produces the segment packet
    (``final=True``), and only when there is something left to say. ``FIRST`` and
    ``PARTIAL`` produce nothing — they are the recognizer clearing its throat, not a
    boundary the model decided to hear.
    """

    def __init__(self) -> None:
        self._segment_id: str = ""
        # The full text of the last packet we emitted (or the segment's start, "").
        self._delivered_text: str = ""
        self._index: int = 0

    def feed(self, event: RecognitionEvent) -> list[DeltaPacket]:
        """Advance the cursor with one recognition event and return the packets it produced."""
        if event.segment_id != self._segment_id:
            # A new segment (a commit cut, or a session that reopened): the cursor belongs to
            # the old one. Nothing carries over — its text is not this segment's text.
            self._segment_id = event.segment_id
            self._delivered_text = ""
            self._index = 0

        if event.phase == RecognitionPhase.CLAUSE:
            return self._emit(event, unit="clause", final=False)
        if event.phase == RecognitionPhase.TAIL:
            return self._emit(event, unit="segment", final=True)
        return []

    def _emit(self, event: RecognitionEvent, *, unit: str, final: bool) -> list[DeltaPacket]:
        full = event.text or ""
        prefix = _common_prefix_len(self._delivered_text, full)
        revised = prefix < len(self._delivered_text)
        delta = full[prefix:]
        if not delta and not revised:
            # Nothing new and nothing corrected: silence is the honest packet.
            return []
        self._index += 1
        # The cursor is the full text, not "delivered_text + delta": after a correction the
        # authoritative version is `full`, and the next packet must diff against it.
        self._delivered_text = full
        return [DeltaPacket(
            segment_id=event.segment_id,
            index=self._index,
            unit=unit,
            delta=delta,
            text=full,
            revised=revised,
            final=final,
            clause_text=event.clause.text if event.clause is not None else "",
        )]
