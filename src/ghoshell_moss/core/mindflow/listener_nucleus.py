"""ListenerNucleus — maps listener-side intent into listener impulses.

The listener side (ListenerController) already decides when to speak up, when to
deliver, at which priority, and with which loss-side mode. This nucleus is the
**hearing** protocol mapping: it turns one listener signal into one impulse,
mechanically, without interpreting anything.

Two mappings, split by ``Signal.complete``:

- ``complete=False`` → onset: high strength + ``thinking_effort='none'`` preempts
  attention and holds the slot, which the deliver packet then fills via same-id
  absorb. The preempt tier itself comes from ``signal.priority``; strength only
  arbitrates at equal priority. Losing the challenge (mode defaults to ``''``)
  suppresses it: the impulse stays peekable and is picked up by a later re-rank —
  only ``attended`` clears the slot. Onset is not emitted during cooldown (suppress
  or attended), keeping multi-source or repeated barge-ins from storming the
  attention.
- ``complete=True`` → deliver: falls back to ``notify`` when mode is empty,
  carries the caller-configured priority (INFO by default), and answers in full.
  Losing the challenge buffers the messages into the mindflow (notify) — the
  content is never lost, and cooldown does not apply to it.

``complete`` and ``interrupt`` are two orthogonal axes, not one thing:

- ``complete`` chooses between holding the slot and delivering;
- ``interrupt`` decides whether the attention the packet wins calls
  ``shell.clear()`` before it starts thinking (see ``Impulse.interrupt`` /
  ``ImpulsePrimitive.interrupt``). Onset and deliver each read it from their own
  signal; False by default.

Winning the challenge already tears the previous attention down, and the action
loop clears the shell's pending commands along that abort path. What ``interrupt``
adds is an unconditional clear at the new attention's first frame, before any new
logos runs: stopping the body stops depending on what the previous attention
happened to be doing.

Same-id chain: the **impulse id** is planned by the listener and travels in the
signal meta (``impulse_id``). One utterance's onset and its final packet share one
id, so mindflow takes the same-id absorb path (onset holds the slot → deliver fills
it) — "speak up → deliver" reads as one continuous turn. Clause packets get a **new
id each**: every clause is its own utterance event, and nothing absorbs it.

``segment_id`` is an **address, not an identity** (the ASR segment, and later the
audio file that holds it). It falls back to being the impulse id only when the
signal carries none, so older producers keep working unchanged.

``paused`` answers one question the model actually asks: **may the speaker have stopped talking?**
``paused=True`` on the packet that closes a segment, ``paused=False`` on the ones that are
still in the middle of it. It is information, not permission — a pause is not an ending,
and speaking is the model's call on every packet.

Two-tier cooldown: ``suppress`` (lost the challenge) is the long one, ``attended``
(won the challenge) the short one — the latter keeps one source from being
preempted by its own next packet right after winning.

Messages are wrapped in ``<listen source="..." created="...">``: ``source`` tells
the origin (asr / wake_word ...), ``created`` is the signal's arrival wall clock.
Speaker, voiceprint and similar extras are fields an impulse does not have — the
listener side organizes them into the message body.
"""
from __future__ import annotations

import time
from typing import Callable, Iterable
from typing_extensions import Self

from ghoshell_container import IoCContainer
from pydantic import Field

from ghoshell_moss.contracts.logger import LoggerItf, get_moss_logger
from ghoshell_moss.core.concepts.channel import Channel
from ghoshell_moss.message import Message, unique_id
from ghoshell_moss.core.blueprint.mindflow import (
    ChallengeMode, Impulse, Nucleus, NucleusMeta, Priority, Signal, SignalMeta, SignalName,
)

__all__ = [
    "ListenerSignal",
    "ListenerNucleus",
    "ListenerNucleusMeta",
    "new_listener_signal",
]

class ListenerSignal(SignalMeta):
    """Listener perception signal — one speech packet.

    ``complete`` is the only protocol branch:

    - ``complete=False`` opens a turn: it asks for attention, and the matching
      ``complete=True`` packet (same ``impulse_id``) fills it.
    - ``complete=True`` delivers: it carries the heard text and closes the turn.

    The fields describe the packet; they are not instructions:

    - ``source``: where the listening came from (asr / wake_word ...); rendered
      as the ``<listen>`` tag's ``source``.
    - ``impulse_id``: the turn identity an onset and its closing packet share.
    - ``segment_id``: the audio segment's address, not an identity; it becomes
      the identity only when ``impulse_id`` is empty.
    - ``paused``: the listening side's reading — true when the stretch may have
      closed, false when more is coming.
    - ``interrupt``: whether the attention this packet wins stops the body first
      via ``shell.clear()``; false by default.
    - ``mode``: loss-side mode (notify / aside / next / ''); empty = onset
      suppresses on loss, deliver falls back to notify.
    - ``logos``: a conditioned-reflex command logos, sent with the impulse.
    - ``low_conf``: characters whose word-level confidence was low; rendered as
      the ``<listen>`` tag's ``low_conf``.
    - ``boost``: a one-shot priority raise; empty = none.
    - ``db``: the packet's mean loudness (dB); rendered as the ``<listen>``
      tag's ``db``.
    - ``strength``: the challenge strength the sending side computed; empty =
      the default strength.
    """

    source: str = Field(
        default="asr",
        description="where the listening came from (asr / wake_word ...); rendered as the "
                    "<listen> tag's source",
    )
    impulse_id: str = Field(
        default="",
        description="the turn identity: an onset and its closing packet share it; each clause "
                    "packet gets its own. Empty = fall back to segment_id.",
    )
    segment_id: str = Field(
        default="",
        description="the audio segment's address, not an identity; it becomes the identity only "
                    "when impulse_id is empty.",
    )
    paused: bool | None = Field(
        default=None,
        description="the listening side's reading: true when the stretch may have closed, false "
                    "when more is coming. Information, not permission.",
    )
    interrupt: bool = Field(
        default=False,
        description="whether the attention this packet wins stops the body first (shell.clear())",
    )
    mode: str = Field(
        default="",
        description="loss-side mode: notify / aside / next / ''(default). An onset defaults to '' "
                    "(suppress on loss); a deliver falls back to notify when empty",
    )
    logos: str = Field(
        default="",
        description="a conditioned-reflex command logos, sent with the impulse",
    )
    boost: Priority | None = Field(
        default=None,
        description="one-shot priority raise: when set, the packet challenges at "
                    "max(priority, boost); after winning it returns to the run tier. Empty = none.",
    )
    low_conf: str = Field(
        default="",
        description="low-confidence characters: word-level confidence below the etiquette's "
                    "deliver.low_confidence; rendered as the <listen> tag's low_conf. Empty = none",
    )
    db: float | None = Field(
        default=None,
        description="the packet's mean loudness (dB); rendered as the <listen> tag's db. "
                    "A measurement, not a judgment.",
    )
    strength: int | None = Field(
        default=None,
        description="the challenge strength the sending side computed; empty = the default strength.",
    )

    @classmethod
    def signal_name(cls) -> SignalName:
        return "listener"

    @classmethod
    def priority(cls) -> Priority:
        return Priority.INFO

    @classmethod
    def xml_tag(cls) -> str:
        """The xml tag wrapping delivered messages — every listening packet shares one
        tag, told apart by the ``source`` attribute."""
        return "listen"


class ListenerNucleus(Nucleus):
    """Listener perception unit — maps one listener signal into one impulse, mechanically.

    It does not accumulate, diff, or decide delivery; the listener side owns that.
    """

    NAME = "listener_nucleus"

    def __init__(
            self,
            *,
            name: str = NAME,
            first_strength: int = 150,
            normal_strength: int = 100,
            suppress_seconds: float = 0.5,
            attended_seconds: float = 0.1,
            strength_decay_seconds: float = 120.0,
            logger: LoggerItf | None = None,
    ):
        self._name = name
        self._logger = logger or get_moss_logger()

        self._first_strength = first_strength        # first-packet strength — wins ties at equal priority
        self._normal_strength = normal_strength      # run strength after attended
        self._suppress_seconds = suppress_seconds    # long cooldown after losing the challenge
        self._attended_seconds = attended_seconds    # short cooldown after winning
        self._strength_decay_seconds = strength_decay_seconds  # strength TTL after winning attention

        self._fire_impulse: Callable[[Impulse], None] | None = None
        self._is_running = False

        # Two-tier cooldown + impulse cache.
        self._suppress_until = 0.0
        self._attended_until = 0.0
        self._impulse_cache: Impulse | None = None

        # Reflexive channel — built lazily and held after the first build.
        self._channel: Channel | None = None

    # ── Nucleus ABC ──

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return "hearing — speech packets, delivered as <listen>"

    def status(self) -> str:
        if self._impulse_cache:
            return f"pending: {self._impulse_cache.description[:50]}"
        if self._in_cooldown():
            return "cooldown"
        return ""

    def signals(self) -> list[SignalName]:
        return [ListenerSignal.signal_name()]

    def clear(self) -> None:
        self._suppress_until = 0.0
        self._attended_until = 0.0
        self._impulse_cache = None

    def add_signal(self, signal: Signal) -> None:
        if not self._is_running:
            # Probe (tail-packet debugging): a signal dropped while the nucleus is not
            # running is silent; log it here.
            self._logger.info(
                "%s signal dropped — nucleus not running (complete=%s)",
                self._name, signal.complete,
            )
            return
        meta = ListenerSignal.from_signal(signal)
        if meta is None:
            self._logger.info(
                "%s signal dropped — meta parse failed (complete=%s)", self._name, signal.complete,
            )
            return
        if signal.complete:
            self._on_deliver(meta, signal)
        else:
            self._on_onset(meta, signal)

    def with_bus(
            self,
            signal_broadcast: Callable[[Signal], None],
            fire_impulse: Callable[[Impulse], None],
    ) -> None:
        self._fire_impulse = fire_impulse

    def suppress(self, suppress_by: Impulse, suppressed: Impulse | None = None) -> None:
        # Contract (``Nucleus.peek``): a suppressed impulse **stays peekable**; it just
        # stops firing during the cooldown — losing the rank is not the end; the next
        # re-rank can pick it up.
        # Do not clear the cache here: ``_rank_best_impulse_from_nuclei`` suppresses
        # every loser in the round. If a losing deliver packet were cleared, its whole
        # content would vanish and the attention it holds would never see ``complete``
        # (the tail cannot arrive → thinking stalls).
        self._suppress_until = time.monotonic() + self._suppress_seconds

    def attended(self, impulse: Impulse) -> Impulse | None:
        if impulse is self._impulse_cache:
            self._impulse_cache = None
        # Winning also enters the short cooldown — one source must not be preempted by
        # its own next packet right after winning.
        self._attended_until = time.monotonic() + self._attended_seconds
        if impulse is None:
            return None
        # Separate challenge from run parameters: after winning, drop to the run
        # priority (INFO) and run strength. A live listener attention stays at the
        # lowest tier so the next packet can enter — a higher tier wins on priority,
        # and an equal tier is decided by strength (same-source challenger ×1.1).
        if impulse.priority != Priority.INFO or impulse.strength != self._normal_strength:
            impulse.priority = Priority.INFO
            impulse.strength = self._normal_strength
            return impulse
        return None

    def peek(self, no_stale: bool = True) -> Impulse | None:
        if self._impulse_cache is None:
            return None
        if no_stale and self._impulse_cache.is_stale():
            self._impulse_cache = None
            return None
        return self._impulse_cache

    def as_channel(self) -> Channel | None:
        if self._channel is None:
            self._channel = self._build_channel()
        return self._channel

    def is_running(self) -> bool:
        return self._is_running

    async def __aenter__(self) -> Self:
        self._is_running = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._is_running = False
        self.clear()

    # ── The two protocol mappings ──

    def _on_onset(self, meta: ListenerSignal, signal: Signal) -> None:
        # Onset: a slot-holding impulse — complete=False for same-id absorb; interrupt
        # is configured by the sender. Not emitted during cooldown (suppress or
        # attended) — this prevents a preemption storm.
        if self._in_cooldown():
            # Probe (tail-packet debugging): an onset dropped in cooldown must be
            # visible — it decides whether there is still a slot to absorb.
            self._logger.info(
                "%s onset dropped in cooldown segment=%s", self._name, meta.segment_id,
            )
            return
        self._logger.info(
            "%s onset ← segment=%s priority=%s interrupt=%s",
            self._name, meta.segment_id, signal.priority, meta.interrupt,
        )
        impulse = Impulse(
            source=self.name(),
            trace_id=self._identity(meta),
            id=self._identity(meta),
            priority=signal.priority,
            strength=self._first_strength,
            strength_decay_seconds=self._strength_decay_seconds,
            complete=False,
            interrupt=meta.interrupt,
            mode=meta.mode,
            thinking_effort='none',
            messages=[],
            description=signal.description,
        )
        self._fire(impulse)

    def _on_deliver(self, meta: ListenerSignal, signal: Signal) -> None:
        # Deliver: empty mode falls back to notify + sender-configured priority. Not
        # held by cooldown — the content must never be lost.
        # Boost: the sender fills it when this packet must win; the challenge uses
        # max(priority, boost); after winning, attended() drops back to the run tier —
        # only this round is won.
        challenge_priority = signal.priority if meta.boost is None else max(
            Priority(signal.priority), Priority(meta.boost),
        )
        # Probe (tail-packet debugging): log the moment the tail reaches the nucleus.
        self._logger.info(
            "%s deliver ← segment=%s len=%d priority=%s boost=%s mode=%s",
            self._name, meta.segment_id, len(self._signal_text(signal)),
            signal.priority, meta.boost, meta.mode or ChallengeMode.notify.value,
        )
        impulse = Impulse(
            source=self.name(),
            trace_id=self._identity(meta),
            id=self._identity(meta),
            priority=challenge_priority,
            strength=self._normal_strength if meta.strength is None else meta.strength,
            strength_decay_seconds=self._strength_decay_seconds,
            complete=True,
            interrupt=meta.interrupt,
            mode=meta.mode or ChallengeMode.notify.value,
            logos=meta.logos,
            hint=signal.hint,
            messages=self._wrap_messages(signal, meta.source, meta.low_conf, meta.paused, meta.db),
            description=signal.description,
            role=signal.role or '',
        )
        self._fire(impulse)

    # ── internals ──

    @staticmethod
    def _identity(meta: ListenerSignal) -> str:
        """This packet's impulse identity: the ``impulse_id`` planned by the listener,
        falling back to ``segment_id`` only when empty.

        Identity and address are two things: an onset and its closing packet share
        one identity (same-id absorb); each clause packet has its own (no absorb);
        ``segment_id`` is only the audio segment's address.
        """
        return meta.impulse_id or meta.segment_id or unique_id()

    def _in_cooldown(self) -> bool:
        now = time.monotonic()
        return now < self._suppress_until or now < self._attended_until

    def _fire(self, impulse: Impulse) -> None:
        self._impulse_cache = impulse
        if self._fire_impulse:
            self._fire_impulse(impulse)

    def _wrap_messages(
            self, signal: Signal, source: str, low_conf: str = "", paused: bool | None = None,
            db: float | None = None,
    ) -> list[Message]:
        """Wrap the signal's text messages into one ``<listen source=... low_conf=... paused=... db=...>`` tag."""
        data = self._signal_text(signal)
        if not data and paused is not True:
            return []
        # Empty body + paused=true is legal (clause delivery: the stretch closed but
        # this packet carries no new text). It answers a mechanism question — the
        # listener decided the stretch stopped, with no new words; the body stays
        # empty. Do not fill it with substitute text: that would translate mechanism
        # semantics into a "fact", and the body must hold only what was actually said.
        # How to read an empty-body packet belongs in the listener instruction.
        return [self._listen_message(data, source, low_conf, signal.created_at, paused, db)]

    def _signal_text(self, signal: Signal) -> str:
        """Collect the text content of every message body in the signal."""
        texts: list[str] = []
        for msg in signal.messages:
            for c in msg.contents:
                if isinstance(c, dict) and c.get('type') == 'text':
                    texts.append(c.get('text', ''))
        return '\n'.join(texts)

    def _listen_message(
            self, data: str, source: str, low_conf: str, created_at,
            paused: bool | None = None, db: float | None = None,
    ) -> Message:
        """One ``<listen ...>`` wrapped message; ``created`` is the signal's arrival wall clock."""
        attributes: dict[str, str] = {}
        if source:
            attributes["source"] = source
        if low_conf:
            attributes["low_conf"] = low_conf
        if paused is not None:
            attributes["paused"] = "true" if paused else "false"
        if db is not None:
            attributes["db"] = f"{db:.1f}"
        message = Message.new(tag=ListenerSignal.xml_tag(), attributes=attributes or None, timestamp=True)
        message.meta.created = created_at
        return message.with_content(data)

    # ── Reflexive channel (minimal surface) ──

    def _build_channel(self) -> Channel:
        from ghoshell_moss.core.blueprint.states_channel import new_prime_channel

        channel = new_prime_channel(self.name(), description=self.description())

        @channel.build.notice
        def notice() -> str:
            return self.status() or "listening ready"

        @channel.build.command(name="configure")
        async def _configure(
                first_strength: int | None = None,
                suppress_seconds: float | None = None,
                attended_seconds: float | None = None,
                strength_decay_seconds: float | None = None,
        ) -> str:
            """Dynamically tune the barge-in parameters (onset strength / cooldowns / attention TTL)."""
            if first_strength is not None:
                self._first_strength = first_strength
            if suppress_seconds is not None:
                self._suppress_seconds = suppress_seconds
            if attended_seconds is not None:
                self._attended_seconds = attended_seconds
            if strength_decay_seconds is not None:
                self._strength_decay_seconds = strength_decay_seconds
            return (
                f"configured: first_strength={self._first_strength} "
                f"suppress={self._suppress_seconds}s attended={self._attended_seconds}s "
                f"decay={self._strength_decay_seconds}s"
            )

        return channel


class ListenerNucleusMeta(NucleusMeta):
    """Factory meta — lets ``moss manifests nuclei`` discover ListenerNucleus."""

    def name(self) -> str:
        return ListenerNucleus.NAME

    def description(self) -> str:
        return "hearing — speech packets, delivered as <listen>"

    def signals(self) -> Iterable[type[SignalMeta]]:
        yield ListenerSignal

    def factory(self, container: IoCContainer) -> Nucleus:
        logger = container.get(LoggerItf)
        return ListenerNucleus(logger=logger)


def new_listener_signal(
        text: str = "",
        *,
        segment_id: str = "",
        source: str = "asr",
        interrupt: bool = False,
        mode: str = "",
        logos: str = "",
        low_conf: str = "",
        impulse_id: str = "",
        paused: bool | None = None,
        complete: bool = True,
        priority: Priority | None = None,
        boost: Priority | None = None,
        db: float | None = None,
        strength: int | None = None,
        description: str = "",
        hint: str = "",
) -> Signal:
    """Helper — construct one ``listener`` signal.

    ``complete=False`` is an onset (holds the attention slot), ``complete=True`` a
    deliver (carries the content). ``interrupt`` / ``mode`` / ``priority`` are
    listener-configured and orthogonal to that split — only ``interrupt=True`` makes
    the attention the packet wins call ``shell.clear()`` (False by default).
    ``segment_id`` is the same-id key shared by onset and deliver.
    """
    return ListenerSignal(
        source=source,
        segment_id=segment_id,
        interrupt=interrupt,
        mode=mode,
        logos=logos,
        boost=boost,
        low_conf=low_conf,
        db=db,
        strength=strength,
        impulse_id=impulse_id,
        paused=paused,
    ).to_signal(
        text,
        description=description or (text[:40] or ("barge-in" if not complete else "deliver")),
        priority=priority,
        hint=hint,
        complete=complete,
    )
