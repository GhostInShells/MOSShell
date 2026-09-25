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
  suppresses it — an onset carries no content, so dropping it loses nothing. Onset
  is not emitted during cooldown (suppress or attended), keeping multi-source or
  repeated barge-ins from storming the attention.
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

Same-id chain: onset and deliver share ``segment_id`` (the ASR segment id) as the
impulse id, so mindflow takes the same-id absorb path (onset holds the slot →
deliver fills it) — "speak up → deliver" reads as one continuous turn. Distinct
listener instances stay apart through segment_id as well.

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
    """Listener perception signal — a single degenerate form, meta fully self-explaining.

    The listener side has already decided the intent; this signal only states it, and
    the nucleus maps it mechanically into an impulse. ``complete`` / ``priority`` /
    ``hint`` / ``description`` are generic ``Signal`` fields (not part of metadata):
    the listener side sets them directly through
    ``to_signal(complete=..., priority=..., hint=...)``.

    Metadata fields (read by the nucleus to fill the impulse):

    - ``source``: where the listening came from (asr / wake_word ...), the ``source``
      attribute of the delivered ``<listen>`` tag.
    - ``segment_id``: the same-id key, matching the ASR segment id (tail-delimited).
      The onset (complete=False) and the deliver (complete=True) share it → same-id
      absorb.
    - ``interrupt``: whether the attention this packet wins stops the body first via
      ``shell.clear()`` — the ``Impulse.interrupt`` protocol. Listener-configurable,
      False by default; it is about stopping the body, not about barge-in.
    - ``mode``: the loss-side mode (notify / aside / next / ''). Listener-configurable.
      An onset defaults to ``''`` (losing suppresses it), a deliver falls back to
      ``notify`` when empty.
    - ``logos``: a conditioned-reflex command logos, listener-configurable.
    - ``low_conf``: 缩写 = "low-confidence" (低置信度) 的字 — the characters whose
      word-level confidence fell below the etiquette's ``deliver.low_confidence``
      threshold. Rendered as the ``low_conf`` attribute of the delivered ``<listen>``
      tag (deliver only). Empty = none.

    Text, speaker, voiceprint and similar extras are fields an impulse does not have —
    the listener side organizes them into the message body.
    """

    source: str = Field(default="asr", description="where the listening came from — the upstream source producing the packet (asr / wake_word ...)")
    segment_id: str = Field(default="", description="same-id key — the ASR segment id, shared by the onset and the deliver packet")
    interrupt: bool = Field(default=False, description="whether the attention won by this packet calls shell.clear() first (Impulse.interrupt)")
    mode: str = Field(default="", description="loss-side mode: notify / aside / next / ''(default). An onset defaults to '' (suppress on loss); a deliver falls back to notify when empty")
    logos: str = Field(default="", description="conditioned-reflex command logos, sent with the impulse")
    low_conf: str = Field(default="", description="缩写 = low-confidence (低置信度): 词级置信度低于阈值 (deliver.low_confidence) 的字, 渲染为 <listen> 的 low_conf 属性; 空 = 无")

    @classmethod
    def signal_name(cls) -> SignalName:
        return "listener"

    @classmethod
    def priority(cls) -> Priority:
        return Priority.INFO

    @classmethod
    def xml_tag(cls) -> str:
        """The xml tag wrapping delivered messages — every listening packet shares one tag, told apart by the ``source`` attribute."""
        return "listen"


class ListenerNucleus(Nucleus):
    """Listener perception unit — a pure protocol mapping from listener signal to impulse.

    It does exactly two things: an onset (complete=False → slot-holding impulse) and a
    deliver (complete=True → content impulse). It does not accumulate clauses, diff
    them, or manage delivery etiquette — all of that understanding lives on the
    listener side.

    ``interrupt`` belongs to neither packet type exclusively: onset and deliver each
    read the field from their own signal (listener-configured, False by default) to
    decide whether the attention they win stops the body. See the module docstring
    for the orthogonal axes.

    Two-tier cooldown: ``suppress`` (long) + ``attended`` (short), both holding back
    onset emission only — deliver is never held back (the content must not be lost).
    same-id = ``segment_id``.
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
            logger: LoggerItf | None = None,
    ):
        self._name = name
        self._logger = logger or get_moss_logger()

        self._first_strength = first_strength        # 首包强度 — 同级强度仲裁时用于抢占
        self._normal_strength = normal_strength      # attended 后降回 (运行强度)
        self._suppress_seconds = suppress_seconds    # 仲裁失败大冷却
        self._attended_seconds = attended_seconds    # 抢占成功小冷却

        self._fire_impulse: Callable[[Impulse], None] | None = None
        self._is_running = False

        # 冷却双档 + impulse cache.
        self._suppress_until = 0.0
        self._attended_until = 0.0
        self._impulse_cache: Impulse | None = None

        # 反身 channel — 惰性构建, 一次生成后持有.
        self._channel: Channel | None = None

    # ── Nucleus ABC ──

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return ("listener sense nucleus — onset (complete=False) and deliver "
                "(complete=True) packets mapped verbatim into impulses")

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
            return
        meta = ListenerSignal.from_signal(signal)
        if meta is None:
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
        # 仲裁失败: 丢弃缓存, 进入大冷却 — 只压首包发射, 发送包不受约束.
        self._impulse_cache = None
        self._suppress_until = time.monotonic() + self._suppress_seconds

    def attended(self, impulse: Impulse) -> Impulse | None:
        if impulse is self._impulse_cache:
            self._impulse_cache = None
        # 抢占成功也进小冷却 — 防同一源抢到后立刻又被自己下一个包抢.
        self._attended_until = time.monotonic() + self._attended_seconds
        if impulse is None:
            return None
        # challenge → run 参数分离: 抢到 attention 后降为运行优先级(INFO) + 运行强度.
        # 运行中的 listener attention 停在最低一档, 下一个包才进得来 — 优先级更高的
        # 包按优先级直接赢, 同为 INFO 级则靠强度仲裁 (同源 challenger ×1.1 加权).
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

    # ── 两套协议映射 ──

    def _on_onset(self, meta: ListenerSignal, signal: Signal) -> None:
        # 首包 (onset): 占坑 impulse — complete=False 供 same-id absorb; interrupt
        # 由端侧配. 冷却期 (suppress 或 attended) 内不发射 — 防抢占风暴.
        if self._in_cooldown():
            return
        impulse = Impulse(
            source=self.name(),
            trace_id=meta.segment_id or unique_id(),
            id=meta.segment_id or unique_id(),
            priority=signal.priority,
            strength=self._first_strength,
            complete=False,
            interrupt=meta.interrupt,
            mode=meta.mode,
            thinking_effort='none',
            messages=[],
            description=signal.description,
        )
        self._fire(impulse)

    def _on_deliver(self, meta: ListenerSignal, signal: Signal) -> None:
        # 发送包 (deliver): mode 为空兜底 notify + 端侧配 priority. 不受冷却约束 —
        # 内容绝不丢.
        impulse = Impulse(
            source=self.name(),
            trace_id=meta.segment_id or unique_id(),
            id=meta.segment_id or unique_id(),
            priority=signal.priority,
            strength=self._normal_strength,
            complete=True,
            interrupt=meta.interrupt,
            mode=meta.mode or ChallengeMode.notify.value,
            logos=meta.logos,
            hint=signal.hint,
            messages=self._wrap_messages(signal, meta.source, meta.low_conf),
            description=signal.description,
        )
        self._fire(impulse)

    # ── internals ──

    def _in_cooldown(self) -> bool:
        now = time.monotonic()
        return now < self._suppress_until or now < self._attended_until

    def _fire(self, impulse: Impulse) -> None:
        self._impulse_cache = impulse
        if self._fire_impulse:
            self._fire_impulse(impulse)

    def _wrap_messages(self, signal: Signal, source: str, low_conf: str = "") -> list[Message]:
        """Wrap the signal's text messages into one ``<listen source=... low_conf=...>`` tag."""
        data = self._signal_text(signal)
        if not data:
            return []
        return [self._listen_message(data, source, low_conf, signal.created_at)]

    def _signal_text(self, signal: Signal) -> str:
        """Collect the text content of every message body in the signal."""
        texts: list[str] = []
        for msg in signal.messages:
            for c in msg.contents:
                if isinstance(c, dict) and c.get('type') == 'text':
                    texts.append(c.get('text', ''))
        return '\n'.join(texts)

    def _listen_message(self, data: str, source: str, low_conf: str, created_at) -> Message:
        """One ``<listen source=... low_conf=... created=...>`` wrapped message; ``created`` is the signal's arrival wall clock."""
        attributes: dict[str, str] = {}
        if source:
            attributes["source"] = source
        if low_conf:
            attributes["low_conf"] = low_conf
        message = Message.new(tag=ListenerSignal.xml_tag(), attributes=attributes or None, timestamp=True)
        message.meta.created = created_at
        return message.with_content(data)

    # ── 反身 channel (最小平面) ──

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
        ) -> str:
            """Dynamically tune the barge-in parameters (onset strength / cooldowns)."""
            if first_strength is not None:
                self._first_strength = first_strength
            if suppress_seconds is not None:
                self._suppress_seconds = suppress_seconds
            if attended_seconds is not None:
                self._attended_seconds = attended_seconds
            return (
                f"configured: first_strength={self._first_strength} "
                f"suppress={self._suppress_seconds}s attended={self._attended_seconds}s"
            )

        return channel


class ListenerNucleusMeta(NucleusMeta):
    """Factory meta — lets ``moss manifests nuclei`` discover ListenerNucleus."""

    def name(self) -> str:
        return ListenerNucleus.NAME

    def description(self) -> str:
        return ("listener nucleus that maps onset (complete=False) and deliver "
                "(complete=True) packets into impulses")

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
        complete: bool = True,
        priority: Priority | None = None,
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
        low_conf=low_conf,
    ).to_signal(
        text,
        description=description or (text[:40] or ("barge-in" if not complete else "deliver")),
        priority=priority,
        hint=hint,
        complete=complete,
    )
