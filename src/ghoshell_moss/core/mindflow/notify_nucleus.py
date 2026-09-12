"""NotifyNucleus — turns ``notify`` signals into ``notify``-mode impulses.

The "must not be lost" perception nucleus, pairing ``ImpulsePrimitive.notify``.
It listens to ``NotifySignalMeta`` (signal name ``"notify"``) and wraps each
signal into an impulse carrying ``mode=notify``: winning the challenge creates a
new attention as usual, while losing it routes the messages into the mindflow's
next-frame percepts instead of suppressing them — notify deviates from
``default`` on the losing side only.

priority is inherited verbatim from ``Signal.priority`` (caller-controlled),
default ``NOTICE``. Canonical case: the user speaks while the ghost is thinking —
no interruption, but the message leaves a trace.
"""
from typing import Callable, Iterable
from typing_extensions import Self

from ghoshell_container import IoCContainer

from ghoshell_moss.contracts.logger import LoggerItf, get_moss_logger
from ghoshell_moss.message import ContextType
from ghoshell_moss.core.blueprint.mindflow import (
    SignalMeta, SignalName, Priority, Signal,
    Nucleus, NucleusMeta, ImpulsePrimitive, Impulse
)

__all__ = ['NotifyNucleus', 'NotifySignalMeta', 'NotifyNucleusMeta', 'new_notify_signal']


class NotifySignalMeta(SignalMeta):
    """Signal meta for ``notify`` — carries messages that must not be lost.

    Winning the challenge creates a new attention as usual; losing it goes through
    notify mode — the messages are buffered into the mindflow (leaving a trace)
    instead of being suppressed, and are consumed by the next frame's percepts.
    Canonical case: the user speaks while the ghost is thinking.
    """

    @classmethod
    def signal_name(cls) -> SignalName:
        return 'notify'

    @classmethod
    def priority(cls) -> Priority:
        return Priority.NOTICE


class NotifyNucleus(Nucleus):
    """Reflex-arc nucleus — turns each ``notify`` signal into a notify-mode impulse.

    The impulse is held as the latest one for mindflow's rank/challenge pull:
    mindflow pulls it via ``peek`` and confirms the outcome via ``attended``.
    notify deviates from ``default`` on the losing side only — the messages are
    buffered into the mindflow instead of being suppressed, so they are not lost.

    priority is inherited verbatim from ``Signal.priority`` (caller-controlled).
    """

    NAME = 'notify_nucleus'

    def __init__(self, *, name: str = NAME, logger: LoggerItf | None = None):
        self._name = name
        self._fire_impulse: Callable[[Impulse], None] | None = None
        self._is_running = False
        self._logger = logger or get_moss_logger()
        self._impulse: Impulse | None = None

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return 'wrap notify signals into notify-mode impulses (message-preserving)'

    def status(self) -> str:
        return ''

    def signals(self) -> list[SignalName]:
        return [NotifySignalMeta.signal_name()]

    def clear(self) -> None:
        self._impulse = None

    def add_signal(self, signal: Signal) -> None:
        if not self._is_running:
            return
        impulse = self.build_impulse(signal)
        if impulse is None:
            return
        # TODO(known, deferred): a newer signal overwrites an un-peeked impulse,
        # dropping its messages before the buffered path can see them. Latent —
        # needs a notify burst inside one loop scheduling window. Shape undecided.
        self._impulse = impulse
        if self._fire_impulse:
            self._fire_impulse(impulse)

    def build_impulse(self, signal: Signal) -> Impulse | None:
        if not NotifySignalMeta.match(signal):
            return None
        impulse = Impulse.from_signal(signal, source=self.name())
        return ImpulsePrimitive.notify(impulse)

    def with_bus(
            self,
            signal_broadcast: Callable[[Signal], None],
            fire_impulse: Callable[[Impulse], None],
    ) -> None:
        self._fire_impulse = fire_impulse

    def suppress(self, suppress_by: Impulse, suppressed: Impulse | None = None) -> None:
        # notify loses the challenge through mindflow's buffered path, so the
        # messages are already accounted for; suppress is only a defensive
        # fallback — clear the cache and wait for the next signal.
        self._impulse = None

    def attended(self, impulse: Impulse) -> None:
        if self._impulse is impulse:
            self._impulse = None

    def peek(self, no_stale: bool = True) -> Impulse | None:
        if self._impulse is None:
            return None
        if no_stale and self._impulse.is_stale():
            self._impulse = None
            return None
        return self._impulse

    def is_running(self) -> bool:
        return self._is_running

    async def __aenter__(self) -> Self:
        self._is_running = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._is_running = False
        self._impulse = None


class NotifyNucleusMeta(NucleusMeta):
    """Factory meta — lets ``moss manifests nuclei`` discover NotifyNucleus."""

    def name(self) -> str:
        return NotifyNucleus.NAME

    def description(self) -> str:
        return 'reflex-arc nucleus that wraps notify signals into notify-mode impulses'

    def signals(self) -> Iterable[type[SignalMeta]]:
        yield NotifySignalMeta

    def factory(self, container: IoCContainer) -> Nucleus:
        logger = container.get(LoggerItf)
        return NotifyNucleus(logger=logger)


def new_notify_signal(
        *messages: ContextType,
        priority: Priority = Priority.NOTICE,
        description: str = '',
        stale_timeout: float = 0,
        hint: str = '',
) -> Signal:
    """Helper — construct a ``notify`` signal in one call."""
    return NotifySignalMeta().to_signal(
        *messages,
        description=description,
        stale_timeout=stale_timeout,
        priority=priority,
        hint=hint,
    )
