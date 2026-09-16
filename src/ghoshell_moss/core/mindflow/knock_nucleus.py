"""KnockNucleus — turns ``knock`` signals into default-mode impulses, dropped on loss.

The one openbox perception entry that is safe to lose: a knock announces that
something is waiting to be fetched, and the ghost answers only when it is free.
Pairs no ``ImpulsePrimitive`` — a knock is an ordinary impulse asking the ghost to
come and look, not a named reflex.

A knock must carry a message; with nothing on it there is nothing to announce, and
the signal is dropped.
"""
from typing import Callable, Iterable
from typing_extensions import Self

from ghoshell_container import IoCContainer

from ghoshell_moss.contracts.logger import LoggerItf, get_moss_logger
from ghoshell_moss.message import ContextType
from ghoshell_moss.core.blueprint.mindflow import (
    SignalMeta, SignalName, Priority, Signal,
    Nucleus, NucleusMeta, Impulse,
)

__all__ = ['KnockNucleus', 'KnockSignalMeta', 'KnockNucleusMeta', 'new_knock_signal']


class KnockSignalMeta(SignalMeta):
    """Signal meta for ``knock`` — a caller at the door, asking the ghost to come and get something.

    If the ghost is free it goes and looks, thinks it over, and takes what is waiting
    itself. If the ghost is busy the knock is simply gone: the caller keeps what it
    brought, and nothing is lost by the ghost not answering.

    A knock carries a message — with nothing on it there is nothing to announce, and
    the knock is dropped.
    """

    @classmethod
    def signal_name(cls) -> SignalName:
        return 'knock'

    @classmethod
    def priority(cls) -> Priority:
        return Priority.NOTICE


class KnockNucleus(Nucleus):
    """Knock channel — a losable request for the ghost's attention, with a message.

    Functional intent: someone knocks and says what they came for. A free ghost
    comes and thinks it over; a busy ghost never hears it, and that is fine — the
    caller still holds what it brought.

    Mechanism: last-impulse cache (last-wins, isomorphic to ``CommandNucleus`` minus
    the logos). ``build_impulse`` produces a ``default`` mode impulse — no logos, no
    effort override — so winning the challenge runs real thinking. ``suppress``
    clears the cache: losing drops the knock outright. There is no cooldown, no
    aggregation and no retry; the next knock is a fresh one.
    """

    NAME = 'knock_nucleus'

    def __init__(self, *, name: str = NAME, logger: LoggerItf | None = None):
        self._name = name
        self._fire_impulse: Callable[[Impulse], None] | None = None
        self._is_running = False
        self._logger = logger or get_moss_logger()
        # Last-impulse cache: 满足 mindflow pull-based 协议.
        self._impulse: Impulse | None = None

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return 'knock — a request the ghost answers only if it is free'

    def status(self) -> str:
        return ''

    def signals(self) -> list[SignalName]:
        return [KnockSignalMeta.signal_name()]

    def clear(self) -> None:
        self._impulse = None

    def add_signal(self, signal: Signal) -> None:
        if not self._is_running:
            return
        impulse = self.build_impulse(signal)
        if impulse is None:
            return
        # Last-wins cache: 覆盖未消费的旧 impulse.
        self._impulse = impulse
        if self._fire_impulse:
            self._fire_impulse(impulse)

    def build_impulse(self, signal: Signal) -> Impulse | None:
        if not KnockSignalMeta.match(signal):
            return None
        if not signal.messages:
            return None
        # 继承 signal 的 priority/strength/messages, 不改 mode / effort / logos —
        # knock 的语义就是"一次普通挑战", 赢了走真实思考.
        return Impulse.from_signal(signal, source=self.name())

    def with_bus(
            self,
            signal_broadcast: Callable[[Signal], None],
            fire_impulse: Callable[[Impulse], None],
    ) -> None:
        self._fire_impulse = fire_impulse

    def suppress(self, suppress_by: Impulse, suppressed: Impulse | None = None) -> None:
        # 抢占失败 → 直接丢. knock 不等待、不重试、不聚合:
        # 端侧自己持有待取的内容, 门没敲开就再敲一次.
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


class KnockNucleusMeta(NucleusMeta):
    """Factory meta — lets ``moss manifests nuclei`` discover KnockNucleus."""

    def name(self) -> str:
        return KnockNucleus.NAME

    def description(self) -> str:
        return 'knock — a request the ghost answers only if it is free'

    def signals(self) -> Iterable[type[SignalMeta]]:
        yield KnockSignalMeta

    def factory(self, container: IoCContainer) -> Nucleus:
        logger = container.get(LoggerItf)
        return KnockNucleus(logger=logger)


def new_knock_signal(
        *messages: ContextType,
        priority: Priority = Priority.NOTICE,
        description: str = '',
        stale_timeout: float = 0,
        hint: str = '',
) -> Signal:
    """Helper — construct a ``knock`` signal in one call.

    A knock without a message is dropped downstream, so callers should always pass
    at least one message.
    """
    return KnockSignalMeta().to_signal(
        *messages,
        description=description,
        stale_timeout=stale_timeout,
        priority=priority,
        hint=hint,
    )
