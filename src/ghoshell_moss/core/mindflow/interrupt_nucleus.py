"""InterruptNucleus — turns ``interrupt`` signals into interrupt-mode impulses.

Functional intent: make the ghost stop whatever it is doing right now — no
response, no thinking, just stop. Pairs ``ImpulsePrimitive.interrupt``.

Mechanism: wraps each signal into ``FATAL + notify + thinking_effort='none' +
interrupt=True``. FATAL guarantees preemption, notify creates a new attention
through the default success path, effort='none' makes articulate return early,
and interrupt=True stops the shell's running logos at the new attention's start.

Structure:
- isomorphic to ``CommandNucleus`` (fire-and-forget, no buffer)
- dual to broadcast (``ImpulsePrimitive.broadcast``, no dedicated nucleus):
  broadcast uses aside to buffer without taking attention; interrupt uses notify
  to take attention then drop it immediately

Reverse suppress (victory-side cooldown, dual to InputSignalNucleus's loss-side):
- attended starts the cooldown, preventing repeated interrupts from churning the
  shell (stop_interpretation + attention rebuild, DDOS-like)
- within the cooldown add_signal silently drops — interrupts have no accumulation
  semantics, several are equivalent to one
"""
import time
from typing import Callable, Iterable
from typing_extensions import Self

from ghoshell_container import IoCContainer

from ghoshell_moss.contracts.logger import LoggerItf, get_moss_logger
from ghoshell_moss.message import ContextType
from ghoshell_moss.core.blueprint.mindflow import (
    SignalMeta, SignalName, Priority, Signal,
    Nucleus, NucleusMeta, ImpulsePrimitive, Impulse
)

__all__ = [
    'InterruptNucleus', 'InterruptSignalMeta', 'InterruptNucleusMeta',
    'new_interrupt_signal',
]


class InterruptSignalMeta(SignalMeta):
    """Signal meta for ``interrupt`` — stop the ghost right now, then let go.

    Dual to ``ImpulsePrimitive.broadcast`` (FATAL + aside: buffer without taking
    attention): interrupt takes attention and stops the shell, then drops it
    without thinking.

    priority is locked to FATAL — there is no "low-priority interrupt".
    """

    @classmethod
    def signal_name(cls) -> SignalName:
        return 'interrupt'

    @classmethod
    def priority(cls) -> Priority:
        return Priority.FATAL


class InterruptNucleus(Nucleus):
    """Interrupt channel — last-impulse cache with victory-side cooldown.

    Functional intent: stop now, then drop the attention without thinking.

    Mechanism: last-wins cache. ``add_signal`` writes ``_impulse``, mindflow pulls
    via ``peek`` and confirms via ``attended`` (which starts the cooldown). Multiple
    interrupts arriving before consumption are equivalent — each preempts with
    FATAL and triggers shell.stop_interpretation.

    Reverse suppress (victory-side, dual to InputSignalNucleus's loss-side): the
    cooldown starts on attended, not on losing. FATAL only "loses" to same-id
    absorb or stale — neither needs a cooldown; the real churn risk is repeated
    successful interrupts. No aggregation: several interrupts carry no extra
    meaning, the first suffices.
    """

    NAME = 'interrupt_nucleus'

    def __init__(
            self,
            *,
            name: str = NAME,
            suppress_seconds: float = 0.5,
            logger: LoggerItf | None = None,
    ):
        self._name = name
        self._suppress_seconds = suppress_seconds
        self._fire_impulse: Callable[[Impulse], None] | None = None
        self._is_running = False
        self._logger = logger or get_moss_logger()
        # 反向 suppress: 胜利后才设, 失败侧不动.
        self._suppress_until: float = 0.0
        self._impulse: Impulse | None = None

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return 'interrupt channel — must-deliver, takes attention then drops it without thinking'

    def status(self) -> str:
        return ''

    def signals(self) -> list[SignalName]:
        return [InterruptSignalMeta.signal_name()]

    def clear(self) -> None:
        self._suppress_until = 0.0
        self._impulse = None

    def add_signal(self, signal: Signal) -> None:
        if not self._is_running:
            return
        # 反向 suppress: 上一次中断刚胜利, 冷静期内静默丢.
        if time.monotonic() < self._suppress_until:
            return
        impulse = self.build_impulse(signal)
        if impulse is None:
            return
        self._impulse = impulse
        if self._fire_impulse:
            self._fire_impulse(impulse)

    def build_impulse(self, signal: Signal) -> Impulse | None:
        if not InterruptSignalMeta.match(signal):
            return None
        impulse = Impulse.from_signal(signal, source=self.name())
        # interrupt primitive 强制 FATAL + notify + effort='none' + interrupt=True.
        # Signal.priority 被覆盖 — interrupt 的语义承诺不可降级.
        return ImpulsePrimitive.interrupt(impulse)

    def with_bus(
            self,
            signal_broadcast: Callable[[Signal], None],
            fire_impulse: Callable[[Impulse], None],
    ) -> None:
        self._fire_impulse = fire_impulse

    def suppress(self, suppress_by: Impulse, suppressed: Impulse | None = None) -> None:
        # 失败侧不进冷静期 — FATAL 仲裁失败只可能是 same-id absorb 或 stale,
        # 这两种都不需要冷静期 (absorb 已被内部处理, stale 在入口丢).
        # 但 cache 仍要清, 让 nucleus 状态正确反映 "没有 pending impulse".
        self._impulse = None

    def attended(self, impulse: Impulse) -> None:
        # 反向 suppress: 仲裁胜利后启动冷静期, 防止 shell churn.
        if not self._is_running:
            return
        if self._impulse is impulse:
            self._impulse = None
        self._suppress_until = time.monotonic() + self._suppress_seconds

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


class InterruptNucleusMeta(NucleusMeta):
    """Factory meta — lets ``moss manifests nuclei`` discover InterruptNucleus."""

    def __init__(self, *, suppress_seconds: float = 0.5):
        self._suppress_seconds = suppress_seconds

    def name(self) -> str:
        return InterruptNucleus.NAME

    def description(self) -> str:
        return 'interrupt channel that turns interrupt signals into FATAL+notify+effort=none+interrupt impulses'

    def signals(self) -> Iterable[type[SignalMeta]]:
        yield InterruptSignalMeta

    def factory(self, container: IoCContainer) -> Nucleus:
        logger = container.get(LoggerItf)
        return InterruptNucleus(suppress_seconds=self._suppress_seconds, logger=logger)


def new_interrupt_signal(
        *messages: ContextType,
        description: str = '',
        stale_timeout: float = 0,
        hint: str = '',
) -> Signal:
    """Helper — construct an ``interrupt`` signal in one call.

    priority is not exposed — interrupt is always FATAL by contract.
    Use ``new_notify_signal`` for soft preemption attempts.
    """
    return InterruptSignalMeta().to_signal(
        *messages,
        description=description,
        stale_timeout=stale_timeout,
        hint=hint,
    )
