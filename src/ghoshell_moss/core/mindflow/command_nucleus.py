"""CommandNucleus — turns ``command`` signals into ``command_only`` impulses.

Functional intent: execute logos now, without the ghost thinking first. Pairs
``ImpulsePrimitive.command_only``.

Mechanism: listens to ``CommandSignalMeta`` (signal name ``"command"``), unwraps
the logos carried by the signal, and routes it through mindflow's
``thinking_effort='none'`` early-return path so the shell executes it without
calling ``ghost.articulate()``.

priority is inherited verbatim from ``Signal.priority`` — no floor is imposed:
callers use ``Priority.FATAL`` for a forced command (equivalent to
``ImpulsePrimitive.fatal_command``) and ``Priority.NOTICE`` for a normal one.
One nucleus covers both.
"""
from typing import Callable, Iterable
from typing_extensions import Self

from ghoshell_container import IoCContainer
from pydantic import Field

from ghoshell_moss.contracts.logger import LoggerItf, get_moss_logger
from ghoshell_moss.message import ContextType
from ghoshell_moss.core.blueprint.mindflow import (
    SignalMeta, SignalName, Priority, Signal,
    Nucleus, NucleusMeta, ImpulsePrimitive, Impulse,
)

__all__ = ['CommandNucleus', 'CommandSignalMeta', 'CommandNucleusMeta', 'new_command_signal']


class CommandSignalMeta(SignalMeta):
    """Signal meta for ``command`` — an instruction to act.

    The ghost does not think it over — it just does it. A newer instruction replaces
    an older one; one that cannot be carried out is dropped.
    """

    logos: str = Field(
        description="logos (usually CTML) for the shell to execute directly, instead of "
                    "the ghost thinking it over.",
    )

    @classmethod
    def signal_name(cls) -> SignalName:
        return 'command'

    @classmethod
    def priority(cls) -> Priority:
        return Priority.NOTICE


class CommandNucleus(Nucleus):
    """Reflex-arc nucleus — sends logos straight to the shell, bypassing thought.

    Functional intent: an instruction to execute now, not a topic to think about.

    Mechanism: last-impulse cache (last-wins, isomorphic to ``_DirectImpulseNucleus``).
    ``add_signal`` wraps the signal into a ``command_only`` impulse and notifies
    mindflow, which pulls it via ``peek`` and confirms via ``attended``. A newer
    command overwrites an unconsumed one — the latest instruction wins.

    Unlike ``InputSignalNucleus`` / ``AsideNucleus``, it does not aggregate or keep
    history. priority is inherited verbatim (no floor); ``Priority.FATAL`` is
    equivalent to ``ImpulsePrimitive.fatal_command``.
    """

    NAME = 'command_nucleus'

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
        return 'send logos directly to the shell, bypassing ghost thinking'

    def status(self) -> str:
        return ''

    def signals(self) -> list[SignalName]:
        return [CommandSignalMeta.signal_name()]

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
        meta = CommandSignalMeta.from_signal(signal)
        if meta is None or not meta.logos:
            return None
        impulse = Impulse.from_signal(signal, source=self.name())
        # 继承 signal.priority 不再强制 floor — 调用方用 Priority.FATAL
        # 等价于 ImpulsePrimitive.fatal_command.
        return ImpulsePrimitive.command_only(impulse, meta.logos)

    def with_bus(
            self,
            signal_broadcast: Callable[[Signal], None],
            fire_impulse: Callable[[Impulse], None],
    ) -> None:
        self._fire_impulse = fire_impulse

    def suppress(self, suppress_by: Impulse, suppressed: Impulse | None = None) -> None:
        # command 抢占失败 (priority 不够) → 清 cache, 让位.
        # command 没有"重试" 语义 — 失败就丢, 等下一条新 command.
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


class CommandNucleusMeta(NucleusMeta):
    """Factory meta — lets ``moss manifests nuclei`` discover CommandNucleus."""

    def name(self) -> str:
        return CommandNucleus.NAME

    def description(self) -> str:
        return 'reflex-arc nucleus that turns command signals into command_only impulses'

    def signals(self) -> Iterable[type[SignalMeta]]:
        yield CommandSignalMeta

    def factory(self, container: IoCContainer) -> Nucleus:
        logger = container.get(LoggerItf)
        return CommandNucleus(logger=logger)


def new_command_signal(
        logos: str,
        *messages: ContextType,
        priority: Priority = Priority.NOTICE,
        description: str = '',
        stale_timeout: float = 0,
) -> Signal:
    """Helper — construct a ``command`` signal from raw logos.

    Convenience over ``CommandSignalMeta(logos=...).to_signal(...)`` for
    one-liner injection in tests / scripts.
    """
    return CommandSignalMeta(logos=logos).to_signal(
        *messages,
        description=description,
        stale_timeout=stale_timeout,
        priority=priority,
    )
