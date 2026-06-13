from typing import Self, Callable

from ghoshell_moss.message import ContextType
from ghoshell_moss.core.blueprint import Impulse
from ghoshell_moss.core.blueprint.mindflow import (
    SignalMeta, SignalName, Priority, Signal,
    Nucleus, ImpulsePrimitive,
)
from pydantic import Field

__all__ = ['CommandNucleus', 'CommandSignalMeta']


class CommandSignalMeta(SignalMeta):
    """
    发送必须执行的 logos 到 Ghost. Ghost 只能默认执行它.
    """

    logos: str = Field(
        description="the logos that "
    )

    @classmethod
    def signal_name(cls) -> SignalName:
        return 'command'

    @classmethod
    def priority(cls) -> Priority:
        return Priority.NOTICE

    def to_signal(
            self,
            *messages: ContextType,
            description: str = '',
            stale_timeout: float = 0,
            priority: int | None = None,
    ) -> Signal:
        signal = super().to_signal(
            *messages,
            description=description,
            stale_timeout=stale_timeout,
            priority=priority,
        )
        signal.logos = self.logos
        return signal


class CommandNucleus(Nucleus):
    """
    单纯发送命令, 让 Ghost 执行的 Nucleus.
    """

    def __init__(self, min_priority: Priority = Priority.NOTICE):
        self._impulse_notify: Callable[[Impulse], None] | None = None
        self._is_running = False
        self._min_priority = min_priority

    def name(self) -> str:
        return 'command_nucleus'

    def description(self) -> str:
        return 'send logos to the shell that bypass the ghost'

    def status(self) -> str:
        return ''

    def signals(self) -> list[SignalName]:
        return [CommandSignalMeta.signal_name()]

    def clear(self) -> None:
        return

    def add_signal(self, signal: Signal) -> None:
        impulse = self.build_impulse(signal)
        if impulse and self._impulse_notify:
            self._impulse_notify(impulse)

    def build_impulse(self, signal: Signal) -> Impulse | None:
        if meta := CommandSignalMeta.from_signal(signal):
            if meta.logos:
                impulse = Impulse.from_signal(signal, source=self.name())
                impulse = ImpulsePrimitive.command_only(impulse, meta.logos)
                impulse.priority = max(impulse.priority, self._min_priority)
                return impulse
        return None

    def with_bus(self, signal_broadcast: Callable[[Signal], None], impulse_notify: Callable[[Impulse], None]) -> None:
        self._impulse_notify = impulse_notify

    def suppress(self, suppress_by: Impulse) -> None:
        return None

    def pop_impulse(self, impulse: Impulse) -> None:
        return None

    def peek(self, no_stale: bool = True) -> Impulse | None:
        return None

    def is_running(self) -> bool:
        return self._is_running

    async def __aenter__(self) -> Self:
        self._is_running = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._is_running = False
