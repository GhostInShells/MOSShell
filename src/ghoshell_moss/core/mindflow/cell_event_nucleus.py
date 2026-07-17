"""CellEventNucleus — 将 'cell_event' signal 转换为 background_notice impulse.

纯 signal→impulse 转换单元, 不依赖 Matrix/mesh/session 等系统抽象.
mesh.on_event → Signal('cell_event') 的生产侧归 channel 层 (cells channel).
"""
from typing import Callable, Iterable
from typing_extensions import Self

from ghoshell_container import IoCContainer

from ghoshell_moss.core.blueprint.mindflow import (
    SignalMeta, SignalName, Priority, Signal,
    Nucleus, NucleusMeta, ImpulsePrimitive, Impulse,
)
from ghoshell_moss.contracts.logger import LoggerItf, get_moss_logger
from ghoshell_moss.signals import CellEventSignalMeta

__all__ = ['CellEventNucleus', 'CellEventNucleusMeta']

NAME = 'cell_event_nucleus'


class CellEventNucleus(Nucleus):
    """'cell_event' signal → Impulse(background_notice).

    与 NotifyNucleus 同构: add_signal 接收 signal, build_impulse 转换,
    impulse_notify 投递到 mindflow. priority 由 signal 携带 (BACKGROUND).
    """

    def __init__(self, *, name: str = NAME, logger: LoggerItf | None = None):
        self._name = name
        self._impulse_notify: Callable[[Impulse], None] | None = None
        self._is_running = False
        self._logger = logger or get_moss_logger()
        self._impulse: Impulse | None = None

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return 'convert cell_event signals into background_notice impulses'

    def status(self) -> str:
        return ''

    def signals(self) -> list[SignalName]:
        return [CellEventSignalMeta.signal_name()]

    def clear(self) -> None:
        self._impulse = None

    def add_signal(self, signal: Signal) -> None:
        if not self._is_running:
            return
        impulse = self.build_impulse(signal)
        if impulse is None:
            return
        self._impulse = impulse
        if self._impulse_notify:
            self._impulse_notify(impulse)

    def build_impulse(self, signal: Signal) -> Impulse | None:
        if not CellEventSignalMeta.match(signal):
            return None
        impulse = Impulse.from_signal(signal, source=self.name())
        return ImpulsePrimitive.background_notice(impulse)

    def with_bus(
            self,
            signal_broadcast: Callable[[Signal], None],
            impulse_notify: Callable[[Impulse], None],
    ) -> None:
        self._impulse_notify = impulse_notify

    def suppress(self, suppress_by: Impulse) -> None:
        self._impulse = None

    def pop_impulse(self, impulse: Impulse) -> None:
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


class CellEventNucleusMeta(NucleusMeta):
    """Factory meta — 让 manifests nuclei 发现 CellEventNucleus."""

    def name(self) -> str:
        return NAME

    def description(self) -> str:
        return 'convert cell_event signals into background_notice impulses'

    def signals(self) -> Iterable[type[SignalMeta]]:
        yield CellEventSignalMeta

    def factory(self, container: IoCContainer) -> Nucleus:
        logger = container.get(LoggerItf)
        return CellEventNucleus(logger=logger)
