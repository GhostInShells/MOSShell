"""CellEventNucleus — converts ``cell_event`` signals into background_notice impulses.

A pure signal→impulse unit with no dependency on Matrix/mesh/session. The producer
side (mesh.on_event → Signal('cell_event')) belongs to the channel layer (mesh
channel, matrix-channel.md §5.2).

SignalMeta and Nucleus live together: CellEventSignalMeta + CellTransition are
defined here; ``ghoshell_moss.signals`` only re-exports them.
"""
import time
from enum import Enum
from typing import Callable, Iterable
from typing_extensions import Self

from pydantic import Field

from ghoshell_container import IoCContainer

from ghoshell_moss.core.blueprint.mindflow import (
    SignalMeta, SignalName, Priority, Signal,
    Nucleus, NucleusMeta, ImpulsePrimitive, Impulse,
)
from ghoshell_moss.contracts.logger import LoggerItf, get_moss_logger

__all__ = [
    'CellEventNucleus', 'CellEventNucleusMeta',
    'CellEventSignalMeta', 'CellTransition',
    'NAME',
]

NAME = 'cell_event_nucleus'


# ==== signal payload =============================================


class CellTransition(str, Enum):
    """Which step a cell's life took — being born, coming online, or going away."""

    SPAWNED = 'spawned'
    """The cell was spawned: its process exists, but it is not on the network yet."""

    READY = 'ready'
    """The cell announced itself and is reachable — a new organ is online."""

    EXITED = 'exited'
    """The cell shut down cleanly (exit_code == 0)."""

    CRASHED = 'crashed'
    """The cell died on an error (exit_code != 0)."""


class CellEventSignalMeta(SignalMeta):
    """Signal meta for ``cell_event`` — something in the system came up, became ready,
    went away, or crashed.

    How much it matters depends on what happened — a crash should not read like a
    routine start.
    """

    address: str = Field(
        default='',
        description="cell address (kind/name/uid) — the subject of the event, and the "
                    "anchor for per-cell grouping. Empty = undetermined / fallback.",
    )
    transition: CellTransition = Field(
        default=CellTransition.READY,
        description="which lifecycle transition happened — how much the event matters is "
                    "read off this.",
    )

    @classmethod
    def signal_name(cls) -> SignalName:
        return 'cell_event'

    @classmethod
    def priority(cls) -> Priority:
        return Priority.BACKGROUND


# ==== nucleus ====================================================


class CellEventNucleus(Nucleus):
    """Cell-lifecycle channel — converts ``cell_event`` into background_notice impulses.

    Functional intent: cell network transitions reach the ghost as low-priority
    awareness, never an interruption.

    Mechanism: isomorphic to NotifyNucleus — add_signal receives, build_impulse
    converts, fire_impulse delivers to mindflow. priority rides with the signal
    (BACKGROUND).
    """

    def __init__(
            self,
            *,
            name: str = NAME,
            logger: LoggerItf | None = None,
            suppress_seconds: float = 0.5,
    ):
        self._name = name
        self._fire_impulse: Callable[[Impulse], None] | None = None
        self._is_running = False
        self._logger = logger or get_moss_logger()
        self._impulse: Impulse | None = None
        self._suppress_seconds = suppress_seconds
        self._suppress_until: float = 0.0

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
        self._suppress_until = 0.0

    def add_signal(self, signal: Signal) -> None:
        if not self._is_running:
            return
        impulse = self.build_impulse(signal)
        if impulse is None:
            return
        if self._impulse is not None and not self._impulse.is_stale():
            # cell_event 同 notify 契约: burst 里后到 signal 的 messages 合并进
            # pending impulse, 不覆盖丢弃 (n 个 node 同时上线时只留最后一条的 bug).
            self._impulse.messages.extend(impulse.messages)
        else:
            self._impulse = impulse
        # suppress 后的 cooldown 内不主动 fire — 由 _loop_attention 下一轮 re-rank 捞回.
        if self._fire_impulse and time.monotonic() > self._suppress_until:
            self._fire_impulse(self._impulse)

    def build_impulse(self, signal: Signal) -> Impulse | None:
        if not CellEventSignalMeta.match(signal):
            return None
        impulse = Impulse.from_signal(signal, source=self.name())
        return ImpulsePrimitive.background_notice(impulse)

    def with_bus(
            self,
            signal_broadcast: Callable[[Signal], None],
            fire_impulse: Callable[[Impulse], None],
    ) -> None:
        self._fire_impulse = fire_impulse

    def suppress(self, suppress_by: Impulse, suppressed: Impulse | None = None) -> None:
        # 契约: suppressed 后 impulse 仍保留、可 peek, 只是 cooldown 内不主动 fire —
        # rank 输掉不等于完结, 由 _loop_attention 下一轮 re-rank 把它捞回.
        self._suppress_until = time.monotonic() + self._suppress_seconds

    def attended(self, impulse: Impulse) -> None:
        if self._impulse is impulse:
            self._impulse = None
            self._suppress_until = 0.0

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
        self._suppress_until = 0.0


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
