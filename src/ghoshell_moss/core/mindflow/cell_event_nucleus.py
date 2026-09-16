"""CellEventNucleus — converts ``cell_event`` signals into background_notice impulses.

A pure signal→impulse unit with no dependency on Matrix/mesh/session. The producer
side (mesh.on_event → Signal('cell_event')) belongs to the channel layer (mesh
channel, matrix-channel.md §5.2).

SignalMeta and Nucleus live together: CellEventSignalMeta + CellTransition are
defined here; ``ghoshell_moss.signals`` only re-exports them.
"""
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
    """Cell 生命周期跃迁类型 (§WW-5 四弧 + spawned 起点).

    nucleus 判决核心: 未来分档时按 transition override 优先级
    (如 CRASHED → 从 BACKGROUND 提到 NOTICE), 一行代码扩展.
    """

    SPAWNED = 'spawned'
    """父进程 spawn 完成, 子进程 pid 已知, 尚未入网."""

    READY = 'ready'
    """子进程 announce presence, 网络上可见 (新器官上线)."""

    EXITED = 'exited'
    """子进程正常退出 (exit_code == 0)."""

    CRASHED = 'crashed'
    """子进程异常退出 (exit_code != 0)."""


class CellEventSignalMeta(SignalMeta):
    """Signal meta for ``cell_event`` — a lifecycle change in the cell network.

    Produced by the mesh channel subscribing to mesh.on_event (matrix-channel.md
    §5.2). priority=BACKGROUND — it never preempts attention, only enters the
    mindflow buffer as a background hint; CellEventNucleus converts it to an
    impulse.

    **The fields are the nucleus's routing signal, not message content** — the
    message body (exit code, stderr tail, diagnostics path) goes through
    to_signal(messages=..., description=...). See the SignalMeta docstring's
    three-scales principle.

    Defaults keep an empty construct valid (tests / fallback):
      CellEventSignalMeta() → address='' + transition=READY, meaning "something happened".
    """

    address: str = Field(
        default='',
        description="cell address (kind/name/uid), 事件主语. "
                    "nucleus 未来按 cell 去重/分组的锚. 空 = 未定/兜底.",
    )
    transition: CellTransition = Field(
        default=CellTransition.READY,
        description="生命周期跃迁类型. nucleus 分档判决的核心依据.",
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

    def __init__(self, *, name: str = NAME, logger: LoggerItf | None = None):
        self._name = name
        self._fire_impulse: Callable[[Impulse], None] | None = None
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
        if self._fire_impulse:
            self._fire_impulse(impulse)

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
