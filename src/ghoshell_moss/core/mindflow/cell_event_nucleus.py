"""CellEventNucleus — 将 'cell_event' signal 转换为 background_notice impulse.

纯 signal→impulse 转换单元, 不依赖 Matrix/mesh/session 等系统抽象.
mesh.on_event → Signal('cell_event') 的生产侧归 channel 层 (mesh channel,
matrix-channel.md §5.2).

SignalMeta 与 Nucleus 同居: CellEventSignalMeta + CellTransition 定义在此,
`ghoshell_moss.signals` 只做 re-export (那是 Signal 的策展导出地图,
不是实现位置).
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
    """Cell 生命周期事件的信号类型.

    由 mesh channel on_startup 订阅 mesh.on_event 桥接产生 (matrix-channel.md
    §5.2), priority=BACKGROUND — 不会抢占 attention, 只作为 background hint
    进 mindflow buffer. CellEventNucleus 消费转为 Impulse.

    **字段是 nucleus 的判决依据, 不是 ghost 看的消息主体** — 消息主体
    (退出码/stderr 尾/诊断入口路径) 走 to_signal(messages=..., description=...).
    详见 SignalMeta docstring 的三尺度原则.

    默认值让空构造合法 (测试 / 兜底信号):
      CellEventSignalMeta() → address='' + transition=READY, 语义 = "有事发生".
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
    """'cell_event' signal → Impulse(background_notice).

    与 NotifyNucleus 同构: add_signal 接收 signal, build_impulse 转换,
    fire_impulse 投递到 mindflow. priority 由 signal 携带 (BACKGROUND).
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

    def suppress(self, suppress_by: Impulse) -> None:
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
