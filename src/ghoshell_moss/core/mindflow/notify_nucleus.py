"""NotifyNucleus — turns ``notify`` signals into ``notify``-mode impulses.

四元 nucleus 之二: 配对 ``ImpulsePrimitive.notify`` 的"不丢消息"入口.
监听 ``NotifySignalMeta`` (signal name = ``"notify"``), 把 signal 包装成
``mode=notify`` 的 impulse — 抢占成功正常创建 attention, 失败时 messages
进 mindflow buffer 而非 suppress (notify 在"抢占失败侧"偏离 default).

priority 完全继承 ``Signal.priority`` (调用方控制). 默认 ``NOTICE`` —
典型用例是 "ghost 思考时用户说话", 不打断就留痕.
"""
from typing import Self, Callable, Iterable

from ghoshell_container import IoCContainer

from ghoshell_moss.contracts.logger import LoggerItf, get_moss_logger
from ghoshell_moss.message import ContextType
from ghoshell_moss.core.blueprint import Impulse
from ghoshell_moss.core.blueprint.mindflow import (
    SignalMeta, SignalName, Priority, Signal,
    Nucleus, NucleusMeta, ImpulsePrimitive,
)

__all__ = ['NotifyNucleus', 'NotifySignalMeta', 'NotifyNucleusMeta', 'new_notify_signal']


class NotifySignalMeta(SignalMeta):
    """Signal meta for ``notify`` — carries messages that must not be lost.

    与 ``InputSignal`` 区别: input 走 default mode (抢占失败 suppress, 信丢);
    notify 走 notify mode (抢占失败 buffer, 留痕). 用例确定要不要丢就选 input,
    确定不能丢就选 notify.
    """

    @classmethod
    def signal_name(cls) -> SignalName:
        return 'notify'

    @classmethod
    def priority(cls) -> Priority:
        return Priority.NOTICE


class NotifyNucleus(Nucleus):
    """Reflex-arc nucleus — turns each ``notify`` signal into a notify-mode
    impulse and pushes it through the bus.

    Fire-and-forget 路径: ``add_signal`` 立即构造 impulse 并调 ``impulse_notify``.
    与 ``InputSignalNucleus`` 的聚合模式不同 — notify 的语义是"每条都要被看到",
    聚合反而会让 mindflow 误判为同 source 的连续 input. 每条 signal 独立成 impulse,
    让 mindflow 的对称表 (notify 抢占失败 → buffer) 自然处理.

    priority 完全继承 ``Signal.priority`` — 调用方决定优先级, nucleus 不强制 floor.
    """

    NAME = 'notify_nucleus'

    def __init__(self, *, name: str = NAME, logger: LoggerItf | None = None):
        self._name = name
        self._impulse_notify: Callable[[Impulse], None] | None = None
        self._is_running = False
        self._logger = logger or get_moss_logger()

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return 'wrap notify signals into notify-mode impulses (message-preserving)'

    def status(self) -> str:
        return ''

    def signals(self) -> list[SignalName]:
        return [NotifySignalMeta.signal_name()]

    def clear(self) -> None:
        return

    def add_signal(self, signal: Signal) -> None:
        if not self._is_running:
            return
        impulse = self.build_impulse(signal)
        if impulse and self._impulse_notify:
            self._impulse_notify(impulse)

    def build_impulse(self, signal: Signal) -> Impulse | None:
        if not NotifySignalMeta.match(signal):
            return None
        impulse = Impulse.from_signal(signal, source=self.name())
        return ImpulsePrimitive.notify(impulse)

    def with_bus(
            self,
            signal_broadcast: Callable[[Signal], None],
            impulse_notify: Callable[[Impulse], None],
    ) -> None:
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
