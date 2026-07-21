"""示范最基础的 Signal 实现 | 系统 | beta """

import asyncio
import time
from typing import Callable, Iterable

from ghoshell_container import IoCContainer
from typing_extensions import Self

from ghoshell_moss.core.blueprint.mindflow import (
    Nucleus, Signal, Impulse, Priority, SignalMeta, NucleusMeta, InputSignalMeta,
)

from ghoshell_moss.contracts.logger import LoggerItf, get_moss_logger
import threading

__all__ = ["InputSignalNucleus", 'InputSignalMeta', 'InputNucleusMeta']


class InputSignalNucleus(Nucleus):
    """
    IM 红点式信号聚合 — 监听 input signal, FIFO 缓冲, pop 时全量返回.

    与 BufferNucleus 的区别:
    - 无 pulse beat 循环, 仅在新信号到达时通知
    - pop 时 Impulse 保留全部消息的 FIFO 顺序
    - status() 返回红点摘要
    """

    def __init__(
            self,
            *,
            name: str = "input_signal_nucleus",
            description: str = "default input signal nucleus",
            default_prompt: str = '',
            suppress_seconds: float = 0.5,
            buffer_size: int = 20,
            min_priority: Priority = Priority.INFO,
            logger: LoggerItf | None = None,
    ):
        self._name = name
        self._description = description
        self._target_signal = InputSignalMeta.signal_name()
        self._suppress_seconds = suppress_seconds
        self._buffer_size = buffer_size
        self._default_prompt = default_prompt
        self._logger = logger or get_moss_logger()
        self._min_priority = min_priority

        self._signals: list[Signal] = []
        self._impulse_cache: Impulse | None = None

        # 未来看到这里的模型协作者需要记得, 关键代码必须加注释.
        # 比如这里如果用了 thread lock, 要考虑所有用锁的程序都要避免搞非计算逻辑的线程阻塞.
        self._data_state_lock = threading.Lock()
        self._suppress_until: float = 0.0
        self._broadcast_cb: Callable[[Signal], None] | None = None
        self._notify_cb: Callable[[Impulse], None] | None = None
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._created_impulse_index: int = 0
        self._running = False

    # -- Nucleus ABC --

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return self._description

    def is_running(self) -> bool:
        return self._running and self._event_loop is not None

    def status(self) -> str:
        count = len(self._signals)
        if count == 0:
            return ""
        latest = max(self._signals, key=lambda s: s.created_at.timestamp())
        desc = f", top: {latest.description[:50]}" if latest.description else ''
        return f"pending: {count}{desc}"

    def signals(self) -> list[str]:
        return [self._target_signal]

    def clear(self) -> None:
        self._signals.clear()
        self._impulse_cache = None

    def with_bus(
            self,
            signal_broadcast: Callable[[Signal], None],
            impulse_notify: Callable[[Impulse], None],
    ) -> None:
        self._broadcast_cb = signal_broadcast
        self._notify_cb = impulse_notify

    def add_signal(self, signal: Signal) -> None:
        if not self.is_running():
            return
        if signal.name != self._target_signal:
            return
        if signal.priority < self._min_priority:
            return
        self._process_signal(signal)

    def suppress(self, suppress_by: Impulse) -> None:
        self._suppress_until = time.monotonic() + self._suppress_seconds
        # default 路径 suppress = 信丢, 清空 buffer 防止 cooldown 后重放.
        # _suppress_until 控制 notify 回调防抖, 不影响 buffer 清空.
        self._atomic_clear_buffer()

    def pop_impulse(self, impulse: Impulse) -> None:
        if not self.is_running():
            return
        self._atomic_clear_buffer()

    def peek(self, no_stale: bool = True) -> Impulse | None:
        if self._impulse_cache is None:
            return None
        if no_stale and self._impulse_cache.is_stale():
            return None
        if time.monotonic() < self._suppress_until:
            return None
        return self._impulse_cache

    # -- lifecycle --

    async def __aenter__(self) -> Self:
        self._running = True
        self._event_loop = asyncio.get_running_loop()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._running = False

    # -- internal --

    def _process_signal(self, signal: Signal) -> None:
        with self._data_state_lock:
            self._signals = [s for s in self._signals if not s.is_stale()]
            if signal.is_stale():
                return

            self._signals.append(signal)
            if len(self._signals) > self._buffer_size:
                self._signals.pop(0)

            self._impulse_cache = self._rebuild_impulse()

            if time.monotonic() > self._suppress_until and self._impulse_cache is not None:
                self._notify_impulse()

    def _notify_impulse(self) -> None:
        if self._notify_cb and self._impulse_cache:
            self._notify_cb(self._impulse_cache)

    def _rebuild_impulse(self) -> Impulse | None:
        valid = [s for s in self._signals if not s.is_stale()]
        if not valid:
            return None

        max_priority = max(s.priority for s in valid)
        max_strength = max(s.strength for s in valid)

        all_msgs = []
        for s in valid:
            all_msgs.extend(s.messages)

        latest = valid[-1]

        self._created_impulse_index += 1
        return Impulse(
            source=self._name,
            source_idx=self._created_impulse_index,
            id=latest.id,
            priority=max_priority,
            strength=max_strength,
            messages=all_msgs,
            description=latest.description,
            hint=latest.hint or self._default_prompt,
            complete=all(s.complete for s in valid),
            stale_timeout=latest.stale_timeout,
        )

    def _atomic_clear_buffer(self) -> None:
        with self._data_state_lock:
            self.clear()


class InputNucleusMeta(NucleusMeta):

    def __init__(
            self,
            *,
            name: str = "input_signal_nucleus",
            description: str = "default input signal nucleus",
            default_prompt: str = '',
            suppress_seconds: float = 0.5,
            buffer_size: int = 20,
            min_priority: Priority = Priority.INFO,
    ):
        self._name = name
        self._description = description
        self._target_signal = InputSignalMeta.signal_name()
        self._suppress_seconds = suppress_seconds
        self._buffer_size = buffer_size
        self._default_prompt = default_prompt
        self._min_priority = min_priority

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return self._description

    def signals(self) -> Iterable[type[SignalMeta]]:
        yield InputSignalMeta

    def factory(self, container: IoCContainer) -> Nucleus:
        logger = container.get(LoggerItf)
        return InputSignalNucleus(
            name=self._target_signal,
            description=self._description,
            default_prompt=self._default_prompt,
            suppress_seconds=self._suppress_seconds,
            buffer_size=self._buffer_size,
            min_priority=self._min_priority,
            logger=logger,
        )
