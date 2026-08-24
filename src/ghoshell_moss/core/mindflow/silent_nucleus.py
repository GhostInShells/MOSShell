"""SilentNucleus — 静默聚合通道, 持续 buffer 不打扰思考.

四元 nucleus 之三: 配对 ``ChallengeMode.silent`` 的"低污染数据流"入口.
监听 ``SilentSignalMeta`` (signal name = ``"silent"``), 把所有进来的 signal
合并进一个内部 buffer, 用 buffer 内最高 priority 作为 impulse 优先级, 产出的
impulse 标记为 ``mode=silent``: 抢占成功时 buffer 进 mindflow 不接管 attention,
抢占失败时 suppress (符合 silent 在"抢占成功侧偏离" 的对称语义).

结构对称于 ``InputSignalNucleus`` (两者都是 buffer + 优先级提取), 差异是:
- InputSignalNucleus: signal 视为离散事件 (FIFO 保留), default mode → 走 articulate
- SilentNucleus: signal 视为数据流 (合并语义), silent mode → 静默 buffer 不思考

这种对称在拓扑层面是有意为之 — 让开发者用"读名字就懂语义"的方式选择路由,
而不需要懂 ChallengeMode × priority × effort 的正交组合.
"""
import asyncio
import time
import threading
from typing import Callable, Iterable

from ghoshell_container import IoCContainer
from typing_extensions import Self

from ghoshell_moss.contracts.logger import LoggerItf, get_moss_logger
from ghoshell_moss.message import ContextType
from ghoshell_moss.core.blueprint.mindflow import (
    Nucleus, NucleusMeta, Signal, SignalMeta, SignalName, Impulse, Priority,
    ChallengeMode,
)

__all__ = ['SilentNucleus', 'SilentSignalMeta', 'SilentNucleusMeta', 'new_silent_signal']


class SilentSignalMeta(SignalMeta):
    """Signal meta for ``silent`` — low-pollution data stream.

    用例: 传感器读数 / 后台监控 / 状态广播 — signal 持续流入, 不希望每条都打断
    ghost 思考, 但累积到一定 priority 时希望下一帧 ghost 看到聚合后的快照.
    """

    @classmethod
    def signal_name(cls) -> SignalName:
        return 'silent'

    @classmethod
    def priority(cls) -> Priority:
        return Priority.NOTICE


class SilentNucleus(Nucleus):
    """Aggregating nucleus — continuous buffer with max-priority extraction.

    结构: 内部维护 ``_signals: list[Signal]``, ``add_signal`` 在线就直接 append +
    重建 impulse cache. ``peek`` 返回当前 cache (silent mode 标记的 impulse).

    Buffer 策略:
    - 持续 buffer, 不区分 signal 独立性
    - max_size 上限 (默认 20), 溢出时丢最早
    - stale 信号在 add / rebuild 两层过滤
    - 优先级与强度都取 buffer 内 max
    - hint / description 取最新一条 (新数据驱动语义)

    Suppress 冷静期: ``suppress_seconds`` 后才再通知 mindflow, 防止仲裁风暴.
    冷静期内 signal 仍持续 buffer, 只是不主动 challenge.
    """

    NAME = 'silent_nucleus'

    def __init__(
            self,
            *,
            name: str = NAME,
            description: str = "silent aggregating channel — buffer signals without preempting thought",
            suppress_seconds: float = 0.5,
            buffer_size: int = 20,
            min_priority: Priority = Priority.BACKGROUND,
            logger: LoggerItf | None = None,
    ):
        self._name = name
        self._description = description
        self._target_signal = SilentSignalMeta.signal_name()
        self._suppress_seconds = suppress_seconds
        self._buffer_size = buffer_size
        self._min_priority = min_priority
        self._logger = logger or get_moss_logger()

        self._signals: list[Signal] = []
        self._impulse_cache: Impulse | None = None

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
        # silent buffer 中, 取当前优先级最高的描述作摘要 — 表达"现在 buffer 中最重的事".
        top = max(self._signals, key=lambda s: s.priority_strength())
        desc = f", top: {top.description[:50]}" if top.description else ''
        return f"buffered: {count}{desc}"

    def signals(self) -> list[str]:
        return [self._target_signal]

    def clear(self) -> None:
        self._signals.clear()
        self._impulse_cache = None

    def with_bus(
            self,
            signal_broadcast: Callable[[Signal], None],
            fire_impulse: Callable[[Impulse], None],
    ) -> None:
        self._broadcast_cb = signal_broadcast
        self._notify_cb = fire_impulse

    def add_signal(self, signal: Signal) -> None:
        if not self.is_running():
            return
        if signal.name != self._target_signal:
            return
        if signal.priority < self._min_priority:
            return
        self._process_signal(signal)

    def suppress(self, suppress_by: Impulse) -> None:
        # silent 抢占失败 → 走 default suppress 分支 (对称表). 进入冷静期防风暴.
        self._suppress_until = time.monotonic() + self._suppress_seconds

    def attended(self, impulse: Impulse) -> None:
        if not self.is_running():
            return
        self._atomic_clear_buffer()

    def peek(self, no_stale: bool = True) -> Impulse | None:
        if self._impulse_cache is None:
            return None
        if no_stale and self._impulse_cache.is_stale():
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
            # 双层 stale 过滤: 加入前清旧, 新 signal 自身过期则丢.
            self._signals = [s for s in self._signals if not s.is_stale()]
            if signal.is_stale():
                return

            self._signals.append(signal)
            if len(self._signals) > self._buffer_size:
                # 溢出丢最早 — silent 数据流场景, 旧数据已被新数据覆盖.
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

        # buffer 中 max priority + max strength — 累积压力作为整体优先级.
        max_priority = max(s.priority for s in valid)
        max_strength = max(s.strength for s in valid)

        all_msgs: list = []
        for s in valid:
            all_msgs.extend(s.messages)

        # 取最新一条作为 description / hint 的语义核心 — 新数据驱动当前快照.
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
            hint=latest.hint,
            complete=all(s.complete for s in valid),
            stale_timeout=latest.stale_timeout,
            # 核心标记 — silent mode 在抢占成功侧偏离 default, 不接管 attention.
            mode=ChallengeMode.silent.value,
        )

    def _atomic_clear_buffer(self) -> None:
        with self._data_state_lock:
            self.clear()


class SilentNucleusMeta(NucleusMeta):
    """Factory meta — lets ``moss manifests nuclei`` discover SilentNucleus."""

    def __init__(
            self,
            *,
            name: str = SilentNucleus.NAME,
            description: str = "silent aggregating channel — buffer signals without preempting thought",
            suppress_seconds: float = 0.5,
            buffer_size: int = 20,
            min_priority: Priority = Priority.BACKGROUND,
    ):
        self._name = name
        self._description = description
        self._suppress_seconds = suppress_seconds
        self._buffer_size = buffer_size
        self._min_priority = min_priority

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return self._description

    def signals(self) -> Iterable[type[SignalMeta]]:
        yield SilentSignalMeta

    def factory(self, container: IoCContainer) -> Nucleus:
        logger = container.get(LoggerItf)
        return SilentNucleus(
            name=self._name,
            description=self._description,
            suppress_seconds=self._suppress_seconds,
            buffer_size=self._buffer_size,
            min_priority=self._min_priority,
            logger=logger,
        )


def new_silent_signal(
        *messages: ContextType,
        priority: Priority = Priority.NOTICE,
        description: str = '',
        stale_timeout: float = 0,
        hint: str = '',
) -> Signal:
    """Helper — construct a ``silent`` signal in one call."""
    return SilentSignalMeta().to_signal(
        *messages,
        description=description,
        stale_timeout=stale_timeout,
        priority=priority,
        hint=hint,
    )
