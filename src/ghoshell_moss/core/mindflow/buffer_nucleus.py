import asyncio
import time
from typing import Callable, Self
from ghoshell_moss.core.concepts.mindflow import Nucleus, Signal, Impulse, Priority
from ghoshell_moss.contracts.logger import LoggerItf, get_moss_logger


class BufferNucleus(Nucleus):
    """
    极简版的 Nucleus, 可以视作一个信号闸门.
    由 Gemini 3 撰写, 人类架构师 review 微调.
    """

    def __init__(
            self,
            *,
            name: str,
            description: str,
            target_signal: str,
            default_prompt: str = '',
            suppress_seconds: float = 2.0,
            buffer_size: int = 5,
            min_priority: Priority = Priority.INFO,
            # 预计的过期时间
            strength_decay_seconds: int = 30,
            pulse_beat_interval: float = 3.0,
            logger: LoggerItf | None = None,
    ):
        self._name = name
        self._description = description
        self._target_signal = target_signal
        self._suppress_seconds = suppress_seconds
        self._buffer_size = buffer_size
        self._default_prompt = default_prompt
        self._logger = logger or get_moss_logger()
        self._min_priority = min_priority
        self._strength_decay_seconds = strength_decay_seconds
        self._pulse_beat_interval = pulse_beat_interval
        self._last_notify_time = 0.0
        self._has_new_impulse = asyncio.Event()

        # signal buffer
        self._signals: list[Signal] = []

        self._impulse_cache: Impulse | None = None

        self._lock = asyncio.Lock()
        self._suppress_until: float = 0.0
        self._broadcast_cb: Callable[[Signal], None] | None = None
        self._notify_cb: Callable[[Impulse], None] | None = None
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._impulse_beat_loop_task: asyncio.Task | None = None
        self._running = False

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return self._description

    def is_running(self) -> bool:
        return self._running and self._event_loop is not None

    def status(self) -> str:
        count = len(self._signals)
        if count == 0: return ""
        # 优先级最高的那条描述
        best = max(self._signals, key=lambda s: s.priority_strength())
        description = f", top: {best.description[:50]}" if best.description else ''
        return f"pending: {count}{description}"

    def signals(self) -> list[str]:
        return [self._target_signal]

    def clear(self) -> None:
        # 立刻生效 .
        self._signals.clear()
        self._impulse_cache = None

    def with_bus(self, signal_broadcast: Callable[[Signal], None], impulse_notify: Callable[[Impulse], None]) -> None:
        self._broadcast_cb = signal_broadcast
        self._notify_cb = impulse_notify

    def on_signal(self, signal: Signal) -> None:
        # 理论上 on signal 来自 mindflow 的回调, 和 mindflow 处于同一个 loop.
        if not self.is_running():
            # 丢弃, 未开始.
            return None
        if signal.name != self._target_signal:
            return None
        elif signal.priority < self._min_priority:
            # 丢弃最低优先级.
            return None

        # 简单异步入口，避免阻塞调用者
        self._event_loop.create_task(self._process_signal(signal))
        return None

    async def _process_signal(self, signal: Signal) -> None:
        # 用 asyncio lock 就不需要用队列了.
        async with self._lock:
            # 1. 过滤陈旧信号
            self._signals = [s for s in self._signals if not s.is_stale()]
            if signal.is_stale():
                return

            # 2. 加入 Buffer
            self._signals.append(signal)
            if len(self._signals) > self._buffer_size:
                self._signals.pop(0)

            # 3. 重新计算 Cache
            self._impulse_cache = self._rebuild_impulse()

            # 4. 判断是否处于冷静期
            if time.monotonic() > self._suppress_until and self._impulse_cache is not None:
                # 主动通知.
                self._notify_impulse()

    def _notify_impulse(self) -> None:
        if self._notify_cb and self._impulse_cache:
            self._notify_cb(self._impulse_cache)
        # 更新最后通知时间.
        self._last_notify_time = time.monotonic()

    def _rebuild_impulse(self) -> Impulse | None:
        if not self._signals:
            return None

        # 排序：按时间戳（保证顺序）
        sorted_signals = sorted(
            [_s for _s in self._signals if not _s.is_stale()],
            key=lambda _s: _s.created_at.timestamp(),
        )

        if len(sorted_signals) == 0:
            return None

        max_priority = 0
        max_strength = 0
        for s in self._signals:
            max_priority = max(max_priority, s.priority)
            max_strength = max(max_strength, s.strength)

        # 取最新的一条作为 Prompt 和 Prompt 语义核心
        latest = sorted_signals[-1]

        # 合并所有消息
        all_msgs = []
        for s in sorted_signals:
            all_msgs.extend(s.messages)

        return Impulse(
            source=self._name,
            id=latest.id,
            priority=max_priority,
            strength=max_strength,
            messages=all_msgs,
            description=latest.description,
            prompt=latest.prompt or self._default_prompt,
            complete=all([s.complete for s in sorted_signals]),
            stale_timeout=latest.stale_timeout,
            strength_decay_seconds=self._strength_decay_seconds,
        )

    async def _impulse_beat_loop(self):
        """创建一个循环"""
        while self._running:
            now = time.monotonic()
            if now < self._suppress_until:
                await asyncio.sleep(self._suppress_until - now)
                continue
            elif (now - self._last_notify_time) > self._pulse_beat_interval:
                # 清理过期信息.
                impulse = self._rebuild_impulse()
                self._impulse_cache = impulse
                if impulse is not None:
                    self._notify_impulse()
            # 无论如何都等待这个心跳.
            await asyncio.sleep(self._pulse_beat_interval)

    def suppress(self, suppress_by: Impulse) -> None:
        self._suppress_until = time.monotonic() + self._suppress_seconds

    def pop_impulse(self, impulse: Impulse) -> None:
        if not self.is_running():
            return None
        # 直接通过 event loop 将清理任务排队，确保它是逻辑上的原子操作
        self._event_loop.create_task(self._atomic_clear_buffer())
        return None

    async def _atomic_clear_buffer(self) -> None:
        async with self._lock:
            # 在锁内进行清理，确保不会被 _process_signal 中断
            self.clear()
            self._last_notify_time = time.monotonic()

    def peek(self, no_stale: bool = True) -> Impulse | None:
        if self._impulse_cache is None:
            return None
        if no_stale and self._impulse_cache.is_stale():
            return None
        return self._impulse_cache

    async def __aenter__(self) -> Self:
        self._running = True
        self._event_loop = asyncio.get_running_loop()
        self._impulse_beat_loop_task = self._event_loop.create_task(self._impulse_beat_loop())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._running = False
        if self._impulse_beat_loop_task and not self._impulse_beat_loop_task.done():
            self._impulse_beat_loop_task.cancel()
            try:
                await self._impulse_beat_loop_task
            except asyncio.CancelledError:
                pass
            self._impulse_beat_loop_task = None
