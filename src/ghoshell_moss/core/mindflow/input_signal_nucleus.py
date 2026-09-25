"""InputSignalNucleus — the default openbox channel for user text input.

Openbox nucleus, not a global mechanism: signals delivered here respect this
nucleus's one aggregation rule. For a different rule, register your own nucleus.
"""

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
    """User-text aggregate buffer — the ghost turns toward the user when the buffer's
    collective urgency wins.

    Functional intent: each user message joins a buffer. When free the ghost turns
    toward the user; when busy with something less urgent the messages wait
    (pending) until a more urgent input arrives.

    Mechanism (openbox aggregation rule): all buffered signals collapse into one
    impulse — priority is the buffer's max priority, strength its max strength,
    messages concatenated in ``created_at`` order. Winning creates a default
    attention; losing keeps the buffer pending under a cooldown that a
    strictly-higher new signal can break through.

    Not a queue: messages are never delivered one-at-a-time, and priority never
    reorders delivery — it only sets the buffer's collective challenge weight.
    Signals arrive out of order, so the nucleus sorts by ``created_at`` itself.

    ``peek`` returns a challenge stub (priority/strength/id/complete, no messages);
    ``attended`` materializes the full impulse from the sorted buffer and returns it.
    """

    NAME = "input_signal_nucleus"

    def __init__(
            self,
            *,
            name: str = NAME,
            description: str = "user text input — aggregate buffer, turn toward the user when it wins",
            default_prompt: str = '',
            suppress_seconds: float = 5.0,
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
        self._last_impulse_weight: int = 0
        self._running = False

        # -- impulse 生命周期观测 (public-internal: 供调试/测试/控制台读取, 不在 Nucleus ABC 契约内) --
        # 计数只在对应的 impulse 回调发生时递增, 是"这个 impulse 的结局是什么"的累计事实.
        self._attended_cnt = 0
        self._ignored_cnt = 0
        self._suppressed_cnt = 0
        # 最近一次对应动作的摘要文本 (简介), 便于定位"刚刚发生了什么".
        self._last_attended: str = ''
        self._last_ignored: str = ''
        self._last_suppressed: str = ''

    # -- Nucleus ABC --

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        # Static label — what this is; the pending count lives in status(), not here.
        return self._description

    def pending_count(self) -> int:
        """尚未交付给 mindflow 的 input signal 数量 (排除已 stale 的信号)."""
        with self._data_state_lock:
            return len([s for s in self._signals if not s.is_stale()])

    # -- impulse 生命周期观测 (public-internal) --

    def attended_count(self) -> int:
        """Impulse 抢占 Attention 成功 (attended) 的次数."""
        with self._data_state_lock:
            return self._attended_cnt

    def ignored_count(self) -> int:
        """Impulse 被忽视 (ignored, 如过期) 的次数."""
        with self._data_state_lock:
            return self._ignored_cnt

    def suppressed_count(self) -> int:
        """Impulse challenge 失败被压制 (suppressed) 的次数."""
        with self._data_state_lock:
            return self._suppressed_cnt

    def counters(self) -> dict[str, int | str]:
        """一次取回全部观测: 三类动作计数 + pending 计数 + 最近一次动作的简介."""
        with self._data_state_lock:
            pending = len([s for s in self._signals if not s.is_stale()])
            return {
                "pending": pending,
                "attended": self._attended_cnt,
                "ignored": self._ignored_cnt,
                "suppressed": self._suppressed_cnt,
                "last_attended": self._last_attended,
                "last_ignored": self._last_ignored,
                "last_suppressed": self._last_suppressed,
            }

    def is_running(self) -> bool:
        return self._running and self._event_loop is not None

    def status(self) -> str:
        # 锁内只做 O(n) 引用级操作 (过滤 + max), 内容提取放锁外 —
        # 保持 _data_state_lock 为快锁, 不阻塞跨线程调用方.
        with self._data_state_lock:
            valid = [s for s in self._signals if not s.is_stale()]
            if not valid:
                return ""
            latest = max(valid, key=lambda s: s.created_at.timestamp())
        preview = ' '.join(self._preview(latest).split())[:50]
        return f"pending: {self.pending_count()}, last: {preview}"

    def signals(self) -> list[str]:
        return [self._target_signal]

    def clear(self) -> None:
        self._signals.clear()
        self._impulse_cache = None
        self._last_impulse_weight = 0

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

    def suppress(self, suppress_by: Impulse, suppressed: Impulse | None = None) -> None:
        self._suppress_until = time.monotonic() + self._suppress_seconds
        with self._data_state_lock:
            # The cache is a challenge stub (no messages) — brief from the buffered
            # signals so the observability summary still reflects real content.
            self._suppressed_cnt += 1
            self._last_suppressed = self._buffer_brief()
            # Keep the stub and _signals: suppress is a cooldown that stops active
            # fire, not a consumption — the impulse stays peekable (rank-visible),
            # and _last_impulse_weight stays as the gate's "strictly higher" baseline.

    def attended(self, impulse: Impulse) -> Impulse | None:
        if not self.is_running():
            return None
        with self._data_state_lock:
            # Materialize the full impulse from the sorted buffer — the stub only
            # carried the challenge weight; attention runs on the real payload.
            full = self._materialize(impulse)
            self._attended_cnt += 1
            self._last_attended = self._brief(full) if full else self._buffer_brief()
            self.clear()
            return full

    def ignored(self, impulse: Impulse) -> None:
        with self._data_state_lock:
            self._ignored_cnt += 1
            self._last_ignored = self._buffer_brief()

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

    @staticmethod
    def _preview(signal: Signal) -> str:
        """Red-dot preview: latest message's plain text, falling back to description."""
        if signal.messages:
            last = signal.messages[-1]
            parts = []
            for content in last.as_contents(with_meta=False, join_text=True):
                if isinstance(content, dict) and content.get('text'):
                    parts.append(str(content['text']))
            text = ' '.join(parts).strip()
            if text:
                return text
        return signal.description or '<input>'

    @staticmethod
    def _brief(impulse: Impulse | None) -> str:
        """Impulse summary: last message's text, falling back to description, then placeholder."""
        if impulse is None:
            return '<no impulse>'
        if impulse.messages:
            last = impulse.messages[-1]
            parts = []
            for content in last.as_contents(with_meta=False, join_text=True):
                if isinstance(content, dict) and content.get('text'):
                    parts.append(str(content['text']))
            text = ' '.join(parts).strip()
            if text:
                return text
        return impulse.description or '<no content>'

    def _buffer_brief(self) -> str:
        """Real content behind a challenge stub — the latest valid buffered signal."""
        valid = [s for s in self._signals if not s.is_stale()]
        if not valid:
            return '<no content>'
        newest = max(valid, key=lambda s: s.created_at.timestamp())
        return self._preview(newest)

    def _process_signal(self, signal: Signal) -> None:
        with self._data_state_lock:
            self._signals = [s for s in self._signals if not s.is_stale()]
            if signal.is_stale():
                return

            self._signals.append(signal)
            if len(self._signals) > self._buffer_size:
                self._signals.pop(0)

            self._impulse_cache = self._build_stub()
            if self._impulse_cache is None:
                return

            # Challenge gate: a strictly-higher aggregate weight breaks the cooldown.
            new_weight = self._impulse_cache.priority_strength()
            if new_weight > self._last_impulse_weight:
                self._suppress_until = 0.0
            self._last_impulse_weight = new_weight

            if time.monotonic() > self._suppress_until:
                self._notify_impulse()

    def _notify_impulse(self) -> None:
        if self._notify_cb and self._impulse_cache:
            self._notify_cb(self._impulse_cache)

    def _build_stub(self) -> Impulse | None:
        """Challenge stub — the aggregate's collective weight, no messages.

        mindflow only needs priority/strength/id/complete to rank and arbitrate;
        the full payload is materialized in ``attended`` (challenge weight vs
        runtime weight bifurcation).
        """
        valid = [s for s in self._signals if not s.is_stale()]
        if not valid:
            return None

        newest = max(valid, key=lambda s: s.created_at.timestamp())
        max_priority = max(s.priority for s in valid)
        max_strength = max(s.strength for s in valid)

        self._created_impulse_index += 1
        return Impulse(
            source=self._name,
            source_idx=self._created_impulse_index,
            id=newest.id,
            priority=max_priority,
            strength=max_strength,
            complete=all(s.complete for s in valid),
            stale_timeout=newest.stale_timeout,
        )

    def _materialize(self, stub: Impulse) -> Impulse | None:
        """Build the full impulse from the sorted buffer, reusing the stub's identity."""
        valid = sorted(
            (s for s in self._signals if not s.is_stale()),
            key=lambda s: s.created_at.timestamp(),
        )
        if not valid:
            return None

        all_msgs = []
        for s in valid:
            all_msgs.extend(s.messages)

        newest = valid[-1]
        return stub.model_copy(update={
            'messages': all_msgs,
            'description': newest.description,
            'hint': newest.hint or self._default_prompt,
        })


class InputNucleusMeta(NucleusMeta):

    def __init__(
            self,
            *,
            name: str = InputSignalNucleus.NAME,
            description: str = "user text input — aggregate buffer, turn toward the user when it wins",
            default_prompt: str = '',
            suppress_seconds: float = 5.0,
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
            name=self._name,
            description=self._description,
            default_prompt=self._default_prompt,
            suppress_seconds=self._suppress_seconds,
            buffer_size=self._buffer_size,
            min_priority=self._min_priority,
            logger=logger,
        )
