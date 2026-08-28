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
    - status() 返回红点摘要: pending 计数 + 最新 pending 消息的预览

    description() 是给模型读的静态标签 (user text input); 计数逻辑收敛在
    public ``pending_count()``, status() 的 f-string 复用, 测试可直接断言.

    impulse 生命周期观测 (public-internal, 不在 Nucleus ABC 契约内):
    ``attended_count()`` / ``ignored_count()`` / ``suppressed_count()`` 分别统计
    impulse 抢占成功 / 被忽视 / challenge 失败被压制的次数; ``counters()`` 一次性
    返回全部计数 + 最近一次动作的简介. 这三个回调是 mindflow 判定 impulse 结局后
    回写给 nucleus 的, 计数即"这个 impulse 的结局"的累计事实, 用于观测与验证.
    """

    def __init__(
            self,
            *,
            name: str = "input_signal_nucleus",
            description: str = "user text input",
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
        # 静态标签 — 稳定描述"这是什么", 不随 pending 变化. 计数在 status().
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
            # 被压制的是我们刚缓存的 impulse — 它没能抢占 attention.
            # 用缓存里的取简介; 若已被清, 回退到压制方.
            brief = self._brief(self._impulse_cache) if self._impulse_cache else self._brief(suppress_by)
            self._suppressed_cnt += 1
            self._last_suppressed = brief
            # 清 cache 让 peek() 返回 None, 但保留 _signals:
            # pop_impulse 才是一次性消费, suppress 只是冷静期,
            # 下个信号到达时从累积的 _signals 重建 impulse.
            self._impulse_cache = None

    def attended(self, impulse: Impulse) -> None:
        if not self.is_running():
            return
        with self._data_state_lock:
            self._attended_cnt += 1
            self._last_attended = self._brief(impulse)
            self.clear()

    def ignored(self, impulse: Impulse) -> None:
        with self._data_state_lock:
            self._ignored_cnt += 1
            self._last_ignored = self._brief(impulse)

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

    @staticmethod
    def _preview(signal: Signal) -> str:
        """红点预览: 最新消息的纯文本, 无消息时回退到 description 字段."""
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
        """impulse 的摘要文本 (简介): 优先最后一条消息, 回退 description, 再回退占位符."""
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


class InputNucleusMeta(NucleusMeta):

    def __init__(
            self,
            *,
            name: str = "input_signal_nucleus",
            description: str = "user text input",
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
