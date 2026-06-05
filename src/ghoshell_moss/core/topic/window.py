"""
Generic deque-backed TopicWindow implementation.

This implementation depends only on the TopicService ABC (Subscriber, TopicModel)
and is shared by both QueueBasedTopicService and ZenohTopicService.
"""

from collections import deque
from typing import Callable, NamedTuple

from ghoshell_moss.core.concepts.topic import (
    TopicWindow, TOPIC_MODEL, TopicModel, Subscriber,
)
from ghoshell_moss.core.helpers import ThreadSafeEvent
from ghoshell_common.contracts import LoggerItf
import asyncio
import logging
import threading
import time

__all__ = ["DequeTopicWindow"]


class _WindowCallback(NamedTuple):
    """Internal callback record with debounce/throttle state."""
    callback: Callable[['DequeTopicWindow'], None]
    debounce: float
    throttle: float
    last_fired: float
    timer: asyncio.Task | None


class DequeTopicWindow(TopicWindow[TOPIC_MODEL]):
    """
    Generic deque-backed TopicWindow — fed via add_item() from a consumer
    coroutine managed by TopicService.create_window().
    """

    def __init__(
            self,
            *,
            topic_name: str,
            max_size: int,
            model: type[TopicModel],
            subscriber: Subscriber,
            subscribing_started: ThreadSafeEvent,
            loop: asyncio.AbstractEventLoop | None = None,
            logger: LoggerItf | None = None,
    ):
        self._max_size = max_size
        self._model = model
        self._deque: deque[TOPIC_MODEL] = deque(maxlen=max_size)
        self._lock = threading.Lock()
        self._subscribing_started = subscribing_started
        self._changed_at: float = 0.0
        self._callbacks: dict[int, _WindowCallback] = {}
        self._callback_counter = 0
        self._loop = loop or asyncio.get_running_loop()
        self._logger = logger or logging.getLogger("moss")
        self._log_prefix = f"[DequeTopicWindow topic={topic_name} max={max_size}]"

    # ── data entry (called from consumer coroutine) ──────

    async def add_item(self, topic: TOPIC_MODEL) -> None:
        """Called by the consumer coroutine when a new typed model arrives."""
        if not isinstance(topic, self._model):
            self._logger.warning(
                "%s add_item type mismatch: expected %s, got %s",
                self._log_prefix, self._model, type(topic),
            )
            return
        with self._lock:
            self._deque.append(topic)
            self._changed_at = time.monotonic()
        self._fire_immediate_callbacks()

    # ── read API (thread-safe) ──────────────────────────

    @property
    def max_size(self) -> int:
        return self._max_size

    def values(self) -> list[TOPIC_MODEL]:
        with self._lock:
            return list(self._deque)

    def __len__(self) -> int:
        # deque.__len__ is atomic in CPython; only the consumer coroutine writes.
        return len(self._deque)

    def changed_at(self) -> float:
        return self._changed_at

    # ── lifecycle ────────────────────────────────────────

    async def wait_started(self) -> None:
        await self._subscribing_started.wait()

    # ── callback API ─────────────────────────────────────

    def on_change(
            self,
            callback: Callable[['TopicWindow'], None],
            *,
            debounce: float = 0,
            throttle: float = 0,
    ) -> Callable[[], None]:
        self._callback_counter += 1
        callback_id = self._callback_counter
        record = _WindowCallback(
            callback=callback,
            debounce=debounce,
            throttle=throttle,
            last_fired=0.0,
            timer=None,
        )
        self._callbacks[callback_id] = record

        def remove():
            rec = self._callbacks.pop(callback_id, None)
            if rec is not None and rec.timer is not None:
                rec.timer.cancel()

        return remove

    def _fire_immediate_callbacks(self):
        """Fire callbacks with debounce=0 and throttle=0 from thread pool."""
        if not self._callbacks:
            return
        for callback_id, rec in list(self._callbacks.items()):
            if callback_id not in self._callbacks:
                continue
            if rec.debounce == 0 and rec.throttle == 0:
                try:
                    self._loop.run_in_executor(None, rec.callback, self)
                except Exception:
                    self._logger.exception("%s callback error", self._log_prefix)
