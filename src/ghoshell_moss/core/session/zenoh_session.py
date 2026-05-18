import contextlib
from typing import Callable, Self

import janus
from ghoshell_moss.message import Message
from ghoshell_moss.contracts import Storage, LoggerItf
from ghoshell_moss.core.concepts.topic import TopicService
from ghoshell_moss.core.helpers import ThreadSafeEvent
from ghoshell_moss.core.blueprint.session import (
    Session, Signal, Role, OutputBuffer, OutputItem, StreamSubscriber,
    Sample
)
from ghoshell_moss.depends import depend_zenoh
from ghoshell_common.helpers import uuid

from typing import Iterable

import threading
import time

depend_zenoh()
import zenoh
import asyncio

__all__ = [
    'MossSessionWithZenoh',
    'SimpleOutputBuffer',
]


class SimpleOutputBuffer(OutputBuffer):

    def __init__(self, maxsize: int, on_change_interval: float = 0.5) -> None:
        self._max = maxsize
        self._on_change_interval = on_change_interval
        self._last_change_at: float = 0.0
        self._closed = False
        self._messages_lock = threading.Lock()
        self._items: list[OutputItem] = []

    def close(self) -> None:
        self._closed = True

    def is_closed(self) -> bool:
        return self._closed

    def add_output(self, item: OutputItem) -> None:
        with self._messages_lock:
            if len(self._items) > 0:
                role = item.role
                last = self._items[-1]
                if last.role == role:
                    last.messages.extend(item.messages)
                else:
                    self._items.append(item)
            else:
                self._items.append(item)
            if len(self._items) > self._max:
                self._items = self._items[len(self._items) - self._max:]
            self._last_change_at = time.time()

    def values(self) -> Iterable[OutputItem]:
        with self._messages_lock:
            items = self._items.copy()
        return items

    def updated_at(self) -> float:
        return self._last_change_at


class MossSessionWithZenoh(Session):
    """
    Session implementation for host
    """

    def __init__(
            self,
            session_scope: str,
            session_storage: Storage,
            logger: LoggerItf,
            zenoh_session: zenoh.Session,
            topic_service: TopicService,
            session_id: str | None = None,
    ):
        self._session_scope = session_scope

        # 子类继承可重写.
        self._output_key_expr = f"MOSS/{session_scope}/outputs"
        self._input_signal_expr = f"MOSS/{session_scope}/signals"
        self._stream_key_expr_prefix = f"MOSS/{session_scope}/streams"

        self._session_storage = session_storage
        self._session_id = session_id or uuid()

        self._zenoh_session = zenoh_session
        if zenoh_session.is_closed():
            raise RuntimeError(f'HostSession receive Zenoh session but closed')

        self._output_sub = zenoh_session.declare_subscriber(self._output_key_expr, self._on_zenoh_output)
        self._input_sub = zenoh_session.declare_subscriber(self._input_signal_expr, self._on_zenoh_signal_input)
        self._logger = logger
        self._log_prefix = f'<Session cls={self.__class__} scope={session_scope} id={self.session_id}>'

        # 注意内存泄漏.
        self._output_listeners: list[Callable[[OutputItem], None]] = []
        # 与生命周期绑定有限个. 这个方法没有解绑的机制. 要考虑未来支持一个最小生命周期 handler.
        self._on_signal_callbacks: list[Callable[[Signal], None]] = []
        self._topic_service = topic_service
        self._closing_event = ThreadSafeEvent()

    @property
    def session_scope(self) -> str:
        return self._session_scope

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def storage(self) -> Storage:
        return self._session_storage

    @property
    def topics(self) -> TopicService:
        return self._topic_service

    def _check_running(self) -> None:
        if self._zenoh_session.is_closed():
            raise RuntimeError(f'HostSession is closed')

    def add_signal(self, signal: Signal) -> None:
        self._check_running()
        # todo: 未来加防蠢限频.
        # 现在有一种深刻的感觉, 不存在过度设计, 只存在过度实现.
        js = signal.to_json()
        self._zenoh_session.put(self._input_signal_expr, js)

    def on_signal(self, callback: Callable[[Signal], None]) -> None:
        self._on_signal_callbacks.append(callback)

    def _on_zenoh_signal_input(self, sample: zenoh.Sample) -> None:
        if len(self._on_signal_callbacks) == 0:
            return None
        try:
            signal = Signal.model_validate_json(sample.payload.to_bytes())
        except Exception as e:
            self._logger.error(
                f"%s failed to handle received signal sample %s: %s",
                self._log_prefix, sample.payload.to_string(), e,
            )
            return None
        for callback in self._on_signal_callbacks:
            try:
                callback(signal)
            except Exception as e:
                self._logger.exception(
                    "%s failed to callback received signal on %s: %s",
                    self._log_prefix, callback, e
                )
        return None

    def output(self, role: str | Role, *messages: Message | str, log: str = '') -> None:
        item = OutputItem.new(role, *messages, log=log)
        js = item.model_dump_json(indent=0, ensure_ascii=False, exclude_none=True, exclude_defaults=True)
        self._zenoh_session.put(self._output_key_expr, js)

    def output_buffer(self, maxsize: int = 100) -> OutputBuffer:
        buffer = SimpleOutputBuffer(maxsize)

        def _output_add_to_buffer(item: OutputItem) -> None:
            nonlocal buffer
            if buffer.is_closed():
                return
            buffer.add_output(item)

        self.on_output(_output_add_to_buffer)
        return buffer

    def _on_zenoh_output(self, sample: zenoh.Sample) -> None:
        if len(self._output_listeners) == 0:
            return
        try:
            item = OutputItem.model_validate_json(sample.payload.to_bytes())
        except Exception as e:
            self._logger.error(
                "%s failed to send output %s: %s",
                self._log_prefix, sample.payload.to_string(), e,
            )
            item = OutputItem.new('error', Message.new().with_content("receive invalid output: %s" % e))
        for listener in self._output_listeners:
            try:
                listener(item)
            except Exception as e:
                self._logger.error(
                    "%s failed to send output %s: %s",
                    self._log_prefix, item.id, e,
                )

    def on_output(self, callback: Callable[[OutputItem], None]) -> None:
        self._output_listeners.append(callback)

    # ── stream 协议 ──────────────────────────────

    def is_running(self) -> bool:
        return not self._zenoh_session.is_closed()

    def self_explain(self) -> str:
        return (
            f"Session:"
            f"  scope: {self._session_scope}\n"
            f"  session_id: {self._session_id}\n"
            f"  transport: zenoh"
            f"  output key: {self._output_key_expr}\n"
            f"  signal key: {self._input_signal_expr}\n"
            f"  stream key prefix: {self._stream_key_expr_prefix}\n"
        )

    def sub_stream(
            self, relative_key: str, callback: Callable[[Sample], None],
    ) -> Callable[[], None]:
        self._check_running()

        stream_key = self.stream_key_expr(relative_key)

        def _on_sample(_sample: zenoh.Sample) -> None:
            if not self.is_running():
                return

            _relative_key = self._parse_stream_relative_key(str(_sample.key_expr))
            if _relative_key is None:
                # todo: 这里有静默失败.
                return
            _moss_sample = Sample(
                relative_key=_relative_key,
                payload=_sample.payload.to_bytes(),
            )
            callback(_moss_sample)

        sub = self._zenoh_session.declare_subscriber(stream_key, _on_sample)

        def _release():
            nonlocal sub
            if not self.is_running() or self._zenoh_session.is_closed():
                return
            try:
                sub.undeclare()
            except Exception:
                return

        return _release

    def _parse_stream_relative_key(self, sample_key: str) -> str | None:
        if sample_key.startswith(self._stream_key_expr_prefix):
            return sample_key[len(self._stream_key_expr_prefix) + 1:]
        return None

    def pub_stream_delta(self, relative_key: str, delta: bytes) -> None:
        self._check_running()
        self._zenoh_session.put(relative_key, delta)

    def stream_key_expr(self, relative_key: str) -> str:
        return "/".join([
            self._stream_key_expr_prefix,
            relative_key.strip('/')
        ])

    def get_stream(self, relative_key: str, *, maxsize: int = 0) -> StreamSubscriber:
        return _SessionStreamSubscriber(
            key_expr_prefix=self._stream_key_expr_prefix,
            relative_key=relative_key,
            maxsize=maxsize,
            zenoh_session=self._zenoh_session,
            session_stop_event=self._closing_event,
        )

    async def __aenter__(self) -> Self:
        self._logger.info("%s session started")

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._closing_event.set()
        self._logger.info("%s session closed")


class _SessionStreamSubscriber(StreamSubscriber):
    """zenoh subscriber 的 StreamHandle 实现"""

    def __init__(
            self,
            key_expr_prefix: str,
            relative_key: str,
            zenoh_session: zenoh.Session,
            session_stop_event: ThreadSafeEvent,
            maxsize: int = 0,
    ) -> None:
        self._zenoh_session = zenoh_session
        self._relative_key = relative_key
        self._key_expr_prefix = key_expr_prefix
        self._full_key = "/".join([self._key_expr_prefix, relative_key])
        self._sub: zenoh.Subscriber | None = None
        self._maxsize = maxsize
        self._session_stop_event = session_stop_event
        self._queue: janus.Queue[Sample | None] | None = None
        self._wait_session_stop_task: asyncio.Task | None = None
        self._closed = False

    def full_key(self) -> str:
        return self._full_key

    def relative_key(self) -> str:
        return self._relative_key

    # def _add_delta(self, sample: zenoh.Sample) -> None:

    def _on_zenoh_sample(self, sample: zenoh.Sample) -> None:
        """做跨线程卸载."""
        if self._closed:
            return
        key_expr = str(sample.key_expr)
        if key_expr.startswith(self._key_expr_prefix):
            relative_key = key_expr[len(self._key_expr_prefix) + 1:]
            moss_sample = Sample(
                relative_key=relative_key,
                payload=sample.payload.to_bytes(),
            )
            try:
                self._queue.sync_q.put(moss_sample)
            except janus.SyncQueueShutDown:
                # todo: 这里也有静默失败问题, 没体现消费能力不足, 消费侧毫无感知.
                self._closed = True

    async def _wait_session_closed(self) -> None:
        await self._session_stop_event.wait()
        # 发送毒丸.
        self._queue.sync_q.put_nowait(None)

    async def __aenter__(self) -> 'StreamSubscriber':
        if self._zenoh_session.is_closed():
            raise RuntimeError('Session is closed')
        elif self._sub is not None:
            raise RuntimeError('Session Stream is already started')
        self._queue = janus.Queue(maxsize=self._maxsize)
        self._sub = self._zenoh_session.declare_subscriber(
            self._full_key,
            self._queue.sync_q.put_nowait,
        )
        self._wait_session_stop_task = asyncio.create_task(self._wait_session_closed())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._closed = True
        if not self._zenoh_session.is_closed():
            try:
                self._sub.undeclare()
            except Exception:
                # zenoh 的 python 包可能有不同类型的异常, 暂时不用处理.
                pass
        if self._wait_session_stop_task is not None:
            self._wait_session_stop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._wait_session_stop_task
            self._wait_session_stop_task = None

    async def __anext__(self) -> Sample:
        if not self._sub and self._queue:
            raise RuntimeError('Session Stream must enter context manager by `async with` first')
        if self._zenoh_session.is_closed() or self._session_stop_event.is_set():
            raise StopAsyncIteration
        try:
            sample = await self._queue.async_q.get()
            if sample is None:
                raise StopAsyncIteration
            return sample
        except janus.AsyncQueueShutDown:
            raise StopAsyncIteration
