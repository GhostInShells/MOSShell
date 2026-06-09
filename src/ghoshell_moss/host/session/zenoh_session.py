import contextlib
import logging
import queue
from typing import Callable
from typing_extensions import Self

import janus
from ghoshell_moss.message import Message
from pathlib import Path

from ghoshell_moss.contracts import Storage, LoggerItf
from ghoshell_moss.contracts.cache import Cache
from ghoshell_moss.core.cache import SqliteCache
from ghoshell_moss.core.blueprint.parameter import ParameterStore
from ghoshell_moss.core.parameter import SessionParameterStore
from ghoshell_moss.core.concepts.topic import TopicService
from ghoshell_moss.core.helpers import ThreadSafeEvent
from ghoshell_moss.core.blueprint.session import (
    Session, Signal, Role, OutputBuffer, OutputItem, StreamSubscriber,
    Sample
)
from ghoshell_moss.core.blueprint.environment import DEFAULT_SESSION_SCOPE
from ghoshell_moss.depends import depend_zenoh
from ghoshell_moss.message import unique_id
from ghoshell_moss.core.session.utils import SimpleOutputBuffer

depend_zenoh()
import zenoh
import asyncio

__all__ = [
    'MossSessionWithZenoh',
    'SimpleOutputBuffer',
]


class MossSessionWithZenoh(Session):
    """
    Session implementation for host
    """

    def __init__(
            self,
            session_scope: str,
            session_root_storage: Storage,
            session_tmp_root_storage: Storage,
            logger: LoggerItf,
            zenoh_session: zenoh.Session,
            topic_service: TopicService,
            session_id: str | None = None,
    ):
        """
        :param session_scope: Moss Matrix 运行时, 所有通讯都围绕同一个 session scope.
        :param session_root_storage: 在当前隔离级别下, Session 拿到的 Root Storage.
        :param session_tmp_storage:  session 的临时存储路径.
        :param logger: 日志模块.
        :param zenoh_session: 依赖 zenoh 通讯.
        :param topic_service: session 持有 topic service. 未来应该是 session 构建它.
        :param session_id: 会话 id, 它实际上在同源 Matrix 所有实例中应该要共享, 从 env 中获取.
        """
        self._session_scope = session_scope or DEFAULT_SESSION_SCOPE  # or 逻辑简单做一个防蠢, 怕 storage 逻辑崩了.

        # 子类继承可重写.
        self._output_key_expr = f"MOSS/{session_scope}/outputs"
        self._input_signal_expr = f"MOSS/{session_scope}/signals"
        self._stream_key_expr_prefix = f"MOSS/{session_scope}/streams"
        self._received_signal_index: int = 0

        self._session_id = session_id or unique_id()

        # session 实例级别的 id.
        self._session_unique_id = unique_id()

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
        self._session_root_storage = session_root_storage
        self._session_tmp_root_storage = session_tmp_root_storage
        self._parameters: ParameterStore | None = None

    @property
    def session_scope(self) -> str:
        return self._session_scope

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def topics(self) -> TopicService:
        return self._topic_service

    @property
    def sessions_root_storage(self) -> Storage:
        return self._session_root_storage

    @property
    def sessions_tmp_root_storage(self) -> Storage:
        return self._session_tmp_root_storage

    @property
    def cache(self) -> Cache:
        if not hasattr(self, '_cache'):
            db_path = Path(self.tmp_storage.abspath()) / 'cache.db'
            self._cache = SqliteCache(db_path)
        return self._cache

    @property
    def parameters(self) -> ParameterStore:
        self._check_running()
        if self._parameters is None:
            self._parameters = SessionParameterStore(self)
        return self._parameters

    def _check_running(self) -> None:
        if self._zenoh_session.is_closed():
            raise RuntimeError(f'HostSession is closed')

    def add_signal(self, signal: Signal) -> None:
        """向 session 总线发布信号。

        调用方负责控制发送频率。本方法不做限频——连续高频调用会直接打满 zenoh
        发布通道，淹没下游 subscriber 回调链。限频应在 Mindflow 的 signal
        ingestion 层实现，而非 transport 层。
        """
        self._check_running()
        js = signal.to_json()
        self._zenoh_session.put(self._input_signal_expr, js)

    def on_signal(self, callback: Callable[[Signal], None]) -> None:
        self._on_signal_callbacks.append(callback)

    def _on_zenoh_signal_input(self, sample: zenoh.Sample) -> None:
        if len(self._on_signal_callbacks) == 0:
            return None
        try:
            signal = Signal.model_validate_json(sample.payload.to_bytes())
            self._received_signal_index += 1
            # 在 session 内流转的都分配一个隐藏的 参数方便 debug. 没有额外性能开销.
            signal.metadata['_session_signal_index'] = self._received_signal_index
        except Exception as e:
            self._logger.error(
                f"%s failed to handle received signal sample %s: %s",
                self._log_prefix, sample.payload.to_string(), e,
            )
            return None
        # 回调感知接口.
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
            f"  transport: zenoh\n"
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
                self._logger.warning(
                    "%s stream subscriber received sample with unexpected key: %s (prefix: %s)",
                    self._log_prefix, str(_sample.key_expr), self._stream_key_expr_prefix,
                )
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
        self._zenoh_session.put(self.stream_key_expr(relative_key), delta)

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
        self._logger.info("%s session started", self._log_prefix)
        # Eager-init parameter store in thread pool — SQLite WAL + Zenoh
        # sub are synchronous and would block the event loop.
        await asyncio.to_thread(lambda: self.parameters)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._closing_event.set()
        if self._parameters is not None:
            self._parameters.close()
        self._logger.info("%s session closed", self._log_prefix)


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

    def _on_zenoh_sample(self, sample: zenoh.Sample) -> None:
        """跨线程卸载：zenoh 回调 → janus 同步队列。

        使用 put_nowait 避免阻塞 zenoh 内部线程。队列满时丢弃并 log，
        优于阻塞 zenoh 影响全局通讯总线。
        """
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
                self._queue.sync_q.put_nowait(moss_sample)
            except janus.SyncQueueShutDown:
                self._closed = True
            except queue.Full:
                pass

    async def _wait_session_closed(self) -> None:
        await self._session_stop_event.wait()
        try:
            self._queue.sync_q.put_nowait(None)
        except janus.SyncQueueShutDown:
            pass

    async def __aenter__(self) -> 'StreamSubscriber':
        if self._zenoh_session.is_closed():
            raise RuntimeError('Session is closed')
        elif self._sub is not None:
            raise RuntimeError('Session Stream is already started')
        self._queue = janus.Queue(maxsize=self._maxsize)
        self._sub = self._zenoh_session.declare_subscriber(
            self._full_key,
            self._on_zenoh_sample,
        )
        self._wait_session_stop_task = asyncio.create_task(self._wait_session_closed())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._closed = True
        if self._sub is not None and not self._zenoh_session.is_closed():
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
        if not self._sub or not self._queue:
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
