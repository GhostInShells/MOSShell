import contextlib
import queue
import janus

from ghoshell_moss.core.helpers import ThreadSafeEvent
from ghoshell_moss.core.blueprint.session import (
    StreamSubscriber,
    Sample
)
from ghoshell_moss.depends import depend_matrix

depend_matrix()
import zenoh
import asyncio

__all__ = ['ZenohStreamSubscriber']


class ZenohStreamSubscriber(StreamSubscriber):
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
        # declare_subscriber 是同步 zenoh 调用, to_thread 卸载 (runtime 路径).
        self._sub = await asyncio.to_thread(
            self._zenoh_session.declare_subscriber,
            self._full_key,
            self._on_zenoh_sample,
        )
        self._wait_session_stop_task = asyncio.create_task(self._wait_session_closed())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._closed = True
        if self._sub is not None and not self._zenoh_session.is_closed():
            try:
                await asyncio.to_thread(self._sub.undeclare)
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
