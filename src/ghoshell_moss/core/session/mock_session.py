"""
进程内 Mock Session，不依赖 zenoh，用于测试 Session 消费者组件。

signal / output / stream 全部在进程内同步交付。
可作为 IoC 替换实现，注入到依赖 Session 的组件中做单元测试。
"""

import asyncio
import tempfile
from pathlib import Path
from typing import Callable

from ghoshell_moss.message import Message
from ghoshell_moss.contracts.workspace import Storage, LocalStorage
from ghoshell_moss.contracts.cache import Cache
from ghoshell_moss.core.cache import SqliteCache
from ghoshell_moss.core.blueprint.parameter import ParameterStore
from ghoshell_moss.core.parameter import SessionParameterStore
from ghoshell_moss.core.concepts.topic import TopicService
from ghoshell_moss.core.concepts.qa import QAManager
from ghoshell_moss.core.blueprint.session import (
    Session, Signal, Role, OutputItem, OutputBuffer, Sample, StreamSubscriber,
)
from ghoshell_moss.core.session.utils import SimpleOutputBuffer

__all__ = ["MockSession", "SimpleOutputBuffer"]


class _MockStreamSubscriber(StreamSubscriber):
    """进程内 stream 订阅者，基于 asyncio.Queue。"""

    def __init__(self, full_key: str, relative_key: str) -> None:
        self._full_key = full_key
        self._relative_key = relative_key
        self._queue: asyncio.Queue[Sample | None] | None = None
        self._entered = False

    def full_key(self) -> str:
        return self._full_key

    def relative_key(self) -> str:
        return self._relative_key

    async def __aenter__(self) -> "_MockStreamSubscriber":
        if self._entered:
            raise RuntimeError("StreamSubscriber already entered")
        self._queue = asyncio.Queue()
        self._entered = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._queue is not None:
            self._queue.put_nowait(None)  # sentinel

    async def __anext__(self) -> Sample:
        if not self._entered or self._queue is None:
            raise RuntimeError("StreamSubscriber must be entered via `async with` first")
        sample = await self._queue.get()
        if sample is None:
            raise StopAsyncIteration
        return sample

    def _push(self, sample: Sample) -> None:
        if self._queue is not None:
            self._queue.put_nowait(sample)


class MockSession(Session):
    """Session ABC 的纯内存实现。signal/output/stream 在进程内同步交付。

    可通过 IoC 替换 MossSessionWithZenoh，用于不依赖 zenoh 的测试场景。

    额外暴露用于测试断言的属性:
    - ``signals``: 所有 add_signal() 的历史
    - ``outputs``: 所有 output() 的历史
    - ``stream_pubs``: 按 key 的 pub_stream_delta 的历史
    """

    def __init__(
            self,
            session_scope: str = "mock_scope",
            *,
            session_id: str | None = None,
            topics: TopicService | None = None,
            qa_manager: QAManager | None = None,
            storage: Storage | None = None,
    ):
        from ghoshell_moss.message import unique_id

        self._session_scope = session_scope
        self._session_id = session_id or unique_id()
        self._topics = topics
        self._qa_manager = qa_manager
        self._running = True
        self._stream_key_prefix = f"MOSS/{session_scope}/streams"

        # storage
        self._session_root_storage = storage or LocalStorage(Path(tempfile.mkdtemp()))
        self._session_storage: Storage | None = None

        # signal 回调链
        self._signal_callbacks: list[Callable[[Signal], None]] = []

        # output 回调链
        self._output_listeners: list[Callable[[OutputItem], None]] = []

        # stream 内部状态
        self._stream_callbacks: dict[str, list[Callable[[Sample], None]]] = {}
        self._stream_queues: dict[str, list[_MockStreamSubscriber]] = {}

        # 历史记录 — 供测试断言
        self.signals: list[Signal] = []
        self.outputs: list[OutputItem] = []
        self.stream_pubs: dict[str, list[bytes]] = {}

    @property
    def storage(self) -> Storage:
        return self._session_root_storage

    @property
    def tmp_storage(self) -> Storage:
        return self._session_root_storage.sub_storage('tmp')

    @property
    def cache(self) -> Cache:
        if not hasattr(self, '_cache'):
            db_path = Path(self.tmp_storage.abspath()) / 'cache.db'
            self._cache = SqliteCache(db_path)
        return self._cache

    @property
    def parameters(self) -> ParameterStore:
        if not hasattr(self, '_parameters'):
            self._parameters = SessionParameterStore(self)
        return self._parameters

    # ── storages ──────────────────────────────

    # ── properties ──────────────────────────────

    @property
    def session_scope(self) -> str:
        return self._session_scope

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def topics(self) -> TopicService:
        return self._topics

    @property
    def qa(self) -> QAManager | None:
        return self._qa_manager

    def _make_session_level_storage(self, storage: Storage) -> Storage:
        scope_level_storage = storage.sub_storage(self._session_scope)
        return scope_level_storage.sub_storage(f"session-{self._session_id}")

    # ── signal ──────────────────────────────────

    def add_signal(self, signal: Signal) -> None:
        self.signals.append(signal)
        for cb in self._signal_callbacks:
            cb(signal)

    def on_signal(self, callback: Callable[[Signal], None]) -> None:
        self._signal_callbacks.append(callback)

    # ── output ──────────────────────────────────

    def output(self, role: str | Role, *messages: Message | str, log: str = "") -> None:
        item = OutputItem.new(role, *messages, log=log)
        self.outputs.append(item)
        for listener in self._output_listeners:
            listener(item)

    def on_output(self, callback: Callable[[OutputItem], None]) -> None:
        self._output_listeners.append(callback)

    def output_buffer(self, maxsize: int = 100) -> OutputBuffer:
        buffer = SimpleOutputBuffer(maxsize)

        def _add_to_buffer(item: OutputItem) -> None:
            if not buffer.is_closed():
                buffer.add_output(item)

        self.on_output(_add_to_buffer)
        return buffer

    # ── stream ──────────────────────────────────

    def is_running(self) -> bool:
        return self._running

    def self_explain(self) -> str:
        return (
            f"Session:\n"
            f"  scope: {self._session_scope}\n"
            f"  session_id: {self._session_id}\n"
            f"  transport: mock (in-process)\n"
            f"  stream key prefix: {self._stream_key_prefix}\n"
        )

    def stream_key_expr(self, relative_key: str) -> str:
        return "/".join([self._stream_key_prefix, relative_key.strip("/")])

    def sub_stream(
            self, relative_key: str, callback: Callable[[Sample], None],
    ) -> Callable[[], None]:
        self._stream_callbacks.setdefault(relative_key, []).append(callback)

        def _stop() -> None:
            cbs = self._stream_callbacks.get(relative_key, [])
            if callback in cbs:
                cbs.remove(callback)

        return _stop

    def pub_stream_delta(self, relative_key: str, delta: bytes) -> None:
        # 历史记录
        self.stream_pubs.setdefault(relative_key, []).append(delta)

        sample = Sample(relative_key=relative_key, payload=delta)

        # 通知 sub_stream 回调
        for cb in self._stream_callbacks.get(relative_key, []):
            cb(sample)

        # 推送到 get_stream 的订阅者
        for sub in self._stream_queues.get(relative_key, []):
            sub._push(sample)

    def get_stream(
            self, relative_key: str, *, maxsize: int = 0,
    ) -> StreamSubscriber:
        full_key = self.stream_key_expr(relative_key)
        sub = _MockStreamSubscriber(full_key=full_key, relative_key=relative_key)
        self._stream_queues.setdefault(relative_key, []).append(sub)
        return sub

    # ── lifecycle ───────────────────────────────

    async def __aenter__(self):
        self._running = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._running = False
        # 通知所有 stream subscriber 结束
        for subs in self._stream_queues.values():
            for sub in subs:
                if sub._queue is not None:
                    try:
                        sub._queue.put_nowait(None)
                    except asyncio.QueueFull:
                        pass
