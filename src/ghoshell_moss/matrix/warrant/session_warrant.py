"""SessionWarrant — concrete Warrant backed by Session QA + filesystem state dir.

State persistence: each PermissionStateData is a JSON file in *states_dir*,
keyed by permission key (``{key}.json``).  All writes go through an ordered
asyncio queue consumed by a lifecycle task spawned in __aenter__.

QA: questions are issued through ``session.qa.asker(WARRANT_NAMESPACE)``.
Answers come from watchers on that namespace — the warrant does not answer
its own questions.
"""

from __future__ import annotations

import asyncio
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

from ghoshell_moss.contracts.logger import get_moss_logger
from ghoshell_moss.core.blueprint.session import Session
from ghoshell_moss.core.blueprint.warrant import (
    Warrant, PermissionStateData,
)
from ghoshell_moss.core.concepts.qa import Question, Answer
from ghoshell_moss.core.concepts.topic import TopicClosedError
from ghoshell_moss.matrix.warrant.topics import WarrantTruth, WarrantWriteRequest

WARRANT_NAMESPACE = "_warrant"

_FILE_SUFFIX = ".json"


class SessionWarrant(Warrant):
    """Concrete Warrant wired through a Session.

    *states_dir* is where permission states are persisted as JSON files.
    The caller decides the path — typically
    ``session.storage.sub_storage("warrants").abspath()``.
    """

    def __init__(
        self,
        session: Session,
        states_dir: Path,
        *,
        namespace: str = WARRANT_NAMESPACE,
    ) -> None:
        self._session = session
        self._states_dir = states_dir
        self._namespace = namespace
        self._logger = get_moss_logger()
        self._running = False
        self._cache: OrderedDict[str, PermissionStateData] = OrderedDict()
        self._flush_queue: asyncio.Queue[PermissionStateData | None] | None = None
        self._flush_task: asyncio.Task | None = None
        self._flush_listeners: list[Callable[[PermissionStateData], None]] = []
        self._sub: Any = None
        self._receive_task: asyncio.Task | None = None

    # -- lifecycle ----------------------------------------------------

    async def __aenter__(self):
        self._running = True
        self._flush_queue = asyncio.Queue()
        self._load_cache()
        self._flush_task = asyncio.ensure_future(self._consume_flush())
        topics = self._session.topics
        if topics is not None:
            self._sub = topics.subscribe_model(WarrantWriteRequest)
            await self._sub.__aenter__()
            self._receive_task = asyncio.ensure_future(self._receive_requests())
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> None:
        self._running = False
        if self._receive_task is not None:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None
        if self._sub is not None:
            await self._sub.__aexit__(None, None, None)
            self._sub = None
        if self._flush_queue is not None:
            await self._flush_queue.put(None)
        if self._flush_task is not None:
            await self._flush_task
        return None

    def is_running(self) -> bool:
        return self._running

    # -- raw materials (template method) ------------------------------

    def states(self) -> dict[str, PermissionStateData]:
        return dict(self._cache)

    async def ask_question(self, question: Question) -> Answer:
        asker = self._session.qa.asker(self._namespace)
        qa = asker.issue(question)
        await qa.wait()
        answer = qa.answer
        if answer is None:
            raise RuntimeError("QA completed without answer")
        return answer

    def store(self, state: PermissionStateData) -> None:
        # host 是序号权威: 未显式给 seq 时, 按缓存 current + 1 分配 (v8).
        current_seq = 0
        current = self._cache.get(state.key)
        if current is not None and current.seq is not None:
            current_seq = current.seq
        seq = state.seq if state.seq is not None else current_seq + 1
        stored = state.model_copy(update={"seq": seq})
        self._cache[stored.key] = stored
        if self._flush_queue is not None:
            self._flush_queue.put_nowait(stored)

    def list_states(self) -> list[PermissionStateData]:
        return list(self._cache.values())

    def on_flushed(
            self,
            callback: Callable[[PermissionStateData], None],
    ) -> Callable[[], None]:
        self._flush_listeners.append(callback)

        def _unsubscribe() -> None:
            if callback in self._flush_listeners:
                self._flush_listeners.remove(callback)

        return _unsubscribe

    # -- internal -----------------------------------------------------

    def _load_cache(self) -> None:
        if not self._states_dir.exists():
            return
        for f in self._states_dir.glob(f"*{_FILE_SUFFIX}"):
            try:
                state = PermissionStateData.model_validate_json(f.read_text())
                self._cache[state.key] = state
            except Exception:
                self._logger.exception("Failed to load warrant state: %s", f)

    def _flush_one(self, state: PermissionStateData) -> None:
        self._states_dir.mkdir(parents=True, exist_ok=True)
        path = self._states_dir / f"{state.key}{_FILE_SUFFIX}"
        path.write_text(state.model_dump_json(indent=2))
        self._notify_flushed(state)
        if state.seq is not None:
            self._pub_truth(state.key, state.seq, state.data)

    def _notify_flushed(self, state: PermissionStateData) -> None:
        for cb in list(self._flush_listeners):
            try:
                cb(state)
            except Exception:
                self._logger.exception("on_flushed listener failed: %s", state.key)

    def _pub_truth(self, key: str, seq: int, data: dict[str, Any]) -> None:
        topics = self._session.topics
        if topics is None:
            return
        topics.pub(WarrantTruth(key=key, seq=seq, data=data))

    # -- host receive-side (reject-retry, v8) -------------------------

    async def _receive_requests(self) -> None:
        if self._sub is None:
            return
        try:
            while True:
                try:
                    req = await self._sub.poll_model()
                except TopicClosedError:
                    break
                if req is not None:
                    self._handle_write_request(req)
        except asyncio.CancelledError:
            pass

    def _handle_write_request(self, req: WarrantWriteRequest) -> None:
        current = self._cache.get(req.key)
        current_seq = current.seq if current is not None and current.seq is not None else 0
        if req.seq == current_seq + 1:
            self.store(PermissionStateData(key=req.key, seq=req.seq, data=req.data))
        elif req.seq == current_seq:
            # duplicate — 幂等, 已在该 truth.
            return
        elif req.seq == 1:
            # 首读: fresh 非 host (空缓存) 写一个 host 已存在的 key — 重播 truth 补全.
            self._pub_truth(req.key, current_seq, current.data)
        elif req.seq < current_seq:
            # 后发先至 (stale): 新 truth 已在路上 — 静默丢弃.
            return
        else:
            # 跳号 (seq > current_seq + 1): 广播当前 truth, 发送方重读再发.
            self._pub_truth(req.key, current_seq, current.data if current is not None else {})

    async def _consume_flush(self) -> None:
        if self._flush_queue is None:
            return
        try:
            while True:
                state = await self._flush_queue.get()
                if state is None:
                    return
                try:
                    self._flush_one(state)
                except Exception:
                    self._logger.exception(
                        "Failed to flush warrant state: %s", state.key,
                    )
        except asyncio.CancelledError:
            pass
