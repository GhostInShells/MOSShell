"""TopicWarrant — concrete Warrant for a non-host cell (topic mode).

Non-host cells do not own storage. store() updates the local cache and
publishes a write-request on the write topic; the host persists it and
broadcasts truth. TopicWarrant subscribes to the truth topic and reconciles
its cache to the authoritative state, firing on_flushed on each truth.

QA (authorization questions) is still wired through ``session.qa`` — the
warrant asks the same way SessionWarrant does. Only the *storage* layer
differs by host/non-host (v8).
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from ghoshell_moss.contracts.logger import get_moss_logger
from ghoshell_moss.core.blueprint.session import Session
from ghoshell_moss.core.blueprint.warrant import (
    Warrant, PermissionStateData,
)
from ghoshell_moss.core.concepts.qa import Question, Answer
from ghoshell_moss.core.concepts.topic import (
    TopicClosedError,
    TopicService,
)
from ghoshell_moss.matrix.warrant.session_warrant import WARRANT_NAMESPACE
from ghoshell_moss.matrix.warrant.topics import WarrantTruth, WarrantWriteRequest


class TopicWarrant(Warrant):
    """non-host concrete. store() -> cache + write-request topic; truth reconciles cache."""

    def __init__(
        self,
        session: Session,
        *,
        namespace: str = WARRANT_NAMESPACE,
    ) -> None:
        self._session = session
        self._namespace = namespace
        self._logger = get_moss_logger()
        self._running = False
        self._cache: dict[str, PermissionStateData] = {}
        self._flush_listeners: list[Callable[[PermissionStateData], None]] = []
        self._sub: Any = None
        self._receive_task: asyncio.Task | None = None

    @property
    def _topics(self) -> TopicService:
        topics = self._session.topics
        if topics is None:
            raise RuntimeError("TopicWarrant requires a Session topic service")
        return topics

    # -- lifecycle ----------------------------------------------------

    async def __aenter__(self):
        self._running = True
        self._sub = self._topics.subscribe_model(WarrantTruth)
        await self._sub.__aenter__()
        self._receive_task = asyncio.ensure_future(self._receive_truth())
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
        return None

    def is_running(self) -> bool:
        return self._running

    # -- raw materials (template method) ------------------------------

    def states(self) -> dict[str, PermissionStateData]:
        return dict(self._cache)

    async def ask_question(self, question: Question) -> Answer:
        qa = self._session.qa
        if qa is None:
            raise RuntimeError("TopicWarrant requires a Session qa manager")
        asker = qa.asker(self._namespace)
        qa_q = asker.issue(question)
        await qa_q.wait()
        answer = qa_q.answer
        if answer is None:
            raise RuntimeError("QA completed without answer")
        return answer

    def store(self, state: PermissionStateData) -> None:
        current_seq = 0
        current = self._cache.get(state.key)
        if current is not None and current.seq is not None:
            current_seq = current.seq
        seq = state.seq if state.seq is not None else current_seq + 1
        stored = state.model_copy(update={"seq": seq})
        self._cache[stored.key] = stored
        self._topics.pub(
            WarrantWriteRequest(key=stored.key, seq=seq, data=stored.data),
        )

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

    # -- truth reconciliation -----------------------------------------

    async def _receive_truth(self) -> None:
        if self._sub is None:
            return
        try:
            while self._running:
                try:
                    truth = await self._sub.poll_model(timeout=0.2)
                except asyncio.TimeoutError:
                    continue
                except TopicClosedError:
                    break
                if truth is not None:
                    self._apply_truth(truth)
        except asyncio.CancelledError:
            pass

    def _apply_truth(self, truth: WarrantTruth) -> None:
        state = PermissionStateData(key=truth.key, seq=truth.seq, data=truth.data)
        self._cache[truth.key] = state
        self._notify_flushed(state)

    def _notify_flushed(self, state: PermissionStateData) -> None:
        for cb in list(self._flush_listeners):
            try:
                cb(state)
            except Exception:
                self._logger.exception("on_flushed listener failed: %s", state.key)
