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
from typing import Any

from ghoshell_moss.contracts.logger import get_moss_logger
from ghoshell_moss.core.blueprint.session import Session
from ghoshell_moss.core.blueprint.warrant import (
    Warrant, PermissionStateData,
)
from ghoshell_moss.core.concepts.qa import Question, Answer

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

    # -- lifecycle ----------------------------------------------------

    async def __aenter__(self):
        self._running = True
        self._flush_queue = asyncio.Queue()
        self._load_cache()
        self._flush_task = asyncio.ensure_future(self._consume_flush())
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> None:
        self._running = False
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
        self._cache[state.key] = state
        if self._flush_queue is not None:
            self._flush_queue.put_nowait(state)

    def list_states(self) -> list[PermissionStateData]:
        return list(self._cache.values())

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
