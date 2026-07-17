"""Data Ghost runtime: Memento-backed conversation, no in-process history."""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING

from ghoshell_container import IoCContainer
from pydantic_ai import Agent
from typing_extensions import Self

from ghoshell_moss.contracts.logger import LoggerItf, get_moss_logger
from ghoshell_moss.core.blueprint.ghost import Ghost, GhostMeta
from ghoshell_moss.core.blueprint.mindflow import Articulator
from ghoshell_moss.core.concepts.channel import Channel
from ghoshell_moss.message import Message

from ._adapter import moment_to_request
from ._channel import new_memento_channel
from ._memory import DataMemory
from ._reflection import DataReflector

if TYPE_CHECKING:
    from ._meta import DataMeta

__all__ = ["Data"]


class Data(Ghost):
    """Ghost prototype whose conversation history lives only in Memento."""

    def __init__(
        self,
        *,
        meta: "DataMeta",
        agent: Agent[IoCContainer],
        container: IoCContainer,
        memory_root: str | Path,
        memory_owner: str,
        memory_detail_n: int,
        memory_summary_m: int,
        auto_commit_every: int,
        reflection_agent: Agent[IoCContainer] | None,
        reflection_max_summary_chars: int,
        reflection_max_source_chars: int,
        reflection_startup_limit: int,
        reflection_enabled: bool,
    ) -> None:
        self._meta = meta
        self._agent = agent
        self._container = container
        self._logger = container.get(LoggerItf) or get_moss_logger()
        self._memory = DataMemory(
            memory_root,
            memory_owner,
            detail_n=memory_detail_n,
            summary_m=memory_summary_m,
            auto_commit_every=auto_commit_every,
        )
        self._reflector = (
            DataReflector(
                reflection_agent,
                max_summary_chars=reflection_max_summary_chars,
                max_source_chars=reflection_max_source_chars,
            )
            if reflection_enabled and reflection_agent is not None
            else None
        )
        self._reflection_startup_limit = reflection_startup_limit
        self._reflection_tasks: set[asyncio.Task] = set()
        self._reflection_inflight: set[str] = set()
        self._reflection_errors: list[str] = []
        self._channel = new_memento_channel(self._memory, on_reflect=self.schedule_reflection)
        self._last_context: dict = {}

    @property
    def meta(self) -> GhostMeta:
        return self._meta

    @property
    def memory(self) -> DataMemory:
        return self._memory

    def system_prompt(self) -> str:
        return self._meta.build_instruction_from_ioc(self._container)

    def memories(self) -> list[Message]:
        return self._memory.messages()

    def channel(self) -> Channel:
        return self._channel

    async def articulate(self, articulator: Articulator) -> AsyncIterator[str]:
        request = moment_to_request(articulator.moment)
        history = self._memory.model_history()
        self._last_context = {
            "system": self.system_prompt(),
            "history_messages": len(history),
            "memory": self._memory.inspect(),
        }
        async with self._agent.run_stream(
            user_prompt=request.parts,
            message_history=history,
            deps=self._container,
        ) as stream:
            async for text in stream.stream_text(delta=True):
                yield text

    def on_articulate_exit(
        self,
        articulator: Articulator,
        logos: str,
        error: Exception | None,
    ) -> None:
        if error is not None:
            self._last_context["memory_write"] = "skipped_on_error"
            return
        commit = self._memory.remember(articulator.moment)
        self._last_context["memory_write"] = "committed" if commit else "staged"
        if commit:
            self._last_context["commit_id"] = commit.id
            self.schedule_reflection([commit])

    def schedule_reflection(self, commits=None) -> None:
        """Schedule reflection without delaying the completed articulate cycle."""
        if self._reflector is None:
            return
        selected = list(commits) if commits is not None else self._memory.reflection_candidates()
        selected = [view for view in selected if view.id not in self._reflection_inflight]
        if not selected:
            return
        self._reflection_inflight.update(view.id for view in selected)
        task = asyncio.create_task(self._reflect_all(selected))
        self._reflection_tasks.add(task)
        task.add_done_callback(self._reflection_tasks.discard)

    async def _reflect_all(self, commits) -> None:
        assert self._reflector is not None
        for view in commits:
            try:
                if view.id not in {candidate.id for candidate in self._memory.reflection_candidates()}:
                    continue
                reflected = await self._reflector.reflect(self._memory, view, self._container)
                if reflected is not None:
                    self._last_context["reflection_commit_id"] = reflected.id
            except asyncio.CancelledError:
                raise
            except Exception as error:  # Reflection is intentionally a non-fatal side path.
                message = f"{type(error).__name__}: {error}"
                self._reflection_errors.append(message)
                self._logger.warning("Data memory reflection failed for %s: %s", view.id, message)
            finally:
                self._reflection_inflight.discard(view.id)

    def inspect_state(self) -> dict:
        return {
            "memory": self._memory.inspect(),
            "reflection": {
                "enabled": self._reflector is not None,
                "running": len(self._reflection_tasks),
                "inflight": len(self._reflection_inflight),
                "errors": list(self._reflection_errors[-3:]),
            },
        }

    def inspect_context(self) -> dict:
        return dict(self._last_context)

    async def __aenter__(self) -> Self:
        if self._reflection_startup_limit > 0:
            self.schedule_reflection(self._memory.reflection_candidates()[:self._reflection_startup_limit])
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        tasks = list(self._reflection_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._memory.close()
