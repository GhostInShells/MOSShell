"""Data Ghost runtime: Memento-backed conversation, no in-process history."""

from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING

from ghoshell_container import IoCContainer
from pydantic_ai import Agent
from typing_extensions import Self

from ghoshell_moss.contracts.logger import LoggerItf, get_moss_logger
from ghoshell_moss.core.blueprint.ghost import Ghost, GhostMeta
from ghoshell_moss.core.blueprint.mindflow import Articulator
from ghoshell_moss.message import Message

from ._adapter import moment_to_request
from ._memory import DataMemory

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

    def inspect_state(self) -> dict:
        return {"memory": self._memory.inspect()}

    def inspect_context(self) -> dict:
        return dict(self._last_context)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._memory.close()
