"""Post-commit reflection that only rewrites Memento's interpretation layer."""

from ghoshell_container import IoCContainer
from pydantic_ai import Agent

from ghoshell_moss.core.memento import CommitView

from ._memory import DataMemory

__all__ = ["DataReflector"]


_INSTRUCTION = """You curate a Ghost's durable memory after a completed conversation segment.
Read only the supplied observable transcript. Return one concise Chinese memory note describing:
the user's goal or stable preference, corrections, unresolved threads, and what matters next.
Do not invent facts. Do not describe hidden reasoning. If evidence is weak, say so briefly.
Return only the note itself, without headings, quotes, or markdown."""


class DataReflector:
    """LLM-backed, retryable-by-replay reflection for frozen mechanical commits."""

    def __init__(
        self,
        agent: Agent[IoCContainer],
        *,
        max_summary_chars: int,
        max_source_chars: int,
    ) -> None:
        self._agent = agent
        self._max_summary_chars = max_summary_chars
        self._max_source_chars = max_source_chars

    async def reflect(self, memory: DataMemory, view: CommitView, container: IoCContainer) -> CommitView | None:
        transcript = memory.commit_transcript(view.id, max_chars=self._max_source_chars)
        if not transcript:
            return None
        result = await self._agent.run(
            f"Commit: {view.id}\n\nObservable transcript:\n{transcript}",
            deps=container,
            instructions=_INSTRUCTION,
        )
        summary = " ".join(str(result.output).split())[:self._max_summary_chars]
        if not summary:
            return None
        return memory.apply_reflection(view.id, summary)
