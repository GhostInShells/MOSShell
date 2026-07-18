"""Post-commit reflection that only rewrites Memento's interpretation layer."""

from ghoshell_container import IoCContainer
from pydantic_ai import Agent

from ghoshell_moss.core.memento import CommitView

from ._memory import AureliusMemory

__all__ = ["AureliusReflector"]


_INSTRUCTION = """You curate a Ghost's durable memory after a completed conversation segment.
Read only the supplied observable transcript. Return one concise Chinese memory note describing:
the user's goal or stable preference, corrections, unresolved threads, and what matters next.
Do not invent facts. Do not describe hidden reasoning. If evidence is weak, say so briefly.
Return only the note itself, without headings, quotes, or markdown."""


class AureliusReflector:
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

    async def reflect(
        self,
        memory: AureliusMemory,
        view: CommitView,
        container: IoCContainer,
    ) -> CommitView | None:
        transcript = memory.commit_transcript(view.id, max_chars=self._max_source_chars)
        if not transcript:
            return None
        # Use the same stream transport as the foreground conversation. Some
        # Anthropic-compatible providers accept a model only with stream=true;
        # reflection remains invisible to the user because we consume locally.
        parts: list[str] = []
        async with self._agent.run_stream(
            f"Commit: {view.id}\n\nObservable transcript:\n{transcript}",
            deps=container,
            instructions=_INSTRUCTION,
        ) as stream:
            async for delta in stream.stream_text(delta=True):
                parts.append(delta)
        summary = " ".join("".join(parts).split())[:self._max_summary_chars]
        if not summary:
            return None
        return memory.apply_reflection(view.id, summary)
