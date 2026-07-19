"""Bypass curation: an async agent that distils frozen commits into a pinned notes file.

This is the reflection pattern applied one layer up. It never parses Moment payloads
into structured claims, never intercepts articulate, and never becomes a source of
truth. It reads the observable commit log, asks a small model to write a human-readable
``facts.md``, and pins that file into a memory Ground so the main model sees it each
frame and can decide for itself what to trust — the evidence stays the frozen Moment,
addressable by ``memory_show``. Failure is non-fatal: a missing or stale ``facts.md``
just means the model falls back to ``memory_search`` over the raw trajectory.
"""

from __future__ import annotations

from pathlib import Path

from ghoshell_container import IoCContainer
from pydantic_ai import Agent

from ._memory import AureliusMemory

__all__ = ["CURATION_FILE_NAME", "AureliusCurator"]

CURATION_FILE_NAME = "facts.md"

_INSTRUCTION = """You maintain a Ghost's running notes file from an observable conversation log.
Write a concise Chinese markdown digest of stable facts, decisions, corrections and open
threads that a future turn should remember. For every concrete fact (codes, ids, names,
places) cite the commit id in parentheses, e.g. 「校验词：雪松 (cmt_...)」, so the reader
can expand the frozen original. Do not invent anything not present in the log. Do not
describe hidden reasoning. Prefer omission over guessing. Return only the markdown body."""


class AureliusCurator:
    """LLM-backed, retryable-by-replay notes writer. Out of the articulate hot path."""

    def __init__(
        self,
        agent: Agent[IoCContainer],
        notes_path: str | Path,
        *,
        max_source_chars: int,
        max_notes_chars: int,
    ) -> None:
        self._agent = agent
        self._notes_path = Path(notes_path)
        self._max_source_chars = max_source_chars
        self._max_notes_chars = max_notes_chars

    @property
    def notes_path(self) -> Path:
        return self._notes_path

    def _commit_log(self, memory: AureliusMemory) -> str:
        lines: list[str] = []
        for view in memory.branch.all_commits():
            transcript = memory.commit_transcript(view.id, max_chars=self._max_source_chars)
            if not transcript:
                continue
            lines.append(f"## commit {view.id} (seq={view.seq}, kind={view.note.kind()})")
            lines.append(transcript)
        return "\n\n".join(lines)[: self._max_source_chars]

    async def curate(self, memory: AureliusMemory, container: IoCContainer) -> Path | None:
        """Rewrite the notes file from the current frozen trajectory. Returns the path or None."""
        source = self._commit_log(memory)
        if not source:
            return None
        result = await self._agent.run(
            f"Owner: {memory.owner}\n\nObservable commit log:\n{source}",
            deps=container,
            instructions=_INSTRUCTION,
        )
        body = str(result.output).strip()[: self._max_notes_chars]
        if not body:
            return None
        self._notes_path.parent.mkdir(parents=True, exist_ok=True)
        self._notes_path.write_text(
            f"# Aurelius 记忆策展笔记\n\n"
            f"> 由旁路 curation 从冻结 commit 自动重写；证据以 `memory_show <commit>` 为准。\n\n"
            f"{body}\n",
            encoding="utf-8",
        )
        return self._notes_path
