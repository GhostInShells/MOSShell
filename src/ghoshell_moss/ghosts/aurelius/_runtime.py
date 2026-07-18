"""Aurelius Ghost runtime: Memento-backed conversation, no in-process history."""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING

from ghoshell_container import IoCContainer
from pydantic_ai import Agent, TextContent
from typing_extensions import Self

from ghoshell_moss.contracts.logger import LoggerItf, get_moss_logger
from ghoshell_moss.core.blueprint.ghost import Ghost, GhostMeta
from ghoshell_moss.core.blueprint.mindflow import Articulator
from ghoshell_moss.core.concepts.channel import Channel
from ghoshell_moss.message import Message

from ._adapter import moment_to_request
from ._channel import new_memento_channel
from ._desktop import AureliusDesktop
from ._knowledge import EvidencePacket, MemoryProjection
from ._memory import AureliusMemory
from ._reflection import AureliusReflector

if TYPE_CHECKING:
    from ._meta import AureliusMeta

__all__ = ["Aurelius"]


class Aurelius(Ghost):
    """Ghost prototype whose conversation history lives only in Memento."""

    def __init__(
        self,
        *,
        meta: "AureliusMeta",
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
        knowledge_enabled: bool,
        knowledge_user_sources: tuple[str, ...],
        knowledge_trusted_tool_sources: tuple[str, ...],
        knowledge_recall_limit: int,
        knowledge_evidence_max_chars: int,
        desktop_enabled: bool,
        desktop_workspace_root: str | Path,
        desktop_root: str | Path,
    ) -> None:
        self._meta = meta
        self._agent = agent
        self._container = container
        self._logger = container.get(LoggerItf) or get_moss_logger()
        self._memory = AureliusMemory(
            memory_root,
            memory_owner,
            detail_n=memory_detail_n,
            summary_m=memory_summary_m,
            auto_commit_every=auto_commit_every,
            index_user_sources=knowledge_user_sources,
        )
        self._reflector = (
            AureliusReflector(
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
        self._knowledge = (
            MemoryProjection(
                self._memory,
                user_sources=knowledge_user_sources,
                trusted_tool_sources=knowledge_trusted_tool_sources,
                recall_limit=knowledge_recall_limit,
                evidence_max_chars=knowledge_evidence_max_chars,
            )
            if knowledge_enabled
            else None
        )
        self._desktop = AureliusDesktop(
            desktop_workspace_root,
            default_root=desktop_root,
            enabled=desktop_enabled,
        )
        self._channel = new_memento_channel(
            self._memory,
            knowledge=self._knowledge,
            desktop=self._desktop,
            on_reflect=self.schedule_reflection,
        )
        self._last_context: dict = {}

    @property
    def meta(self) -> GhostMeta:
        return self._meta

    @property
    def memory(self) -> AureliusMemory:
        return self._memory

    @property
    def knowledge(self) -> MemoryProjection | None:
        return self._knowledge

    @property
    def desktop(self) -> AureliusDesktop:
        return self._desktop

    def system_prompt(self) -> str:
        return self._meta.build_instruction_from_ioc(self._container)

    def memories(self) -> list[Message]:
        return self._memory.messages()

    def channel(self) -> Channel:
        return self._channel

    async def articulate(self, articulator: Articulator) -> AsyncIterator[str]:
        request = moment_to_request(articulator.moment)
        history = self._memory.model_history()
        request_parts = []
        ground_instruction = self._desktop.instruction()
        ground_context = await self._desktop.context()
        if ground_context:
            request_parts.append(TextContent(content=f"<current-ground>\n{ground_context}\n</current-ground>"))

        query = "\n".join(articulator.moment.percepts_texts())
        evidence: EvidencePacket | None = None
        if self._knowledge is not None and self._knowledge.is_recall_question(query):
            try:
                evidence = self._knowledge.recall(query)
            except Exception as error:  # A derived projection must fail closed, not break ordinary Memento.
                self._logger.warning("Aurelius knowledge projection failed: %s", error)
                evidence = EvidencePacket(query=query, status="unknown", reason=f"projection failed: {error}")
            if evidence.status == "ok":
                try:
                    evidence_text = self._knowledge.render_packet(evidence)
                except ValueError as error:
                    evidence = EvidencePacket(
                        query=query,
                        requested_keys=evidence.requested_keys,
                        status="unknown",
                        reason=str(error),
                    )
                else:
                    request_parts.append(TextContent(content=evidence_text))
        request_parts.extend(request.parts)

        self._last_context = {
            "system": self.system_prompt(),
            "history_messages": len(history),
            "memory": self._memory.inspect(),
            "knowledge": self._knowledge_state(),
            "desktop": self._desktop.inspect(),
            "ground_context_chars": len(ground_context),
        }
        if evidence is not None:
            self._last_context["recall"] = evidence.model_dump(mode="json")
            if evidence.status != "ok":
                self._last_context["answer_verification"] = {
                    "accepted": True,
                    "reason": f"safe {evidence.status} response without model generation",
                }
                yield evidence.safe_answer()
                return

        buffered: list[str] | None = [] if evidence is not None else None
        async with self._agent.run_stream(
            user_prompt=request_parts,
            message_history=history,
            instructions=ground_instruction or None,
            deps=self._container,
        ) as stream:
            async for text in stream.stream_text(delta=True):
                if buffered is None:
                    yield text
                else:
                    buffered.append(text)

        if buffered is not None:
            answer = "".join(buffered)
            if self._knowledge is None or evidence is None:  # Defensive invariant; construction decides both.
                raise RuntimeError("buffered memory answer has no projection or evidence packet")
            try:
                verification = self._knowledge.verify(answer, evidence)
            except Exception as error:  # Verification failure must not leak the unverified buffered answer.
                self._logger.warning("Aurelius memory answer verification failed: %s", error)
                self._last_context["answer_verification"] = {
                    "accepted": False,
                    "reason": f"verifier failed: {error}",
                }
                yield "记忆证据校验未通过，暂不回答该事实。"
                return
            self._last_context["answer_verification"] = verification.model_dump(mode="json")
            if verification.accepted:
                yield answer
            else:
                yield "记忆证据校验未通过，暂不回答该事实。"

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
                self._logger.warning("Aurelius memory reflection failed for %s: %s", view.id, message)
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
            "knowledge": self._knowledge_state(),
            "desktop": self._desktop.inspect(),
        }

    def inspect_context(self) -> dict:
        return dict(self._last_context)

    def _knowledge_state(self) -> dict:
        if self._knowledge is None:
            return {"enabled": False}
        try:
            return {"enabled": True, **self._knowledge.inspect()}
        except Exception as error:
            return {"enabled": True, "error": f"{type(error).__name__}: {error}"}

    async def __aenter__(self) -> Self:
        await self._desktop.__aenter__()
        if self._reflection_startup_limit > 0:
            self.schedule_reflection(self._memory.reflection_candidates()[:self._reflection_startup_limit])
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        tasks = list(self._reflection_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        try:
            await self._desktop.__aexit__(exc_type, exc_val, exc_tb)
        finally:
            self._memory.close()
