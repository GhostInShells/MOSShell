"""Aurelius Ghost runtime: Memento-backed conversation, no in-process history."""

import asyncio
import contextlib
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
from ._budget import estimate_history_tokens, is_context_overflow
from ._channel import new_memento_channel
from ._curation import AureliusCurator
from ._desktop import AureliusDesktop
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
        curation_agent: Agent[IoCContainer] | None,
        curation_notes_path: str | Path,
        curation_max_source_chars: int,
        curation_max_notes_chars: int,
        curation_enabled: bool,
        curation_index_sources: tuple[str, ...],
        desktop_enabled: bool,
        desktop_workspace_root: str | Path,
        desktop_root: str | Path,
        context_budget_enabled: bool = True,
        context_input_budget: int = 0,
        context_min_detail_n: int = 2,
        context_fixed_overhead_tokens: int = 2048,
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
            index_user_sources=curation_index_sources,
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
        self._curator = (
            AureliusCurator(
                curation_agent,
                curation_notes_path,
                max_source_chars=curation_max_source_chars,
                max_notes_chars=curation_max_notes_chars,
            )
            if curation_enabled and curation_agent is not None
            else None
        )
        self._curation_task: asyncio.Task | None = None
        self._curation_errors: list[str] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._desktop = AureliusDesktop(
            desktop_workspace_root,
            default_root=desktop_root,
            enabled=desktop_enabled,
        )
        self._channel = new_memento_channel(
            self._memory,
            desktop=self._desktop,
            on_reflect=self.schedule_reflection,
            on_curate=self.schedule_curation,
        )
        # Input-token budget: context_window - max_output_tokens - margin, resolved by
        # the meta from the model contract. 0 disables budgeting (injected test models).
        self._context_budget_enabled = context_budget_enabled and context_input_budget > 0
        self._context_input_budget = context_input_budget
        self._context_min_detail_n = context_min_detail_n
        self._context_fixed_overhead = context_fixed_overhead_tokens
        self._last_context: dict = {}

    @property
    def meta(self) -> GhostMeta:
        return self._meta

    @property
    def memory(self) -> AureliusMemory:
        return self._memory

    @property
    def desktop(self) -> AureliusDesktop:
        return self._desktop

    def system_prompt(self) -> str:
        return self._meta.build_instruction_from_ioc(self._container)

    def memories(self) -> list[Message]:
        return self._memory.messages()

    def channel(self) -> Channel:
        return self._channel

    def _budgeted_history(self) -> tuple[list, dict]:
        """Render history, shrinking the window until it fits the input-token budget.

        Frame-count windows (detail_n/summary_m) say nothing about tokens: unbounded
        summaries and multimodal frames can blow the context window on their own.
        Shrink order — detail first (full frames fold into already-present summaries,
        biggest token win per step), then the oldest summaries (their originals stay
        on disk, `memory_show` pages them back). Nothing is destroyed: this narrows
        what is *rendered into context*, exactly the Memento fold-don't-drop stance.
        """
        detail_n = self._memory.detail_n
        summary_m = self._memory.summary_m
        history = self._memory.model_history()
        report = {
            "budget": self._context_input_budget,
            "detail_n": detail_n,
            "summary_m": summary_m,
            "shrunk": False,
        }
        if not self._context_budget_enabled:
            report["estimated_tokens"] = None
            return history, report

        budget = max(self._context_input_budget - self._context_fixed_overhead, 1)
        estimated = estimate_history_tokens(history)
        while estimated > budget:
            if detail_n > self._context_min_detail_n:
                detail_n = max(self._context_min_detail_n, detail_n // 2)
            elif summary_m == -1:
                summary_m = 32  # unbounded summaries are the usual long-run offender
            elif summary_m > 0:
                summary_m //= 2  # reaches 0 = no summaries; originals stay addressable
            else:
                break  # floor reached; the overflow-retry fallback is the last resort
            history = self._memory.model_history(detail_n=detail_n, summary_m=summary_m)
            estimated = estimate_history_tokens(history)
            report["shrunk"] = True
        report.update(detail_n=detail_n, summary_m=summary_m, estimated_tokens=estimated)
        return history, report

    async def articulate(self, articulator: Articulator) -> AsyncIterator[str]:
        request = moment_to_request(articulator.moment)
        request_parts = []
        ground_instruction = self._desktop.instruction()
        ground_context = await self._desktop.context()
        if ground_context:
            request_parts.append(TextContent(content=f"<current-ground>\n{ground_context}\n</current-ground>"))
        request_parts.extend(request.parts)

        history, budget_report = self._budgeted_history()
        self._last_context = {
            "system": self.system_prompt(),
            "history_messages": len(history),
            "memory": self._memory.inspect(),
            "curation": self._curation_state(),
            "desktop": self._desktop.inspect(),
            "ground_context_chars": len(ground_context),
            "context_budget": budget_report,
        }

        # Overflow fallback: if the provider still rejects the input (estimation is
        # heuristic), halve the window and retry — but only while nothing has been
        # yielded yet. Once tokens streamed we cannot un-say them; and an attention
        # abort is never overflow-worded, so it propagates to the driver untouched.
        detail_n = budget_report["detail_n"]
        summary_m = budget_report["summary_m"]
        attempts_left = 3
        while True:
            started_streaming = False
            try:
                async for text in self._stream_once(request_parts, history, ground_instruction):
                    started_streaming = True
                    yield text
                return
            except Exception as error:
                if (
                    started_streaming
                    or attempts_left <= 0
                    or not is_context_overflow(error)
                ):
                    raise
                attempts_left -= 1
                detail_n = max(self._context_min_detail_n, detail_n // 2)
                summary_m = 0 if summary_m == -1 else summary_m // 2
                history = self._memory.model_history(detail_n=detail_n, summary_m=summary_m)
                self._last_context["context_budget"] = {
                    **self._last_context.get("context_budget", {}),
                    "overflow_retry": True,
                    "detail_n": detail_n,
                    "summary_m": summary_m,
                    "estimated_tokens": estimate_history_tokens(history),
                }
                self._logger.warning(
                    "Aurelius input overflowed the model context; retrying with "
                    "detail_n=%d summary_m=%d: %s",
                    detail_n,
                    summary_m,
                    error,
                )

    async def _stream_once(self, request_parts, history, ground_instruction) -> AsyncIterator[str]:
        # An abort during streaming (new input preempts this frame) raises out of the
        # `yield` below and unwinds through run_stream's __aexit__. That teardown re-enters
        # the still-running SSE generator and raises "asynchronous generator is already
        # running", which escapes to the event loop as an unhandled exception. Draining the
        # text iterator inside its own aclosing() lets us close it deterministically before
        # the context manager tears down the underlying stream; the abort itself still
        # propagates and is handled by the articulate driver.
        async with self._agent.run_stream(
            user_prompt=request_parts,
            message_history=history,
            instructions=ground_instruction or None,
            deps=self._container,
        ) as stream:
            text_stream = stream.stream_text(delta=True)
            try:
                async for text in text_stream:
                    yield text
            finally:
                with contextlib.suppress(Exception):
                    await text_stream.aclose()

    def on_articulate_exit(
        self,
        articulator: Articulator,
        logos: str,
        error: Exception | None,
    ) -> None:
        # A failed frame is still a trajectory event: the Ghost saw the percept and
        # tried to answer. Dropping it would make the next incarnation blind to what
        # the user said. Record it, tagged "failed" so it reads as an attempt, not a
        # completed turn — we do not dress failure up as a finished memory.
        threads = ("failed",) if error is not None else ()
        commit = self._memory.remember(articulator.moment, threads=threads)
        if error is not None:
            self._last_context["memory_write"] = "committed_failed" if commit else "staged_failed"
        else:
            self._last_context["memory_write"] = "committed" if commit else "staged"
        if commit:
            self._last_context["commit_id"] = commit.id
            self.schedule_reflection([commit])

    def _spawn(self, coro) -> None:
        """Create a background task on the ghost loop, safe from CTML worker threads.

        CTML commands run under ``asyncio.to_thread`` where ``create_task`` has no running
        loop; ``memory_reflect``/``memory_curate`` would otherwise raise. Route through the
        loop captured at ``__aenter__``.
        """
        loop = self._loop
        if loop is None:
            coro.close()
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            self._track(loop.create_task(coro))
        else:
            loop.call_soon_threadsafe(lambda: self._track(loop.create_task(coro)))

    def _track(self, task: asyncio.Task) -> None:
        self._reflection_tasks.add(task)
        task.add_done_callback(self._reflection_tasks.discard)

    def schedule_reflection(self, commits=None) -> None:
        """Schedule reflection without delaying the completed articulate cycle."""
        if self._reflector is None:
            return
        selected = list(commits) if commits is not None else self._memory.reflection_candidates()
        selected = [view for view in selected if view.id not in self._reflection_inflight]
        if not selected:
            return
        self._reflection_inflight.update(view.id for view in selected)
        self._spawn(self._reflect_all(selected))

    def schedule_curation(self) -> None:
        """Rewrite the pinned notes file from the current trajectory, off the hot path."""
        if self._curator is None:
            return
        if self._curation_task is not None and not self._curation_task.done():
            return
        self._spawn(self._curate_once())

    async def _curate_once(self) -> None:
        assert self._curator is not None
        self._curation_task = asyncio.current_task()
        try:
            notes_path = await self._curator.curate(self._memory, self._container)
            if notes_path is not None:
                self._last_context["curation_notes"] = str(notes_path)
                await self._pin_curation_notes(notes_path)
        except asyncio.CancelledError:
            raise
        except Exception as error:  # Curation is a non-fatal side path; raw trajectory still recalls.
            message = f"{type(error).__name__}: {error}"
            self._curation_errors.append(message)
            self._logger.warning("Aurelius memory curation failed: %s", message)

    async def _pin_curation_notes(self, notes_path) -> None:
        """Pin facts.md into the primary Ground so the main model sees it each frame."""
        ground = self._desktop.primary
        if ground is None:
            return
        try:
            addr = str(notes_path.resolve().relative_to(ground.root))
        except ValueError:
            return  # Notes live outside the Ground root; skip rather than escape the boundary.
        try:
            self._desktop.pin(ground.label, addr, note="memory curation notes")
        except Exception as error:  # Pinning is best-effort; memory_search remains the fallback.
            self._logger.warning("Aurelius could not pin curation notes: %s", error)

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
            "curation": self._curation_state(),
            "desktop": self._desktop.inspect(),
        }

    def inspect_context(self) -> dict:
        return dict(self._last_context)

    def _curation_state(self) -> dict:
        if self._curator is None:
            return {"enabled": False}
        return {
            "enabled": True,
            "notes_path": str(self._curator.notes_path),
            "notes_exists": self._curator.notes_path.exists(),
            "running": self._curation_task is not None and not self._curation_task.done(),
            "errors": list(self._curation_errors[-3:]),
        }

    async def __aenter__(self) -> Self:
        self._loop = asyncio.get_running_loop()
        await self._desktop.__aenter__()
        if self._reflection_startup_limit > 0:
            self.schedule_reflection(self._memory.reflection_candidates()[:self._reflection_startup_limit])
        self.schedule_curation()
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
