"""
memento_pydantic_agent impl — MementoAgent driven by pydantic-ai + Sandbox.

The runner shape (v1):
- reading side: window rendered from memento line (summaries + detail frames),
  injected into the instruction so the model sees its own past
- instruction = meta narrative + verbatim source + optional __interfaces__
  appendix + window (when memento exists) + memory-truth preamble
- model's only tool = sandbox_exec (registered by factory)
- pydantic-ai drives the model loop; sandbox holds task state
- each invoke records new messages to memento staging as a single MomentRecord
  with the final answer as the content field
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import time
from pathlib import Path
from types import ModuleType
from typing import Any

from ghoshell_moss.depends import depend_ghost

depend_ghost()
from pydantic_ai import Agent
from pydantic_ai.agent import AgentRunResult
from ulid import ULID

from ghoshell_moss.agents._instruction import assemble_instruction
from ghoshell_moss.agents.contract import InvocationRecord, MementoAgent
from ghoshell_moss.agents.pydantic_ai_utils import serialize_messages
from ghoshell_moss.core.codex.sandbox import Sandbox
from ghoshell_moss.memento.abc import BranchWindow, Memento, MomentRecord

__all__ = ["MementoPydanticAgentImpl"]

logger = logging.getLogger("moss.memento_agent")

_MESSAGE_TYPE: str = "pydantic_ai.messages/v2"
_DEFAULT_DETAIL_N: int = 10
_DEFAULT_SUMMARY_M: int = -1
_DETAIL_MAX_CHARS: int = 500


def _new_moment_id() -> str:
    return f"mmt_{ULID()}"


class MementoPydanticAgentImpl(MementoAgent):
    """v1 impl. Each invoke records one moment to staging.

    The window is rendered from memento on every instruction composition —
    the model always sees its latest history before composing a response.
    """

    def __init__(
        self,
        *,
        agent: Agent,
        dry_run_agent: Agent,
        sandbox: Sandbox,
        compiled_module: ModuleType,
        source: str,
        name: str,
        description: str,
    ) -> None:
        self._agent: Agent = agent
        self._dry_run_agent: Agent = dry_run_agent
        self._sandbox: Sandbox = sandbox
        self._compiled: ModuleType = compiled_module
        self._source: str = source
        self._name: str = name
        self._description: str = description

    # ── MementoAgent contract ──

    async def invoke(
        self,
        *,
        user_prompt: str,
        memento: Memento | None = None,
        line_name: str = "",
        cwd: Path | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Run one interaction. Records new messages to memento staging."""
        instruction: str = self.compose_instruction(memento, line_name)
        logger.info(
            "invoke start: agent=%s line=%s memento=%s",
            self._name, line_name or "(none)", "yes" if memento else "no",
        )
        try:
            result: AgentRunResult[Any] = await self._agent.run(
                user_prompt, instructions=instruction
            )
        except Exception as e:
            logger.exception("invoke failed: agent=%s", self._name)
            raise RuntimeError(f"agent {self._name!r} invoke failed: {e}") from e

        # Degraded baseline, explicit at the invoke layer: record only when a
        # store exists AND a line is bound. memento=None or empty line_name =
        # pure in-memory single round, no storage write.
        # None output (run ended without a final text) collapses to "" — the
        # CLI treats empty stdout as failure (exit 1), keeping the invoke
        # protocol honest: exit 0 guarantees a non-empty final answer.
        output: str = str(result.output or "")
        if memento is not None and line_name:
            self._record(memento, line_name, result, output)

        logger.info("invoke done: agent=%s output=%d chars", self._name, len(output))
        return output

    async def dry_run(
        self,
        *,
        user_prompt: str,
        memento: Memento | None = None,
        line_name: str = "",
        cwd: Path | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> InvocationRecord:
        """Pure probe — real generation, paused at the tool call, no side effects."""
        from pydantic_ai.tools import DeferredToolRequests

        instruction: str = self.compose_instruction(memento, line_name)
        logger.info(
            "dry run start: agent=%s line=%s memento=%s",
            self._name, line_name or "(none)", "yes" if memento else "no",
        )
        start = time.perf_counter()
        try:
            result: AgentRunResult[Any] = await self._dry_run_agent.run(
                user_prompt, instructions=instruction
            )
        except Exception as e:
            logger.exception("dry run failed: agent=%s", self._name)
            raise RuntimeError(f"agent {self._name!r} dry run failed: {e}") from e
        elapsed = time.perf_counter() - start

        deferred = result.output if isinstance(result.output, DeferredToolRequests) else None
        tool_calls: list[dict[str, Any]] = []
        if deferred is not None:
            for call in deferred.approvals:
                tool_calls.append({"tool_name": call.tool_name, "args": call.args})

        output = "" if deferred is not None else str(result.output or "")

        logger.info(
            "dry run done: agent=%s tool_calls=%d output=%d chars",
            self._name, len(tool_calls), len(output),
        )
        return InvocationRecord(
            output=output,
            content=_extract_text(result),
            usage=dataclasses.asdict(result.usage) if result.usage else {},
            cast=elapsed,
            tool_calls=tool_calls,
            messages=serialize_messages(result.all_messages()),
        )

    def export_context_md(self, memento: Memento, line_name: str) -> str:
        """Current composed instruction including the window — the exact
        system text the model sees, formatted as markdown."""
        return self.compose_instruction(memento, line_name)

    def describe_line(self, memento: Memento, line_name: str) -> str:
        """Agent-perspective line summary. Returns the first commit summary
        in the window, or a placeholder when the line is empty."""
        try:
            window = _try_window(memento, line_name)
        except Exception:
            return f"{line_name} (unavailable)"
        if window is None:
            return f"{line_name} (empty — no commits yet)"
        if window.summaries:
            return window.summaries[0].summary() or f"{line_name} (untitled)"
        if window.details:
            preview = window.details[0].content
            if preview:
                return preview[:120].replace("\n", " ")
        return f"{line_name} (empty — no commits yet)"

    # ── instruction assembly (invoke-time) ──

    def compose_instruction(
        self, memento: Memento | None = None, line_name: str = ""
    ) -> str:
        """The exact system text the model will see on the next invoke.

        When a memento store AND line are present, the window is rendered and
        injected together with the memory-truth preamble. When either is
        missing (degraded baseline), the instruction is pure — source +
        interfaces, no memory mention.
        """
        window_text: str | None = None
        if memento is not None and line_name:
            window_text = self._render_window(memento, line_name)
        return assemble_instruction(
            name=self._name,
            source=self._source,
            module=self._compiled,
            window_text=window_text,
        )

    def instruction_sha(self) -> str:
        """Short sha256 fingerprint of the degraded instruction (source +
        interfaces, no window). Used by `moss memento agent parse` for the
        parse-vs-run parity guarantee — a debug tool, not trajectory data.
        """
        text = self.compose_instruction()  # degraded: no memento
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    # ── window rendering (read-side, (b) phase) ──

    @staticmethod
    def _render_window(memento: Memento, line_name: str) -> str | None:
        """Render the sliding window as model-readable text.

        Falls back to None when the line cannot be read; the caller treats
        None as "no window to show".
        """
        window = _try_window(memento, line_name)
        if window is None:
            return None
        lines: list[str] = []

        if window.summaries:
            lines.append("[History — past checkpoints]")
            for i, cv in enumerate(window.summaries):
                summary = cv.summary() or "(untitled)"
                lines.append(f"  [{i}] {summary}")
        elif not window.details:
            return None  # nothing to remember

        if window.details:
            lines.append("")
            lines.append("[Recent frames]")
            for moment in window.details:
                content = moment.content.strip() if moment.content else ""
                if not content:
                    continue
                if len(content) > _DETAIL_MAX_CHARS:
                    content = content[:_DETAIL_MAX_CHARS] + "..."
                lines.append(f"  [{moment.type}] {content}")

        return "\n".join(lines).strip()

    # ── memento recording ──

    def _record(
        self,
        memento: Memento,
        line_name: str,
        result: AgentRunResult[Any],
        output: str,
    ) -> None:
        """Dump new messages as a MomentRecord and write to staging.

        Caller (invoke) guards the degraded baseline; this is only reached
        when a store exists and a line is bound. Any failure here is a real
        trajectory loss — raise loudly, never swallow.
        """
        try:
            line = memento.get_line(line_name)
            raw_bytes: bytes = result.new_messages_json()
            messages: Any = json.loads(raw_bytes.decode())

            record: MomentRecord = MomentRecord(
                id=_new_moment_id(),
                type=_MESSAGE_TYPE,
                content=output,
                payload={
                    "messages": messages,
                },
            )
            line.record(record)
            logger.info(
                "recorded moment %s to %s/%s", record.id, self._name, line_name
            )
        except Exception as e:
            logger.exception("moment record failed: agent=%s line=%s", self._name, line_name)
            raise RuntimeError(
                f"agent {self._name!r}: failed to record moment to line "
                f"{line_name!r}: {e}"
            ) from e


# ── window helpers ────────────────────────────────────────────────────────────


def _try_window(memento: Memento, line_name: str) -> BranchWindow | None:
    """Read the window from a line, returning None when the line cannot be
    resolved — a soft read that degrades gracefully."""
    try:
        line = memento.get_line(line_name)
        return line.window(
            detail_n=_DEFAULT_DETAIL_N,
            summary_m=_DEFAULT_SUMMARY_M,
        )
    except Exception:
        return None


def _extract_text(result: AgentRunResult[Any]) -> str:
    """Concatenate all TextParts across the message history."""
    from pydantic_ai.messages import TextPart

    parts: list[str] = []
    for msg in result.all_messages():
        for part in msg.parts:
            if isinstance(part, TextPart):
                parts.append(part.content)
    return "".join(parts)
