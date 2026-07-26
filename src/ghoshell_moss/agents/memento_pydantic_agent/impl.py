"""
memento_pydantic_agent impl — MementoAgent driven by pydantic-ai + Sandbox.

The runner shape (v1):
- instruction = meta narrative + verbatim source + optional __interfaces__
  appendix, composed on demand by assemble_instruction()
- model's only tool = sandbox_exec (registered by factory)
- pydantic-ai drives the model loop; sandbox holds task state
- each invoke records new messages to memento staging as a single MomentRecord
"""

from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.agent import AgentRunResult
from ulid import ULID

from ghoshell_moss.agents._instruction import assemble_instruction, prompt_sha
from ghoshell_moss.agents.contract import MementoAgent
from ghoshell_moss.core.codex.sandbox import Sandbox
from ghoshell_moss.memento.abc import Memento, MomentRecord

__all__ = ["MementoPydanticAgentImpl"]

_MESSAGE_TYPE: str = "pydantic_ai.messages/v2"


def _new_moment_id() -> str:
    return f"mmt_{ULID()}"


class MementoPydanticAgentImpl(MementoAgent):
    """v1 impl. Each invoke records one moment to staging."""

    def __init__(
        self,
        *,
        agent: Agent,
        sandbox: Sandbox,
        compiled_module: ModuleType,
        source: str,
        name: str,
        description: str,
    ) -> None:
        self._agent: Agent = agent
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
        instruction: str = self.compose_instruction()
        try:
            result: AgentRunResult[Any] = await self._agent.run(
                user_prompt, instructions=instruction
            )
        except Exception as e:
            raise RuntimeError(f"agent {self._name!r} invoke failed: {e}") from e

        self._record(memento, line_name, result, metadata)

        return str(result.output)

    def export_context_md(self, memento: Memento, line_name: str) -> str:
        raise NotImplementedError("export_context_md will land in a later step")

    def describe_line(self, memento: Memento, line_name: str) -> str:
        raise NotImplementedError("describe_line will land in a later step")

    # ── introspection surface (used by CLI parse + memento metadata) ──

    def compose_instruction(self) -> str:
        """The exact system text the model will see on the next invoke."""
        return assemble_instruction(
            name=self._name,
            source=self._source,
            module=self._compiled,
        )

    def instruction_sha(self) -> str:
        """SHA-256 (16-hex prefix) of the composed instruction."""
        return prompt_sha(self.compose_instruction())

    # ── memento recording ──

    def _record(
        self,
        memento: Memento | None,
        line_name: str,
        result: AgentRunResult[Any],
        metadata: dict[str, Any] | None,
    ) -> None:
        """Dump new messages as a MomentRecord and write to staging."""
        if memento is None or not line_name:
            return

        try:
            line = memento.get_line(line_name)
        except Exception:
            return

        raw_bytes: bytes = result.new_messages_json()
        messages: Any = json.loads(raw_bytes.decode())

        prompt_sha_val: str = ""
        if metadata:
            prompt_sha_val = metadata.get("prompt_sha", "")

        record: MomentRecord = MomentRecord(
            id=_new_moment_id(),
            type=_MESSAGE_TYPE,
            payload={
                "messages": messages,
                "prompt_sha": prompt_sha_val,
            },
        )
        line.record(record)
