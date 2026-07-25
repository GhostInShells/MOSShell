"""
memento_pydantic_agent impl — MementoAgent driven by pydantic-ai + Sandbox.

The runner shape (v1):
- instruction = meta narrative + verbatim source + optional __interfaces__
  appendix, composed on demand by `_assemble_instruction()`
- model's only tool = sandbox_exec (registered by factory)
- pydantic-ai drives the model loop; sandbox holds task state
- memento wiring: deferred to a later step; invoke still accepts the params
  per the ABC so the CLI can pass them
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any

from pydantic_ai import Agent

from ghoshell_moss.agents._instruction import assemble_instruction, prompt_sha
from ghoshell_moss.agents.contract import MementoAgent
from ghoshell_moss.core.codex.sandbox import Sandbox
from ghoshell_moss.memento.abc import Memento

__all__ = ["MementoPydanticAgentImpl"]


class MementoPydanticAgentImpl(MementoAgent):
    """v1 impl. memento recording is deferred to the next step."""

    def __init__(
        self,
        *,
        agent: Agent,
        sandbox: Sandbox,
        compiled_module: ModuleType,
        source: str,
        name: str,
        description: str,
    ):
        self._agent = agent
        self._sandbox = sandbox
        self._compiled = compiled_module
        self._source = source
        self._name = name
        self._description = description

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
        """Run one interaction. Returns the final answer text.

        memento / line_name / cwd accepted per the ABC but not yet consumed
        — recording lands in a follow-up step.
        """
        instruction = self.compose_instruction()
        try:
            result = await self._agent.run(user_prompt, instructions=instruction)
        except Exception as e:
            raise RuntimeError(f"agent {self._name!r} invoke failed: {e}") from e
        return str(result.output)

    def export_context_md(self, memento: Memento, line_name: str) -> str:
        raise NotImplementedError("export_context_md will land in a later step")

    def describe_line(self, memento: Memento, line_name: str) -> str:
        raise NotImplementedError("describe_line will land in a later step")

    # ── introspection surface (used by CLI parse + memento metadata) ──

    def compose_instruction(self) -> str:
        """The exact system text the model will see on the next invoke.

        parse-vs-run parity is the point: `moss memento agent PATH` prints
        the return of THIS function, and `invoke` sends the return of THIS
        function to the model. One truth, two arities.
        """
        return assemble_instruction(
            name=self._name,
            source=self._source,
            module=self._compiled,
        )

    def instruction_sha(self) -> str:
        """SHA-256 (16-hex prefix) of the composed instruction."""
        return prompt_sha(self.compose_instruction())
