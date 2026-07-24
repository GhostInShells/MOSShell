"""
memento_pydantic_agent impl — MementoAgent driven by pydantic-ai + Sandbox.

The runner shape (v1 hello world):
- system prompt = sandbox.get_interface() (reflection = compressed prompt)
- model's only tool = sandbox_exec (registered at factory time)
- pydantic-ai drives the model loop; sandbox holds task state
- memento operations: deferred (step F/G merges them in)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic_ai import Agent

from ghoshell_moss.agents.contract import MementoAgent
from ghoshell_moss.core.codex.sandbox import Sandbox
from ghoshell_moss.memento.abc import Memento

__all__ = ["MementoPydanticAgentImpl"]


class MementoPydanticAgentImpl(MementoAgent):
    """v1 hello world impl. memento wiring is deferred to a later step."""

    def __init__(
        self,
        *,
        agent: Agent,
        sandbox: Sandbox,
        name: str,
        description: str,
    ):
        self._agent = agent
        self._sandbox = sandbox
        self._name = name
        self._description = description

    # ── MementoAgent contract ──

    async def invoke(
        self,
        *,
        instruction: str,
        prompt: str = "",
        memento: Memento | None = None,
        line_name: str = "",
        cwd: Path | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Run one interaction. Returns final answer text.

        v1 shape: reflection of the sandbox is the system prompt; the
        `prompt` param is ignored (kept in signature per ABC). memento /
        line_name / cwd accepted but not yet consumed — recording lands
        in the next step.
        """
        system_prompt = self._build_system_prompt()
        try:
            result = await self._agent.run(instruction, instructions=system_prompt)
        except Exception as e:
            raise RuntimeError(f"agent {self._name!r} invoke failed: {e}") from e
        return str(result.output)

    def compact(self, memento: Memento, line_name: str) -> None:
        raise NotImplementedError("compact is deferred beyond v1 scope (see FEATURE §10.2)")

    def export_context_md(self, memento: Memento, line_name: str) -> str:
        raise NotImplementedError("export_context_md will land in a later step")

    def describe_line(self, memento: Memento, line_name: str) -> str:
        raise NotImplementedError("describe_line will land in a later step")

    # ── internal ──

    def _build_system_prompt(self) -> str:
        """Reflect the sandbox to produce the model's system prompt."""
        interface = self._sandbox.get_interface()
        return (
            "You are a MementoAgent running in a Python sandbox. Your capabilities "
            "are declared in the sandbox namespace shown below. To do anything, call "
            "the `sandbox_exec` tool with a Python code string — the sandbox will run "
            "it and return stdout / exceptions / the value assigned to `__result__`.\n"
            "\n"
            "Sandbox state persists across `sandbox_exec` calls: variables you set "
            "remain available in later calls. When you have a final answer, respond "
            "in plain text (not via `sandbox_exec`).\n"
            "\n"
            "## Sandbox Interface\n"
            "\n"
            f"{interface}"
        )
