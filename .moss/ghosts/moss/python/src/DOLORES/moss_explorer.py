"""A read-only explorer of the MOSShell project.

Its context is the project-level GROUND.md. `quick_answer` is a no-tools op;
`explore` may use read-only tools; `describe` is a plain method delegating to an op.
"""

from __future__ import annotations

from pathlib import Path

from ghoshell_moss.agents.blueprint.agentic_class import (
    AgentConfig,
    agent_class,
    agent_method,
)
from ghoshell_moss.ghosts.dolores.agentic_class_driver import DoloresAgentClassDriver


@agent_class(AgentConfig(effort="low", readonly=True), driver=DoloresAgentClassDriver)
class MossExplorer:
    """A read-only explorer of the MOSShell project, grounded in GROUND.md."""

    def __init__(self, project_home: str) -> None:
        self.project_home = project_home

    async def __context__(self) -> str:
        """Return the project-level ground for the first instruction."""
        ground = Path(self.project_home) / "GROUND.md"
        try:
            return ground.read_text(encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 — a missing ground is data, not a crash
            return f"# project ground unavailable at {ground}: {exc}"

    async def describe(self) -> str:
        """A plain method (not an op) that delegates to the quick_answer op."""
        return await self.quick_answer("用一句话描述你自己, 以及你现在能做什么。")

    @agent_method
    async def quick_answer(self, question: str) -> str:
        """Answer immediately from the project ground; return a <run> block; one or two sentences."""
        ...

    @agent_method
    async def explore(self, question: str) -> str:
        """Explore the MOSShell project with read-only tools (bash/read/grep), then answer with evidence.
        Try `moss features list`; report whether it can be called and how the call feels.
        """
        ...
