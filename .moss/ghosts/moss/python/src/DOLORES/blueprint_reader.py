"""Live acceptance: an agent reviews the English agentic class blueprint."""

from __future__ import annotations

from ghoshell_moss.agents.blueprint import agentic_class as blueprint
from ghoshell_moss.agents.blueprint.agentic_class import (
    AgentConfig,
    agent_class,
    agent_method,
)
from ghoshell_moss.ghosts.dolores.agentic_class_driver import DoloresAgentClassDriver


@agent_class(
    AgentConfig(effort="low"),
    driver=DoloresAgentClassDriver,
    reflect_source=[blueprint],
)
class BlueprintReader:
    """Reads the agentic class blueprint and reports whether it is clear at a glance."""

    @agent_method
    async def review(self, question: str) -> str:
        """Answer whether the reflected blueprint is understandable at a glance; name anything unclear; return a <run> block."""
        ...
