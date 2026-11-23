"""Evaluate pydantic: reflected source vs generic interface."""

from __future__ import annotations

from ghoshell_moss.agents.blueprint.agentic_class import (
    AgentConfig,
    agent_class,
    agent_method,
)
from ghoshell_moss.ghosts.dolores.agentic_class_driver import DoloresAgentClassDriver

from DOLORES.agent_contracts import ExplorerQuestion


@agent_class(
    AgentConfig(effort="low"),
    driver=DoloresAgentClassDriver,
    reflect_source=[ExplorerQuestion],
)
class PydanticSourceDemo:
    """Answer a structured question; the pydantic contract is reflected as source."""

    @agent_method
    async def answer(self, question: ExplorerQuestion) -> str:
        """Answer the question in one sentence."""
        ...


@agent_class(
    AgentConfig(effort="low"),
    driver=DoloresAgentClassDriver,
    reflect_interface=[ExplorerQuestion],
)
class PydanticInterfaceDemo:
    """Answer a structured question; the pydantic contract is reflected as a generic interface."""

    @agent_method
    async def answer(self, question: ExplorerQuestion) -> str:
        """Answer the question in one sentence."""
        ...
