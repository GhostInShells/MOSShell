"""Dry-run fixture: explicit imported-source and imported-interface hints."""

from __future__ import annotations

from ghoshell_moss.agents.blueprint.agentic_class import (
    AgentConfig,
    agent_class,
    agent_method,
)
from ghoshell_moss.ghosts.dolores.agentic_class_driver import DoloresAgentClassDriver

from DOLORES.agent_contracts import ExplorerQuestion, QuestionPlan, expand_question


@agent_class(
    AgentConfig(effort="low"),
    driver=DoloresAgentClassDriver,
    reflect_source=[AgentConfig, ExplorerQuestion],  # imported pydantic: source is the API
    reflect_interface=[QuestionPlan, expand_question],  # imported plain class + function
)
class ReflectionDemo:
    """A tiny class for the dry-run reflection debug."""

    @agent_method
    async def answer(self, question: ExplorerQuestion) -> str:
        """Answer one structured question in one sentence."""
        ...
