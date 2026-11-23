"""Agentic class demo — a counter with two blocking ops.

The class source is the op table; each call names one op. Try it from dsh's bash:

    python -c "import asyncio; from DOLORES.agentic_demo import Counter; ..."
"""

from __future__ import annotations

from ghoshell_moss.agents.blueprint.agentic_class import (
    AgentConfig,
    agent_class,
    agent_method,
)
from ghoshell_moss.ghosts.dolores.agentic_class_driver import DoloresAgentClassDriver


@agent_class(AgentConfig(effort="low"), driver=DoloresAgentClassDriver)
class Counter:
    """A counter whose ops are implemented by a module agent."""

    def __init__(self, start: int = 0) -> None:
        self.value = start

    @agent_method
    async def add(self, n: int) -> int:
        """Add n to the counter and return the new value."""
        ...

    @agent_method
    async def get(self) -> int:
        """Return the current value."""
        ...
