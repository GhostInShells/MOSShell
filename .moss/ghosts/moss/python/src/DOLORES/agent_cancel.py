"""A slow probe: one op holds a bash turn so cancellation can be observed live."""

from __future__ import annotations

from ghoshell_moss.agents.blueprint.agentic_class import (
    AgentConfig,
    agent_class,
    agent_method,
)
from ghoshell_moss.ghosts.dolores.agentic_class_driver import DoloresAgentClassDriver


@agent_class(AgentConfig(effort="low"), driver=DoloresAgentClassDriver)
class Slow:
    """A slow probe; its stall op deliberately holds a bash turn so cancellation can be observed."""

    def __init__(self, name: str) -> None:
        self.name = name

    @agent_method
    async def stall(self) -> str:
        """You MUST first run `sleep 45` in bash (a real tool call), then return 'done'. Do not answer before the sleep finishes."""
        ...

    @agent_method
    async def ping(self) -> str:
        """Return 'pong' immediately; return a <run> block."""
        ...
