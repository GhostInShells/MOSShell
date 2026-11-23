"""A composition case: Composer holds a Probe; its op calls the probe's op three times."""

from __future__ import annotations

from ghoshell_moss.agents.blueprint.agentic_class import (
    AgentConfig,
    agent_class,
    agent_method,
)
from ghoshell_moss.ghosts.dolores.agentic_class_driver import DoloresAgentClassDriver


@agent_class(AgentConfig(effort="low"), driver=DoloresAgentClassDriver)
class Probe:
    """A one-question probe agent; each `step` is one blocking op."""

    def __init__(self, name: str) -> None:
        self.name = name

    @agent_method
    async def step(self, question: str) -> str:
        """Answer one probe in one or two sentences, from your own judgment; return a <run> block."""
        ...


@agent_class(AgentConfig(effort="low"), driver=DoloresAgentClassDriver)
class Composer:
    """An agent that holds a Probe and composes its three answers into a conclusion."""

    def __init__(self, probe: Probe) -> None:
        self.probe = probe

    @agent_method
    async def investigate(self, question: str) -> str:
        """Call `self.probe.step` exactly three times with three different probes about the question; then return your own conclusion as a <run> block. Do not call it more than three times."""
        ...
