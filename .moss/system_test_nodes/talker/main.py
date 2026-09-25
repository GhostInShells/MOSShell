"""Talker node — a conversational memento agent driven by node harness code.

Start:  moss nodes run .moss/system_test_nodes/talker
Debug:  python main.py

The node holds the harness: it builds the memento store, builds the agent
from talker.agent.py, and exposes a single `talk` command that drives the
agent through one invocation. The agent itself owns nothing — session
(memento), branch (line_name), and working directory (cwd) are all supplied
by this node code.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ghoshell_moss.agents.memento_pydantic_agent import factory
from ghoshell_moss.core.blueprint.channel_builder import new_channel
from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.memento.fs_memento import new_filesystem_memento

logger = logging.getLogger("talker")


async def main(matrix: Matrix):
    node_dir = Path(__file__).parent
    agent_path = node_dir / "talker.agent.py"

    # Harness-owned state: the memento store lives in this cell's persistent
    # home. The agent never constructs or owns it — it is passed in.
    memento_root = matrix.home / "memento"
    memento = new_filesystem_memento(memento_root, "talker")

    # Build the agent once. Its instruction is reflected from the .py source;
    # its model comes from __model__ / ANTHROPIC_MODEL.
    agent = factory(agent_path, cwd=node_dir)

    channel = new_channel(
        name="talker",
        description="A conversational agent. Each talk() is one agent invocation "
        "recorded into a memento line for continuity across turns.",
    )

    @channel.build.command(always_observe=True)
    async def talk(text__: str) -> str:
        """Have a conversation turn with the talker agent.  text__: your message.

        Returns the agent's reply. Past turns persist via the memento line,
        so later turns see earlier ones through the window.
        """
        return await agent.invoke(
            user_prompt=text__,
            memento=memento,
            line_name="main",
            cwd=node_dir,
        )

    await matrix.provide_channel(channel)


if __name__ == "__main__":
    Matrix.discover().run(main)
