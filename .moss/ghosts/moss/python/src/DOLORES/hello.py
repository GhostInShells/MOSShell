"""A hand-written capability used by the agent_func acceptance.

The first line is the session title.
"""

from __future__ import annotations

from ghoshell_moss.ghosts.dolores.agent_func import agent_func


async def __context__() -> str:
    """The module context the agent sees when its session is created."""
    return "This module belongs to the agent_func acceptance; keep answers short."


@agent_func
async def say_helloworld() -> str:
    """Return a friendly hello world."""
    raise NotImplementedError


@agent_func
async def greet(name: str, *, excited: bool = False) -> str:
    """Greet the given name; shout when excited is true."""
    raise NotImplementedError


@agent_func(mode="per_call")
async def say_ephemeral() -> str:
    """Return the single word "ephemeral"."""
    raise NotImplementedError


@agent_func
async def leave_a_note() -> str:
    """Return "noted"; before returning, call agent_ctx.emit("CTX-NOTE-1")."""
    raise NotImplementedError
