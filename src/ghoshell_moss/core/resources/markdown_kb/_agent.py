"""
Recall agent for resources — uses pydantic-ai to match user queries against
a flat context of resource metas.  Gated by ANTHROPIC_SMALL_FAST_MODEL env var.

AnthropicModel auto-reads ANTHROPIC_BASE_URL / ANTHROPIC_API_KEY from env.

Usage:
    if not recall_available():
        fallback to keyword matching
    result = await recall(kb, "how do i create a new how-to doc?")
    # result.locators → ["markdown-kb://moss-howto/how-to-make-how-to.md"]
"""

import os

from ghoshell_moss.contracts.resource import Recollection

__all__ = ["recall_available", "recall"]


def recall_available() -> bool:
    """Check whether ANTHROPIC_SMALL_FAST_MODEL is set (gate)."""
    return bool(os.environ.get("ANTHROPIC_SMALL_FAST_MODEL") or os.environ.get("ANTHROPIC_MODEL"))


async def recall(kb, question: str) -> Recollection:
    """Query a resource storage (kb) with a natural-language question.

    Internally:
        1. dump all metas via list_metas(limit=-1)
        2. build a flat context string
        3. create a pydantic-ai Agent with result_type=QueryResult
        4. run the agent, return the structured result
    """
    from ghoshell_moss.depends import depend_pydantic_ai

    depend_pydantic_ai()

    from pydantic_ai import Agent
    from pydantic_ai.models.anthropic import AnthropicModel

    metas = await kb.list_metas(limit=-1)

    # Build context: flattened lines, each is a resource candidate
    context_lines = []
    for m in metas:
        context_lines.append(f"  {m.locator}: {m.description}")
    context = "\n".join(context_lines)

    # System prompt describing the task
    system_prompt = (
        "You are a recall agent. Match the user question against "
        "the available resource descriptions below. "
        "Return the most relevant locators with a brief reasoning.\n\n"
        "Available resources:\n"
        f"{context}"
    )

    model_name = os.environ.get("ANTHROPIC_SMALL_FAST_MODEL") or os.environ.get("ANTHROPIC_MODEL")
    if not model_name:
        raise RuntimeError(f"depending on env var ANTHROPIC_SMALL_FAST_MODEL or ANTHROPIC_MODEL")
    model = AnthropicModel(model_name=model_name)
    agent = Agent(
        model=model,
        system_prompt=system_prompt,
    )

    result = await agent.run(question, output_type=Recollection)
    return result.output
