"""Pydantic-ai agent anchor — the agent-level cognitive anchor.

Freeze a pydantic-ai agent's cognitive conditions into a self-explaining
file: ``instruction`` + ``tools`` protocol + ``model`` + ``turns`` (one frame
of the cognitive stream). The value is review — past (why the agent decided
as it did) and future (introspection / avatar) — not replay.

Rebuild-path recoverable, not 100% reconstructable: tools are protocol
declarations (name + signature + description, not implementations), model is
keyless (name + thinking, not api_key/base_url). A consumer curls ``ref`` to
learn the payload shape and how to walk the rebuild path back.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from ghoshell_moss.anchor import AnchorModel

__all__ = ["PydanticAIAgentAnchor"]

_ANCHOR_REF = (
    "https://github.com/GhostInShells/MOSShell/blob/main/"
    "src/ghoshell_moss/agents/pydantic_ai_utils/anchor.py"
)


class PydanticAIAgentAnchor(AnchorModel):
    """Anchor payload of one pydantic-ai agent's cognitive frame.

    Four parts:

    - ``instruction`` — composed system text (meta + source + interfaces +
      window, already folded). Pure text, so the anchor does not depend on
      memento's rendering types.
    - ``tools`` — protocol declarations the model can call (name + signature
      + description). Not implementations.
    - ``model_name`` / ``thinking`` — keyless model config. Not a ModelRef
      because memento's factory reads env directly (not LLMConfig); upgrade
      to ModelRef when the dead-config problem is resolved.
    - ``turns`` — message trajectory in pydantic-ai standard serialization
      (thinking/text/tool preserved). One frame of the cognitive stream.
    """

    instruction: str = Field(
        default="",
        description="composed system text — window already folded in",
    )
    tools: list[dict[str, Any]] = Field(
        default_factory=list,
        description="tool protocol declarations (name + signature + description)",
    )
    model_name: str = Field(
        default="",
        description="model name — keyless",
    )
    thinking: bool = Field(
        default=True,
        description="thinking on/off",
    )
    turns: list[dict[str, Any]] = Field(
        default_factory=list,
        description="message trajectory in pydantic-ai standard serialization — one frame of the cognitive stream",
    )

    @classmethod
    def ref(cls) -> str:
        return _ANCHOR_REF
