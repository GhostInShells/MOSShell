"""Call anchor — the anchor payload for a single LLMFuncs call.

The key frame of a model call: ``instruction`` + ``turns`` — the message
history (request/response pairs) in pydantic-ai's standard serialization.
A model that reads an anchor file curls ``CallAnchor.ref()`` to learn how the
payload is shaped and how the call is reconstructed — code-as-prompt at the
protocol layer (anchor SPECIFICATION §5).
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from ghoshell_moss.anchor import AnchorModel
from ghoshell_moss.contracts.llms import Effort, ModelRef

__all__ = ["CallAnchor"]

# The URL a model curls to learn this payload's structure. Points at this
# file on main — pin to a commit/tag when long-term fidelity matters
# (SPEC §5: stable-or-versioned is a consumer decision).
_CALL_ANCHOR_REF = (
    "https://github.com/GhostInShells/MOSShell/blob/main/"
    "src/ghoshell_moss/llms/call_anchor.py"
)


class CallAnchor(AnchorModel):
    """Anchor payload of one model call — instruction + turns[request/response].

    ``turns`` is the full message history serialized with pydantic-ai's
    standard protocol (``ModelMessagesTypeAdapter``) — every part survives
    (system/user prompts, thinking, text, tool calls). This is the fidelity
    record a consumer needs to reconstruct the call and re-inject the model's
    own reasoning (introspection) as history. ``model`` is a ``ModelRef``
    (no secrets — api_key/base_url stripped at projection). ``result_type``
    is the ``module:attr`` output-schema pointer. ``result`` is the typed
    output dict — a convenience summary, also present in the turns.
    """

    instruction: str = Field(
        default="",
        description="system prompt — also the first turn's system-prompt part",
    )
    model: ModelRef = Field(
        description="model reference — no secrets",
    )
    result_type: str = Field(
        default="",
        description="module:attr of the output schema — resolve it to learn the result shape",
    )
    effort: Effort | None = Field(
        default=None,
        description="thinking effort (none..max)",
    )
    turns: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "message history in pydantic-ai standard serialization — "
            "request/response parts incl thinking, text, tool calls"
        ),
    )
    result: dict[str, Any] | None = Field(
        default=None,
        description="typed output dict — convenience summary; also present in turns",
    )

    @classmethod
    def ref(cls) -> str:
        return _CALL_ANCHOR_REF
