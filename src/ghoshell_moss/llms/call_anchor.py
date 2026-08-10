"""Call anchor — the anchor payload for a single LLMFuncs call.

The key frame of a model call: the request (instruction + prompt + model +
output schema) and, after the call completes, the observed result. A model
that reads an anchor file curls ``CallAnchor.ref()`` to learn how the payload
is shaped and how the call is reconstructed — code-as-prompt at the protocol
layer (anchor SPECIFICATION §5).
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
    """Anchor payload of one model call — request frame + observed result.

    ``model`` is a ``ModelRef`` (no secrets — api_key/base_url stripped at
    projection). ``result_type`` is a ``module:attr`` pointer to the output
    schema — the "tool call json schema" of the call; a model resolves it to
    learn the result shape. ``result`` / ``content`` are the observed output,
    filled after the call; before that they are None and dropped from the
    serialized request frame.
    """

    instruction: str = Field(
        default="",
        description="system prompt sent to the model",
    )
    prompt: str = Field(
        default="",
        description="user message sent to the model",
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
    result: dict[str, Any] | None = Field(
        default=None,
        description="structured output dict — filled after the call",
    )
    content: str | None = Field(
        default=None,
        description="raw text output — filled after the call",
    )

    @classmethod
    def ref(cls) -> str:
        return _CALL_ANCHOR_REF
