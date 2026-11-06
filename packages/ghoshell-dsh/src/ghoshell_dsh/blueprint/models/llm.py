"""The model side of a call: what it is allowed to cost, and what it may be asked as.

Two shapes, one idea each.

`LlmConfigPatch` is a **patch, not a configuration**. A caller that wants this one call to think
less, spend less, or answer with a different model should say only that — not restate provider,
model, and effort from scratch. The base is always what the session already holds in its request
header, so the call stays inside the conversation it borrowed; the patch is the delta. `None`
means "leave it alone", which is also why the fields are optional rather than defaulted: a
default would silently override a session setting the caller never looked at.

`LlmProvider` is a **resource**, not a route: what this dsh can actually talk to. A caller
picking a model from thin air is a caller about to be corrected by an error message, and the
correction costs a round trip. Listing first is cheaper.

Both are declared here rather than in a generation because the question they answer — "what does
this call cost, and what can this dsh reach" — does not change between dsh versions; only the
plumbing that answers it does.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["LlmConfigPatch", "LlmProvider"]


class LlmConfigPatch(BaseModel):
    """A delta on the model configuration the session already holds.

    Merge, never replace: an absent field means "keep what the session has". That is the whole
    difference between this and a config object — a config says what everything is, a patch says
    what about it is different *here*.

    `reasoningEffort="off"` is a real instruction (thinking disabled), not an absence, which is
    why absence has to be `None` rather than a sentinel default.
    """

    model_config = ConfigDict(extra="allow")

    provider: str | None = Field(default=None, description="route key; see `LlmProvider.id`.")
    model: str | None = Field(default=None, description="model id within that route.")
    reasoningEffort: str | None = Field(
        default=None, description="thinking depth; `off` disables thinking rather than meaning 'unset'."
    )
    maxTokens: int | None = Field(default=None, description="hard ceiling for this one answer.")
    temperature: float | None = Field(default=None)
    stop: list[str] | None = Field(default=None)

    def applied_to(self, base: dict[str, Any]) -> dict[str, Any]:
        """The base config with this patch's set fields written over it.

        Absent fields are skipped, so a patch never erases a session setting by omission. This is
        the only place the merge is defined — an implementation that merges its own way would
        drift from this one the first time a field is added.
        """
        merged = dict(base)
        for key, value in self.model_dump(exclude_none=True).items():
            merged[key] = value
        return merged


class LlmProvider(BaseModel):
    """One model route this dsh can reach, and whether it is live right now.

    `live` is the difference between "declared" and "usable": an adapter may ship the knowledge
    that a route exists and still have no credentials for it. Both are reported because a caller
    deciding where to send a call needs to tell "wrong name" from "not configured yet".
    """

    model_config = ConfigDict(extra="allow")

    id: str = Field(default="", description="the route key a call names as `provider`.")
    name: str = Field(default="", description="human-readable name, for a picker or a log.")
    live: bool = Field(default=False, description="registered with an adapter right now.")
    declared: bool | None = Field(
        default=None, description="the adapter knows the route but configuration owns it."
    )
    error: str | None = Field(default=None, description="a configuration diagnostic, when one exists.")
