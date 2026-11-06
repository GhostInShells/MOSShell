"""Attachments — files that travel with a conversation, addressed by their content.

dsh keeps attachments as content-addressed objects under the dsh home
(`attachments/v1/objects/<digest[:2]>/<digest>`), and its wire form for a reference is a single
opaque dict — `{"attachmentId": "sha256:<hex>"}` inside an `image` content block. Opaque is fine
for the wire; it is not fine for a driver, because a caller that wants to look at what was
attached needs a shape it can rely on.

So this is our shape, deliberately small: the content address, and enough to decide what to do
with it. Everything else about an attachment — where it is stored, how it is fetched — belongs
to the implementation.

Why the id is the address and not a serial number: an attachment is its bytes. Two prompts that
attach the same file produce the same id, and a session store that already has it costs nothing
to reference again. A serial id would make that a copy.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["Attachment"]


class Attachment(BaseModel):
    """One file, identified by the content it contains."""

    model_config = ConfigDict(extra="allow")

    attachment_id: str = Field(
        description='content address — `sha256:<hex of the bytes>`. The same bytes, the same id.',
    )
    media_type: str = Field(
        default="",
        description='MIME type when known, e.g. `image/png`. dsh also carries it on the block.',
    )
    name: str | None = Field(default=None, description="the filename it came in with, if any.")
    size: int | None = Field(default=None, description="byte length, when the implementation knows it.")
