"""Identifiers this package mints itself.

The request id dsh persists on an accepted user message is client-minted: an opaque string
whose only requirement is uniqueness on our side. It used to come from MOSShell's message
layer, which minted ULIDs — nothing in the dsh protocol depends on that shape.
"""

from __future__ import annotations

import uuid

__all__ = ["new_request_id"]


def new_request_id() -> str:
    """A fresh client-minted request id — 32 hex characters, unique enough, dependency-free."""
    return uuid.uuid4().hex
