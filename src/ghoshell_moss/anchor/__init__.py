"""Cognitive anchor protocol — freeze model-call key frames as self-explaining files.

The protocol defines a file format and a data structure, nothing more. The
payload structure is the job of ``ref``, the protocol's single key
proposition: an HTTP URL a model curls to reconstruct the call.

See ``SPECIFICATION.md`` for the full spec.
"""

from .contract import Anchor, AnchorMeta, AnchorModel

__all__ = [
    "Anchor",
    "AnchorMeta",
    "AnchorModel",
]
