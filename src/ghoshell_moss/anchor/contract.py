"""Anchor contract — data structures for the cognitive anchor protocol.

One-line promise: freeze the complete cognitive conditions of a model call
into a self-explaining file; a consumer curls ``ref`` to reconstruct the call.

Structure:
1. AnchorMeta — meta info. Minimal top-level fields: uid / name / description / ref / created / metadata
2. Anchor     — meta + payload. payload structure is defined by ``meta.ref``, not by this protocol
3. AnchorModel — self-explaining payload carrier (code-as-prompt, mirrors TopicModel).
   A typed subclass declares its own ``ref`` and converts to/from the weak
   Anchor container via to_anchor/from_anchor

The protocol's single key proposition: ``ref`` points to an HTTP URL; a model
curls it to reconstruct the payload.

SPEC: ``ghoshell_moss.anchor.SPECIFICATION.md``.
"""

from __future__ import annotations

import yaml
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from ulid import ULID
from typing_extensions import Self

__all__ = [
    "Anchor",
    "AnchorMeta",
    "AnchorModel",
]


def _new_uid() -> str:
    return str(ULID())


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class AnchorMeta(BaseModel):
    """Anchor meta info. Top-level fields are minimal — nothing beyond need.

    ``uid`` is the primary key (ULID), stored in meta, not the filename — a
    filename full of ids has governance cost (half of an ``ls`` becomes
    opaque). The filename uses the human-readable ``name``; collisions
    resolve by ``uid``. ``ref`` is the protocol's single key proposition: an
    HTTP URL a model curls to reconstruct the payload structure and the call.
    """

    uid: str = Field(
        default_factory=_new_uid,
        description="primary key (ULID). In meta, not filename.",
    )
    name: str = Field(
        description="human-readable name. Filename stem in file storage.",
    )
    description: str = Field(
        default="",
        description="one-line note — what the anchor is, or the cognitive scene.",
    )
    ref: str = Field(
        description="HTTP URL defining the payload structure. A model curls it to reconstruct the call.",
    )
    created: datetime = Field(
        default_factory=_utc_now,
        description="ISO 8601 timestamp.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="escape hatch — free-form, uninterpreted. e.g. model_generation / anchor_type / labels.",
    )


class Anchor(BaseModel):
    """A cognitive snapshot of a model call — two parts: meta + payload.

    ``meta`` is the protocol layer, readable by every consumer. ``payload``
    is protocol-native data whose structure is defined by ``meta.ref``, not
    by this protocol.
    """

    meta: AnchorMeta
    payload: Any = Field(
        default=None,
        description="protocol-native request data. Structure defined by meta.ref.",
    )

    def dump_to_dir(self, dir: Path, name: str, *, suffix: str = ".anchor.yml") -> Path:
        """Serialize the anchor to a directory.

        Code-as-prompt self-explaining sample, not a strong constraint: a
        consumer may store anchors however it wishes, as long as the on-disk
        format matches SPEC §3/§4.

        YAML with a ``---`` separator: section 1 = meta (top-level keys),
        section 2 = payload. Filename = ``name + suffix``, glob-friendly via
        ``**/*.anchor.yml``.
        """
        dir = Path(dir)
        dir.mkdir(parents=True, exist_ok=True)
        path = dir / f"{name}{suffix}"

        meta = self.meta.model_dump(exclude_none=True)
        created = meta.get("created")
        if isinstance(created, datetime):
            meta["created"] = created.isoformat()
        section1 = yaml.safe_dump(
            meta, allow_unicode=True, sort_keys=False, default_flow_style=False,
        )
        section2 = yaml.safe_dump(
            self.payload, allow_unicode=True, sort_keys=False, default_flow_style=False,
        )
        path.write_text(section1 + "---\n" + section2, encoding="utf-8")
        return path


class AnchorModel(BaseModel, ABC):
    """Self-explaining anchor payload convention — code-as-prompt.

    Mirrors the TopicModel pattern: a typed carrier that carries its own
    meta, declares its ``ref``, and converts to/from the weak Anchor
    container via to_anchor/from_anchor. A model that sees an anchor curls
    the ``ref`` URL (usually this subclass's file) to reconstruct the payload
    structure.

    Subclasses declare only their payload fields and ``ref()``; the
    conversion logic is inherited.
    """

    meta: AnchorMeta = Field(
        default_factory=lambda: AnchorMeta(name="", ref=""),
        description="meta information. ref is overwritten by ref() in to_anchor.",
    )

    @classmethod
    @abstractmethod
    def ref(cls) -> str:
        """HTTP URL pointing to this data structure's definition.

        A model curls it to learn how the payload is shaped and how the call
        is reconstructed. See SPEC §5.
        """
        pass

    def to_anchor(self, *, name: str = "", description: str = "") -> Anchor:
        """Typed → weak Anchor. Fills meta; payload is the field dict."""
        meta = self.meta.model_copy()
        if name:
            meta.name = name
        if description:
            meta.description = description
        meta.ref = self.ref()
        return Anchor(
            meta=meta,
            payload=self.model_dump(exclude_none=True, exclude={"meta"}),
        )

    @classmethod
    def from_anchor(cls, anchor: Anchor) -> Self:
        """Weak Anchor → typed. Rebuilds from the payload dict."""
        return cls.model_validate(anchor.payload)
