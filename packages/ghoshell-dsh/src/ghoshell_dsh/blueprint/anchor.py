"""Where a dsh session *was* — a coordinate, written down, that can be read back anywhere.

An anchor is not a snapshot of the conversation; it is the address of a point in one. Reading
what happened there is the consumer's job (cat the file, or ask the session). That is what
makes an anchor small, portable, and safe to keep: it costs a few lines, it never goes stale
the way a copied transcript does, and it says enough about itself that a reader who has never
seen the session can tell what it refers to.

Four rules this type exists to satisfy:

1. **It exports itself.** `dump()` writes a file; `from_file()` reads one back. No side table,
   no id registry, no "you must have the session object to interpret this".
2. **It is easy to parse.** One YAML document, flat keys, no nesting, no opaque blob. A human
   with an editor and a model with a file reader both get the same thing out of it.
3. **What you store and what you load are the same type.** Not a DTO, not a file format plus a
   converter: `DshSessionAnchor` on both sides, so nothing has to be re-assembled.
4. **It is honest about what it does not know.** Every session-identifying field beyond
   `session_id` and the coordinate is optional and explicitly "for the reader", not for
   reconstruction.

Naming: coordinates are `seq` first (`turn` is derived from it and only carried for humans) —
`seq` is what a log position is, and a turn is a range of them.

### The file it writes

Flat, one YAML document, no nesting — the format is meant to be read by an eye and by a
parser, and a nested blob is only friendly to the second:

```yaml
session_id: s-abc
seq: 42                     # omitted when the anchor points at the session's start
turn: 7
workspace_id: ws-1
workspace_path: /repo       # host-side realpath, the same one a workspace row carries
model: deepseek-chat
reasoning_effort: high
agent_preset: web
cwd: /repo/src
title: refactor the launcher contract
note: design settled — launcher hands back a connection, it does not inherit one
created: '2026-11-06T09:26:23.459437+00:00'
```

Every key except `session_id` and `created` is omittable: an anchor is still valid when it knows
only the coordinate. Extra keys are preserved on load, so a file written by a newer version is
not rejected by an older reader.

### Why the richer fields

An anchor is knowledge, not a pointer. `model` and `reasoning_effort` say what was thinking;
`workspace_path` says where; `note` says what was decided. If the note were left out, an anchor
would only locate a moment — and its whole use is coming back later and reviewing what the code
did against what was decided here.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from .models.refs import DshSessionRef

__all__ = ["ANCHOR_SUFFIX", "DshSessionAnchor"]

#: File suffix for a dumped anchor — glob-friendly, and it says what is inside.
ANCHOR_SUFFIX = ".anchor.yml"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class DshSessionAnchor(BaseModel):
    """A point in one dsh session, in a form that survives leaving the process.

    Two halves, and the difference matters:

    - **the coordinate** — `session_id` and, for the position, `seq` or `turn`: enough to go back
      and take up the thread. `seq` is exact and wins; `turn` is the fallback for a caller that
      only ever recorded the turn (a ghost's commit ref carries turns, and its seq may be empty).
    - **the self-description** — `agent_preset`, `cwd`, `title`, `created`: enough for a reader to
      know what this point is about without connecting to anything. None of it is used to
      reconstruct the conversation, so none of it has to be trusted.

    The anchor is deliberately *not* self-sufficient for reading. A transcript copied into a
    file is a snapshot that rots; an anchor stays true by pointing at the thing that keeps
    being true.
    """

    model_config = ConfigDict(extra="allow")

    # ---- the coordinate ---- #
    session_id: str = Field(description="which session this point belongs to.")
    seq: int | None = Field(
        default=None,
        description="log position — the point itself. None means 'wherever the session starts'.",
    )
    turn: int | None = Field(
        default=None,
        description=(
            "the turn whose end this point is, when no seq was recorded. Read as a coordinate, "
            "not the seq's decoded turn: whichever the caller has, a reader can cut at it."
        ),
    )

    # ---- for the reader, not for reconstruction ---- #
    workspace_id: str | None = Field(default=None, description="dsh workspace this session belongs to.")
    workspace_path: str | None = Field(default=None, description="the workspace's host-side path.")
    model: str | None = Field(default=None, description="the model answering at this point.")
    reasoning_effort: str | None = Field(default=None, description="its thinking depth, as configured.")
    agent_preset: str | None = Field(default=None, description="which preset shaped the agent.")
    cwd: str | None = Field(default=None, description="working directory the agent ran in.")
    title: str | None = Field(default=None, description="the session's title, if it had one.")
    note: str | None = Field(default=None, description="what was decided here — the anchor's own memory.")
    created: datetime = Field(default_factory=_utc_now, description="when the anchor was taken.")

    # ---- export / load ---- #

    def to_dict(self) -> dict[str, Any]:
        """The anchor as plain data — ISO timestamp, no objects."""
        data = self.model_dump(exclude_none=True)
        created = data.get("created")
        if isinstance(created, datetime):
            data["created"] = created.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DshSessionAnchor:
        """Read an anchor back from plain data. Unknown keys are kept, not rejected: an anchor
        written by a newer version should still load."""
        return cls.model_validate(data)

    def dump(self, path: str | Path) -> Path:
        """Write the anchor to one YAML file and return where it landed.

        A path ending in a directory separator, or an existing directory, gets the default
        filename derived from the coordinate; anything else is used as the file path.
        """
        target = Path(path)
        if target.is_dir() or str(path).endswith(("/", "\\")):
            target = target / f"{self.session_id}-{self.seq if self.seq is not None else 'start'}{ANCHOR_SUFFIX}"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            yaml.safe_dump(self.to_dict(), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        return target

    @classmethod
    def from_ref(
        cls,
        ref: DshSessionRef,
        *,
        at: Literal["start", "end"] = "end",
        note: str | None = None,
    ) -> DshSessionAnchor:
        """Take an anchor from one end of a ref's range.

        A ref is a *span* — it says which stretch of turns something covered, which is what
        reading that stretch back needs. An anchor is a *point*. `at="end"` is the default
        because a range is usually recorded after the fact, when the thing it covers finished;
        `at="start"` anchors where the range opened, which is the point a reader coming back to
        "what did we know before this" wants.
        """
        if at == "end":
            seq, turn = ref.end_seq, ref.end_turn
        elif at == "start":
            seq, turn = ref.start_seq, ref.start_turn
        else:
            raise ValueError(f"at must be 'start' or 'end', got {at!r}")
        return cls(
            session_id=ref.session_id,
            seq=seq,
            turn=turn,
            agent_preset=ref.preset,
            cwd=ref.cwd,
            title=ref.title,
            created=ref.created,
            note=note,
        )

    @classmethod
    def from_file(cls, path: str | Path) -> DshSessionAnchor:
        """Load an anchor from a file written by `dump()` — or written by hand, which is the
        point of keeping the format flat."""
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"anchor file is not a mapping: {path}")
        return cls.from_dict(data)

    def describe(self) -> str:
        """One line, for a log, a prompt, or a person: what point this is."""
        where = f"{self.session_id}@{self.seq if self.seq is not None else 'start'}"
        extras = [f"turn {self.turn}" if self.turn is not None else "",
                  self.model or "",
                  self.workspace_path or self.workspace_id or "",
                  self.note or ""]
        tail = ", ".join(part for part in extras if part)
        return f"{where} ({tail})" if tail else where
