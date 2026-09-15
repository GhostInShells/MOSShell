"""Axis-1 data structures for the file-editor thread node.

Pure dataclasses + pure functions — no IO, no MOSS deps, unit-testable in
isolation. The interaction is a :class:`Thread`: a dialogue line bound to one
readable text file. Both sides (``g`` = ghost/model, ``u`` = user/human) append
actions; mutations apply in FIFO order per thread, and each confirmed mutation
appends a :class:`Version` to a linear, append-only chain.

Rules encoded here:

- ``reference`` has no side effect — it never advances the version chain.
- rejecting an action cascades to every later action on the same thread
  (they were computed against a state that no longer holds).
- ``rewind`` is an ordinary action whose result content equals an older
  version's content; the chain stays linear, no branching.
"""

from __future__ import annotations

import difflib
import time
from dataclasses import dataclass, field
from typing import Callable, Literal

Author = Literal["g", "u"]
Kind = Literal["reference", "write", "str_replace", "insert", "rewind", "export"]
ActionState = Literal["streaming", "tailed", "cancelled", "error"]
Verdict = Literal["pending", "confirmed", "rejected"]
Anchor = Literal["intent", "effect"]

#: kinds that advance the version chain when confirmed.
MUTATING = ("write", "str_replace", "insert", "rewind")


@dataclass(frozen=True, order=True)
class Seq:
    """An action's position on a thread's FIFO line."""

    thread_id: str
    n: int


@dataclass
class Effect:
    """The change an action produces: before / after full texts + unified diff."""

    before: str
    after: str
    diff: str


@dataclass
class Version:
    """One state of the thread's file, produced by a confirmed mutation.

    ``v0`` is the loaded baseline (``action_seq`` is ``None``).
    """

    id: str
    thread_id: str
    parent: str | None
    action_seq: Seq | None
    content: str
    effect: Effect


@dataclass
class Reply:
    """A dialogue entry under an action."""

    n: int
    author: Author
    anchor: Anchor
    diff: str | None = None
    text: str = ""
    verdict: Verdict | None = None
    at: float = field(default_factory=time.time)


@dataclass
class Action:
    """One item on a thread's line — an utterance or a mutation."""

    seq: Seq
    author: Author
    kind: Kind
    description: str
    payload: str
    from_version: str | None = None
    state: ActionState = "tailed"
    effect: Effect | None = None
    verdict: Verdict = "pending"
    replies: list[Reply] = field(default_factory=list)
    at: float = field(default_factory=time.time)


@dataclass
class Thread:
    """A dialogue line bound to one readable text file."""

    id: str
    label: str
    motivation: str = ""
    path: str | None = None
    versions: list[Version] = field(default_factory=list)
    order: list[Seq] = field(default_factory=list)
    head: str | None = None
    state: Literal["open", "closed"] = "open"
    created_at: float = field(default_factory=time.time)

    @property
    def head_version(self) -> Version | None:
        return self.versions[-1] if self.versions else None


# -- pure functions --


def diff_of(
    before: str,
    after: str,
    from_label: str = "before",
    to_label: str = "after",
) -> str:
    a = before.splitlines(keepends=True)
    b = after.splitlines(keepends=True)
    return "".join(difflib.unified_diff(a, b, fromfile=from_label, tofile=to_label))


def is_mutating(kind: Kind) -> bool:
    return kind in MUTATING


def effect_of(
    before: str,
    after: str,
    from_label: str = "before",
    to_label: str = "after",
) -> Effect:
    return Effect(before=before, after=after, diff=diff_of(before, after, from_label, to_label))


def result_content(
    action: Action,
    base_content: str,
    resolve_version: Callable[[str], str],
) -> str:
    """The full content an action yields.

    - mutation kinds carry the result in ``payload`` (already full text);
    - ``rewind``'s ``payload`` is a version id, resolved via ``resolve_version``;
    - non-mutating kinds (``reference`` / ``export``) leave content unchanged.
    """
    if action.kind == "rewind":
        return resolve_version(action.payload)
    if is_mutating(action.kind):
        return action.payload
    return base_content


def cascade_seqs(thread: Thread, seq: Seq) -> list[Seq]:
    """``seq`` plus every later action on the same thread."""
    return [s for s in thread.order if s.n >= seq.n]
