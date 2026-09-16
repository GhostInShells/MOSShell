"""Axis-1 data structures for the file-editor thread node.

Pure dataclasses + pure functions — no IO, no MOSS deps, unit-testable in
isolation.

An interaction is a :class:`Thread`: a dialogue line bound to one readable text
file. Both sides (``g`` = ghost/model, ``u`` = user/human) append
:class:`Action` objects, and together those actions *are* the append-only log.
Every action carries the :class:`Effect` it would produce, computed when it is
appended — never when it is confirmed.

Rules encoded here:

- **Actions are self-contained.** An effect follows from the action's own
  payload plus the line's tail, never from the moment a verdict is given.
  ``reference`` carries no effect; ``rewind`` resolves its target's content;
  ``export`` carries the content it would write.
- **``before`` is the pending tail**, not the last confirmed state — the last
  non-rejected action's ``after``. This is what lets an effect exist before any
  human decision, and it is why confirmation never has to compute anything.
- **The head is derived**, never stored: the last confirmed action that actually
  changed the content. So is the version list — a version is not an entity, its
  identity is the action's :class:`Seq`.
- **Reject cascades** to every later action on the thread (they were computed
  against a state that no longer holds).
"""

from __future__ import annotations

import difflib
import time
from dataclasses import dataclass, field
from typing import Literal

Author = Literal["g", "u"]
Kind = Literal["reference", "write", "str_replace", "insert", "rewind", "export"]
ActionState = Literal["streaming", "tailed", "cancelled", "error"]
Verdict = Literal["pending", "confirmed", "rejected"]
Anchor = Literal["intent", "effect"]

BASE = "base"
"""``rewind`` payload selecting the loaded baseline — no action produced it."""


@dataclass(frozen=True, order=True)
class Seq:
    """An action's position on a thread's line — the shared coordinate.

    Also the version identity: a version is a confirmed action, not an object.
    """

    thread_id: str
    n: int


@dataclass
class Effect:
    """The change an action carries: before / after full texts + unified diff."""

    before: str
    after: str
    diff: str


@dataclass
class Reply:
    """A dialogue entry under an action.

    Commentary only — a reply never decides anything. Whoever replies may go on
    to append an action of their own; that action, not the reply, changes the
    line.
    """

    n: int
    author: Author
    anchor: Anchor
    diff: str | None = None
    text: str = ""
    at: float = field(default_factory=time.time)


@dataclass
class Action:
    """One item on a thread's line — an utterance, a mutation, or a rewind."""

    seq: Seq
    author: Author
    kind: Kind
    description: str
    payload: str
    state: ActionState = "tailed"
    effect: Effect | None = None
    verdict: Verdict = "pending"
    verdict_by: Author | None = None
    replies: list[Reply] = field(default_factory=list)
    at: float = field(default_factory=time.time)


@dataclass
class Thread:
    """A dialogue line bound to one readable text file.

    ``actions`` maps ``n`` to :class:`Action` in append order. ``n`` is
    monotonic and never removed, so insertion order *is* line order — the log
    needs no separate index.
    """

    id: str
    label: str
    motivation: str = ""
    path: str | None = None
    base_content: str = ""
    actions: dict[int, Action] = field(default_factory=dict)
    state: Literal["open", "closed"] = "open"
    created_at: float = field(default_factory=time.time)

    @property
    def head(self) -> Action | None:
        """The last confirmed action that actually changed the content."""
        for action in reversed(self.actions.values()):
            if action.verdict == "confirmed" and changes_content(action):
                return action
        return None

    @property
    def content(self) -> str:
        """The line's current text: the head's result, or the baseline."""
        head = self.head
        return self.base_content if head is None else head.effect.after

    @property
    def versions(self) -> list[Action]:
        """Derived view: every confirmed action that changed the content."""
        return [
            a
            for a in self.actions.values()
            if a.verdict == "confirmed" and changes_content(a)
        ]


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


def effect_of(
    before: str,
    after: str,
    from_label: str = "before",
    to_label: str = "after",
) -> Effect:
    return Effect(before=before, after=after, diff=diff_of(before, after, from_label, to_label))


def changes_content(action: Action) -> bool:
    """Whether an action moves the thread's text.

    The one predicate behind both the head and the version list, so a no-op
    (``export``, a write of identical text, a rewind to where we already are)
    advances neither.
    """
    return action.effect is not None and action.effect.after != action.effect.before


def tail_content(thread: Thread) -> str:
    """The line's pending tail — the last non-rejected action's result.

    Deliberately not the last *confirmed* result: an action's effect has to
    exist before any human decision, so the basis is the whole non-rejected
    line, pending actions included.
    """
    for action in reversed(thread.actions.values()):
        if action.verdict != "rejected" and action.effect is not None:
            return action.effect.after
    return thread.base_content


def rewind_target(thread: Thread, payload: str) -> Action | None:
    """Resolve a ``rewind`` payload to the action it points at.

    ``None`` means the loaded baseline. The target may still be pending — going
    back to an earlier proposal inside the same burst is the ordinary case.
    """
    if payload == BASE:
        return None
    try:
        n = int(payload)
    except ValueError:
        raise ValueError(
            f"rewind payload must be {BASE!r} or a seq number, got {payload!r}"
        )
    target = thread.actions.get(n)
    if target is None:
        raise ValueError(f"no action {n} on thread {thread.id!r}")
    if target.verdict == "rejected":
        raise ValueError(f"action {n} on thread {thread.id!r} was rejected")
    if target.effect is None:
        raise ValueError(f"action {n} on thread {thread.id!r} carries no content")
    return target


def compute_effect(thread: Thread, kind: Kind, payload: str) -> Effect | None:
    """The effect an action of ``kind``/``payload`` would carry if appended now.

    Called at append time. The result depends on the line's tail, never on
    verdicts given later — which is what keeps confirmation a bookkeeping step.
    """
    if kind == "reference":
        return None
    before = tail_content(thread)
    if kind == "export":
        # Export writes the content it was authored against; the line itself
        # does not move. Its ``before`` is the text that would land on disk.
        return effect_of(before, before)
    if kind == "rewind":
        target = rewind_target(thread, payload)
        after = thread.base_content if target is None else target.effect.after
        return effect_of(before, after)
    return effect_of(before, payload)


def cascade_seqs(thread: Thread, n: int) -> list[Seq]:
    """``n`` plus every later action on the same thread."""
    return [a.seq for a in thread.actions.values() if a.seq.n >= n]
