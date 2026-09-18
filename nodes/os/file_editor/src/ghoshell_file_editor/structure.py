"""Axis-1 data structures for the file-editor node.

Pure dataclasses + pure functions — no IO, no MOSS deps, unit-testable in
isolation.

A **thread** is an editable object: a working copy of some text, plus the line
of **actions** both sides appended to it. Every action is also a card on the
human surface, and ``n`` — the action's sequence number on its thread — is the
shared coordinate for all three tabs (effect / full / history) and for
``rewind``.

The one rule that shapes everything here: **an action's effect is computed when
it is appended, from the content that existed at that moment.** Nothing waits on
a human decision. Edits land in memory immediately; the only action that ever
asks permission is ``export``, because landing on disk is the only real side
effect. That is why ``Effect`` is a field of ``Action`` and why the content of a
thread is a derivation (``Thread.content``) rather than a stored value — a
rewind is just another action whose effect happens to point backwards.
"""

from __future__ import annotations

import difflib
import time
from dataclasses import dataclass, field
from typing import Literal

Author = Literal["g", "u"]
"""``g`` = ghost (the model), ``u`` = the human."""

Kind = Literal[
    "read", "write", "str_replace", "append", "rewind", "export"
]
"""What an action does. ``read`` changes nothing; ``export`` is the only kind
that reaches the disk."""

ActionState = Literal[
    "streaming", "applied", "awaiting", "written", "rejected", "failed",
    "cancelled",
]
"""``streaming`` while the model is still writing the payload; ``applied`` once
it has landed in memory; ``awaiting`` only for an export proposal; then
``written`` / ``rejected`` / ``failed``; ``cancelled`` if the model withdrew it
mid-stream."""

ThreadState = Literal["live", "exported", "closed"]
"""``live`` while it can still be edited; ``exported`` once its text landed on
disk (the final chapter — but the line stays traceable); ``closed`` when
abandoned."""

STREAMING_KINDS: frozenset[str] = frozenset({"write", "append"})
"""The kinds whose payload the model writes token by token."""

SIDE_EFFECTS: dict[str, str] = {
    "read": "none — read only",
    "write": "in-memory only — working copy replaced",
    "str_replace": "in-memory only — working copy edited",
    "append": "in-memory only — working copy appended",
    "rewind": "in-memory only — working copy rolled back",
    "export": "disk — writes the file",
}
"""The mechanical verdict shown on every card: what this kind does to the world.
One string per kind, derived from nothing — the point is that the human can
classify an action without reading its content."""


@dataclass
class Effect:
    """The change an action carried: before / after full texts + unified diff."""

    before: str
    after: str
    diff: str


@dataclass
class Dialogue:
    """One line of conversation on a card, appended by either side.

    ``ask`` is the human's whole vocabulary for a card that needs no verdict:
    they talk, the model answers by acting.
    """

    author: Author
    text: str
    at: float = field(default_factory=time.time)


@dataclass
class Action:
    """One action on a thread — and one card on the human surface.

    ``text`` is the markdown source the effect tab renders; ``payload`` is the
    machine-readable input the kind needs to be replayed or explained (the JSON
    ops of a ``str_replace``, the target of a ``rewind``, the path of an
    ``export``).
    """

    n: int
    kind: Kind
    author: Author
    label: str = ""
    """The human-facing name of what this action did — the card's title."""

    state: ActionState = "applied"
    text: str = ""
    payload: str = ""
    effect: Effect | None = None
    dialogue: list[Dialogue] = field(default_factory=list)
    at: float = field(default_factory=time.time)


@dataclass
class Thread:
    """An editable object: a working copy plus the line of actions on it.

    ``path`` is the file this thread exports to by default; it may be None for a
    thread that was never bound to a file (an editable object does not have to
    come from a document). ``draft`` is the working copy's name under the
    drafts directory — the crash net.
    """

    id: str
    label: str
    path: str | None = None
    state: ThreadState = "live"
    auto: bool = False
    """Per-thread trust: exports to the established target need no approval.

    Only meaningful when ``path`` is set — auto without a final target would be
    trust without an object, so a pathless thread can never be auto."""
    draft: str = ""
    base: str = ""
    """The text at open time — what v0 of the line holds."""

    exported_to: str = ""
    actions: list[Action] = field(default_factory=list)
    created: float = field(default_factory=time.time)

    @property
    def head(self) -> Action | None:
        """The last action that moved the text. Versions are not objects."""
        for action in reversed(self.actions):
            if action.effect is not None:
                return action
        return None

    @property
    def content(self) -> str:
        """The working copy's current text: the head's result, or the baseline."""
        head = self.head
        return self.base if head is None else head.effect.after

    @property
    def version(self) -> int:
        """The coordinates of the current text — 0 = the baseline."""
        head = self.head
        return 0 if head is None else head.n

    @property
    def live(self) -> bool:
        return self.state == "live"

    def get(self, n: int) -> Action | None:
        for action in self.actions:
            if action.n == n:
                return action
        return None


# -- pure functions --


def diff_of(
    before: str,
    after: str,
    from_label: str = "before",
    to_label: str = "after",
) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=from_label,
            tofile=to_label,
        )
    )


def effect_of(
    before: str,
    after: str,
    from_label: str = "before",
    to_label: str = "after",
) -> Effect:
    return Effect(
        before=before,
        after=after,
        diff=diff_of(before, after, from_label, to_label),
    )


def replace_once(text: str, old: str, new: str) -> str:
    """Swap the single occurrence of ``old`` for ``new``.

    Exactly once: an edit that matches nowhere or matches twice is a mistake in
    the proposal, not something to guess at. This is the str_replace protocol's
    own rule, kept here so the store never has to validate prose.
    """
    if not old:
        raise ValueError("str_replace needs a non-empty old_str")
    found = text.count(old)
    if found == 0:
        raise ValueError("old_str does not appear in the text")
    if found > 1:
        raise ValueError(
            f"old_str appears {found} times — extend it with surrounding context "
            f"so it is unique"
        )
    return text.replace(old, new, 1)


def content_at(thread: Thread, n: int) -> str:
    """The thread's text after action ``n`` (the baseline if nothing moved by then).

    This is what the "full" tab shows when you click a card: the document as
    that action left it.
    """
    for action in reversed(thread.actions):
        if action.n <= n and action.effect is not None:
            return action.effect.after
    return thread.base


def render_source(action: Action) -> str:
    """The markdown the effect tab renders for this action.

    What reads as "the change" differs by kind: a read shows the fragment it
    read, an append shows the segment it wrote, a write or a rewind shows the
    whole resulting text (it moved everything), an export shows what would land
    on disk.
    """
    if action.kind in ("write", "rewind"):
        return action.effect.after if action.effect is not None else ""
    return action.text


def side_effect(action: Action) -> str:
    """The mechanical effect line for a card. See :data:`SIDE_EFFECTS`."""
    base = SIDE_EFFECTS[action.kind]
    if action.kind == "export" and action.payload:
        return f"disk — writes {action.payload}"
    return base


def line_count(text: str) -> int:
    return len(text.splitlines())


def slice_region(text: str, region: str) -> str:
    """Resolve a ``"start-end"`` line spec (1-based, inclusive) against ``text``.

    An empty region means the whole text. Raises ValueError on a malformed spec
    or one that falls outside the text.
    """
    if not region.strip():
        return text
    spec = region.strip()
    if "-" in spec:
        head, _, tail = spec.partition("-")
        start_s, end_s = head, tail
    else:
        start_s = end_s = spec
    try:
        start = int(start_s)
        end = int(end_s) if end_s.strip() else start
    except ValueError:
        raise ValueError(f"region must look like 'start-end', got {region!r}")
    lines = text.splitlines(keepends=True)
    if start < 1 or end < start or end > len(lines):
        raise ValueError(
            f"region {region!r} is outside the text ({len(lines)} lines)"
        )
    return "".join(lines[start - 1 : end])
