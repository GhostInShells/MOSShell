"""Web view kind — identity, condition, and the business keys between them.

A *web view* is a cell that serves a page for a human and tells the mesh where it
lives.  The kind carries nothing else: an identity plus a small mutable condition.
What to do with them is the consumer's business.

The two lifetimes are kept apart for a structural reason, not a stylistic one.
``ZenohServiceTerminal`` turns the declaration into the service meta once, when
the provider is created, and serves that snapshot to every discoverer forever —
anything runtime-varying put in a declaration would be stale for its whole life.

What the condition is, and what it deliberately is not:

* **The server records facts; the consumer owns policy.**  A cell sees only its
  own condition — no component holds a global view of the mesh — so "what belongs
  in front" is not answerable server-side and is not attempted here.  Ordering,
  region and focus belong to whatever renders: a screen, a list, an orb.

* **No priority.**  A self-claimed scalar rank is the obvious first idea and it is
  rejected.  With no arbiter, every node inflates to the top and the scalar stops
  discriminating; a rank also conflates standing placement with recent activity
  with unread; and it compels a node to understand the consumer's layout.
  Recency replaces it — time is a self-limiting arbiter: only the most recent
  claim ranks first, and it cannot be claimed without actually being active.

* **Ordering and focus stay apart.**  ``last_activity_at`` moves a view within a
  variable region; ``focused_at`` marks what someone is actually looking at.  A
  notification bumps the first without stealing the second — an IM conversation
  rises in the list without opening itself under your hand.

* **``touch`` and ``notify`` are the two self-claims.**  "I just did something"
  (activity, no badge) and "there is something for you" (activity plus unread).
  Both are statements of fact about the view, never a message to a consumer.
  They are method calls on the serve side, not wire messages: only the *result*
  travels, and only once.
"""

import time
from pydantic import BaseModel, Field

from ghoshell_moss.core.blueprint.service import ServiceDeclaration

__all__ = [
    'KIND', 'STATE_KEY', 'ACTIVATE_KEY',
    'WebViewDeclaration', 'WebViewState',
]

KIND = 'webview'

STATE_KEY = 'state'
"""Business key of the condition: a one-shot snapshot plus a change stream."""

ACTIVATE_KEY = 'activate'
"""Business key of the focus request (request-reply)."""

# ``meta`` is reserved by the operator, which auto-registers the discovery
# queryable.  ``state`` and ``activate`` are the whole business surface: the
# serve side reaches the mesh through nothing else.


class WebViewDeclaration(ServiceDeclaration):
    """Identity of a web view.  Frozen into the meta at provide() time."""

    url: str
    """Where the page is served.  The capability itself."""

    title: str
    """Human-facing name."""

    description: str = ""
    """Human-facing explanation."""

    icon: str | None = None
    """Rendering hint for consumers that draw a glyph rather than text."""

    created: float = Field(default_factory=time.time)
    """When the view came up.

    Not a stable identity across restarts — ``address`` is.  This is the birth
    stamp a variable region falls back on when nothing has happened yet, and the
    final tie-break in ordering.
    """

    @classmethod
    def kind(cls) -> str:
        return KIND


class WebViewState(BaseModel):
    """Condition of a web view.  Replaced wholesale, never merged in place."""

    last_activity_at: float = Field(default_factory=time.time)
    """Most recent activity — the ordering key of a variable region."""

    focused_at: float | None = None
    """Most recent focus.

    ``None`` until someone activates it.  Focus is the maximum across views, so
    there is no way to release it: a later activation simply outranks.
    """

    unread_count: int = 0
    """How much has arrived since anyone last looked at this view."""

    last_message: str | None = None
    """The most recent thing said about this view, for tooltips.

    Survives being read: having seen it does not erase what it was.
    """

    def encode(self) -> bytes:
        return self.model_dump_json().encode()

    @classmethod
    def decode(cls, payload: bytes) -> 'WebViewState':
        """Inverse of ``encode``.

        Snapshots, replies and stream samples all travel in this one encoding, so
        a consumer parses exactly one format.
        """
        return cls.model_validate_json(payload)
