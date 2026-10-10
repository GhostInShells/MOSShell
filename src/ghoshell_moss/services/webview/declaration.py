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

* **A face is a condition too, and it is a *line* plus a *still image address*.**
  ``face_line`` is what the tile says; ``face_art`` is a url the consumer draws,
  when it draws at all.  Both are conditions rather than identity for the same
  structural reason as everything else here: what a view looks like *right now*
  changes while the view lives, and a declaration is frozen at provide time.
  Consequences worth stating, because each one was a decision:

  - **Words first, pixels optional.**  Any cell can produce a line with no
    rendering ability whatsoever; an image needs one.  A consumer must therefore
    be able to draw a useful tile from ``face_line`` alone, and treat
    ``face_art`` as an upgrade, never a requirement.
  - **The image is an address, not a payload.**  Inlining bytes would put a
    picture on the wire every time a condition is published, and would force the
    producer to render even when nobody is looking.  A url lets the consumer
    decide when to fetch — lazily, only for a tile that is actually visible.
  - **Dynamism is the producer's business, not the protocol's.**  The contract
    promises *a still image*; it promises neither animation nor freshness.  A
    cell that wants its picture to change changes the address (or serves it
    uncached).  This is what keeps a wall of tiles from becoming N decoders:
    the cost the pool exists to avoid.
  - **A summary beats a screenshot** — text inside a thumbnail is unreadable at
    tile size, one line is not.  That is why the line is the primary face.
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

    face_line: str | None = None
    """One line about what this view is doing right now — what a tile says.

    A *condition*, not an identity: it changes as the thing behind the view
    changes, which is exactly why it cannot live in the declaration (frozen at
    provide time, stale for its whole life). Keep it short — a phrase someone
    reads at a glance, not a sentence they read.
    """

    face_art: str | None = None
    """A still image for the tile — a url, resolved by whoever renders.

    Deliberately **still**: an animated one turns a wall of quiet tiles into N
    decoders and N contenders for attention, which is the cost a pool exists to
    avoid. A summary beats a screenshot: text in a thumbnail is unreadable,
    one line is not.
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
