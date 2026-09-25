"""Consumer side of the web view kind — keep every live view and its condition.

Connection is three steps, and none of them is optional::

    discovery  who exists    — liveness; static identity rides in the meta
    snapshot   what state    — pulled once per view
    stream     what changes  — followed from then on

The snapshot is not a convenience.  ``pub`` is not retained: a view that
published before this consumer declared its subscription is invisible to the
stream, so without the pull a screen that starts late would render zeros until
the next thing happened.  Subscribing first and pulling second keeps the reverse
race (a change published while the pull is in flight) from being lost.

The client itself holds no policy.  It reports; the consumer decides where things
go, and whether to hold an order steady while a hand is on it.
"""

import inspect
from dataclasses import dataclass
from typing import Awaitable, Callable

from typing_extensions import Self

from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.core.blueprint.service import (
    Handle,
    Sample,
    ServiceClient,
    ServiceDeclaration,
    ServiceMeta,
    ServiceOperator,
)

from .declaration import ACTIVATE_KEY, KIND, STATE_KEY, WebViewDeclaration, WebViewState

__all__ = ['WebViewClient', 'WebViewItem']


@dataclass(frozen=True)
class WebViewItem:
    """One live web view as a consumer sees it."""

    meta: ServiceMeta
    """Raw discovery identity — kept so the view stays addressable."""

    declaration: WebViewDeclaration
    state: WebViewState

    @property
    def address(self) -> str:
        return self.meta['address']


class _CallbackHandle(Handle):
    """Close handle for a consumer-side callback registration.

    Nothing travels behind it — the operator's handles close transport entities,
    this one just deregisters a local function.
    """

    def __init__(self, key: str, close_fn: Callable[[], None]):
        self._key = key
        self._close_fn = close_fn
        self._closed = False

    @property
    def key(self) -> str:
        return self._key

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._close_fn()


class WebViewClient(ServiceClient):
    """Tracks the live web views and their conditions, in discovery order of events."""

    def __init__(self, operator_factory: Callable[[], Awaitable[ServiceOperator]]):
        self._operator_factory = operator_factory
        self._operator: ServiceOperator | None = None
        self._items: dict[str, WebViewItem] = {}
        self._change_callbacks: list[Callable[[], Awaitable[None] | None]] = []
        self._handles: list[Handle] = []

    # -- ServiceClient interface -----------------------------------------

    @classmethod
    def new(cls, matrix: Matrix) -> Self:
        return cls(lambda: matrix.service_operator())

    @classmethod
    def from_operator(cls, operator: ServiceOperator) -> Self:
        """Construct against an already-built operator (test seam)."""

        async def _operator() -> ServiceOperator:
            return operator

        return cls(_operator)

    async def get_connected(self) -> list[ServiceDeclaration]:
        """ABC accessor.  Consumers want ``items()`` — that is where addresses live."""
        return [item.declaration for item in self.items()]

    async def __aenter__(self) -> Self:
        operator = await self._operator_factory()
        self._operator = operator

        # Subscribe before pulling (see the module docstring), and subscribe to
        # the wildcard: one subscription covers every view, present and future.
        self._handles.append(operator.on_service_start(KIND, self._on_start))
        self._handles.append(operator.on_service_stop(KIND, self._on_stop))
        self._handles.append(operator.sub(KIND, STATE_KEY, self._on_state))

        for meta in await operator.get_services_by_kind(KIND):
            await self._adopt(meta)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        for handle in self._handles:
            handle.close()
        self._handles.clear()
        self._change_callbacks.clear()
        self._items.clear()
        self._operator = None

    # -- reading ---------------------------------------------------------

    def items(self) -> list[WebViewItem]:
        """Live views, most recently active first.

        This ordering is a default, not a policy — it is what a variable region
        (a sidebar, a list) wants.  A consumer that owns its layout is free to
        reorder, including holding the order still while a hand is on it.
        """
        return sorted(
            self._items.values(),
            key=lambda item: (-item.state.last_activity_at, item.declaration.created),
        )

    def on_change(self, callback: Callable[[], Awaitable[None] | None]) -> Handle:
        """Run ``callback`` after anything changes, for view or condition.

        The callback takes no arguments and re-reads ``items()`` — it can then
        never render a list that is already superseded.  Sync callbacks run
        inline on the loop; async ones are awaited.
        """
        def _close() -> None:
            try:
                self._change_callbacks.remove(callback)
            except ValueError:
                pass

        handle = _CallbackHandle('webview/change', _close)
        self._change_callbacks.append(callback)
        return handle

    # -- acting ----------------------------------------------------------

    async def activate(self, address: str) -> WebViewState | None:
        """Ask a view to take focus; returns the condition the server settled on.

        Consumers act optimistically and let this settle them: the reply is
        authoritative for the caller, and every other consumer converges through
        the published condition.
        """
        item = self._items.get(address)
        operator = self._operator
        if item is None or operator is None:
            return None
        replies = await operator.get(KIND, ACTIVATE_KEY, None, item.meta)
        if not replies:
            return None
        state = WebViewState.decode(replies[0]['payload'])
        self._apply(address, state)
        return state

    # -- discovery events ------------------------------------------------

    async def _on_start(self, meta: ServiceMeta) -> None:
        await self._adopt(meta)

    async def _on_stop(self, meta: ServiceMeta) -> None:
        address = meta['address']
        if self._items.pop(address, None) is None:
            return
        await self._notify_change()

    async def _on_state(self, sample: Sample) -> None:
        address = sample['address']
        if address not in self._items:
            return  # a condition from a view not yet discovered; the pull will catch it
        if self._apply(address, WebViewState.decode(sample['payload'])):
            await self._notify_change()

    # -- internals -------------------------------------------------------

    async def _adopt(self, meta: ServiceMeta) -> None:
        declaration = WebViewDeclaration.from_meta(meta)
        if declaration is None:
            return
        address = meta['address']
        # Placeholder condition until the pull lands: born active, so the view
        # has a defined (not arbitrary) place in a recency-ordered region.
        placeholder = WebViewState(last_activity_at=declaration.created)
        current = self._items.get(address)
        if current is None:
            self._items[address] = WebViewItem(meta, declaration, placeholder)

        snapshot = await self._fetch_state(meta)
        if snapshot is not None and self._apply(address, snapshot):
            await self._notify_change()
        elif current is None:
            await self._notify_change()

    async def _fetch_state(self, meta: ServiceMeta) -> WebViewState | None:
        """Pull one view's condition; ``None`` when the view does not answer."""
        operator = self._operator
        if operator is None:
            return None
        replies = await operator.get(KIND, STATE_KEY, None, meta)
        if not replies:
            return None
        return WebViewState.decode(replies[0]['payload'])

    def _apply(self, address: str, state: WebViewState) -> bool:
        """Merge one observation of a condition; True if it changed anything.

        Observations arrive from three places — the initial pull, an activate
        reply, the change stream — and their arrival order says nothing about
        which is newer.  Every mutation on the serve side advances
        ``last_activity_at``, so that is the ordering used here, and an
        observation older than what is already held is discarded.
        """
        item = self._items.get(address)
        if item is None:
            return False
        if state.last_activity_at < item.state.last_activity_at:
            return False
        self._items[address] = WebViewItem(item.meta, item.declaration, state)
        return True

    async def _notify_change(self) -> None:
        for callback in list(self._change_callbacks):
            result = callback()
            if inspect.isawaitable(result):
                await result
