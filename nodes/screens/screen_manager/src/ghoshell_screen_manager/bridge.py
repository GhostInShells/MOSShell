"""Bridge: webview services on the mesh → items in the screen store.

The screen consumes the ``webview`` service kind (matrix-operator's reference
kind). It does not invent a protocol: ``WebViewClient`` already does discovery,
snapshot and stream, and reports change by callback. This module is the thin
translation from "the set of live views changed" to "materialize / release items".

Two rules the model relies on:

* **Adopt onto the desktop, never into a group.** Choosing where things go is the
  model's job; a discoverer that quietly arranged would be stealing that move.
* **A destroy is a tombstone.** When the model/human destroys an adopted window,
  the store records its address so the next change event does not resurrect it;
  the bridge only adopts what the store reports it lacks.

The bridge starts and stops with the screen node. If no webview provider ever
appears, it simply holds an empty set.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from ghoshell_moss.core.blueprint.cell import CellAddressCodec
from ghoshell_moss.services.webview import WebViewClient

from . import projection as P
from .model import ScreenModel

__all__ = ["WebViewBridge"]

FrameSink = Callable[[dict[str, Any]], Awaitable[None]]


def _item_id(address: str) -> str:
    """A stable, unique handle for a mesh address.

    Prefer the cell's short form (``name_uid[-6:]``) when the address is a strict
    ``role/name/uid``; fall back to a safe normalization for anything else. The
    mesh never guarantees the strict form on the wire, only that an address is
    unique.
    """
    try:
        return CellAddressCodec(address).short
    except ValueError:
        return CellAddressCodec.normalize(address)


class WebViewBridge:
    """Keeps the store in step with the live web views on the mesh.

    Construct with a connected ``WebViewClient`` (via ``matrix.connect_service``);
    call ``start`` to subscribe. Every change re-reads the client's ``items()`` and
    reconciles: new addresses materialize onto the desktop, gone addresses release.
    """

    def __init__(self, model: ScreenModel, client: WebViewClient, *, emit: FrameSink) -> None:
        self._model = model
        self._client = client
        self._emit = emit
        self._handle: Any = None

    async def start(self) -> None:
        """Subscribe to the client's changes. Idempotent."""
        if self._handle is not None:
            return
        self._handle = self._client.on_change(self._on_change)
        await self._on_change()

    def stop(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    # -- reconcile ---------------------------------------------------------

    async def _on_change(self) -> None:
        live = {item.address: item for item in self._client.items()}
        changed = False

        for address, item in live.items():
            if self._model.item_of_service(address) is not None:
                continue
            adopted = self._model.adopt(
                address,
                item.declaration.url,
                label=item.declaration.title,
                item_id=_item_id(address),
                icon=item.declaration.icon,
            )
            if adopted is not None:
                await self._emit(P.open_frame(adopted))
                changed = True

        known = self._model.adopted()
        for address in [a for a in known if a not in live]:
            item_id = known[address]
            if self._model.release(address) is not None:
                await self._emit(P.close_frame(item_id))
                changed = True

        if changed:
            await self._emit(P.state_frame(self._model))

    # -- notice ------------------------------------------------------------

    def notice(self) -> str:
        """A warm fragment: how many views are floating with nothing arranged."""
        floating = [
            i for i in self._model.desktop_items()
            if self._model.service_of(i)
        ]
        if not floating:
            return ""
        return f"{len(floating)} view(s) floating: {', '.join(floating)}"
