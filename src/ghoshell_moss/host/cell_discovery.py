"""Matrix cell discovery over Zenoh — queryable-based.

Key space: ``MOSS/{session_scope}/cells/{address}`` — per-cell queryable
           ``MOSS/{session_scope}/cells/query``   — host query portal

Each cell declares a queryable at its address key.  The host cell
additionally declares the query portal, whose handler returns cached
cell data (live ``get`` inside a queryable handler is disallowed by Zenoh).
Live queries happen through :meth:`query_cells`, called by MatrixImpl
via ``asyncio.to_thread`` outside any handler context.

Plugs into MatrixImpl as a composable module.  Matrix owns the cell
registry (dict[str, Cell]); CellDiscovery owns the Zenoh machinery:
key expressions, queryable declare/undeclare, wildcard aggregation.

Pattern reference: ``FractalKeyExpressions`` in ``host/fractal/_base.py``.
"""

from ghoshell_moss.depends import depend_zenoh

depend_zenoh()

import zenoh
import contextlib
import json
from typing import Callable

__all__ = ["CellDiscovery"]


class CellDiscovery:
    """Zenoh queryable-based cell discovery.

    Usage in MatrixImpl::

        self._cell_discovery = CellDiscovery(session_scope)

        # in _session_communication_bus_ctx_manager:
        zenoh_session = self._container.force_fetch(zenoh.Session)
        self._exit_stack.enter_context(zenoh_session)
        self._exit_stack.enter_context(
            self._cell_discovery.announce_cell(
                zenoh_session, self._this_cell.address, cell_info,
            )
        )
        if self._is_main:
            self._exit_stack.enter_context(
                self._cell_discovery.serve_query_portal(zenoh_session)
            )
    """

    def __init__(self, session_scope: str):
        self._session_scope = session_scope

    # -- key expressions ------------------------------------------------ #

    def cell_prefix(self) -> str:
        """``MOSS/{scope}/cells`` — all cell keys share this prefix."""
        return f"MOSS/{self._session_scope}/cells"

    def cell_key(self, address: str) -> str:
        """``MOSS/{scope}/cells/{address}`` — single cell queryable key."""
        return "/".join([self.cell_prefix(), address])

    def query_portal_key(self) -> str:
        """``MOSS/{scope}/cells/query`` — host query portal."""
        return "/".join([self.cell_prefix(), "query"])

    # -- announce this cell --------------------------------------------- #

    @contextlib.contextmanager
    def announce_cell(self, session: "zenoh.Session", address: str, cell_info: dict):
        """Declare this cell's queryable.  Undeclares on exit.

        *cell_info* must contain at least ``"address"`` so the portal
        can index replies.
        """
        key = self.cell_key(address)

        def _handler(query: zenoh.Query):
            query.reply(query.key_expr, json.dumps(cell_info))

        q = session.declare_queryable(key, _handler)
        try:
            yield
        finally:
            q.undeclare()

    # -- serve query portal (host only) --------------------------------- #

    @contextlib.contextmanager
    def serve_query_portal(
            self,
            session: "zenoh.Session",
            cells_provider: "Callable[[], dict[str, dict]]",
    ):
        """Declare the host query portal.  Undeclares on exit.

        The portal handler returns cached cell data from *cells_provider*
        instead of doing a live ``session.get`` — Zenoh disallows nested
        ``get`` inside a queryable handler on the same session.

        Live queries go through :meth:`query_cells`, called by MatrixImpl
        via ``asyncio.to_thread`` outside any handler context.
        """

        def _handler(query: zenoh.Query):
            cells = cells_provider()
            query.reply(query.key_expr, json.dumps(cells))

        q = session.declare_queryable(self.query_portal_key(), _handler)
        try:
            yield
        finally:
            q.undeclare()

    # -- query (blocking — call via asyncio.to_thread) ------------------ #

    def _query_all_cells(self, session: "zenoh.Session") -> dict[str, dict]:
        """Wildcard get all per-cell queryables.  Returns {address: info}."""
        replies = session.get(
            f"{self.cell_prefix()}/**",
            target=zenoh.QueryTarget.ALL,
            consolidation=zenoh.QueryConsolidation(zenoh.ConsolidationMode.NONE),
        )
        result: dict[str, dict] = {}
        for r in replies:
            if r.ok is not None:
                info = json.loads(r.ok.payload.to_string())
                addr = info.get("address")
                if addr:
                    result[addr] = info
        return result

    def query_cells(self, session: "zenoh.Session") -> dict[str, dict]:
        """Public entry point for network cell query (blocking).

        Caller is responsible for running this via ``asyncio.to_thread``
        to avoid blocking the event loop.
        """
        return self._query_all_cells(session)
