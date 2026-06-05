"""Matrix cell liveness discovery over Zenoh.

Key space: ``MOSS/{session_scope}/cell/liveness/{address}``

Plugs into MatrixImpl as a composable module.  Matrix owns the cell
registry (dict[str, Cell] + dict[str, threading.Event]); CellDiscovery
owns the Zenoh liveness machinery: key expressions, token declare,
subscriber management, initial wildcard query.

Pattern reference: ``FractalKeyExpressions`` in ``host/fractal/_base.py``.
"""

from ghoshell_moss.depends import depend_zenoh

depend_zenoh()

import zenoh
import contextlib
import threading

__all__ = ["CellDiscovery"]


class CellDiscovery:
    """Zenoh liveness discovery for Matrix cells.

    Usage in MatrixImpl::

        self._cell_discovery = CellDiscovery(session_scope)

        # in _session_communication_bus_ctx_manager:
        zenoh_session = self._container.force_fetch(zenoh.Session)
        self._exit_stack.enter_context(zenoh_session)
        self._exit_stack.enter_context(
            self._cell_discovery.discover_cells(
                zenoh_session, self._cells, self._cell_alive_events,
                this_address=self._this_cell.address,
            )
        )
        self._exit_stack.enter_context(
            self._cell_discovery.declare_this_cell(
                zenoh_session, self._this_cell.address,
            )
        )
    """

    def __init__(self, session_scope: str):
        self._session_scope = session_scope

    # -- key expressions ------------------------------------------------ #

    def liveness_prefix(self) -> str:
        """``MOSS/{scope}/cell/liveness`` — all cell liveness keys share this."""
        return f"MOSS/{self._session_scope}/cell/liveness"

    def liveness_key(self, address: str) -> str:
        """``MOSS/{scope}/cell/liveness/{address}`` — single cell liveness key."""
        return "/".join([self.liveness_prefix(), address])

    def liveness_wildcard(self) -> str:
        """``MOSS/{scope}/cell/liveness/**`` — wildcard for initial query."""
        return "/".join([self.liveness_prefix(), "**"])

    # -- declare this cell ---------------------------------------------- #

    @contextlib.contextmanager
    def declare_this_cell(self, session: "zenoh.Session", address: str):
        """Declare this cell's liveness token.  Undeclares on exit."""
        key = self.liveness_key(address)
        token = session.liveliness().declare_token(key)
        try:
            yield
        finally:
            token.undeclare()

    # -- discover known cells ------------------------------------------- #

    @contextlib.contextmanager
    def discover_cells(
            self,
            session: "zenoh.Session",
            cells: dict[str, "Cell"],
            alive_events: dict[str, threading.Event],
            *,
            this_address: str,
    ):
        """Subscribe to liveness of every known cell + run initial wildcard query.

        Skips *this_address* (self).  On exit, undeclares all subscribers.
        """
        if session.is_closed():
            raise RuntimeError("zenoh session closed")

        subscribers: list[zenoh.Subscriber] = []
        for address in cells:
            if address == this_address:
                alive_events[this_address].set()
                continue
            sub = self._register_listener(session, address, alive_events[address])
            subscribers.append(sub)

        self._query_initial(session, alive_events)
        try:
            yield
        finally:
            for sub in subscribers:
                if not session.is_closed():
                    sub.undeclare()

    def _register_listener(
            self,
            session: "zenoh.Session",
            address: str,
            event: threading.Event,
    ) -> zenoh.Subscriber:
        key = self.liveness_key(address)

        def _on_sample(sample: zenoh.Sample) -> None:
            if sample.kind == zenoh.SampleKind.PUT:
                event.set()
            else:
                event.clear()

        return session.liveliness().declare_subscriber(key, _on_sample)

    def _query_initial(
            self,
            session: "zenoh.Session",
            alive_events: dict[str, threading.Event],
    ) -> None:
        """Wildcard query all existing liveness tokens; set matching events."""
        prefix = self.liveness_prefix()
        for sample in session.liveliness().get(self.liveness_wildcard()):
            key = str(sample.result.key_expr)
            if not key.startswith(prefix):
                continue
            address = key[len(prefix) + 1:]
            if address in alive_events:
                alive_events[address].set()
