"""CalendarSurface — one port serving both the page and the WebSocket.

The page is the human's half of the body: it is where the calendar is *seen*. It binds an
ephemeral port by default and reports the real address back through the channel's ``url``
notice, so two calendar instances never fight over a port.

Downlink frames describe state changes; the page renders them. Uplink frames are the
human editing — those are written to the store here and announced to the ghost by the
injected ``on_human_edit`` hook, so the ghost learns about a change the moment it happens
rather than by polling.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Callable

from websockets.asyncio.server import ServerConnection, serve

from .store import CalendarStore


class CalendarSurface:
    def __init__(
        self,
        store: CalendarStore,
        *,
        html_path: Path,
        on_human_edit: Callable[[str], None] | None = None,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        self._store = store
        self._html_path = html_path
        self._on_human_edit = on_human_edit
        self._host = host
        self.port = port
        self._clients: set[ServerConnection] = set()
        self._server: Any = None

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self.port}"

    # -- lifecycle --

    async def start(self) -> None:
        self._server = await serve(
            self._handler,
            self._host,
            self.port,
            process_request=self._process_request,
        )
        sockets = getattr(self._server, "sockets", None)
        if sockets:
            self.port = sockets[0].getsockname()[1]

    async def stop(self) -> None:
        if self._server is None:
            return
        self._server.close()
        await self._server.wait_closed()
        self._server = None

    # -- http --

    def _process_request(self, connection: ServerConnection, request: Any):
        path = request.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            try:
                html = self._html_path.read_text(encoding="utf-8")
            except OSError:
                return connection.respond(500, "index.html not found")
            response = connection.respond(200, html)
            # respond() defaults Content-Type to text/plain and __setitem__ appends, so the
            # default has to be deleted first or the page renders as source text.
            del response.headers["Content-Type"]
            response.headers["Content-Type"] = "text/html; charset=utf-8"
            return response
        if path == "/ws":
            return None
        return connection.respond(404, "not found")

    # -- websocket --

    async def _handler(self, connection: ServerConnection) -> None:
        self._clients.add(connection)
        try:
            await connection.send(_encode({"type": "state", **_snapshot(self._store)}))
            async for raw in connection:
                try:
                    frame = json.loads(raw)
                except (TypeError, ValueError):
                    continue
                try:
                    await self._apply(frame)
                except Exception as exc:  # keep one bad frame from dropping the socket
                    await connection.send(_encode({"type": "error", "error": str(exc)}))
        finally:
            self._clients.discard(connection)

    async def _apply(self, frame: dict[str, Any]) -> None:
        kind = frame.get("type")
        if kind == "event.add":
            event = self._store.add(
                title=str(frame.get("title") or "untitled"),
                start_ts=float(frame["start_ts"]),
                end_ts=_opt_float(frame.get("end_ts")),
                all_day=bool(frame.get("all_day")),
                notes=str(frame.get("notes") or ""),
                level=int(frame.get("level", 1)),
                remind_before=_opt_float(frame.get("remind_before")),
            )
            self._store.log_edit("human", f"added {event['title']}")
            await self.broadcast({"type": "event.upsert", "event": event})
            self._announce(f"human added {event['title']}")
        elif kind == "event.update":
            patch = {
                k: frame[k]
                for k in ("title", "start_ts", "end_ts", "all_day", "notes", "level", "remind_before")
                if k in frame
            }
            event = self._store.update(int(frame["id"]), **patch)
            if event is None:
                return
            self._store.log_edit("human", f"updated {event['title']}")
            await self.broadcast({"type": "event.upsert", "event": event})
            self._announce(f"human updated {event['title']}")
        elif kind == "event.remove":
            if self._store.remove(int(frame["id"])):
                self._store.log_edit("human", f"removed event {frame['id']}")
                await self.broadcast({"type": "event.remove", "id": int(frame["id"])})
                self._announce(f"human removed event {frame['id']}")
        elif kind == "event.done":
            event = self._store.set_done(int(frame["id"]), bool(frame.get("done", True)))
            if event is None:
                return
            self._store.log_edit("human", f"marked {event['title']} done")
            await self.broadcast({"type": "event.upsert", "event": event})
            self._announce(f"human marked {event['title']} done")

    def _announce(self, text: str) -> None:
        if self._on_human_edit is not None:
            self._on_human_edit(text)

    # -- broadcast --

    async def broadcast(self, frame: dict[str, Any]) -> None:
        if not self._clients:
            return
        payload = _encode(frame)
        await asyncio.gather(*(self._send(c, payload) for c in list(self._clients)))

    async def _send(self, connection: ServerConnection, payload: str) -> None:
        try:
            await connection.send(payload)
        except Exception:
            self._clients.discard(connection)


def _encode(frame: dict[str, Any]) -> str:
    return json.dumps(frame, ensure_ascii=False)


def _opt_float(value: Any) -> float | None:
    return None if value is None or value == "" else float(value)


def _snapshot(store: CalendarStore) -> dict[str, Any]:
    """Everything the page needs to render from cold: recent events plus the edit tail.

    Bounded on both ends — the page shows a month grid, and shipping the whole table to
    every new tab would grow without limit as the calendar ages.
    """
    horizon = store.now()
    events = store.between(horizon - 60 * 86400, horizon + 365 * 86400)
    return {"events": events, "edits": store.recent_edits(10)}
